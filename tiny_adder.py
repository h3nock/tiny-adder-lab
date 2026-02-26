#!/usr/bin/env python3
"""
Tiny adder lab: one-file project.

Contains:
- tiny autoregressive transformer (causal attention required by AdderBoard)
- train/eval/predict CLI
- optional W&B sweep entrypoint
- AdderBoard submission hooks: build_model() and add()
"""

import argparse
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Task constants
# ----------------------------
NUM_DIGITS = 10
SUM_DIGITS = 11
MAX_ADDEND = 10**NUM_DIGITS - 1
MAX_OPERAND = 10**NUM_DIGITS

# Vocab: 0-9, +, =, <EOS>
TOKENS = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "+": 10,
    "=": 11,
    "<EOS>": 12,
}
VOCAB_SIZE = len(TOKENS)
EOS_ID = TOKENS["<EOS>"]

PROMPT_LEN = NUM_DIGITS + 1 + NUM_DIGITS + 1  # "aaaaaaaaaa+bbbbbbbbbb="
TARGET_LEN = SUM_DIGITS + 1  # reversed 11-digit sum + EOS
MODEL_INPUT_LEN = PROMPT_LEN + SUM_DIGITS  # teacher-forced LM length (33)

POW10_11 = torch.tensor([10**i for i in range(SUM_DIGITS)], dtype=torch.int64)
POW10_10_DESC = torch.tensor([10 ** (NUM_DIGITS - 1 - i) for i in range(NUM_DIGITS)], dtype=torch.int64)
MAX_BY_DIGITS = [10**d - 1 for d in range(NUM_DIGITS + 1)]
DEFAULT_PHASE1_RATIO = 2000.0 / 27000.0
DEFAULT_PHASE2_RATIO = 7000.0 / 27000.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pair_hash(a: int, b: int) -> int:
    return a * MAX_OPERAND + b


def preprocess_prompt(a: int, b: int) -> list[int]:
    if not (0 <= a <= MAX_ADDEND and 0 <= b <= MAX_ADDEND):
        raise ValueError(f"a and b must be in [0, {MAX_ADDEND}]")
    s = f"{a:010d}+{b:010d}="
    return [TOKENS[ch] for ch in s]


def decode_generated_sum(tokens: list[int]) -> int:
    digits: list[str] = []
    for t in tokens:
        tid = int(t)
        if tid == EOS_ID:
            break
        if 0 <= tid <= 9:
            digits.append(str(tid))
        else:
            break
    if not digits:
        return 0
    while len(digits) < SUM_DIGITS:
        digits.append("0")
    digits = digits[:SUM_DIGITS]
    return int("".join(digits)[::-1])


def preprocess_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    bsz = a.shape[0]
    a_digits = ((a[:, None] // POW10_10_DESC[None, :]) % 10).to(torch.long)
    b_digits = ((b[:, None] // POW10_10_DESC[None, :]) % 10).to(torch.long)
    plus = torch.full((bsz, 1), TOKENS["+"], dtype=torch.long)
    equals = torch.full((bsz, 1), TOKENS["="], dtype=torch.long)
    return torch.cat([a_digits, plus, b_digits, equals], dim=1)


def encode_batch(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prompt = preprocess_batch(a, b)
    sums = a + b
    rev_digits = ((sums[:, None] // POW10_11[None, :]) % 10).to(torch.long)
    eos = torch.full((a.shape[0], 1), EOS_ID, dtype=torch.long)
    target = torch.cat([rev_digits, eos], dim=1)

    full = torch.cat([prompt, target], dim=1)  # [bsz, 34]
    x = full[:, :-1].clone()  # [bsz, 33]
    y = full[:, 1:].clone()   # [bsz, 33]

    # Mask prompt loss; we only care about generated sum tokens.
    y[:, : PROMPT_LEN - 1] = -100
    return x, y


# ----------------------------
# Model
# ----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.a = nn.Parameter(torch.empty(in_features, rank))
        self.b = nn.Parameter(torch.empty(rank, out_features))
        nn.init.normal_(self.a, std=math.sqrt(2.0 / (in_features + rank)))
        nn.init.normal_(self.b, std=math.sqrt(2.0 / (rank + out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.a @ self.b


class LowRankEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        self.a = nn.Parameter(torch.empty(num_embeddings, rank))
        self.b = nn.Parameter(torch.empty(rank, embedding_dim))
        nn.init.normal_(self.a, std=0.02)
        nn.init.normal_(self.b, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.a) @ self.b


class FixedSinCosEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, embedding_dim: int, period: float):
        super().__init__()
        if period <= 0:
            raise ValueError(f"period must be > 0, got {period}")
        half = embedding_dim // 2
        if half <= 0:
            raise ValueError(f"embedding_dim must be >= 2 for sin/cos PE, got {embedding_dim}")

        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        freq_index = torch.arange(half, dtype=torch.float32)
        base_omega = (2.0 * math.pi) / period
        exponents = freq_index / max(1, half)
        inv_freq = base_omega / torch.pow(10000.0, exponents)
        angles = positions * inv_freq.unsqueeze(0)
        pe = torch.zeros(max_seq_len, embedding_dim, dtype=torch.float32)
        pe[:, 0 : 2 * half : 2] = torch.sin(angles)
        pe[:, 1 : 2 * half : 2] = torch.cos(angles)
        if embedding_dim % 2 == 1:
            pe[:, -1] = 0.0
        self.register_buffer("table", pe, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.table[idx]


@dataclass
class ModelConfig:
    n_layer: int = 1
    d_model: int = 4
    n_head: int = 1
    d_ff: int = 8
    max_seq_len: int = MODEL_INPUT_LEN
    vocab_size: int = VOCAB_SIZE
    pe_kind: str = "learned"  # learned, rope, sincos
    pe_period: float = 11.0
    pos_rank: int = 3
    qkv_rank: int = 3
    attn_out_rank: int = 3
    ffn_rank: int = 3
    tie_qkv: str = "shareA_tieKV"  # "shareA_tieKV" or "none"
    use_rmsnorm: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.tie_qkv = cfg.tie_qkv
        self.use_rope = cfg.pe_kind == "rope"
        if self.use_rope:
            if cfg.pe_period <= 0:
                raise ValueError(f"rope requires pe_period > 0, got {cfg.pe_period}")
            if self.head_dim % 2 != 0:
                raise ValueError(
                    f"rope requires even head_dim, got d_model={cfg.d_model}, n_head={cfg.n_head}, head_dim={self.head_dim}"
                )
            half = self.head_dim // 2
            base_omega = (2.0 * math.pi) / float(cfg.pe_period)
            exponents = torch.arange(half, dtype=torch.float32) / max(1, half)
            inv_freq = base_omega / torch.pow(10000.0, exponents)
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        if self.tie_qkv == "shareA_tieKV":
            if cfg.qkv_rank <= 0:
                raise ValueError("shareA_tieKV requires qkv_rank > 0")
            self.qkv_a = nn.Parameter(torch.empty(cfg.d_model, cfg.qkv_rank))
            self.qkv_bq = nn.Parameter(torch.empty(cfg.qkv_rank, cfg.d_model))
            self.qkv_bkv = nn.Parameter(torch.empty(cfg.qkv_rank, cfg.d_model))
            nn.init.normal_(self.qkv_a, std=math.sqrt(2.0 / (cfg.d_model + cfg.qkv_rank)))
            nn.init.normal_(self.qkv_bq, std=math.sqrt(2.0 / (cfg.qkv_rank + cfg.d_model)))
            nn.init.normal_(self.qkv_bkv, std=math.sqrt(2.0 / (cfg.qkv_rank + cfg.d_model)))
        elif self.tie_qkv == "none":
            if cfg.qkv_rank > 0:
                self.qkv = LowRankLinear(cfg.d_model, 3 * cfg.d_model, cfg.qkv_rank)
            else:
                self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        else:
            raise ValueError(f"Unsupported tie_qkv: {self.tie_qkv}")

        if cfg.attn_out_rank > 0:
            self.proj = LowRankLinear(cfg.d_model, cfg.d_model, cfg.attn_out_rank)
        else:
            self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def _apply_rope(self, x: torch.Tensor, seqlen: int) -> torch.Tensor:
        pos = torch.arange(seqlen, device=x.device, dtype=x.dtype)
        inv_freq = self.rope_inv_freq.to(device=x.device, dtype=x.dtype)
        angles = torch.outer(pos, inv_freq)
        sin = angles.sin().unsqueeze(0).unsqueeze(0)
        cos = angles.cos().unsqueeze(0).unsqueeze(0)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rot_even
        out[..., 1::2] = rot_odd
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        if self.tie_qkv == "shareA_tieKV":
            h = x @ self.qkv_a
            q = h @ self.qkv_bq
            kv = h @ self.qkv_bkv
            k, v = kv, kv
        else:
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        if self.use_rope:
            q = self._apply_rope(q, seqlen)
            k = self._apply_rope(k, seqlen)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:seqlen, :seqlen], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.ffn_rank > 0:
            self.up = LowRankLinear(cfg.d_model, cfg.d_ff, cfg.ffn_rank)
            self.down = LowRankLinear(cfg.d_ff, cfg.d_model, cfg.ffn_rank)
        else:
            self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        norm_cls = RMSNorm if cfg.use_rmsnorm else nn.LayerNorm
        self.ln1 = norm_cls(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = norm_cls(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyAdderTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pe_kind == "learned":
            self.pos_emb: Optional[nn.Module] = (
                LowRankEmbedding(cfg.max_seq_len, cfg.d_model, cfg.pos_rank)
                if cfg.pos_rank > 0
                else nn.Embedding(cfg.max_seq_len, cfg.d_model)
            )
        elif cfg.pe_kind == "sincos":
            self.pos_emb = FixedSinCosEmbedding(cfg.max_seq_len, cfg.d_model, cfg.pe_period)
        elif cfg.pe_kind == "rope":
            self.pos_emb = None
        else:
            raise ValueError(f"Unsupported pe_kind: {cfg.pe_kind!r}")
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        norm_cls = RMSNorm if cfg.use_rmsnorm else nn.LayerNorm
        self.ln_f = norm_cls(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # tied output head
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        _, seqlen = idx.shape
        if seqlen > self.cfg.max_seq_len:
            raise ValueError(f"Input length {seqlen} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.token_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(seqlen, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            out = idx
            for _ in range(max_new_tokens):
                if out.shape[1] > self.cfg.max_seq_len:
                    out = out[:, -self.cfg.max_seq_len :]
                logits, _ = self(out)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                out = torch.cat([out, next_token], dim=1)
            return out
        finally:
            if was_training:
                self.train()


def unique_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------
# Train / eval
# ----------------------------
@dataclass
class TrainConfig:
    seed: int = 34
    device: str = "cpu"
    steps: int = 162000
    batch_size: int = 512
    lr: float = 0.02
    min_lr_ratio: float = 0.1
    warmup_steps: int = 1350
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_interval: int = 1000
    eval_batch_size: int = 512
    val_size: int = 5000
    test_size: int = 10000
    holdout_seed: int = 2025
    ckpt_out: str = "best.pt"
    last_ckpt_out: str = ""
    resume: str = ""
    finetune: str = ""
    curriculum_mode: str = "absolute"  # absolute, percent, none
    phase1_end: int = 2000
    phase2_end: int = 7000
    phase1_ratio: float = DEFAULT_PHASE1_RATIO
    phase2_ratio: float = DEFAULT_PHASE2_RATIO


def build_holdout(
    val_size: int, test_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    total = val_size + test_size
    pairs: list[tuple[int, int]] = []
    seen: set[int] = set()
    while len(pairs) < total:
        a = rng.randint(0, MAX_ADDEND)
        b = rng.randint(0, MAX_ADDEND)
        h = pair_hash(a, b)
        if h in seen:
            continue
        seen.add(h)
        pairs.append((a, b))
    val = pairs[:val_size]
    test = pairs[val_size:]
    val_a = torch.tensor([x[0] for x in val], dtype=torch.int64)
    val_b = torch.tensor([x[1] for x in val], dtype=torch.int64)
    test_a = torch.tensor([x[0] for x in test], dtype=torch.int64)
    test_b = torch.tensor([x[1] for x in test], dtype=torch.int64)
    return val_a, val_b, test_a, test_b


def resolve_curriculum_phases(cfg: TrainConfig) -> list[tuple[int, int, int]]:
    mode = cfg.curriculum_mode.strip().lower()
    if mode == "none":
        return [(1, 10, 10**18)]
    if mode == "absolute":
        p1 = max(1, int(cfg.phase1_end))
        p2 = int(cfg.phase2_end)
        if p2 <= p1:
            raise ValueError(
                f"Invalid absolute curriculum boundaries: phase2_end ({p2}) must be > phase1_end ({p1})."
            )
        return [(1, 3, p1), (1, 6, p2), (1, 10, 10**18)]
    if mode == "percent":
        r1 = float(cfg.phase1_ratio)
        r2 = float(cfg.phase2_ratio)
        if r2 <= r1:
            raise ValueError(
                f"Invalid percent curriculum boundaries: phase2_ratio ({r2}) must be > phase1_ratio ({r1})."
            )
        p1 = max(1, int(round(cfg.steps * r1)))
        p2 = int(round(cfg.steps * r2))
        if p2 <= p1:
            raise ValueError(
                f"Invalid percent curriculum step boundaries after rounding: phase1_end ({p1}), phase2_end ({p2}). "
                "Increase phase2_ratio or steps."
            )
        return [(1, 3, p1), (1, 6, p2), (1, 10, 10**18)]
    raise ValueError(f"Unsupported curriculum_mode: {cfg.curriculum_mode!r}")


class CurriculumSampler:
    def __init__(
        self,
        batch_size: int,
        seed: int,
        reserved_pairs: set[int],
        phases: list[tuple[int, int, int]],
    ):
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.reserved = reserved_pairs
        self.phases = phases

    def _range_for_step(self, step: int) -> tuple[int, int]:
        for lo, hi, stop in self.phases:
            if step < stop:
                return lo, hi
        return 1, 10

    def state_dict(self) -> dict[str, Any]:
        return {"rng_state": self.rng.getstate(), "phases": list(self.phases)}

    def load_state_dict(self, blob: dict[str, Any]) -> None:
        rng_state = blob.get("rng_state")
        if rng_state is not None:
            self.rng.setstate(rng_state)

    def sample(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = self._range_for_step(step)
        a = torch.zeros(self.batch_size, dtype=torch.int64)
        b = torch.zeros(self.batch_size, dtype=torch.int64)
        rng = self.rng
        reserved = self.reserved
        for i in range(self.batch_size):
            while True:
                digits = rng.randint(lo, hi)
                max_val = MAX_BY_DIGITS[digits]
                ai = rng.randint(0, max_val)
                bi = rng.randint(0, max_val)
                if pair_hash(ai, bi) not in reserved:
                    a[i] = ai
                    b[i] = bi
                    break
        return a, b


def cosine_lr(step: int, total_steps: int, peak_lr: float, warmup_steps: int, min_lr_ratio: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    p = min(max(p, 0.0), 1.0)
    c = 0.5 * (1.0 + math.cos(math.pi * p))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * c)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(blob: dict[str, Any]) -> None:
    py = blob.get("python")
    if py is not None:
        random.setstate(py)
    t_cpu = blob.get("torch_cpu")
    if t_cpu is not None:
        torch.set_rng_state(t_cpu)
    t_cuda = blob.get("torch_cuda")
    if t_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(t_cuda)


@torch.no_grad()
def evaluate_exact(
    model: TinyAdderTransformer,
    a: torch.Tensor,
    b: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    n = int(a.numel())
    exact_ok = 0
    token_ok = 0
    token_total = n * SUM_DIGITS
    for st in range(0, n, batch_size):
        ed = min(st + batch_size, n)
        aa = a[st:ed]
        bb = b[st:ed]
        prompt = preprocess_batch(aa, bb).to(device)
        gen = model.generate(prompt, max_new_tokens=TARGET_LEN)
        pred_digits = gen[:, -TARGET_LEN:-1].to("cpu")  # generated 11 sum digits
        tgt_digits = ((aa + bb)[:, None] // POW10_11[None, :]) % 10
        tgt_digits = tgt_digits.to(torch.long)
        matches = pred_digits.eq(tgt_digits)
        token_ok += int(matches.sum().item())
        exact_ok += int(matches.all(dim=1).sum().item())
    return exact_ok / n, token_ok / token_total


def save_checkpoint(
    ckpt_path: Path,
    model: TinyAdderTransformer,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    best_val_exact: float,
    best_val_token_acc: float,
    best_step: int,
    step: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    sampler: Optional[CurriculumSampler] = None,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "best_val_exact": best_val_exact,
        "best_val_token_acc": best_val_token_acc,
        "best_step": best_step,
        "step": step,
        "params": unique_parameter_count(model),
        "rng_state": capture_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if sampler is not None:
        payload["sampler_state"] = sampler.state_dict()
    torch.save(payload, ckpt_path)


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[TinyAdderTransformer, dict]:
    try:
        blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        blob = torch.load(ckpt_path, map_location=device)
    cfg = ModelConfig(**blob["model_config"])
    model = TinyAdderTransformer(cfg).to(device)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, blob


def run_train(
    model_cfg: ModelConfig, train_cfg: TrainConfig, wandb_run: Optional[Any] = None
) -> dict:
    set_seed(train_cfg.seed)
    device = torch.device(train_cfg.device)
    phases = resolve_curriculum_phases(train_cfg)

    val_a, val_b, test_a, test_b = build_holdout(
        train_cfg.val_size, train_cfg.test_size, train_cfg.holdout_seed
    )
    reserved = {pair_hash(int(x), int(y)) for x, y in zip(val_a.tolist(), val_b.tolist())}
    reserved |= {pair_hash(int(x), int(y)) for x, y in zip(test_a.tolist(), test_b.tolist())}

    model = TinyAdderTransformer(model_cfg).to(device)
    params = unique_parameter_count(model)
    best_ckpt_path = Path(train_cfg.ckpt_out)
    if train_cfg.last_ckpt_out.strip():
        last_ckpt_path = Path(train_cfg.last_ckpt_out)
    else:
        last_ckpt_path = best_ckpt_path.with_name("last.pt")
    print(f"params={params} device={device} curriculum={phases}")
    if wandb_run is not None:
        wandb_run.summary["params"] = int(params)
        wandb_run.summary["curriculum_phases"] = str(phases)
        wandb_run.summary["best_ckpt"] = str(best_ckpt_path)
        wandb_run.summary["last_ckpt"] = str(last_ckpt_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    sampler = CurriculumSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved, phases)

    best_val_exact = -1.0
    best_val_token_acc = -1.0
    best_step = -1
    start_step = 0
    t0 = time.time()

    if train_cfg.resume.strip() and train_cfg.finetune.strip():
        raise ValueError("Cannot use both --resume and --finetune. Pick one.")

    if train_cfg.resume.strip():
        resume_path = Path(train_cfg.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        try:
            resume_blob = torch.load(resume_path, map_location=device, weights_only=True)
        except TypeError:
            resume_blob = torch.load(resume_path, map_location=device)
        resume_cfg = ModelConfig(**resume_blob["model_config"])
        if asdict(resume_cfg) != asdict(model_cfg):
            raise ValueError(
                "Resume checkpoint architecture does not match current model config. "
                "Use matching model args when resuming."
            )
        model.load_state_dict(resume_blob["model_state"])
        if resume_blob.get("optimizer_state") is not None:
            optimizer.load_state_dict(resume_blob["optimizer_state"])
        if resume_blob.get("sampler_state") is not None:
            sampler.load_state_dict(resume_blob["sampler_state"])
        if resume_blob.get("rng_state") is not None:
            restore_rng_state(resume_blob["rng_state"])

        best_val_exact = float(resume_blob.get("best_val_exact", resume_blob.get("val_exact", -1.0)))
        best_val_token_acc = float(resume_blob.get("best_val_token_acc", -1.0))
        best_step = int(resume_blob.get("best_step", resume_blob.get("step", -1)))
        start_step = int(resume_blob.get("step", -1)) + 1
        print(
            f"resumed from {resume_path} at step={start_step} "
            f"(best_val_exact={best_val_exact:.5f} best_val_token_acc={best_val_token_acc:.5f} "
            f"best_step={best_step})"
        )
        if wandb_run is not None:
            wandb_run.summary["resumed_from"] = str(resume_path)

        if not best_ckpt_path.exists():
            print(
                f"warning: {best_ckpt_path} missing while resuming; "
                "best metrics will be recomputed from the resumed trajectory."
            )
            best_val_exact = -1.0
            best_val_token_acc = -1.0
            best_step = -1

    elif train_cfg.finetune.strip():
        ft_path = Path(train_cfg.finetune)
        if not ft_path.exists():
            raise FileNotFoundError(f"Finetune checkpoint not found: {ft_path}")
        try:
            ft_blob = torch.load(ft_path, map_location=device, weights_only=True)
        except TypeError:
            ft_blob = torch.load(ft_path, map_location=device)
        ft_cfg = ModelConfig(**ft_blob["model_config"])
        if asdict(ft_cfg) != asdict(model_cfg):
            raise ValueError(
                "Finetune checkpoint architecture does not match current model config. "
                "Use matching model args when fine-tuning."
            )
        model.load_state_dict(ft_blob["model_state"])
        print(
            f"fine-tuning from {ft_path} "
            f"(fresh optimizer, lr={train_cfg.lr}, steps={train_cfg.steps})"
        )
        if wandb_run is not None:
            wandb_run.summary["finetuned_from"] = str(ft_path)

    last_trained_step = start_step - 1
    for step in range(start_step, train_cfg.steps):
        last_trained_step = step
        model.train()
        a, b = sampler.sample(step)
        x, y = encode_batch(a, b)
        x = x.to(device)
        y = y.to(device)

        lr_now = cosine_lr(step, train_cfg.steps, train_cfg.lr, train_cfg.warmup_steps, train_cfg.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if step % train_cfg.eval_interval == 0 or step == train_cfg.steps - 1:
            val_exact, val_tok = evaluate_exact(model, val_a, val_b, train_cfg.eval_batch_size, device)
            elapsed = time.time() - t0
            loss_val = float(loss.item())
            print(
                f"step={step:6d} loss={loss_val:.4f} val_exact={val_exact:.5f} "
                f"val_tok={val_tok:.5f} lr={lr_now:.3e} t={elapsed:.1f}s"
            )
            improved = (val_exact > best_val_exact) or (
                val_exact == best_val_exact and val_tok > best_val_token_acc
            )
            if improved:
                best_val_exact = float(val_exact)
                best_val_token_acc = float(val_tok)
                best_step = step
                save_checkpoint(
                    best_ckpt_path,
                    model,
                    model_cfg,
                    train_cfg,
                    best_val_exact,
                    best_val_token_acc,
                    best_step,
                    step,
                    optimizer=optimizer,
                    sampler=sampler,
                )
                print(f"  saved best {best_ckpt_path}")
                if wandb_run is not None:
                    wandb_run.summary["best_ckpt"] = str(best_ckpt_path)
                    wandb_run.summary["best_step"] = int(best_step)
                    wandb_run.summary["best_val_exact"] = float(best_val_exact)
                    wandb_run.summary["best_val_token_acc"] = float(best_val_token_acc)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "step": step,
                        "train_loss": loss_val,
                        "val_exact": float(val_exact),
                        "val_token_acc": float(val_tok),
                        "lr": float(lr_now),
                        "elapsed_sec": float(elapsed),
                        "best_val_exact": float(best_val_exact),
                        "best_val_token_acc": float(best_val_token_acc),
                    },
                    step=step,
                )
            save_checkpoint(
                last_ckpt_path,
                model,
                model_cfg,
                train_cfg,
                best_val_exact,
                best_val_token_acc,
                best_step,
                step,
                optimizer=optimizer,
                sampler=sampler,
            )

    final_step = max(last_trained_step, start_step - 1, 0)
    save_checkpoint(
        last_ckpt_path,
        model,
        model_cfg,
        train_cfg,
        best_val_exact,
        best_val_token_acc,
        best_step,
        final_step,
        optimizer=optimizer,
        sampler=sampler,
    )

    if not best_ckpt_path.exists():
        val_exact, val_tok = evaluate_exact(model, val_a, val_b, train_cfg.eval_batch_size, device)
        best_val_exact = float(val_exact)
        best_val_token_acc = float(val_tok)
        best_step = int(final_step)
        save_checkpoint(
            best_ckpt_path,
            model,
            model_cfg,
            train_cfg,
            best_val_exact,
            best_val_token_acc,
            best_step,
            final_step,
            optimizer=optimizer,
            sampler=sampler,
        )

    best_model, blob = load_checkpoint(best_ckpt_path, device)
    test_exact, test_tok = evaluate_exact(best_model, test_a, test_b, train_cfg.eval_batch_size, device)

    summary = {
        "params": int(blob.get("params", params)),
        "best_val_exact": float(best_val_exact),
        "best_val_token_acc": float(best_val_token_acc),
        "best_step": int(best_step),
        "test_exact": float(test_exact),
        "test_token_acc": float(test_tok),
        "ckpt_out": str(best_ckpt_path),
        "last_ckpt_out": str(last_ckpt_path),
        "start_step": int(start_step),
    }
    print(f"done: {summary}")
    if wandb_run is not None:
        wandb_run.summary.update(summary)
    return summary


@torch.no_grad()
def run_eval(
    ckpt: Path,
    device_str: str,
    test_size: int,
    seed: int,
    batch_size: int,
    val_size: int,
) -> None:
    device = torch.device(device_str)
    model, blob = load_checkpoint(ckpt, device)
    train_blob = blob.get("train_config", {})
    holdout_seed = int(train_blob.get("holdout_seed", 2025)) if seed < 0 else int(seed)
    holdout_val_size = int(train_blob.get("val_size", 5000)) if val_size < 0 else int(val_size)
    holdout_test_size = int(train_blob.get("test_size", 10000)) if test_size < 0 else int(test_size)
    _, _, test_a, test_b = build_holdout(holdout_val_size, holdout_test_size, holdout_seed)
    exact, tok = evaluate_exact(model, test_a, test_b, batch_size, device)
    print(f"ckpt={ckpt} params={blob.get('params', unique_parameter_count(model))}")
    print(
        f"test_exact={exact:.5f} test_tok={tok:.5f} n={holdout_test_size} "
        f"seed={holdout_seed} val_offset={holdout_val_size}"
    )


@torch.no_grad()
def run_predict(ckpt: Path, device_str: str, a: int, b: int) -> None:
    device = torch.device(device_str)
    model, _ = load_checkpoint(ckpt, device)
    prompt = torch.tensor([preprocess_prompt(a, b)], dtype=torch.long, device=device)
    gen = model.generate(prompt, max_new_tokens=TARGET_LEN)
    pred = decode_generated_sum(gen[0, -TARGET_LEN:].tolist())
    print(f"{a} + {b} = {pred} (expected {a+b})")


# ----------------------------
# W&B sweep mode
# ----------------------------
def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool from {x!r}")


def run_sweep(project: str, entity: str, group: str, device: str, output_root: Path, tags: str) -> None:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required. Install with: pip install wandb") from exc

    defaults = {
        "seed": 34,
        "n_layer": 1,
        "d_model": 4,
        "n_head": 1,
        "d_ff": 8,
        "pe_kind": "learned",
        "pe_period": 11.0,
        "pos_rank": 3,
        "qkv_rank": 3,
        "attn_out_rank": 3,
        "ffn_rank": 3,
        "tie_qkv": "shareA_tieKV",
        "use_rmsnorm": True,
        "steps": 162000,
        "batch_size": 512,
        "lr": 0.02,
        "min_lr_ratio": 0.1,
        "warmup_steps": 1350,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "curriculum_mode": "absolute",
        "phase1_end": 2000,
        "phase2_end": 7000,
        "phase1_ratio": DEFAULT_PHASE1_RATIO,
        "phase2_ratio": DEFAULT_PHASE2_RATIO,
        "eval_interval": 1000,
        "eval_batch_size": 512,
        "val_size": 5000,
        "test_size": 10000,
        "holdout_seed": 2025,
        "max_params": 0,
    }

    run = wandb.init(
        project=project,
        entity=(entity.strip() or None),
        group=group,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
        config=defaults,
        job_type="train",
    )
    try:
        cfg = dict(run.config)
        model_cfg = ModelConfig(
            n_layer=int(cfg["n_layer"]),
            d_model=int(cfg["d_model"]),
            n_head=int(cfg["n_head"]),
            d_ff=int(cfg["d_ff"]),
            pe_kind=str(cfg.get("pe_kind", "learned")),
            pe_period=float(cfg.get("pe_period", 11.0)),
            pos_rank=int(cfg["pos_rank"]),
            qkv_rank=int(cfg["qkv_rank"]),
            attn_out_rank=int(cfg["attn_out_rank"]),
            ffn_rank=int(cfg["ffn_rank"]),
            tie_qkv=str(cfg["tie_qkv"]),
            use_rmsnorm=_as_bool(cfg["use_rmsnorm"]),
        )

        try:
            probe = TinyAdderTransformer(model_cfg)
        except ValueError as exc:
            run.summary["skipped"] = True
            run.summary["skip_reason"] = f"invalid_config: {exc}"
            print(run.summary["skip_reason"])
            return
        params = unique_parameter_count(probe)
        del probe
        run.summary["params"] = int(params)

        max_params = int(cfg.get("max_params", 0))
        if max_params > 0 and params > max_params:
            run.summary["skipped"] = True
            run.summary["skip_reason"] = f"params={params} > max_params={max_params}"
            print(run.summary["skip_reason"])
            return

        run_dir = output_root / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_out = run_dir / "best.pt"
        last_ckpt_out = run_dir / "last.pt"
        train_cfg = TrainConfig(
            seed=int(cfg["seed"]),
            device=device,
            steps=int(cfg["steps"]),
            batch_size=int(cfg["batch_size"]),
            lr=float(cfg["lr"]),
            min_lr_ratio=float(cfg["min_lr_ratio"]),
            warmup_steps=int(cfg["warmup_steps"]),
            weight_decay=float(cfg["weight_decay"]),
            grad_clip=float(cfg["grad_clip"]),
            curriculum_mode=str(cfg.get("curriculum_mode", "absolute")),
            phase1_end=int(cfg.get("phase1_end", 2000)),
            phase2_end=int(cfg.get("phase2_end", 7000)),
            phase1_ratio=float(cfg.get("phase1_ratio", DEFAULT_PHASE1_RATIO)),
            phase2_ratio=float(cfg.get("phase2_ratio", DEFAULT_PHASE2_RATIO)),
            eval_interval=int(cfg["eval_interval"]),
            eval_batch_size=int(cfg["eval_batch_size"]),
            val_size=int(cfg["val_size"]),
            test_size=int(cfg["test_size"]),
            holdout_seed=int(cfg["holdout_seed"]),
            ckpt_out=str(ckpt_out),
            last_ckpt_out=str(last_ckpt_out),
        )
        run_train(model_cfg, train_cfg, wandb_run=run)
    finally:
        run.finish()


# ----------------------------
# AdderBoard hooks
# ----------------------------
class SubmissionBundle:
    def __init__(self, model: TinyAdderTransformer, device: torch.device):
        self.model = model
        self.device = device


def build_model():
    ckpt = Path(os.environ.get("ADDER_CKPT", "best.pt"))
    device = torch.device(os.environ.get("ADDER_DEVICE", "cpu"))
    if not ckpt.exists():
        model = TinyAdderTransformer(ModelConfig()).to(device)
        metadata = {
            "name": "tiny-adder-lab",
            "author": "your-name",
            "params": unique_parameter_count(model),
            "architecture": "untrained fallback (set ADDER_CKPT)",
            "tricks": ["causal attention", "autoregressive decoding"],
        }
        return SubmissionBundle(model, device), metadata

    model, blob = load_checkpoint(ckpt, device)
    cfg = ModelConfig(**blob["model_config"])
    metadata = {
        "name": "tiny-adder-lab",
        "author": "your-name",
        "params": int(blob.get("params", unique_parameter_count(model))),
        "architecture": (
            f"{cfg.n_layer}L decoder d={cfg.d_model} h={cfg.n_head} ff={cfg.d_ff} "
            f"ranks=({cfg.pos_rank},{cfg.qkv_rank},{cfg.attn_out_rank},{cfg.ffn_rank})"
        ),
        "tricks": [
            "tied token/output embedding",
            "low-rank factorization",
            f"qkv_tie={cfg.tie_qkv}",
            "autoregressive decoding",
        ],
    }
    return SubmissionBundle(model, device), metadata


@torch.no_grad()
def add(bundle: SubmissionBundle, a: int, b: int) -> int:
    prompt = torch.tensor([preprocess_prompt(a, b)], dtype=torch.long, device=bundle.device)
    gen = bundle.model.generate(prompt, max_new_tokens=TARGET_LEN)
    return decode_generated_sum(gen[0, -TARGET_LEN:].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tiny adder lab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train model")
    t.add_argument("--seed", type=int, default=34)
    t.add_argument("--device", type=str, default="cpu")
    t.add_argument("--steps", type=int, default=162000)
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--lr", type=float, default=0.02)
    t.add_argument("--min-lr-ratio", type=float, default=0.1)
    t.add_argument("--warmup-steps", type=int, default=1350)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--eval-interval", type=int, default=1000)
    t.add_argument("--eval-batch-size", type=int, default=512)
    t.add_argument("--val-size", type=int, default=5000)
    t.add_argument("--test-size", type=int, default=10000)
    t.add_argument("--holdout-seed", type=int, default=2025)
    t.add_argument("--ckpt-out", type=str, default="best.pt")
    t.add_argument("--last-ckpt-out", type=str, default="")
    t.add_argument("--resume", type=str, default="")
    t.add_argument("--finetune", type=str, default="")

    t.add_argument("--curriculum-mode", type=str, default="absolute", choices=["absolute", "percent", "none"])
    t.add_argument("--phase1-end", type=int, default=2000)
    t.add_argument("--phase2-end", type=int, default=7000)
    t.add_argument("--phase1-ratio", type=float, default=DEFAULT_PHASE1_RATIO)
    t.add_argument("--phase2-ratio", type=float, default=DEFAULT_PHASE2_RATIO)

    t.add_argument("--n-layer", type=int, default=1)
    t.add_argument("--d-model", type=int, default=4)
    t.add_argument("--n-head", type=int, default=1)
    t.add_argument("--d-ff", type=int, default=8)
    t.add_argument("--pe-kind", type=str, default="learned", choices=["learned", "rope", "sincos"])
    t.add_argument("--pe-period", type=float, default=11.0)
    t.add_argument("--pos-rank", type=int, default=3)
    t.add_argument("--qkv-rank", type=int, default=3)
    t.add_argument("--attn-out-rank", type=int, default=3)
    t.add_argument("--ffn-rank", type=int, default=3)
    t.add_argument("--tie-qkv", type=str, default="shareA_tieKV", choices=["none", "shareA_tieKV"])
    t.add_argument("--use-rmsnorm", action="store_true", default=True)
    t.add_argument("--no-rmsnorm", action="store_false", dest="use_rmsnorm")

    t.add_argument("--wandb", action="store_true", default=False)
    t.add_argument("--project", type=str, default="tiny-adder-lab")
    t.add_argument("--entity", type=str, default="")
    t.add_argument("--group", type=str, default="manual")
    t.add_argument("--run-name", type=str, default="")
    t.add_argument("--tags", type=str, default="tiny,adder,manual")

    e = sub.add_parser("eval", help="Evaluate checkpoint")
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument("--device", type=str, default="cpu")
    e.add_argument("--test-size", type=int, default=-1)
    e.add_argument("--seed", type=int, default=-1)
    e.add_argument("--val-size", type=int, default=-1)
    e.add_argument("--batch-size", type=int, default=512)

    p = sub.add_parser("predict", help="Predict one sample")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--a", type=int, required=True)
    p.add_argument("--b", type=int, required=True)

    s = sub.add_parser("sweep", help="W&B sweep runner")
    s.add_argument("--project", type=str, default="tiny-adder-lab")
    s.add_argument("--entity", type=str, default="")
    s.add_argument("--group", type=str, default="sweep")
    s.add_argument("--device", type=str, default="cuda")
    s.add_argument("--output-root", type=Path, default=Path("runs/wandb"))
    s.add_argument("--tags", type=str, default="tiny,adder,sweep")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        model_cfg = ModelConfig(
            n_layer=args.n_layer,
            d_model=args.d_model,
            n_head=args.n_head,
            d_ff=args.d_ff,
            pe_kind=args.pe_kind,
            pe_period=args.pe_period,
            pos_rank=args.pos_rank,
            qkv_rank=args.qkv_rank,
            attn_out_rank=args.attn_out_rank,
            ffn_rank=args.ffn_rank,
            tie_qkv=args.tie_qkv,
            use_rmsnorm=args.use_rmsnorm,
        )
        train_cfg = TrainConfig(
            seed=args.seed,
            device=args.device,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            min_lr_ratio=args.min_lr_ratio,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            curriculum_mode=args.curriculum_mode,
            phase1_end=args.phase1_end,
            phase2_end=args.phase2_end,
            phase1_ratio=args.phase1_ratio,
            phase2_ratio=args.phase2_ratio,
            eval_interval=args.eval_interval,
            eval_batch_size=args.eval_batch_size,
            val_size=args.val_size,
            test_size=args.test_size,
            holdout_seed=args.holdout_seed,
            ckpt_out=args.ckpt_out,
            last_ckpt_out=args.last_ckpt_out,
            resume=args.resume,
            finetune=args.finetune,
        )
        if args.wandb:
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError("wandb missing. Install with: pip install wandb") from exc
            run = wandb.init(
                project=args.project,
                entity=(args.entity.strip() or None),
                group=args.group,
                name=(args.run_name.strip() or None),
                tags=[t.strip() for t in args.tags.split(",") if t.strip()],
                config={**asdict(model_cfg), **asdict(train_cfg)},
            )
            try:
                run_train(model_cfg, train_cfg, wandb_run=run)
            finally:
                run.finish()
        else:
            run_train(model_cfg, train_cfg)
    elif args.cmd == "eval":
        run_eval(args.ckpt, args.device, args.test_size, args.seed, args.batch_size, args.val_size)
    elif args.cmd == "predict":
        run_predict(args.ckpt, args.device, args.a, args.b)
    else:
        run_sweep(args.project, args.entity, args.group, args.device, args.output_root, args.tags)


if __name__ == "__main__":
    main()
