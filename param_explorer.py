#!/usr/bin/env python3
"""Parameter budget explorer for sub-311 adder configs.

Computes exact parameter counts for different architecture configurations
using sincos/rope PE (0 learnable params) to free budget for model capacity.
"""

from dataclasses import dataclass


@dataclass
class Config:
    d_model: int
    d_ff: int
    qkv_rank: int
    attn_out_rank: int
    ffn_rank: int  # 0 = full-rank MLP
    pe_kind: str  # "sincos" or "rope" (0 params) or "learned" (has cost)
    pos_rank: int = 3  # only matters for pe_kind="learned"
    vocab_size: int = 13
    max_seq_len: int = 33
    n_layer: int = 1
    n_head: int = 1


def param_count(c: Config) -> dict[str, int]:
    breakdown = {}

    # Token embedding (tied with output head, counted once)
    breakdown["token_emb"] = c.vocab_size * c.d_model

    # Position embedding
    if c.pe_kind == "learned":
        if c.pos_rank > 0:
            breakdown["pos_emb"] = c.max_seq_len * c.pos_rank + c.pos_rank * c.d_model
        else:
            breakdown["pos_emb"] = c.max_seq_len * c.d_model
    else:
        breakdown["pos_emb"] = 0  # sincos/rope = free

    per_layer = {}
    # RMSNorm x2 per layer
    per_layer["ln1"] = c.d_model
    per_layer["ln2"] = c.d_model

    # Attention: shareA_tieKV
    # A: d_model x qkv_rank, Bq: qkv_rank x d_model, Bkv: qkv_rank x d_model
    per_layer["qkv"] = c.d_model * c.qkv_rank + c.qkv_rank * c.d_model + c.qkv_rank * c.d_model

    # Attention output projection
    if c.attn_out_rank > 0:
        per_layer["attn_out"] = c.d_model * c.attn_out_rank + c.attn_out_rank * c.d_model
    else:
        per_layer["attn_out"] = c.d_model * c.d_model

    # MLP
    if c.ffn_rank > 0:
        per_layer["mlp_up"] = c.d_model * c.ffn_rank + c.ffn_rank * c.d_ff
        per_layer["mlp_down"] = c.d_ff * c.ffn_rank + c.ffn_rank * c.d_model
    else:
        per_layer["mlp_up"] = c.d_model * c.d_ff
        per_layer["mlp_down"] = c.d_ff * c.d_model

    layer_total = sum(per_layer.values())
    breakdown["layers"] = layer_total * c.n_layer
    breakdown["layer_detail"] = per_layer

    # Final RMSNorm
    breakdown["ln_f"] = c.d_model

    # Output head (tied, 0 extra)
    breakdown["lm_head"] = 0

    breakdown["total"] = (
        breakdown["token_emb"]
        + breakdown["pos_emb"]
        + breakdown["layers"]
        + breakdown["ln_f"]
    )
    return breakdown


def print_config(label: str, c: Config) -> None:
    b = param_count(c)
    detail = b.pop("layer_detail")
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  d_model={c.d_model}, d_ff={c.d_ff}, pe={c.pe_kind}")
    print(f"  qkv_rank={c.qkv_rank}, attn_out_rank={c.attn_out_rank}, ffn_rank={c.ffn_rank}")
    print(f"{'='*60}")
    print(f"  token_emb:  {c.vocab_size}x{c.d_model} = {b['token_emb']}")
    print(f"  pos_emb:    {b['pos_emb']}")
    for k, v in detail.items():
        print(f"    {k}: {v}")
    print(f"  layer_total: {b['layers']}")
    print(f"  ln_f:       {b['ln_f']}")
    print(f"  lm_head:    {b['lm_head']} (tied)")
    print(f"  ─────────────────────")
    print(f"  TOTAL:      {b['total']} params")
    status = "OK" if b["total"] < 311 else "OVER"
    print(f"  [{status}] budget=310, delta={310 - b['total']}")


def search_all_under(budget: int = 310) -> list[tuple[int, Config]]:
    """Enumerate all valid configs under the parameter budget."""
    results = []
    for d_model in [3, 4, 5, 6, 7, 8]:
        for d_ff in range(4, 40):
            for qkv_rank in [1, 2, 3, 4]:
                if qkv_rank > d_model:
                    continue
                for attn_out_rank in [0, 1, 2, 3, 4]:
                    if attn_out_rank > d_model and attn_out_rank != 0:
                        continue
                    for ffn_rank in [0, 1, 2, 3, 4]:
                        if ffn_rank > min(d_model, d_ff) and ffn_rank != 0:
                            continue
                        c = Config(
                            d_model=d_model,
                            d_ff=d_ff,
                            qkv_rank=qkv_rank,
                            attn_out_rank=attn_out_rank,
                            ffn_rank=ffn_rank,
                            pe_kind="sincos",
                        )
                        b = param_count(c)
                        if b["total"] <= budget:
                            results.append((b["total"], c))
    results.sort(key=lambda x: (-x[1].d_model, -x[1].d_ff, x[0]))
    return results


if __name__ == "__main__":
    # Reference: rezabyt's 311-param model (learned PE)
    print_config(
        "REFERENCE: rezabyt 311p (learned PE)",
        Config(d_model=4, d_ff=8, qkv_rank=3, attn_out_rank=3, ffn_rank=3,
               pe_kind="learned", pos_rank=3, vocab_size=14),
    )

    # Reference: our 335p model (learned PE, vocab=14)
    print_config(
        "REFERENCE: our 335p (learned PE, vocab=14)",
        Config(d_model=4, d_ff=12, qkv_rank=3, attn_out_rank=3, ffn_rank=3,
               pe_kind="learned", pos_rank=3, vocab_size=14),
    )

    print("\n" + "="*60)
    print("  CANDIDATE CONFIGS (sincos/rope PE, vocab=13)")
    print("="*60)

    # --- Same as rezabyt but with sincos PE ---
    print_config(
        "A: rezabyt-equivalent + sincos PE (massive savings)",
        Config(d_model=4, d_ff=8, qkv_rank=3, attn_out_rank=3, ffn_rank=3, pe_kind="sincos"),
    )

    # --- Redirect savings to bigger FFN ---
    print_config(
        "B: d=4, big FFN (d_ff=20), rank-3",
        Config(d_model=4, d_ff=20, qkv_rank=3, attn_out_rank=3, ffn_rank=3, pe_kind="sincos"),
    )

    # --- Full-rank MLP with freed budget ---
    print_config(
        "C: d=4, full-rank FFN (d_ff=23)",
        Config(d_model=4, d_ff=23, qkv_rank=3, attn_out_rank=3, ffn_rank=0, pe_kind="sincos"),
    )

    # --- Bigger d_model ---
    print_config(
        "D: d=5, d_ff=16, rank-3",
        Config(d_model=5, d_ff=16, qkv_rank=3, attn_out_rank=3, ffn_rank=3, pe_kind="sincos"),
    )

    print_config(
        "E: d=6, d_ff=12, rank-3",
        Config(d_model=6, d_ff=12, qkv_rank=3, attn_out_rank=3, ffn_rank=3, pe_kind="sincos"),
    )

    # --- Compact attention, big FFN ---
    print_config(
        "F: d=4, qkv_r=2, attn_r=2, full FFN d_ff=25",
        Config(d_model=4, d_ff=25, qkv_rank=2, attn_out_rank=2, ffn_rank=0, pe_kind="sincos"),
    )

    print_config(
        "G: d=5, qkv_r=2, attn_r=2, full FFN d_ff=18",
        Config(d_model=5, d_ff=18, qkv_rank=2, attn_out_rank=2, ffn_rank=0, pe_kind="sincos"),
    )

    # --- Max-capacity within budget ---
    print_config(
        "H: d=4, qkv_r=3, attn_r=2, full FFN d_ff=26",
        Config(d_model=4, d_ff=26, qkv_rank=3, attn_out_rank=2, ffn_rank=0, pe_kind="sincos"),
    )

    # --- d_model=4 with NO attn_out projection ---
    # (identity output = skip proj entirely)
    print_config(
        "I: d=4, no attn_out, full FFN d_ff=28",
        Config(d_model=4, d_ff=28, qkv_rank=3, attn_out_rank=0, ffn_rank=0, pe_kind="sincos"),
    )

    # --- Minimal viable: try to push d_model down ---
    print_config(
        "J: d=3, d_ff=20, rank-2, sincos",
        Config(d_model=3, d_ff=20, qkv_rank=2, attn_out_rank=2, ffn_rank=2, pe_kind="sincos"),
    )

    # --- Systematic search ---
    print("\n" + "="*60)
    print("  TOP CONFIGS BY d_model (automated search, budget=310)")
    print("="*60)
    results = search_all_under(310)

    # Show best configs per d_model (highest d_ff for each)
    seen = set()
    shown = 0
    for total, c in results:
        key = (c.d_model, c.d_ff, c.ffn_rank)
        if key in seen:
            continue
        seen.add(key)
        suffix = f"d={c.d_model} ff={c.d_ff} qr={c.qkv_rank} ar={c.attn_out_rank} fr={c.ffn_rank}"
        print(f"  {total:3d}p | {suffix}")
        shown += 1
        if shown >= 30:
            break

    print(f"\n  Total valid configs: {len(results)}")
