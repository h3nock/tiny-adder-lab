# tiny-adder-lab

Small PyTorch repo for 10-digit addition with a tiny autoregressive transformer.

## Setup
```bash
pip install torch wandb
```

## Core Commands
Train:
```bash
python tiny_adder.py train --device cuda --steps 162000 --batch-size 512 --ckpt-out best.pt
```

Eval:
```bash
python tiny_adder.py eval --ckpt best.pt --device cuda
```

Predict:
```bash
python tiny_adder.py predict --ckpt best.pt --device cuda --a 1234567890 --b 9876543210
```

## W&B Sweep
```bash
wandb sweep sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

Use `sweep.yaml` for main search and `sweep_pe.yaml` for PE-focused search.

## 305-Parameter Reference Run
Run: https://wandb.ai/henokwondimu/tiny-adder-lab/runs/l6voiqpn

- params: `305`
- best_step: `386000`
- best_val_exact: `1.0`
- best_val_token_acc: `1.0`
- test_exact: `0.9998`
- test_token_acc: `0.9999818`

This run trained for `512000` steps.

Reproduce:
```bash
python tiny_adder.py train \
  --device cuda \
  --seed 43 \
  --n-layer 1 --d-model 4 --n-head 1 --d-ff 9 \
  --pe-kind learned --pos-rank 3 \
  --qkv-rank 3 --attn-out-rank 2 --ffn-rank 3 \
  --tie-qkv shareA_tieKV --use-rmsnorm \
  --batch-size 512 --steps 512000 \
  --lr 0.01808611797410989 --min-lr-ratio 0.1 \
  --warmup-steps 2000 --weight-decay 0.003 --grad-clip 1 \
  --curriculum-mode absolute --phase1-end 2000 --phase2-end 7000 \
  --eval-interval 2000 --eval-batch-size 512 \
  --val-size 5000 --test-size 10000 --holdout-seed 2025 \
  --ckpt-out runs/wandb/l6voiqpn/best.pt \
  --last-ckpt-out runs/wandb/l6voiqpn/last.pt
```

## AdderBoard verify integration
`tiny_adder.py` exposes `build_model()` and `add(model, a, b)`.

```bash
ADDER_CKPT=best.pt python verify.py tiny_adder.py --seed 2025
```
