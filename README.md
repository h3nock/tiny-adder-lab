# tiny-adder-lab

Minimal PyTorch repo for tiny autoregressive transformers that perform 10-digit addition.

## Install
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

## 239-Parameter Frontier Result
This result came from branch `exp/split-frontier-explore` and sweep `tiny-adder-split-frontier` (`r11xj7pr`).

Reference runs:
- `blooming-sweep-4` ([w0yxzaqp](https://wandb.ai/henokwondimu/tiny-adder-lab/runs/w0yxzaqp)): `params=239`, `best_val_exact=1.0`, `test_exact=0.9999`
- `sandy-sweep-14` ([zw3h3adk](https://wandb.ai/henokwondimu/tiny-adder-lab/runs/zw3h3adk)): `params=239`, `best_val_exact=1.0`, `test_exact=1.0`

Reproduce `blooming-sweep-4` config:
```bash
python tiny_adder.py train \
  --device cuda \
  --seed 34 \
  --arch split \
  --n-layer 1 --d-model 6 --d-ff 2 \
  --split-tok-dim 3 --split-pos-dim 3 \
  --split-n-head 2 --split-head-dim 3 \
  --split-ffn-bias --split-spiral-init \
  --split-pos-kind learned_shared \
  --prompt-order lsd \
  --use-rmsnorm \
  --batch-size 512 --steps 512000 \
  --lr 0.004080483478864221 \
  --min-lr-ratio 0.1 --warmup-steps 2000 \
  --weight-decay 0.01 --grad-clip 1 \
  --curriculum-mode absolute --phase1-end 2000 --phase2-end 7000 \
  --eval-interval 2000 --eval-batch-size 512 \
  --val-size 5000 --test-size 10000 --holdout-seed 2025 \
  --ckpt-out runs/wandb/w0yxzaqp/best.pt \
  --last-ckpt-out runs/wandb/w0yxzaqp/last.pt
```

## AdderBoard Verify
`tiny_adder.py` exposes `build_model()` and `add(model, a, b)`.

```bash
ADDER_CKPT=runs/wandb/zw3h3adk/best.pt ADDER_DEVICE=cuda python verify.py tiny_adder.py --seed 2025
```

Expected verify headline for `zw3h3adk`: `10010/10010 correct (100.00%)`.

## Sweep Files
- `sweep.yaml`: broad factorized search (sub-334)
- `sweep_split.yaml`: split-architecture frontier search
- `sweep_split_sub200.yaml`: sub-200 follow-up search
