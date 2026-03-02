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

## 239-Parameter Result (From Sub-300 Search)
This configuration came from a sub-300 search, where one run reached 239 parameters with perfect validation exact match.

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
  --lr 0.00408 \
  --min-lr-ratio 0.1 --warmup-steps 2000 \
  --weight-decay 0.01 --grad-clip 1 \
  --curriculum-mode absolute --phase1-end 2000 --phase2-end 7000 \
  --eval-interval 2000 --eval-batch-size 512 \
  --val-size 5000 --test-size 10000 --holdout-seed 2025 \
  --ckpt-out best_239.pt --last-ckpt-out last_239.pt
```

## AdderBoard Verify
Benchmark: [AdderBoard](https://github.com/anadim/AdderBoard)

`tiny_adder.py` exposes `build_model()` and `add(model, a, b)`.

```bash
ADDER_CKPT=best_239.pt ADDER_DEVICE=cuda python verify.py tiny_adder.py --seed 2025
```

## Sweep Files
- `sweep_split.yaml`: split sub-300 search that produced the 239-parameter result
