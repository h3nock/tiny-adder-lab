# tiny-adder-lab

Minimal PyTorch repo for AdderBoard-style tiny transformer addition.

## Files
- `tiny_adder.py`: model + train/eval/predict + W&B sweep mode + AdderBoard hooks
- `sweep.yaml`: broad sweep config

## Install
```bash
pip install torch wandb
```

## Train (single run)
```bash
python tiny_adder.py train --device cuda --steps 162000 --batch-size 512 --ckpt-out best.pt
```

## Resume training
The trainer saves:
- `best.pt`: best validation checkpoint
- `last.pt`: latest resumable checkpoint (includes optimizer/RNG/sampler state)

Resume exactly from the last checkpoint:
```bash
python tiny_adder.py train --device cuda --steps 512000 --ckpt-out best.pt --resume last.pt
```

Resume from a run directory:
```bash
python tiny_adder.py train --device cuda --steps 320000 \
  --ckpt-out runs/wandb/<run_id>/best.pt \
  --last-ckpt-out runs/wandb/<run_id>/last.pt \
  --resume runs/wandb/<run_id>/last.pt
```

## Fine-tuning
Use `--finetune` (not `--resume`) for fine-tuning. This loads model weights only,
with a fresh optimizer, fresh LR schedule, and training from step 0:
```bash
python tiny_adder.py train --finetune best.pt \
  --lr 0.001 --steps 100000 --warmup-steps 500 \
  --min-lr-ratio 0.01 --curriculum-mode none \
  --ckpt-out ft1.pt
```

Multi-round fine-tuning (progressively lower LR):
```bash
python tiny_adder.py train --finetune ft1.pt \
  --lr 0.0003 --steps 50000 --warmup-steps 500 \
  --min-lr-ratio 0.01 --curriculum-mode none \
  --ckpt-out ft2.pt
```

## Curriculum control
You can control phase boundaries directly from CLI/config:

- `--curriculum-mode absolute`: use `--phase1-end` and `--phase2-end`
- `--curriculum-mode percent`: use `--phase1-ratio` and `--phase2-ratio`
- `--curriculum-mode none`: always sample 1-10 digit operands

Examples:
```bash
# Absolute cutoffs (default)
python tiny_adder.py train --curriculum-mode absolute --phase1-end 2000 --phase2-end 7000

# Percent-based cutoffs (scales with total steps)
python tiny_adder.py train --steps 512000 --curriculum-mode percent \
  --phase1-ratio 0.074074 --phase2-ratio 0.259259
```

## Positional Encoding Options
`tiny_adder.py train` supports:
- `--pe-kind learned` (default, trainable positional embeddings with `--pos-rank`)
- `--pe-kind rope` (rotary attention; no learned positional params)
- `--pe-kind sincos --pe-period 11` (fixed sinusoidal embeddings; no learned positional params)

## Evaluate
```bash
python tiny_adder.py eval --ckpt best.pt --device cuda
```

By default, eval reuses the holdout split settings saved in the checkpoint
(`holdout_seed`, `val_size`, `test_size`) so standalone eval matches train-time test split.

## Predict one pair
```bash
python tiny_adder.py predict --ckpt best.pt --device cuda --a 1234567890 --b 9876543210
```

## W&B sweep
```bash
wandb login
wandb sweep sweep.yaml
CUDA_VISIBLE_DEVICES=0 wandb agent <entity>/<project>/<sweep_id>
```

Recommended split:
- `sweep.yaml`: main learned-PE search (fixed `min_lr_ratio=0.1`)
- `sweep_pe.yaml`: focused PE search (`sincos`/`rope` with period sweep)

## AdderBoard verify integration
`tiny_adder.py` exposes `build_model()` and `add(model, a, b)`.

```bash
ADDER_CKPT=best.pt python verify.py tiny_adder.py --seed 2025
```
