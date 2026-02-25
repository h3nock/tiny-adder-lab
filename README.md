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

## Evaluate
```bash
python tiny_adder.py eval --ckpt best.pt --device cuda --test-size 10000 --seed 2025
```

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

## AdderBoard verify integration
`tiny_adder.py` exposes `build_model()` and `add(model, a, b)`.

```bash
ADDER_CKPT=best.pt python verify.py tiny_adder.py --seed 2025
```
