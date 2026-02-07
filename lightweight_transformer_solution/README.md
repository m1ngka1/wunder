# Lightweight Transformer Solution

This solution provides a compact, non-causal transformer for faster training and inference.
It operates directly on the raw 32 LOB features (no feature engineering) with a short
context window (32 steps).

## Highlights
- Context length: 32
- Encoder layers: 2
- Attention heads: 4 (non-causal)
- Weighted Pearson correlation loss

## Training

```bash
python train.py \
  --train-path ../datasets/train.parquet \
  --valid-path ../datasets/valid.parquet \
  --out-dir . \
  --device cpu
```

Key defaults are tuned for speed:
- `--epochs 15`
- `--batch-size 512`
- `--d-model 32`
- `--num-layers 2`
- `--nhead 4`

Artifacts are saved under `./artifacts/` for inference.
