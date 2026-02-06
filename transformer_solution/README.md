# Transformer Solution

This folder contains a Transformer-based solution compatible with the competition interface.

## Files

- `solution.py`: Submission entrypoint with `PredictionModel`.
- `model.py`: `LOBTransformer` architecture.
- `train.py`: Training script that saves model + normalization artifacts.

## Train

From the project root:

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n wunder python transformer_solution/train.py \
  --train-path datasets/train.parquet \
  --valid-path datasets/valid.parquet \
  --out-dir transformer_solution \
  --max-train-seqs 2000 \
  --max-valid-seqs 500 \
  --device cpu
```

For a quick smoke test, use small subsets:

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n wunder python transformer_solution/train.py \
  --train-path datasets/train.parquet \
  --valid-path datasets/valid.parquet \
  --out-dir transformer_solution \
  --max-train-seqs 128 \
  --max-valid-seqs 64 \
  --epochs 1 \
  --batch-size 128
```

## Validate

```bash
KMP_DUPLICATE_LIB_OK=TRUE conda run -n wunder python transformer_solution/solution.py
```

## Package submission

From inside `transformer_solution/`:

```bash
zip -r ../solution.zip .
```

