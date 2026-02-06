# Transformer Solution

This folder contains a Transformer-based solution compatible with the competition interface.

## Files

- `solution.py`: Submission entrypoint with `PredictionModel`.
- `model.py`: `LOBTransformer` architecture.
- `train.py`: Training script that saves model + normalization artifacts.
- `feature_engineering.py`: Feature engineering helpers for LOB-derived signals.
- `environment.yml`: Clean conda environment recipe (no `KMP_DUPLICATE_LIB_OK` workaround needed).

## Environment setup

From the project root:

```bash
conda env create -f transformer_solution/environment.yml
conda activate wunder2
```

## Train

From the project root:

```bash
conda run -n wunder2 python transformer_solution/train.py \
  --train-path datasets/train.parquet \
  --valid-path datasets/valid.parquet \
  --out-dir transformer_solution \
  --max-train-seqs 2000 \
  --max-valid-seqs 500 \
  --device cpu
```

For a quick smoke test, use small subsets:

```bash
conda run -n wunder2 python transformer_solution/train.py \
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
conda run -n wunder2 python transformer_solution/solution.py
```

## Package submission

From inside `transformer_solution/`:

```bash
zip -r ../solution.zip .
```
