# Kaggle GPU workflow (Transformer)

This folder provides scripts to push a solution's training code to Kaggle, run training on a GPU, and pull back trained artifacts for submission.

By default, the scripts stage `train.py`, `model.py`, `feature_engineering.py`, and `solution.py` from the selected `MODEL_DIR`. If your future solution uses different filenames, update the scripts accordingly.

## Prerequisites

- Kaggle API credentials configured locally (`~/.kaggle/kaggle.json`).
- Kaggle CLI installed (`pip install kaggle`).

## Environment variables

Set these before running the scripts:

- `KAGGLE_KERNEL_ID`: Your Kaggle kernel identifier, e.g. `username/wunder-transformer-train`.
- `KAGGLE_COMPETITION`: Competition slug that hosts the dataset (e.g. `wunder-challenge-2`).
  - Alternatively, set `KAGGLE_DATASET` to a Kaggle dataset slug if you uploaded the data as a dataset.
- Optional: `KAGGLE_KERNEL_TITLE` to override the kernel display title.
- Optional: `MODEL_DIR` to choose which solution folder to stage (defaults to `transformer_solution`).
- Optional: `WUNDER_*` environment variables can override training hyperparameters (e.g. `WUNDER_EPOCHS=12`).
- Optional: `WUNDER_SKIP_VALIDATION=1` to skip validation during long runs (enabled by default in `kaggle_train.py`).

## Upload train/valid parquet once as a Kaggle dataset

To avoid re-uploading parquet files for every kernel push:

```bash
cd kaggle_tools
export KAGGLE_DATASET_ID="mingkaijia/wunder-train-valid-parquet"
./push_dataset.sh
```

Then use this dataset in training pushes:

```bash
export KAGGLE_DATASET="mingkaijia/wunder-train-valid-parquet"
```

## Push + run training on Kaggle

```bash
cd kaggle_tools
export KAGGLE_KERNEL_ID="mingkaijia/wunder-transformer-train"
export KAGGLE_DATASET="mingkaijia/wunder-train-valid-parquet"
export WUNDER_EPOCHS=50
export WUNDER_SKIP_VALIDATION=1
./push_kernel.sh
```

This creates a staging directory, writes `kernel-metadata.json`, and pushes the kernel to Kaggle. Kaggle will start a run automatically.

The push script also writes `kaggle_train_config.json` so the kernel knows which `/kaggle/input/...` path to use without setting `WUNDER_DATA_DIR` manually.

## Pull trained artifacts back

```bash
cd kaggle_tools
./pull_outputs.sh
```

This downloads the latest kernel outputs into `kaggle_outputs/`.

If you want to sync the artifacts back into the main transformer solution folder for local inference or packaging, set:

```bash
export WUNDER_SYNC_DIR="../transformer_solution"
./pull_outputs.sh
```

That will copy the trained artifacts into `<solution_dir>/artifacts/`.

## Notes

- The Kaggle run writes artifacts into `/kaggle/working/outputs/artifacts` and packages `artifacts/solution.zip`.
- You can monitor kernel status via `kaggle kernels status $KAGGLE_KERNEL_ID`.

## GitHub Actions setup (for cloud/phone triggering)

You can trigger runs from GitHub Actions without local files by using the workflow:

- `.github/workflows/kaggle-run.yml`

### 1. Add repository secrets

In GitHub UI: `Settings` -> `Secrets and variables` -> `Actions` -> `New repository secret`:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

### 2. Trigger the workflow

In GitHub UI: `Actions` -> `Kaggle Run` -> `Run workflow`.

Set inputs such as:

- `kernel_id`: `mingkaijia/wunder-transformer-gpu-train`
- `dataset_id`: `mingkaijia/wunder-train-valid-parquet`
- `epochs`: `2`
- `nhead`: `4`
- `skip_validation`: `0`

The workflow bootstraps Kaggle auth via `kaggle_tools/bootstrap_kaggle_auth.sh` and pushes a kernel run using `kaggle_tools/push_kernel.sh`.
