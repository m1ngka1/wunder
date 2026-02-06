# Kaggle GPU workflow (Transformer)

This folder provides scripts to push the Transformer training code to Kaggle, run training on a GPU, and pull back trained artifacts for submission.

## Prerequisites

- Kaggle API credentials configured locally (`~/.kaggle/kaggle.json`).
- Kaggle CLI installed (`pip install kaggle`).

## Environment variables

Set these before running the scripts:

- `KAGGLE_KERNEL_ID`: Your Kaggle kernel identifier, e.g. `username/wunder-transformer-train`.
- `KAGGLE_COMPETITION`: Competition slug that hosts the dataset (e.g. `wunder-challenge-2`).
  - Alternatively, set `KAGGLE_DATASET` to a Kaggle dataset slug if you uploaded the data as a dataset.
- Optional: `KAGGLE_KERNEL_TITLE` to override the kernel display title.
- Optional: `WUNDER_*` environment variables can override training hyperparameters (e.g. `WUNDER_EPOCHS=12`).

## Push + run training on Kaggle

```bash
cd transformer_solution/kaggle_tools
./push_kernel.sh
```

This creates a staging directory, writes `kernel-metadata.json`, and pushes the kernel to Kaggle. Kaggle will start a run automatically.

The push script also writes `kaggle_train_config.json` so the kernel knows which `/kaggle/input/...` path to use without setting `WUNDER_DATA_DIR` manually.

## Pull trained artifacts back

```bash
cd transformer_solution/kaggle_tools
./pull_outputs.sh
```

This downloads the latest kernel outputs into `transformer_solution/kaggle_outputs/`.

If you want to sync the artifacts back into the main transformer solution folder for local inference or packaging, set:

```bash
export WUNDER_SYNC_DIR="../"
./pull_outputs.sh
```

That will copy the trained artifacts into `transformer_solution/`.

## Notes

- The Kaggle run writes artifacts into `/kaggle/working/outputs` and packages a `solution.zip` containing the trained model + inference code.
- You can monitor kernel status via `kaggle kernels status $KAGGLE_KERNEL_ID`.
