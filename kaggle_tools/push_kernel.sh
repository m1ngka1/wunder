#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${KAGGLE_KERNEL_ID:-}" ]]; then
  echo "KAGGLE_KERNEL_ID is required (e.g. username/wunder-transformer-train)." >&2
  exit 1
fi

if [[ -z "${KAGGLE_COMPETITION:-}" && -z "${KAGGLE_DATASET:-}" ]]; then
  echo "Set KAGGLE_COMPETITION or KAGGLE_DATASET so Kaggle can mount the data." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-transformer_solution}"
MODEL_PATH="${REPO_ROOT}/${MODEL_DIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "${MODEL_PATH}" && pwd)"
STAGING_DIR="${SCRIPT_DIR}/staging"

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"
export STAGING_DIR

cp "${SCRIPT_DIR}/kaggle_train.py" "${STAGING_DIR}/kaggle_train.py"
cp "${PROJECT_ROOT}/train.py" "${STAGING_DIR}/train.py"
cp "${PROJECT_ROOT}/model.py" "${STAGING_DIR}/model.py"
cp "${PROJECT_ROOT}/feature_engineering.py" "${STAGING_DIR}/feature_engineering.py"
cp "${PROJECT_ROOT}/solution.py" "${STAGING_DIR}/solution.py"
cp "${REPO_ROOT}/utils.py" "${STAGING_DIR}/utils.py"

python - <<'PY'
import json
import os
from pathlib import Path

kernel_id = os.environ["KAGGLE_KERNEL_ID"]
kernel_title = os.environ.get("KAGGLE_KERNEL_TITLE", "Wunder Transformer GPU Train")
competition = os.environ.get("KAGGLE_COMPETITION")
dataset = os.environ.get("KAGGLE_DATASET")

metadata = {
    "id": kernel_id,
    "title": kernel_title,
    "code_file": "kaggle_train.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
}

if competition:
    metadata["competition_sources"] = [competition]
if dataset:
    metadata["dataset_sources"] = [dataset]

output_path = Path(os.environ["STAGING_DIR"]) / "kernel-metadata.json"
output_path.write_text(json.dumps(metadata, indent=2))

data_slug = None
if competition:
    data_slug = competition
elif dataset:
    data_slug = dataset.split("/")[-1]

if data_slug:
    config = {"data_dir": f"/kaggle/input/{data_slug}"}
    config_path = Path(os.environ["STAGING_DIR"]) / "kaggle_train_config.json"
    config_path.write_text(json.dumps(config, indent=2))
PY

kaggle kernels push -p "${STAGING_DIR}"

cat <<INFO
Kernel pushed. Monitor status with:
  kaggle kernels status ${KAGGLE_KERNEL_ID}
INFO
