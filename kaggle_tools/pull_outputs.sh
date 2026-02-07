#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../kaggle_outputs"
KAGGLE_PYTHON_BIN="${KAGGLE_PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-transformer_solution}"
CONFIG_PATH="${SCRIPT_DIR}/solution_kaggle_config.json"
export CONFIG_PATH
export MODEL_DIR

if [[ -z "${KAGGLE_KERNEL_ID:-}" && -f "${CONFIG_PATH}" ]]; then
  eval "$("${KAGGLE_PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

config_path = Path(os.environ["CONFIG_PATH"])
model_dir = os.environ["MODEL_DIR"]
if not config_path.exists():
    raise SystemExit(0)

config = json.loads(config_path.read_text(encoding="utf-8"))
entry = config.get(model_dir, {})
exports = []

if not os.environ.get("KAGGLE_KERNEL_ID") and entry.get("kernel_id"):
    exports.append(f"KAGGLE_KERNEL_ID='{entry['kernel_id']}'")

print("; ".join(exports))
PY
)"
fi

if [[ -z "${KAGGLE_KERNEL_ID:-}" ]]; then
  echo "KAGGLE_KERNEL_ID is required (e.g. username/wunder-transformer-train)." >&2
  echo "Set it directly or update ${CONFIG_PATH} for ${MODEL_DIR}." >&2
  exit 1
fi

kaggle_cli() {
  "${KAGGLE_PYTHON_BIN}" -m kaggle.cli "$@"
}

mkdir -p "${OUTPUT_DIR}"

kaggle_cli kernels output "${KAGGLE_KERNEL_ID}" -p "${OUTPUT_DIR}"

# Kaggle may place files directly under OUTPUT_DIR or under OUTPUT_DIR/outputs.
PULLED_DIR="${OUTPUT_DIR}"
if [[ -d "${OUTPUT_DIR}/outputs" ]]; then
  PULLED_DIR="${OUTPUT_DIR}/outputs"
fi

ARTIFACT_SOURCE="${PULLED_DIR}"
if [[ -d "${PULLED_DIR}/artifacts" ]]; then
  ARTIFACT_SOURCE="${PULLED_DIR}/artifacts"
fi

if [[ -n "${WUNDER_SYNC_DIR:-}" ]]; then
  echo "Syncing artifacts into ${WUNDER_SYNC_DIR}"
  TARGET_ARTIFACT_DIR="${WUNDER_SYNC_DIR}/artifacts"
  mkdir -p "${TARGET_ARTIFACT_DIR}"
  for filename in \
    transformer_model.pt \
    transformer_training_bundle.pt \
    feature_stats.npz \
    config.npz \
    train_config.json \
    train_history.json \
    solution.zip \
    kaggle_run_report.json
  do
    if [[ -f "${ARTIFACT_SOURCE}/${filename}" ]]; then
      cp "${ARTIFACT_SOURCE}/${filename}" "${TARGET_ARTIFACT_DIR}/${filename}"
    elif [[ -f "${PULLED_DIR}/${filename}" ]]; then
      cp "${PULLED_DIR}/${filename}" "${TARGET_ARTIFACT_DIR}/${filename}"
    fi
  done
  if [[ -d "${ARTIFACT_SOURCE}/checkpoints" ]]; then
    mkdir -p "${TARGET_ARTIFACT_DIR}/checkpoints"
    rsync -a --delete "${ARTIFACT_SOURCE}/checkpoints/" "${TARGET_ARTIFACT_DIR}/checkpoints/"
  fi
fi

# Always archive pulled results under archived/<timestamp>.
TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="${REPO_ROOT}/archived/${TS}"
mkdir -p "${ARCHIVE_DIR}"
ARCHIVE_ARTIFACT_DIR="${ARCHIVE_DIR}/artifacts"
mkdir -p "${ARCHIVE_ARTIFACT_DIR}"

for filename in \
  transformer_model.pt \
  transformer_training_bundle.pt \
  feature_stats.npz \
  config.npz \
  train_config.json \
  train_history.json \
  solution.zip \
  kaggle_run_report.json \
  solution.py \
  model.py \
  feature_engineering.py \
  utils.py
do
  if [[ -f "${ARTIFACT_SOURCE}/${filename}" ]]; then
    cp "${ARTIFACT_SOURCE}/${filename}" "${ARCHIVE_ARTIFACT_DIR}/${filename}"
  elif [[ -f "${PULLED_DIR}/${filename}" ]]; then
    cp "${PULLED_DIR}/${filename}" "${ARCHIVE_ARTIFACT_DIR}/${filename}"
  fi
done

if [[ -d "${ARTIFACT_SOURCE}/checkpoints" ]]; then
  rsync -a --delete "${ARTIFACT_SOURCE}/checkpoints/" "${ARCHIVE_ARTIFACT_DIR}/checkpoints/"
fi

echo "Archived pulled outputs to ${ARCHIVE_DIR}"
