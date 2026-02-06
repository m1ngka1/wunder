#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SOURCE_DATA_DIR="${REPO_ROOT}/datasets"
STAGING_DIR="${SCRIPT_DIR}/dataset_staging"
KAGGLE_PYTHON_BIN="${KAGGLE_PYTHON_BIN:-python}"

kaggle_cli() {
  "${KAGGLE_PYTHON_BIN}" -m kaggle.cli "$@"
}

if [[ -z "${KAGGLE_DATASET_ID:-}" ]]; then
  echo "KAGGLE_DATASET_ID is required (e.g. username/wunder-train-valid-parquet)." >&2
  exit 1
fi

if [[ ! -f "${SOURCE_DATA_DIR}/train.parquet" || ! -f "${SOURCE_DATA_DIR}/valid.parquet" ]]; then
  echo "Expected train.parquet and valid.parquet under ${SOURCE_DATA_DIR}" >&2
  exit 1
fi

DATASET_SLUG="${KAGGLE_DATASET_ID#*/}"
DATASET_TITLE="${KAGGLE_DATASET_TITLE:-Wunder Train+Valid Parquet}"
VERSION_MSG="${KAGGLE_DATASET_MESSAGE:-Update train/valid parquet files}"

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

# Use hard links when possible to avoid large file copy overhead.
ln "${SOURCE_DATA_DIR}/train.parquet" "${STAGING_DIR}/train.parquet" 2>/dev/null || cp "${SOURCE_DATA_DIR}/train.parquet" "${STAGING_DIR}/train.parquet"
ln "${SOURCE_DATA_DIR}/valid.parquet" "${STAGING_DIR}/valid.parquet" 2>/dev/null || cp "${SOURCE_DATA_DIR}/valid.parquet" "${STAGING_DIR}/valid.parquet"

cat > "${STAGING_DIR}/dataset-metadata.json" <<EOF
{
  "id": "${KAGGLE_DATASET_ID}",
  "title": "${DATASET_TITLE}",
  "licenses": [
    { "name": "CC0-1.0" }
  ]
}
EOF

if kaggle_cli datasets status "${KAGGLE_DATASET_ID}" >/dev/null 2>&1; then
  kaggle_cli datasets version -p "${STAGING_DIR}" -m "${VERSION_MSG}" -r skip
  echo "Updated existing dataset: ${KAGGLE_DATASET_ID}"
else
  kaggle_cli datasets create -p "${STAGING_DIR}" -r skip
  echo "Created new dataset: ${KAGGLE_DATASET_ID}"
fi

cat <<INFO
Use this dataset for kernel runs:
  export KAGGLE_DATASET=${KAGGLE_DATASET_ID}
INFO
