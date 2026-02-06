#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${KAGGLE_KERNEL_ID:-}" ]]; then
  echo "KAGGLE_KERNEL_ID is required (e.g. username/wunder-transformer-train)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../kaggle_outputs"

mkdir -p "${OUTPUT_DIR}"

kaggle kernels output "${KAGGLE_KERNEL_ID}" -p "${OUTPUT_DIR}"

if [[ -f "${OUTPUT_DIR}/solution.zip" ]]; then
  unzip -o "${OUTPUT_DIR}/solution.zip" -d "${OUTPUT_DIR}/solution_unpacked"
fi

if [[ -n "${WUNDER_SYNC_DIR:-}" ]]; then
  echo "Syncing artifacts into ${WUNDER_SYNC_DIR}"
  rsync -av "${OUTPUT_DIR}/solution_unpacked/" "${WUNDER_SYNC_DIR}/"
fi
