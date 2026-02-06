#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  echo "KAGGLE_USERNAME and KAGGLE_KEY must be set." >&2
  exit 1
fi

KAGGLE_DIR="${HOME}/.kaggle"
KAGGLE_JSON="${KAGGLE_DIR}/kaggle.json"

mkdir -p "${KAGGLE_DIR}"
cat > "${KAGGLE_JSON}" <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
chmod 600 "${KAGGLE_JSON}"

python -m pip install --quiet --upgrade kaggle
python -m kaggle.cli --version

