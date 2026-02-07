#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${KAGGLE_COMPETITION:-}" && -z "${KAGGLE_DATASET:-}" ]]; then
  echo "Set KAGGLE_COMPETITION or KAGGLE_DATASET so Kaggle can mount the data." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-transformer_solution}"
MODEL_PATH="${REPO_ROOT}/${MODEL_DIR}"
KAGGLE_PYTHON_BIN="${KAGGLE_PYTHON_BIN:-python}"
CONFIG_PATH="${SCRIPT_DIR}/solution_kaggle_config.json"
export CONFIG_PATH
export MODEL_DIR

if [[ -f "${CONFIG_PATH}" ]]; then
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
if not os.environ.get("KAGGLE_KERNEL_TITLE") and entry.get("kernel_title"):
    exports.append(f"KAGGLE_KERNEL_TITLE='{entry['kernel_title']}'")
if not os.environ.get("KAGGLE_NOTEBOOK_ID") and entry.get("notebook_id"):
    exports.append(f"KAGGLE_NOTEBOOK_ID='{entry['notebook_id']}'")
if not os.environ.get("KAGGLE_NOTEBOOK_TITLE") and entry.get("notebook_title"):
    exports.append(f"KAGGLE_NOTEBOOK_TITLE='{entry['notebook_title']}'")

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

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "MODEL_DIR does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

PROJECT_ROOT="$(cd "${MODEL_PATH}" && pwd)"
STAGING_DIR="${SCRIPT_DIR}/staging"

rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"
export STAGING_DIR
export SCRIPT_DIR
export REPO_ROOT
export PROJECT_ROOT

cp "${SCRIPT_DIR}/kaggle_train.py" "${STAGING_DIR}/kaggle_train.py"
cp "${PROJECT_ROOT}/train.py" "${STAGING_DIR}/train.py"
cp "${PROJECT_ROOT}/model.py" "${STAGING_DIR}/model.py"
cp "${PROJECT_ROOT}/solution.py" "${STAGING_DIR}/solution.py"
cp "${REPO_ROOT}/utils.py" "${STAGING_DIR}/utils.py"
if [[ -f "${PROJECT_ROOT}/feature_engineering.py" ]]; then
  cp "${PROJECT_ROOT}/feature_engineering.py" "${STAGING_DIR}/feature_engineering.py"
fi

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

staging_dir = Path(os.environ["STAGING_DIR"])
project_root = Path(os.environ["PROJECT_ROOT"])
repo_root = Path(os.environ["REPO_ROOT"])

files_to_embed = {}
files_to_embed["train.py"] = (project_root / "train.py").read_text(encoding="utf-8")
files_to_embed["model.py"] = (project_root / "model.py").read_text(encoding="utf-8")
files_to_embed["solution.py"] = (project_root / "solution.py").read_text(encoding="utf-8")
feature_engineering = project_root / "feature_engineering.py"
if feature_engineering.exists():
    files_to_embed["feature_engineering.py"] = feature_engineering.read_text(encoding="utf-8")
files_to_embed["utils.py"] = (repo_root / "utils.py").read_text(encoding="utf-8")

embedded_file_list = [
    name for name in ("solution.py", "model.py", "feature_engineering.py") if name in files_to_embed
]

kaggle_train_path = staging_dir / "kaggle_train.py"
kaggle_train_template = kaggle_train_path.read_text(encoding="utf-8")
kaggle_train_filled = kaggle_train_template.replace(
    "EMBEDDED_FILES = {}", f"EMBEDDED_FILES = {files_to_embed!r}", 1
)
kaggle_train_filled = kaggle_train_filled.replace(
    "EMBEDDED_FILE_LIST = []", f"EMBEDDED_FILE_LIST = {embedded_file_list!r}", 1
)

wunder_overrides = {k: v for k, v in os.environ.items() if k.startswith("WUNDER_")}
kaggle_train_filled = kaggle_train_filled.replace(
    "DEFAULT_ENV_OVERRIDES = {}", f"DEFAULT_ENV_OVERRIDES = {wunder_overrides!r}", 1
)

kaggle_train_path.write_text(kaggle_train_filled, encoding="utf-8")

data_slug = None
if competition:
    data_slug = competition
elif dataset:
    data_slug = dataset.split("/")[-1]

if data_slug:
    config = {"data_dir": f"/kaggle/input/{data_slug}"}
    config_path = Path(os.environ["STAGING_DIR"]) / "kaggle_train_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    kaggle_train_filled = kaggle_train_filled.replace(
        'DEFAULT_DATA_DIR = ""', f'DEFAULT_DATA_DIR = "/kaggle/input/{data_slug}"', 1
    )
    kaggle_train_path.write_text(kaggle_train_filled, encoding="utf-8")
PY

kaggle_cli kernels push -p "${STAGING_DIR}"

cat <<INFO
Kernel pushed. Monitor status with:
  kaggle kernels status ${KAGGLE_KERNEL_ID}
INFO
