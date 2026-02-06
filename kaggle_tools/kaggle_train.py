import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
WORK_ROOT = Path("/kaggle/working/wunder_kernel_src")
PROJECT_ROOT = WORK_ROOT
EMBEDDED_FILES = {}
DEFAULT_DATA_DIR = ""
DEFAULT_ENV_OVERRIDES = {}

def _materialize_embedded_files(project_root: Path) -> None:
    for relative_path, content in EMBEDDED_FILES.items():
        path = project_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(value) if value else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    for key, value in DEFAULT_ENV_OVERRIDES.items():
        os.environ.setdefault(key, str(value))

    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    _materialize_embedded_files(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    from train import TrainConfig, train  # noqa: E402

    data_root = os.environ.get("WUNDER_DATA_DIR")
    config_path = CURRENT_DIR / "kaggle_train_config.json"
    if not data_root and config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        data_root = config.get("data_dir")
    if not data_root and DEFAULT_DATA_DIR:
        data_root = DEFAULT_DATA_DIR

    if not data_root:
        raise RuntimeError(
            "WUNDER_DATA_DIR is not set. Provide the Kaggle dataset directory, "
            "e.g. /kaggle/input/wunder-challenge-2"
        )

    data_root_path = Path(data_root)
    train_path = data_root_path / "train.parquet"
    valid_path = data_root_path / "valid.parquet"

    output_dir = Path(os.environ.get("WUNDER_OUTPUT_DIR", "/kaggle/working/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        train_path=str(train_path),
        valid_path=str(valid_path),
        out_dir=str(output_dir),
        max_train_seqs=_env_int("WUNDER_MAX_TRAIN_SEQS", 0),
        max_valid_seqs=_env_int("WUNDER_MAX_VALID_SEQS", 0),
        context_len=_env_int("WUNDER_CONTEXT_LEN", 128),
        d_model=_env_int("WUNDER_D_MODEL", 128),
        nhead=_env_int("WUNDER_NHEAD", 8),
        num_layers=_env_int("WUNDER_NUM_LAYERS", 3),
        dim_feedforward=_env_int("WUNDER_DIM_FEEDFORWARD", 256),
        dropout=_env_float("WUNDER_DROPOUT", 0.1),
        batch_size=_env_int("WUNDER_BATCH_SIZE", 256),
        epochs=_env_int("WUNDER_EPOCHS", 3),
        lr=_env_float("WUNDER_LR", 2e-4),
        weight_decay=_env_float("WUNDER_WEIGHT_DECAY", 1e-4),
        seed=_env_int("WUNDER_SEED", 42),
        device=_env_str("WUNDER_DEVICE", "cuda"),
        num_workers=_env_int("WUNDER_NUM_WORKERS", 2),
        skip_validation=_env_bool("WUNDER_SKIP_VALIDATION", True),
        log_interval=_env_int("WUNDER_LOG_INTERVAL", 100),
        early_stopping_patience=_env_int("WUNDER_EARLY_STOPPING_PATIENCE", 3),
        early_stopping_min_delta=_env_float("WUNDER_EARLY_STOPPING_MIN_DELTA", 0.0),
    )

    train(cfg)

    artifacts = [
        "transformer_model.pt",
        "transformer_training_bundle.pt",
        "feature_stats.npz",
        "config.npz",
        "train_config.json",
        "train_history.json",
    ]

    for filename in artifacts:
        src = output_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing artifact {src}")

    for filename in [
        "solution.py",
        "model.py",
        "feature_engineering.py",
    ]:
        shutil.copy2(PROJECT_ROOT / filename, output_dir / filename)

    utils_path = PROJECT_ROOT / "utils.py"
    if utils_path.exists():
        shutil.copy2(utils_path, output_dir / "utils.py")

    solution_zip = output_dir / "solution.zip"
    with zipfile.ZipFile(solution_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in [
            "solution.py",
            "model.py",
            "feature_engineering.py",
            "transformer_model.pt",
            "feature_stats.npz",
            "config.npz",
            "train_config.json",
        ]:
            archive.write(output_dir / filename, arcname=filename)
        if (output_dir / "utils.py").exists():
            archive.write(output_dir / "utils.py", arcname="utils.py")

    report = {
        "output_dir": str(output_dir),
        "solution_zip": str(solution_zip),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
    }
    with open(output_dir / "kaggle_run_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
