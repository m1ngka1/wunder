import json
import os
import shutil
import sys
import zipfile
import inspect
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
WORK_ROOT = Path("/kaggle/working/wunder_kernel_src")
PROJECT_ROOT = WORK_ROOT
EMBEDDED_FILES = {}
EMBEDDED_FILE_LIST = []
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

def _build_train_config(TrainConfig, train_path: Path, valid_path: Path, output_dir: Path):
    # Always pass core paths; all other parameters are opt-in via WUNDER_* env vars.
    candidate_kwargs = {
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "out_dir": str(output_dir),
    }
    env_specs = {
        "max_train_seqs": ("WUNDER_MAX_TRAIN_SEQS", int),
        "max_valid_seqs": ("WUNDER_MAX_VALID_SEQS", int),
        "context_len": ("WUNDER_CONTEXT_LEN", int),
        "d_model": ("WUNDER_D_MODEL", int),
        "nhead": ("WUNDER_NHEAD", int),
        "num_layers": ("WUNDER_NUM_LAYERS", int),
        "dim_feedforward": ("WUNDER_DIM_FEEDFORWARD", int),
        "dropout": ("WUNDER_DROPOUT", float),
        "batch_size": ("WUNDER_BATCH_SIZE", int),
        "epochs": ("WUNDER_EPOCHS", int),
        "lr": ("WUNDER_LR", float),
        "weight_decay": ("WUNDER_WEIGHT_DECAY", float),
        "seed": ("WUNDER_SEED", int),
        "device": ("WUNDER_DEVICE", str),
        "num_workers": ("WUNDER_NUM_WORKERS", int),
        "skip_validation": ("WUNDER_SKIP_VALIDATION", lambda v: v.strip().lower() in {"1", "true", "yes", "y", "on"}),
        "log_interval": ("WUNDER_LOG_INTERVAL", int),
        "early_stopping_patience": ("WUNDER_EARLY_STOPPING_PATIENCE", int),
        "early_stopping_min_delta": ("WUNDER_EARLY_STOPPING_MIN_DELTA", float),
        "hybrid_loss_alpha": ("WUNDER_HYBRID_LOSS_ALPHA", float),
    }
    for key, (env_name, caster) in env_specs.items():
        if env_name in os.environ:
            candidate_kwargs[key] = caster(os.environ[env_name])

    accepted = set(inspect.signature(TrainConfig).parameters.keys())
    filtered_kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}

    dropped = sorted(set(candidate_kwargs) - set(filtered_kwargs))
    if dropped:
        print(f"[INFO] Ignoring unsupported TrainConfig args: {', '.join(dropped)}")

    return TrainConfig(**filtered_kwargs)


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
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_train_config(
        TrainConfig=TrainConfig,
        train_path=train_path,
        valid_path=valid_path,
        output_dir=output_dir,
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
        src = artifact_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing artifact {src}")

    default_files = [
        "solution.py",
        "model.py",
        "feature_engineering.py",
    ]
    project_files = EMBEDDED_FILE_LIST or [
        filename for filename in default_files if (PROJECT_ROOT / filename).exists()
    ]
    for filename in project_files:
        shutil.copy2(PROJECT_ROOT / filename, output_dir / filename)

    utils_path = PROJECT_ROOT / "utils.py"
    if utils_path.exists():
        shutil.copy2(utils_path, output_dir / "utils.py")

    solution_zip = artifact_dir / "solution.zip"
    with zipfile.ZipFile(solution_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in project_files:
            archive.write(output_dir / filename, arcname=filename)
        for filename in [
            "transformer_model.pt",
            "feature_stats.npz",
            "config.npz",
            "train_config.json",
        ]:
            archive.write(artifact_dir / filename, arcname=f"artifacts/{filename}")
        if (output_dir / "utils.py").exists():
            archive.write(output_dir / "utils.py", arcname="utils.py")

    report = {
        "output_dir": str(output_dir),
        "artifact_dir": str(artifact_dir),
        "solution_zip": str(solution_zip),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
    }
    with open(artifact_dir / "kaggle_run_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
