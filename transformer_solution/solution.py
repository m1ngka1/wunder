import os
import sys
from collections import deque

import numpy as np
import torch

# Adjust path to import utils from parent directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils import DataPoint, ScorerStepByStep  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402
from model import LOBTransformer  # noqa: E402


class PredictionModel:
    """
    Transformer-based streaming predictor.

    Expected artifacts next to this file:
      - transformer_model.pt
      - feature_stats.npz
      - config.npz
    """

    def __init__(self, model_path: str = ""):
        self.current_seq_ix = None
        self.device = torch.device("cpu")
        self._loaded = False

        if model_path:
            artifact_dir = model_path
        else:
            artifact_dir = CURRENT_DIR

        config_path = os.path.join(artifact_dir, "config.npz")
        stats_path = os.path.join(artifact_dir, "feature_stats.npz")
        ckpt_path = os.path.join(artifact_dir, "transformer_model.pt")

        # Defaults let code run even if artifacts are missing.
        self.context_len = 128
        self.feature_dim = 51
        self.feature_mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.feature_std = np.ones(self.feature_dim, dtype=np.float32)
        self.model = LOBTransformer(input_dim=self.feature_dim, max_len=self.context_len)
        self.history = deque(maxlen=self.context_len)
        self.feature_engineer = FeatureEngineer()

        try:
            if os.path.exists(config_path):
                cfg = np.load(config_path)
                self.context_len = int(cfg["context_len"])
                if "feature_dim" in cfg:
                    self.feature_dim = int(cfg["feature_dim"])
                d_model = int(cfg["d_model"])
                nhead = int(cfg["nhead"])
                num_layers = int(cfg["num_layers"])
                dim_feedforward = int(cfg["dim_feedforward"])
                dropout = float(cfg["dropout"])
            else:
                d_model = 128
                nhead = 8
                num_layers = 3
                dim_feedforward = 256
                dropout = 0.1

            self.model = LOBTransformer(
                input_dim=self.feature_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_len=self.context_len,
            ).to(self.device)
            self.model.eval()
            self.history = deque(maxlen=self.context_len)
            self.feature_engineer = FeatureEngineer()

            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                self.feature_mean = stats["mean"].astype(np.float32)
                self.feature_std = stats["std"].astype(np.float32)
                self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
                if self.feature_mean.shape[0] != self.feature_dim:
                    print(
                        "[WARN] Feature stats dimension mismatch. "
                        "Falling back to default normalization."
                    )
                    self.feature_mean = np.zeros(self.feature_dim, dtype=np.float32)
                    self.feature_std = np.ones(self.feature_dim, dtype=np.float32)

            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location="cpu")
                self.model.load_state_dict(state_dict, strict=True)
                self._loaded = True
        except Exception as exc:
            print(f"[WARN] Failed to load transformer artifacts: {exc}")
            self._loaded = False

    def _reset_sequence(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.history.clear()
        self.feature_engineer.reset()

    def _prepare_model_input(self) -> torch.Tensor:
        window = np.asarray(self.history, dtype=np.float32)
        window = (window - self.feature_mean) / self.feature_std
        tensor = torch.from_numpy(window).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self._reset_sequence(data_point.seq_ix)

        engineered = self.feature_engineer.update(np.asarray(data_point.state, dtype=np.float32))
        self.history.append(engineered)

        if not data_point.need_prediction:
            return None

        if not self._loaded:
            return np.zeros(2, dtype=np.float32)

        with torch.no_grad():
            x = self._prepare_model_input()
            pred = self.model(x).cpu().numpy()[0]
        return pred.astype(np.float32)


if __name__ == "__main__":
    # Local testing
    test_file = f"{CURRENT_DIR}/../datasets/valid.parquet"

    if os.path.exists(test_file):
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        print("Testing Transformer solution...")
        results = scorer.score(model)
        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for target in scorer.targets:
            print(f"  {target}: {results[target]:.6f}")
    else:
        print("Valid parquet not found for testing.")
