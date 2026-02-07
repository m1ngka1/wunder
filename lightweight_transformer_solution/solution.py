import os
import sys
from collections import deque

import numpy as np
import torch
try:
    import onnxruntime as ort
except Exception:
    ort = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils import DataPoint  # noqa: E402
from model import LightweightLOBTransformer  # noqa: E402


class PredictionModel:
    """
    Lightweight transformer streaming predictor (no feature engineering).

    Expected artifacts under `artifacts/` next to this file:
      - artifacts/transformer_model.pt
      - artifacts/feature_stats.npz
      - artifacts/config.npz
    """

    def __init__(self, model_path: str = ""):
        self.current_seq_ix = None
        self.device = torch.device("cpu")
        self._loaded = False
        self._use_onnx = False
        self._ort_session = None
        self._ort_input_name = None

        if model_path:
            if os.path.isdir(os.path.join(model_path, "artifacts")):
                artifact_dir = os.path.join(model_path, "artifacts")
            else:
                artifact_dir = model_path
        else:
            default_artifact_dir = os.path.join(CURRENT_DIR, "artifacts")
            artifact_dir = default_artifact_dir if os.path.isdir(default_artifact_dir) else CURRENT_DIR

        config_path = os.path.join(artifact_dir, "config.npz")
        stats_path = os.path.join(artifact_dir, "feature_stats.npz")
        ckpt_path = os.path.join(artifact_dir, "transformer_model.pt")
        onnx_path = os.path.join(artifact_dir, "transformer_model.onnx")

        # Defaults let code run even if artifacts are missing.
        self.context_len = 32
        self.feature_dim = 32
        self.feature_mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.feature_std = np.ones(self.feature_dim, dtype=np.float32)
        self.model = LightweightLOBTransformer(input_dim=self.feature_dim, max_len=self.context_len)
        self.history = deque(maxlen=self.context_len)

        try:
            d_model = 32
            nhead = 4
            num_layers = 2
            dim_feedforward = 128
            dropout = 0.1
            cfg_has_feature_dim = False

            if os.path.exists(config_path):
                cfg = np.load(config_path)
                self.context_len = int(cfg["context_len"])
                if "feature_dim" in cfg:
                    self.feature_dim = int(cfg["feature_dim"])
                    cfg_has_feature_dim = True
                if "d_model" in cfg:
                    d_model = int(cfg["d_model"])
                if "nhead" in cfg:
                    nhead = int(cfg["nhead"])
                if "num_layers" in cfg:
                    num_layers = int(cfg["num_layers"])
                if "dim_feedforward" in cfg:
                    dim_feedforward = int(cfg["dim_feedforward"])
                if "dropout" in cfg:
                    dropout = float(cfg["dropout"])

            stats_mean = None
            stats_std = None
            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                stats_mean = stats["mean"].astype(np.float32)
                stats_std = stats["std"].astype(np.float32)
                stats_std = np.where(stats_std < 1e-6, 1.0, stats_std)
                if not cfg_has_feature_dim:
                    self.feature_dim = int(stats_mean.shape[0])

            self.model = LightweightLOBTransformer(
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

            if stats_mean is not None and stats_std is not None:
                self.feature_mean = stats_mean
                self.feature_std = stats_std
                if self.feature_mean.shape[0] != self.feature_dim:
                    print(
                        "[WARN] Feature stats dimension mismatch. "
                        "Falling back to default normalization."
                    )
                    self.feature_mean = np.zeros(self.feature_dim, dtype=np.float32)
                    self.feature_std = np.ones(self.feature_dim, dtype=np.float32)

            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location="cpu")
                if "input_proj.weight" in state_dict:
                    ckpt_feature_dim = int(state_dict["input_proj.weight"].shape[1])
                    if ckpt_feature_dim != self.feature_dim:
                        raise ValueError(
                            "Checkpoint/model feature dim mismatch: "
                            f"checkpoint={ckpt_feature_dim}, configured={self.feature_dim}"
                        )
                self.model.load_state_dict(state_dict, strict=True)
                self._loaded = True
                self._init_onnx_runtime(ckpt_path=ckpt_path, onnx_path=onnx_path)
        except Exception as exc:
            print(f"[WARN] Failed to load lightweight transformer artifacts: {exc}")
            self._loaded = False

    def _export_onnx_if_needed(self, ckpt_path: str, onnx_path: str):
        if ort is None:
            return
        needs_export = True
        if os.path.exists(onnx_path):
            needs_export = os.path.getmtime(onnx_path) < os.path.getmtime(ckpt_path)
        if not needs_export:
            return

        self.model.eval()
        dummy_input = torch.zeros(
            1, self.context_len, self.feature_dim, dtype=torch.float32, device=self.device
        )
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=["x"],
            output_names=["y"],
            opset_version=17,
            do_constant_folding=True,
        )

    def _init_onnx_runtime(self, ckpt_path: str, onnx_path: str):
        if ort is None:
            return
        try:
            self._export_onnx_if_needed(ckpt_path=ckpt_path, onnx_path=onnx_path)
            if not os.path.exists(onnx_path):
                return
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._ort_session = ort.InferenceSession(
                onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
            )
            self._ort_input_name = self._ort_session.get_inputs()[0].name
            self._use_onnx = True
            print(f"[INFO] Using ONNX Runtime with {onnx_path}")
        except Exception as exc:
            print(f"[WARN] ONNX runtime init failed, fallback to torch: {exc}")
            self._use_onnx = False
            self._ort_session = None
            self._ort_input_name = None

    def _reset_sequence(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.history.clear()

    def _prepare_model_input_np(self) -> np.ndarray:
        window = np.zeros((self.context_len, self.feature_dim), dtype=np.float32)
        if len(self.history) > 0:
            hist_arr = np.asarray(self.history, dtype=np.float32)
            use_len = min(hist_arr.shape[0], self.context_len)
            window[-use_len:] = hist_arr[-use_len:]
        window = (window - self.feature_mean) / self.feature_std
        return window

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self._reset_sequence(data_point.seq_ix)

        self.history.append(np.asarray(data_point.state, dtype=np.float32))

        if not data_point.need_prediction:
            return None

        if not self._loaded:
            return np.zeros(2, dtype=np.float32)

        x_np = self._prepare_model_input_np()[None, ...]  # [1, context_len, feature_dim]

        if self._use_onnx and self._ort_session is not None and self._ort_input_name is not None:
            pred = self._ort_session.run(None, {self._ort_input_name: x_np})[0][0]
            return pred.astype(np.float32)

        with torch.no_grad():
            x = torch.from_numpy(x_np).to(self.device)
            pred = self.model(x).cpu().numpy()[0]
            return pred.astype(np.float32)
