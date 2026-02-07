import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from model import LightweightLOBTransformer


@dataclass
class TrainConfig:
    train_path: str
    valid_path: str
    out_dir: str
    max_train_seqs: int
    max_valid_seqs: int
    context_len: int
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    seed: int
    device: str
    num_workers: int
    skip_validation: bool
    log_interval: int
    early_stopping_patience: int
    early_stopping_min_delta: float


class SequenceWindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        context_len: int,
    ):
        """
        features: [n_seq, 1000, feature_dim]
        targets: [n_seq, 1000, 2]
        """
        feat = torch.from_numpy(features).contiguous()
        tgt = torch.from_numpy(targets).contiguous()
        n_seq, seq_len, feat_dim = feat.shape

        # Pre-pad once so every sample window is a fixed contiguous slice.
        pad = torch.zeros((n_seq, context_len - 1, feat_dim), dtype=feat.dtype)
        self.features_padded = torch.cat([pad, feat], dim=1).contiguous()
        self.targets = tgt
        self.context_len = context_len
        self.seq_len = seq_len
        self.start_idx = context_len - 1
        self.samples_per_seq = seq_len - self.start_idx

    def __len__(self):
        return self.targets.shape[0] * self.samples_per_seq

    def __getitem__(self, index: int):
        seq_idx = index // self.samples_per_seq
        pred_idx = self.start_idx + (index % self.samples_per_seq)

        # With prefix padding, window always exists and has fixed length.
        start = pred_idx
        end = pred_idx + self.context_len
        window = self.features_padded[seq_idx, start:end]
        y = self.targets[seq_idx, pred_idx]
        return window, y


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log(message: str):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def save_artifacts(
    artifact_dir: str,
    epoch: int,
    model_state_dict: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    history: dict,
    best_score: float,
    mean: np.ndarray,
    std: np.ndarray,
    cfg: TrainConfig,
    feature_dim: int,
):
    checkpoints_dir = os.path.join(artifact_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    epoch_dir = os.path.join(checkpoints_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    bundle = {
        "epoch": epoch,
        "state_dict": model_state_dict,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
        "best_score": float(best_score),
    }

    # Root-level latest artifacts for inference and quick inspection.
    torch.save(model_state_dict, os.path.join(artifact_dir, "transformer_model.pt"))
    torch.save(bundle, os.path.join(artifact_dir, "transformer_training_bundle.pt"))
    np.savez(os.path.join(artifact_dir, "feature_stats.npz"), mean=mean, std=std)
    np.savez(
        os.path.join(artifact_dir, "config.npz"),
        context_len=cfg.context_len,
        feature_dim=feature_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    )
    with open(os.path.join(artifact_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
    with open(os.path.join(artifact_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Epoch-specific checkpoint for resume/recovery after timeout/cancel.
    torch.save(model_state_dict, os.path.join(epoch_dir, "transformer_model.pt"))
    torch.save(bundle, os.path.join(epoch_dir, "transformer_training_bundle.pt"))
    np.savez(os.path.join(epoch_dir, "feature_stats.npz"), mean=mean, std=std)
    np.savez(
        os.path.join(epoch_dir, "config.npz"),
        context_len=cfg.context_len,
        feature_dim=feature_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    )
    with open(os.path.join(epoch_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)
    with open(os.path.join(epoch_dir, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def read_and_reshape(path: str, max_seqs: int | None = None):
    df = pd.read_parquet(path)
    feature_cols = df.columns[3:35]
    target_cols = df.columns[35:]

    # Ensure strict temporal ordering within each sequence.
    df = df.sort_values(["seq_ix", "step_in_seq"], kind="mergesort")

    seq_ids = df["seq_ix"].drop_duplicates().values
    if max_seqs is not None and max_seqs > 0:
        seq_ids = seq_ids[:max_seqs]
        df = df[df["seq_ix"].isin(seq_ids)]

    n_seq = len(seq_ids)
    if n_seq == 0:
        raise ValueError(f"No sequences loaded from {path}")

    feat = df[feature_cols].values.astype(np.float32).reshape(n_seq, 1000, len(feature_cols))
    tgt = df[target_cols].values.astype(np.float32).reshape(n_seq, 1000, len(target_cols))
    return feat, tgt


def weighted_pearson_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    y_pred = torch.clamp(y_pred, -6.0, 6.0)
    weights = torch.abs(y_true).clamp_min(1e-8)
    sum_w = weights.sum(dim=0)

    mean_true = (y_true * weights).sum(dim=0) / sum_w
    mean_pred = (y_pred * weights).sum(dim=0) / sum_w

    dev_true = y_true - mean_true
    dev_pred = y_pred - mean_pred

    cov = (weights * dev_true * dev_pred).sum(dim=0) / sum_w
    var_true = (weights * dev_true.pow(2)).sum(dim=0) / sum_w
    var_pred = (weights * dev_pred.pow(2)).sum(dim=0) / sum_w

    corr = cov / (torch.sqrt(var_true) * torch.sqrt(var_pred) + 1e-8)
    return -corr.mean()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
):
    model.eval()
    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(x)
                loss = weighted_pearson_loss(y, pred)
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
    return loss_sum / max(n, 1)


def train(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    artifact_dir = os.path.join(cfg.out_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    set_seed(cfg.seed)
    log(f"Training config: {cfg}")

    torch.set_float32_matmul_precision("high")
    if cfg.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    t0 = time.time()
    log(f"Loading train parquet: {cfg.train_path}")
    train_feat, train_tgt = read_and_reshape(cfg.train_path, cfg.max_train_seqs)
    log(
        f"Loaded train data in {time.time() - t0:.1f}s | "
        f"shape={train_feat.shape}, targets={train_tgt.shape}"
    )

    if not cfg.skip_validation:
        t0 = time.time()
        log(f"Loading valid parquet: {cfg.valid_path}")
        valid_feat, valid_tgt = read_and_reshape(cfg.valid_path, cfg.max_valid_seqs)
        log(
            f"Loaded valid data in {time.time() - t0:.1f}s | "
            f"shape={valid_feat.shape}, targets={valid_tgt.shape}"
        )

    log("Computing normalization stats from train features...")
    mean = train_feat.reshape(-1, train_feat.shape[-1]).mean(axis=0).astype(np.float32)
    std = train_feat.reshape(-1, train_feat.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    log("Applying normalization...")
    train_feat = ((train_feat - mean) / std).astype(np.float32)
    if not cfg.skip_validation:
        valid_feat = ((valid_feat - mean) / std).astype(np.float32)
    feature_dim = train_feat.shape[-1]

    log("Building datasets...")
    train_ds = SequenceWindowDataset(train_feat, train_tgt, cfg.context_len)
    del train_feat
    if not cfg.skip_validation:
        valid_ds = SequenceWindowDataset(valid_feat, valid_tgt, cfg.context_len)
        del valid_feat

    log(
        f"Dataset ready | train_samples={len(train_ds)}"
        + ("" if cfg.skip_validation else f", valid_samples={len(valid_ds)}")
    )

    pin_memory = cfg.device != "cpu"
    loader_workers = max(cfg.num_workers, 0)
    train_loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    if loader_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    if not cfg.skip_validation:
        valid_loader_kwargs = dict(
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=pin_memory,
        )
        if loader_workers > 0:
            valid_loader_kwargs["persistent_workers"] = True
            valid_loader_kwargs["prefetch_factor"] = 2
        valid_loader = DataLoader(valid_ds, **valid_loader_kwargs)

    device = torch.device(cfg.device)
    log(f"Using device={device}")
    model = LightweightLOBTransformer(
        input_dim=feature_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_len=cfg.context_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    log(f"AMP enabled={use_amp}")

    best_val = float("inf")
    best_state = None
    no_improve_epochs = 0
    history = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "lr": [],
        "elapsed_sec": [],
    }
    train_start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"epoch {epoch:02d}/{cfg.epochs}",
            leave=False,
            mininterval=2.0,
        )
        for step, (x, y) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(x)
                loss = weighted_pearson_loss(y, pred)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item() * x.size(0)
            seen += x.size(0)

            if step % max(cfg.log_interval, 1) == 0 or step == len(train_loader):
                avg_loss = running / max(seen, 1)
                pbar.set_postfix({"train_loss": f"{avg_loss:.6f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
                log(
                    f"epoch={epoch:02d} step={step}/{len(train_loader)} "
                    f"train_loss={avg_loss:.6f} lr={optimizer.param_groups[0]['lr']:.3e}"
                )

        scheduler.step()
        train_loss = running / max(seen, 1)
        if cfg.skip_validation:
            val_loss = None
            log(f"epoch={epoch:02d} train_weighted_pearson_loss={train_loss:.6f}")
            score_to_track = train_loss
        else:
            val_loss = evaluate(model, valid_loader, device, use_amp)
            log(
                f"epoch={epoch:02d} train_weighted_pearson_loss={train_loss:.6f} "
                f"valid_weighted_pearson_loss={val_loss:.6f}"
            )
            score_to_track = val_loss

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["valid_loss"].append(None if val_loss is None else float(val_loss))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["elapsed_sec"].append(float(time.time() - train_start))
        log(f"epoch={epoch:02d} elapsed_sec={time.time() - epoch_start:.1f}")

        if score_to_track < (best_val - cfg.early_stopping_min_delta):
            best_val = score_to_track
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        latest_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        save_artifacts(
            artifact_dir=artifact_dir,
            epoch=epoch,
            model_state_dict=latest_state,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            history=history,
            best_score=best_val,
            mean=mean,
            std=std,
            cfg=cfg,
            feature_dim=feature_dim,
        )
        log(f"Saved checkpoint for epoch={epoch:02d}")

        if cfg.early_stopping_patience > 0 and no_improve_epochs >= cfg.early_stopping_patience:
            log(
                f"Early stopping triggered at epoch={epoch:02d} "
                f"(patience={cfg.early_stopping_patience}, min_delta={cfg.early_stopping_min_delta})"
            )
            break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Keep best model at root for inference packaging.
    torch.save(best_state, os.path.join(artifact_dir, "transformer_model.pt"))

    log(f"Saved model artifacts to: {artifact_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="../datasets/train.parquet")
    parser.add_argument("--valid-path", default="../datasets/valid.parquet")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--max-train-seqs", type=int, default=2000)
    parser.add_argument("--max-valid-seqs", type=int, default=500)
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(
        train_path=args.train_path,
        valid_path=args.valid_path,
        out_dir=args.out_dir,
        max_train_seqs=args.max_train_seqs,
        max_valid_seqs=args.max_valid_seqs,
        context_len=args.context_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        skip_validation=args.skip_validation,
        log_interval=args.log_interval,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    train(config)
