import argparse
import json
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from feature_engineering import engineer_features
from model import LOBTransformer


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
        self.features = features
        self.targets = targets
        self.context_len = context_len
        self.samples_per_seq = 901  # steps [99..999]

    def __len__(self):
        return self.features.shape[0] * self.samples_per_seq

    def __getitem__(self, index: int):
        seq_idx = index // self.samples_per_seq
        pred_idx = 99 + (index % self.samples_per_seq)

        start = max(0, pred_idx - self.context_len + 1)
        window = self.features[seq_idx, start : pred_idx + 1]

        if window.shape[0] < self.context_len:
            pad_len = self.context_len - window.shape[0]
            pad = np.zeros((pad_len, window.shape[1]), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)

        y = self.targets[seq_idx, pred_idx]
        w = np.abs(y).astype(np.float32)
        w = np.maximum(w, 1e-4)
        return torch.from_numpy(window), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(w)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    mse_sum = 0.0
    n = 0
    with torch.no_grad():
        for x, y, w in loader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            pred = model(x)
            loss = ((pred - y) ** 2 * w).mean()
            mse_sum += loss.item() * x.size(0)
            n += x.size(0)
    return mse_sum / max(n, 1)


def train(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    train_feat, train_tgt = read_and_reshape(cfg.train_path, cfg.max_train_seqs)
    valid_feat, valid_tgt = read_and_reshape(cfg.valid_path, cfg.max_valid_seqs)

    train_feat = engineer_features(train_feat)
    valid_feat = engineer_features(valid_feat)

    mean = train_feat.reshape(-1, train_feat.shape[-1]).mean(axis=0).astype(np.float32)
    std = train_feat.reshape(-1, train_feat.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)

    train_feat = ((train_feat - mean) / std).astype(np.float32)
    valid_feat = ((valid_feat - mean) / std).astype(np.float32)

    train_ds = SequenceWindowDataset(train_feat, train_tgt, cfg.context_len)
    valid_ds = SequenceWindowDataset(valid_feat, valid_tgt, cfg.context_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "cpu"),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device != "cpu"),
    )

    device = torch.device(cfg.device)
    model = LOBTransformer(
        input_dim=train_feat.shape[-1],
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_len=cfg.context_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for x, y, w in train_loader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = ((pred - y) ** 2 * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        scheduler.step()
        train_loss = running / max(seen, 1)
        val_loss = evaluate(model, valid_loader, device)

        print(f"epoch={epoch:02d} train_weighted_mse={train_loss:.6f} valid_weighted_mse={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    torch.save(best_state, os.path.join(cfg.out_dir, "transformer_model.pt"))
    np.savez(os.path.join(cfg.out_dir, "feature_stats.npz"), mean=mean, std=std)
    np.savez(
        os.path.join(cfg.out_dir, "config.npz"),
        context_len=cfg.context_len,
        feature_dim=train_feat.shape[-1],
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
    )

    with open(os.path.join(cfg.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print(f"Saved model artifacts to: {cfg.out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="../datasets/train.parquet")
    parser.add_argument("--valid-path", default="../datasets/valid.parquet")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--max-train-seqs", type=int, default=2000)
    parser.add_argument("--max-valid-seqs", type=int, default=500)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
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
    )
    train(config)
