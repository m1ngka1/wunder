import numpy as np

EPS = 1e-6


def _split_state(state: np.ndarray):
    prices = state[..., :12]
    volumes = state[..., 12:24]
    trade_prices = state[..., 24:28]
    trade_volumes = state[..., 28:32]
    return prices, volumes, trade_prices, trade_volumes


def _basic_features(state: np.ndarray):
    prices, volumes, trade_prices, trade_volumes = _split_state(state)

    best_bid = prices[..., 0]
    best_ask = prices[..., 6]
    mid = (best_bid + best_ask) * 0.5
    spread = best_ask - best_bid
    log_spread = np.log1p(np.maximum(spread, 0.0))

    depth_bid = volumes[..., :6].sum(axis=-1)
    depth_ask = volumes[..., 6:].sum(axis=-1)
    depth_imb = (depth_bid - depth_ask) / (depth_bid + depth_ask + EPS)

    top_bid = volumes[..., 0]
    top_ask = volumes[..., 6]
    top_imb = (top_bid - top_ask) / (top_bid + top_ask + EPS)

    microprice = (best_bid * top_ask + best_ask * top_bid) / (top_bid + top_ask + EPS)
    micro_dev = microprice - mid

    top3_bid = volumes[..., :3].sum(axis=-1)
    top3_ask = volumes[..., 6:9].sum(axis=-1)
    top3_imb = (top3_bid - top3_ask) / (top3_bid + top3_ask + EPS)

    gap_bid = prices[..., 1] - prices[..., 0]
    gap_ask = prices[..., 7] - prices[..., 6]

    trade_vol_total = trade_volumes.sum(axis=-1)
    trade_imb = (
        trade_volumes[..., 0]
        + trade_volumes[..., 1]
        - trade_volumes[..., 2]
        - trade_volumes[..., 3]
    ) / (trade_vol_total + EPS)

    trade_vwap = (trade_prices * trade_volumes).sum(axis=-1) / (trade_vol_total + EPS)
    trade_vwap_dev = trade_vwap - mid

    extra = np.stack(
        [
            mid,
            spread,
            log_spread,
            depth_bid,
            depth_ask,
            depth_imb,
            top_imb,
            microprice,
            micro_dev,
            top3_imb,
            gap_bid,
            gap_ask,
            trade_vol_total,
            trade_imb,
            trade_vwap_dev,
        ],
        axis=-1,
    )

    return extra, mid, spread, depth_imb, micro_dev


def engineer_features(states: np.ndarray) -> np.ndarray:
    """Build feature matrix from raw LOB+trade state.

    Accepts 2D arrays (T, 32) or 3D arrays (N, T, 32).
    Returns same leading shape with additional engineered features appended.
    """
    if states.ndim == 2:
        return _engineer_sequence(states)
    if states.ndim == 3:
        return np.stack([_engineer_sequence(seq) for seq in states], axis=0)
    raise ValueError(f"Expected 2D or 3D input, got shape {states.shape}")


def _engineer_sequence(sequence: np.ndarray) -> np.ndarray:
    extra, mid, spread, depth_imb, micro_dev = _basic_features(sequence)

    mid_ret = np.concatenate([[0.0], np.diff(mid)])
    spread_change = np.concatenate([[0.0], np.diff(spread)])
    depth_imb_change = np.concatenate([[0.0], np.diff(depth_imb)])
    micro_dev_change = np.concatenate([[0.0], np.diff(micro_dev)])

    lagged = np.stack(
        [mid_ret, spread_change, depth_imb_change, micro_dev_change],
        axis=-1,
    )

    return np.concatenate([sequence, extra, lagged], axis=-1).astype(np.float32)


class FeatureEngineer:
    """Stateful streaming feature builder for step-by-step inference."""

    def __init__(self):
        self.prev_mid = None
        self.prev_spread = None
        self.prev_depth_imb = None
        self.prev_micro_dev = None

    def reset(self):
        self.prev_mid = None
        self.prev_spread = None
        self.prev_depth_imb = None
        self.prev_micro_dev = None

    def update(self, state: np.ndarray) -> np.ndarray:
        extra, mid, spread, depth_imb, micro_dev = _basic_features(state)

        if self.prev_mid is None:
            mid_ret = 0.0
            spread_change = 0.0
            depth_imb_change = 0.0
            micro_dev_change = 0.0
        else:
            mid_ret = float(mid - self.prev_mid)
            spread_change = float(spread - self.prev_spread)
            depth_imb_change = float(depth_imb - self.prev_depth_imb)
            micro_dev_change = float(micro_dev - self.prev_micro_dev)

        self.prev_mid = float(mid)
        self.prev_spread = float(spread)
        self.prev_depth_imb = float(depth_imb)
        self.prev_micro_dev = float(micro_dev)

        lagged = np.array(
            [mid_ret, spread_change, depth_imb_change, micro_dev_change],
            dtype=np.float32,
        )

        return np.concatenate([state.astype(np.float32), extra.astype(np.float32), lagged])
