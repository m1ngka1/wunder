from collections import deque

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

    mid_ret_5 = _lagged_delta(mid, 5)
    mid_ret_10 = _lagged_delta(mid, 10)
    spread_mean_5 = _rolling_mean(spread, 5)
    depth_imb_mean_5 = _rolling_mean(depth_imb, 5)
    micro_dev_mean_5 = _rolling_mean(micro_dev, 5)
    mid_ret_std_10 = _rolling_std(mid_ret, 10)

    rolling = np.stack(
        [
            mid_ret_5,
            mid_ret_10,
            spread_mean_5,
            depth_imb_mean_5,
            micro_dev_mean_5,
            mid_ret_std_10,
        ],
        axis=-1,
    )

    return np.concatenate([sequence, extra, lagged, rolling], axis=-1).astype(np.float32)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    cumsum = np.cumsum(values, dtype=np.float64)
    result = np.empty_like(values, dtype=np.float64)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        total = cumsum[idx] - (cumsum[start - 1] if start > 0 else 0.0)
        result[idx] = total / (idx - start + 1)
    return result.astype(np.float32)


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    cumsum = np.cumsum(values, dtype=np.float64)
    cumsum_sq = np.cumsum(values**2, dtype=np.float64)
    result = np.empty_like(values, dtype=np.float64)
    for idx in range(values.shape[0]):
        start = max(0, idx - window + 1)
        count = idx - start + 1
        total = cumsum[idx] - (cumsum[start - 1] if start > 0 else 0.0)
        total_sq = cumsum_sq[idx] - (cumsum_sq[start - 1] if start > 0 else 0.0)
        mean = total / count
        var = max(total_sq / count - mean**2, 0.0)
        result[idx] = np.sqrt(var)
    return result.astype(np.float32)


def _lagged_delta(values: np.ndarray, lag: int) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] > lag:
        result[lag:] = values[lag:] - values[:-lag]
    return result.astype(np.float32)


class FeatureEngineer:
    """Stateful streaming feature builder for step-by-step inference."""

    def __init__(self):
        self.prev_mid = None
        self.prev_spread = None
        self.prev_depth_imb = None
        self.prev_micro_dev = None
        self.mid_history = deque(maxlen=11)
        self.spread_history = deque(maxlen=5)
        self.depth_imb_history = deque(maxlen=5)
        self.micro_dev_history = deque(maxlen=5)
        self.mid_ret_history = deque(maxlen=10)

    def reset(self):
        self.prev_mid = None
        self.prev_spread = None
        self.prev_depth_imb = None
        self.prev_micro_dev = None
        self.mid_history.clear()
        self.spread_history.clear()
        self.depth_imb_history.clear()
        self.micro_dev_history.clear()
        self.mid_ret_history.clear()

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

        self.mid_history.append(float(mid))
        self.spread_history.append(float(spread))
        self.depth_imb_history.append(float(depth_imb))
        self.micro_dev_history.append(float(micro_dev))
        self.mid_ret_history.append(float(mid_ret))

        mid_ret_5 = mid - self.mid_history[-6] if len(self.mid_history) > 5 else 0.0
        mid_ret_10 = mid - self.mid_history[0] if len(self.mid_history) > 10 else 0.0
        spread_mean_5 = float(np.mean(self.spread_history))
        depth_imb_mean_5 = float(np.mean(self.depth_imb_history))
        micro_dev_mean_5 = float(np.mean(self.micro_dev_history))
        mid_ret_std_10 = float(np.std(self.mid_ret_history, ddof=0))

        lagged = np.array(
            [mid_ret, spread_change, depth_imb_change, micro_dev_change],
            dtype=np.float32,
        )

        rolling = np.array(
            [
                mid_ret_5,
                mid_ret_10,
                spread_mean_5,
                depth_imb_mean_5,
                micro_dev_mean_5,
                mid_ret_std_10,
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [state.astype(np.float32), extra.astype(np.float32), lagged, rolling]
        )
