"""Shared data handling for every model: load, split by time, cut sliding windows.

Notation (index = one 10-minute step):
    tau      target time step we want to predict
    t0       = tau - horizon, the "now" (last step the model is allowed to see)
    window   inputs are speed[t0 - window + 1 ... t0]

Splits are defined on the TARGET time:
    train : Oct-Nov targets, first (1 - val_ratio)
    val   : Oct-Nov targets, last val_ratio
    test  : every December target (identical for all models / windows / horizons)
Inputs may reach back into the previous split (the past is always known), never forward.
"""
from dataclasses import dataclass

import jpholiday
import numpy as np
import pandas as pd
import yaml


@dataclass
class TrafficData:
    speed: np.ndarray      # [T, N] float32, gaps filled
    observed: np.ndarray   # [T, N] bool, True = real reading
    times: pd.DatetimeIndex
    tfeat: np.ndarray      # [T, 4] time features of each step
    val_start: int
    test_start: int

    @property
    def n_links(self):
        return self.speed.shape[1]

    def targets(self, split, window, horizon):
        """All valid target steps tau for a split."""
        first = window + horizon - 1          # earliest tau with a full input window
        lo, hi = {"train": (first, self.val_start),
                  "val": (self.val_start, self.test_start),
                  "test": (self.test_start, len(self.times))}[split]
        return np.arange(max(lo, first), hi)


def time_features(times):
    """sin/cos time of day, day of week (0-1), Japanese holiday or weekend flag."""
    minutes = times.hour * 60 + times.minute
    angle = 2 * np.pi * minutes / 1440
    off = (times.weekday >= 5) | np.array([jpholiday.is_holiday(t.date()) for t in times])
    return np.stack([np.sin(angle), np.cos(angle), times.weekday / 6, off], 1).astype(np.float32)


def load_data(config_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(config_path))
    d = np.load(f"{cfg['data']['processed_dir']}/speed.npz")
    times = pd.to_datetime(d["timestamps"])
    test_start = int(np.argmax(times.strftime("%Y%m") == cfg["split"]["test_month"]))
    val_start = int(test_start * (1 - cfg["split"]["val_ratio"]))
    return TrafficData(d["speed"], d["observed"], times, time_features(times), val_start, test_start)


# ---------- tabular features for Linear Regression / XGBoost ----------

def tabular(data, tau, links, window, horizon, link_mean):
    """One row per (tau, link): [lag_0 .. lag_{w-1}, 4 time features of tau, link mean speed].

    lag_0 is the most recent speed s(t0), lag_1 = s(t0-1), ...
    Returns X [S, window+5], y [S], mask [S] (mask = target is a real reading).
    """
    t0 = tau - horizon
    lag_idx = t0[:, None] - np.arange(window)[None, :]
    lags = data.speed[lag_idx, links[:, None]]
    X = np.hstack([lags, data.tfeat[tau], link_mean[links, None]]).astype(np.float32)
    return X, data.speed[tau, links], data.observed[tau, links]


def train_link_mean(data):
    """Mean observed speed of each link over the TRAINING period only (no test leakage)."""
    s, o = data.speed[:data.val_start], data.observed[:data.val_start]
    return ((s * o).sum(0) / o.sum(0)).astype(np.float32)


def feature_names(window):
    return [f"lag_{i}" for i in range(window)] + ["tod_sin", "tod_cos", "dow", "offday", "link_mean"]


def sample_rows(data, split, window, horizon, n, rng):
    """Random (tau, link) pairs from a split, only where the target is a real reading."""
    tau_all = data.targets(split, window, horizon)
    tau = rng.choice(tau_all, size=int(n * 1.01) + 10)
    links = rng.integers(0, data.n_links, size=tau.size)
    keep = data.observed[tau, links]
    return tau[keep][:n], links[keep][:n]


def iter_split_rows(data, split, window, horizon, chunk_steps=300):
    """Yield every (tau, link) of a split in chunks of time steps (keeps RAM low)."""
    tau_all = data.targets(split, window, horizon)
    N = data.n_links
    for i in range(0, tau_all.size, chunk_steps):
        tau_c = tau_all[i:i + chunk_steps]
        yield np.repeat(tau_c, N), np.tile(np.arange(N), tau_c.size)
