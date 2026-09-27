"""Turn the raw EXPY-TKY CSVs into one clean speed matrix.

Output: data/processed/speed.npz with
    speed      float32 [T, N]  cleaned speed (km/h), gaps interpolated
    observed   bool    [T, N]  True where the original reading was real (not 0 / missing)
    timestamps str     [T]     one per 10-minute step
    link_ids   int64   [N]     road link id of each column
Usage:  python -m src.data.preprocess
"""
import os

import numpy as np
import pandas as pd
import yaml


def load_month(path, n_total, subset):
    df = pd.read_csv(path)
    # Rows come as (t0,link0), (t0,link1), ... so one reshape gives [T, all links].
    speed = df["speed"].to_numpy(np.float32).reshape(-1, n_total)
    times = df["timestamp"].to_numpy()[::n_total]
    link_ids = df["linkid"].to_numpy()[:n_total]
    # Same clipping rule as the MegaCRN authors (keeps results comparable).
    speed[speed < 0] = 0
    speed[speed > 200] = 100
    return speed[:, subset], times, link_ids[subset]


def preprocess(config_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(config_path))["data"]
    raw, out = cfg["raw_dir"], cfg["processed_dir"]
    subset = np.loadtxt(os.path.join(raw, cfg["subset_file"])).astype(int)

    parts = [load_month(os.path.join(raw, f"expy-tky_{m}.csv.gz"), cfg["n_links_total"], subset)
             for m in cfg["months"]]
    speed = np.concatenate([p[0] for p in parts])
    times = np.concatenate([p[1] for p in parts])
    link_ids = parts[0][2]

    # Speed 0 means "no probe car passed" -> treat as missing.
    observed = speed > 0
    filled = pd.DataFrame(np.where(observed, speed, np.nan))
    # Fill gaps from neighbouring time steps of the same link; edges use nearest value.
    filled = filled.interpolate(axis=0, limit_direction="both")
    speed_clean = filled.to_numpy(np.float32)

    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "speed.npz"), speed=speed_clean, observed=observed,
                        timestamps=times.astype(str), link_ids=link_ids)
    print(f"speed matrix: {speed_clean.shape} (time steps x links)")
    print(f"missing values filled: {(~observed).sum()} ({(~observed).mean() * 100:.2f}%)")
    print(f"time range: {times[0]} -> {times[-1]}")


if __name__ == "__main__":
    preprocess()
