"""Train + evaluate one model for one (window, horizon) setting.

Usage:  python -m src.train --model xgboost --window 6 --horizon 3
Appends one row per split (val, test) to results/all_results.csv.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import yaml

from src.data import windows as W
from src.evaluate import Metrics
from src.models import baselines

RESULTS = "results/all_results.csv"


def evaluate_split(model, data, split, window, horizon, link_mean):
    m = Metrics()
    for tau, links in W.iter_split_rows(data, split, window, horizon):
        X, y, mask = W.tabular(data, tau, links, window, horizon, link_mean)
        m.update(y, model.predict(X), mask)
    return m.result()


def run(model_name, window, horizon, cfg, data=None):
    data = data or W.load_data()
    rng = np.random.default_rng(cfg["baselines"]["seed"])
    link_mean = W.train_link_mean(data)

    X_tr = y_tr = X_va = y_va = None
    if model_name != "persistence":
        tau, links = W.sample_rows(data, "train", window, horizon, cfg["baselines"]["train_samples"], rng)
        X_tr, y_tr, _ = W.tabular(data, tau, links, window, horizon, link_mean)
        tau, links = W.sample_rows(data, "val", window, horizon, cfg["baselines"]["val_samples"], rng)
        X_va, y_va, _ = W.tabular(data, tau, links, window, horizon, link_mean)

    print(f"\n=== {model_name} | window={window} ({window * 10} min) | horizon={horizon} ({horizon * 10} min) ===")
    start = time.time()
    model = baselines.build(model_name, cfg).fit(X_tr, y_tr, X_va, y_va)
    train_sec = time.time() - start

    rows = []
    for split in ["val", "test"]:
        res = evaluate_split(model, data, split, window, horizon, link_mean)
        print(f"  {split:4s}  MAE {res['MAE']:.3f}  RMSE {res['RMSE']:.3f}  "
              f"MAPE {res['MAPE']:.2f}%  R2 {res['R2']:.4f}")
        rows.append({"model": model_name, "window": window, "horizon": horizon, "split": split,
                     **res, "train_sec": round(train_sec, 1)})

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS, mode="a", header=not os.path.exists(RESULTS), index=False)
    return model, rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["persistence", "linear", "xgboost"])
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--horizon", type=int, default=None)
    a = p.parse_args()
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(a.model, a.window or cfg["experiment"]["window"], a.horizon or cfg["experiment"]["horizon"], cfg)
