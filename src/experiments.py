"""Run the full window x horizon grid for the chosen models.

Usage:  python -m src.experiments --models persistence linear xgboost
Runs already present in results/all_results.csv are skipped, so it can be restarted safely.
"""
import argparse
import os

import pandas as pd
import yaml

from src import train
from src.data import windows as W


def done_runs():
    if not os.path.exists(train.RESULTS):
        return set()
    df = pd.read_csv(train.RESULTS)
    return set(zip(df.model, df.window, df.horizon))


def main(models):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    data = W.load_data()                      # load once, reuse for every run
    done = done_runs()
    for model in models:
        for window in cfg["experiment"]["windows"]:
            for horizon in cfg["experiment"]["horizons"]:
                if (model, window, horizon) in done:
                    print(f"[skip] {model} w={window} h={horizon}")
                    continue
                train.run(model, window, horizon, cfg, data)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["persistence", "linear", "xgboost", "lstm", "stgcn"])
    main(p.parse_args().models)
