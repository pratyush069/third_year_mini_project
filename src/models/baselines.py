"""Persistence, Linear Regression and XGBoost with one shared interface.

Every model:  fit(X_train, y_train, X_val, y_val)  then  predict(X) -> speeds (km/h)
Input rows come from src.data.windows.tabular (column 0 = most recent speed).
"""
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression


class Persistence:
    """'The future equals now': predict s(t+h) = s(t). Sanity floor for every other model."""

    def fit(self, *args):
        return self

    def predict(self, X):
        return X[:, 0]


class Linear:
    """Ordinary least squares: prediction = weighted sum of the features + bias."""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X).astype(np.float32)


class XGB:
    """Gradient-boosted decision trees; uses the GPU automatically when one is available."""

    def __init__(self, early_stopping_rounds=30, **params):
        device = "cuda" if _has_cuda() else "cpu"
        self.model = xgb.XGBRegressor(tree_method="hist", device=device,
                                      early_stopping_rounds=early_stopping_rounds, **params)

    def fit(self, X, y, X_val, y_val):
        self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=100)
        return self

    def predict(self, X):
        return self.model.predict(X).astype(np.float32)


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def build(name, cfg):
    if name == "persistence":
        return Persistence()
    if name == "linear":
        return Linear()
    if name == "xgboost":
        return XGB(random_state=cfg["baselines"]["seed"], **cfg["baselines"]["xgb"])
    raise ValueError(f"unknown model {name}")
