"""Masked error metrics, computed only on real (observed) readings.

Metrics are accumulated chunk by chunk so the whole test set never has to sit in RAM.
    MAE  = mean |error|                         (km/h)
    RMSE = sqrt(mean error^2)                   (km/h, punishes big misses)
    MAPE = mean |error| / true  * 100           (%, only where true speed >= 1 km/h)
    R2   = 1 - SSE / SST                        (share of variation explained)
"""
import numpy as np

MAPE_MIN_SPEED = 1.0


class Metrics:
    def __init__(self):
        self.n = self.abs = self.sq = self.y = self.y2 = 0.0
        self.n_mape = self.pct = 0.0

    def update(self, y_true, y_pred, mask):
        yt, err = y_true[mask].astype(np.float64), (y_pred[mask] - y_true[mask]).astype(np.float64)
        self.n += yt.size
        self.abs += np.abs(err).sum()
        self.sq += (err ** 2).sum()
        self.y += yt.sum()
        self.y2 += (yt ** 2).sum()
        m = yt >= MAPE_MIN_SPEED
        self.n_mape += m.sum()
        self.pct += (np.abs(err[m]) / yt[m]).sum()

    def result(self):
        sst = self.y2 - self.y ** 2 / self.n
        return {"MAE": self.abs / self.n,
                "RMSE": np.sqrt(self.sq / self.n),
                "MAPE": 100 * self.pct / self.n_mape,
                "R2": 1 - self.sq / sst}
