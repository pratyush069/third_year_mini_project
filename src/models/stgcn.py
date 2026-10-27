"""Training adapter for the adapted PyTorch STGCN implementation.

Upstream STGCN implementation: https://github.com/hazdzz/stgcn (LGPL-2.1).
The upstream temporal/graph blocks are retained under third_party_stgcn/;
this file connects them to EXPY-TKY, its sparse road graph and our splits.
"""
import copy
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml

from src.data import windows as W
from src.evaluate import Metrics
from src.models.third_party_stgcn.models import STGCNChebGraphConv


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Use CPU on Apple Silicon: sparse graph training with MPS is unstable in
    # some PyTorch/macOS builds. Kaggle's CUDA device is selected automatically.
    return torch.device("cpu")


def _scaled_laplacian(adjacency):
    """Create the scaled symmetric Laplacian used by Chebyshev graph filters."""
    graph = sp.csr_matrix(adjacency, dtype=np.float32)
    graph = graph.maximum(graph.T)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    positive = degree > 0
    inv_sqrt[positive] = degree[positive] ** -0.5
    D = sp.diags(inv_sqrt)
    normalized = D @ graph @ D
    laplacian = sp.eye(graph.shape[0], dtype=np.float32, format="csr") - normalized
    # A symmetric normalized Laplacian has spectrum in [0, 2]. Using 2 as the
    # scale is the standard safe bound and avoids a platform-specific ARPACK
    # eigenvalue solve that can crash on some macOS SciPy builds.
    scaled = laplacian - sp.eye(graph.shape[0], dtype=np.float32, format="csr")
    scaled.sum_duplicates()
    scaled.eliminate_zeros()
    coo = scaled.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    # CSR -> COO preserves sorted unique coordinates after sum_duplicates().
    return torch.sparse_coo_tensor(indices, values, coo.shape, is_coalesced=True)


def _make_batch(data, target_times, window, horizon, speed_mean, speed_std, device):
    """Build [batch, features, history, links] inputs and [batch, links] targets."""
    target_times = np.asarray(target_times, dtype=np.int64)
    last_input = target_times - horizon
    offsets = np.arange(window - 1, -1, -1)
    input_times = last_input[:, None] - offsets[None, :]

    # Channel 0 is speed. Channels 1–4 are known calendar features at each step.
    speeds = data.speed[input_times]                         # [B, W, N]
    calendars = data.tfeat[input_times]                      # [B, W, 4]
    calendars = np.broadcast_to(
        calendars[:, :, None, :],
        (len(target_times), window, data.n_links, 4),
    )
    features = np.concatenate([speeds[..., None], calendars], axis=-1)
    features = features.transpose(0, 3, 1, 2).copy()         # [B, 5, W, N]
    features[:, 0] = (features[:, 0] - speed_mean) / speed_std

    targets = (data.speed[target_times] - speed_mean) / speed_std
    mask = data.observed[target_times]
    x_tensor = torch.from_numpy(features.astype(np.float32)).to(device)
    y_tensor = torch.from_numpy(targets.astype(np.float32)).to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)
    return x_tensor, y_tensor, mask_tensor


class STGCNTrainer:
    def __init__(self, window, config, adjacency, device):
        self.window = window
        self.config = config
        self.device = device
        self.gso = _scaled_laplacian(adjacency).to(device)

        # The adapted reference model uses two T-G-T blocks followed by an output block.
        args = SimpleNamespace(
            Kt=2,
            Ks=3,
            n_his=window,
            act_func="glu",
            graph_conv_type="cheb_graph_conv",
            gso=self.gso,
            enable_bias=True,
            droprate=float(config.get("dropout", 0.2)),
        )
        blocks = [[5], [32, 16, 32], [32, 16, 32], [64, 64], [1]]
        self.model = STGCNChebGraphConv(args, blocks, adjacency.shape[0]).to(device)
        self.speed_mean = None
        self.speed_std = None

    def fit(self, data, window, horizon, config):
        observed = data.observed[:data.val_start]
        train_speed = data.speed[:data.val_start]
        self.speed_mean = float(train_speed[observed].mean())
        self.speed_std = float(max(train_speed[observed].std(), 1e-6))

        rng = np.random.default_rng(int(config.get("seed", 42)))
        train_times = data.targets("train", window, horizon)
        val_times = data.targets("val", window, horizon)
        val_count = min(int(config.get("val_time_samples", 128)), len(val_times))
        val_sample = np.sort(rng.choice(val_times, val_count, replace=False))

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-5)),
        )
        batch_size = int(config.get("batch_size", 2))
        samples_per_epoch = min(int(config.get("train_time_samples", 256)), len(train_times))
        best_loss, best_weights, stale = float("inf"), None, 0
        patience = int(config.get("patience", 6))

        for epoch in range(int(config.get("epochs", 30))):
            selected = rng.choice(train_times, samples_per_epoch, replace=False)
            selected = rng.permutation(selected)
            self.model.train()
            train_loss_sum, train_count = 0.0, 0
            for start in range(0, len(selected), batch_size):
                tau = selected[start:start + batch_size]
                x, y, mask = _make_batch(
                    data, tau, window, horizon, self.speed_mean, self.speed_std, self.device
                )
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(x)[:, 0, 0, :]
                count = mask.sum().clamp_min(1)
                loss = (((prediction - y) ** 2) * mask).sum() / count
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * count.item()
                train_count += int(count.item())

            val_loss = self._masked_mse(data, val_sample, window, horizon, batch_size)
            print(f"  STGCN epoch {epoch + 1:03d}: train masked MSE "
                  f"{train_loss_sum / max(train_count, 1):.5f}; validation masked MSE {val_loss:.5f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"  Early stopping after {epoch + 1} epochs")
                    break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        return self

    def _masked_mse(self, data, target_times, window, horizon, batch_size):
        self.model.eval()
        squared_error, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(target_times), batch_size):
                tau = target_times[start:start + batch_size]
                x, y, mask = _make_batch(
                    data, tau, window, horizon, self.speed_mean, self.speed_std, self.device
                )
                prediction = self.model(x)[:, 0, 0, :]
                squared_error += ((((prediction - y) ** 2) * mask).sum()).item()
                count += int(mask.sum().item())
        return squared_error / max(count, 1)

    def evaluate(self, data, split, window, horizon, batch_size):
        metrics = Metrics()
        target_times = data.targets(split, window, horizon)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(target_times), batch_size):
                tau = target_times[start:start + batch_size]
                x, _, mask = _make_batch(
                    data, tau, window, horizon, self.speed_mean, self.speed_std, self.device
                )
                prediction = self.model(x)[:, 0, 0, :].cpu().numpy()
                prediction = prediction * self.speed_std + self.speed_mean
                actual = data.speed[tau]
                metrics.update(actual.ravel(), prediction.ravel(), mask.cpu().numpy().ravel().astype(bool))
        return metrics.result()


def run_stgcn(window, horizon, cfg, data=None):
    data = data or W.load_data()
    config = dict(cfg.get("stgcn", {}))
    config.setdefault("seed", cfg["baselines"]["seed"])
    device = _device()
    adjacency = np.load(f"{cfg['data']['processed_dir']}/adj.npy")
    print(f"\n=== stgcn | window={window} ({window * 10} min) | "
          f"horizon={horizon} ({horizon * 10} min) | device={device} ===")

    start = time.time()
    trainer = STGCNTrainer(window, config, adjacency, device).fit(data, window, horizon, config)
    train_sec = time.time() - start
    os.makedirs("results/checkpoints", exist_ok=True)
    checkpoint = f"results/checkpoints/stgcn_w{window}_h{horizon}.pt"
    torch.save({"state_dict": trainer.model.state_dict(), "speed_mean": trainer.speed_mean,
                "speed_std": trainer.speed_std, "window": window, "horizon": horizon}, checkpoint)

    rows = []
    for split in ["val", "test"]:
        result = trainer.evaluate(data, split, window, horizon,
                                  int(config.get("batch_size", 2)))
        print(f"  {split:4s}  MAE {result['MAE']:.3f}  RMSE {result['RMSE']:.3f}  "
              f"MAPE {result['MAPE']:.2f}%  R2 {result['R2']:.4f}")
        rows.append({"model": "stgcn", "window": window, "horizon": horizon, "split": split,
                     **result, "train_sec": round(train_sec, 1)})

    result_path = "results/comparison_results.csv"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(result_path, mode="a", header=not os.path.exists(result_path), index=False)
    print("  checkpoint:", checkpoint)
    return trainer, rows
