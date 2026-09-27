"""Build the road-network graph (adjacency matrix) used by STGCN.

w_ij = exp(-(d_ij / sigma)^2), kept only if w_ij >= epsilon,
plus direct road connections from adj01 (made symmetric).
Output: data/processed/adj.npy  float32 [N, N], zero diagonal.
Usage:  python -m src.data.graph
"""
import os

import numpy as np
import yaml


def build_adjacency(config_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(config_path))
    raw, out, g = cfg["data"]["raw_dir"], cfg["data"]["processed_dir"], cfg["graph"]
    subset = np.loadtxt(os.path.join(raw, cfg["data"]["subset_file"])).astype(int)

    dist = np.load(os.path.join(raw, "expy-tky_adjdis.npy"))[subset][:, subset]  # metres
    W = np.exp(-(dist / g["sigma_m"]) ** 2)
    W[W < g["epsilon"]] = 0

    if g["add_road_links"]:
        road = np.load(os.path.join(raw, "expy-tky_adj01.npy"))[subset][:, subset] > 0
        road = road | road.T               # one-way links -> treat as undirected neighbours
        W = np.maximum(W, road.astype(W.dtype))

    np.fill_diagonal(W, 0)                 # self-loops are added inside the model instead

    # A few links are far from all others: link each to its nearest link so none is isolated.
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    for i in np.where((W > 0).sum(1) == 0)[0]:
        j = d[i].argmin()
        W[i, j] = W[j, i] = g["epsilon"]
    W = W.astype(np.float32)

    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "adj.npy"), W)
    deg = (W > 0).sum(1)
    print(f"adjacency: {W.shape}, symmetric={np.allclose(W, W.T)}")
    print(f"neighbours per link: mean {deg.mean():.1f}, min {deg.min()}, max {deg.max()}, isolated {(deg == 0).sum()}")
    return W


if __name__ == "__main__":
    build_adjacency()
