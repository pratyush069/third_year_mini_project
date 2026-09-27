"""Exploratory figures for the synopsis / paper.

Usage:  python -m src.eda      -> results/figures/*.png
"""
import os

import jpholiday
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

OUT = "results/figures"


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load("data/processed/speed.npz")
    speed, obs = d["speed"], d["observed"]
    times = pd.to_datetime(d["timestamps"])
    info = pd.read_csv("data/raw/expy-tky_link_info_add_end_latlon.csv", index_col=0)
    info = info.set_index("link_id").loc[d["link_ids"]]

    # 1) Network map coloured by mean speed
    segs = np.stack([info[["start_lon", "start_lat"]].values, info[["end_lon", "end_lat"]].values], 1)
    fig, ax = plt.subplots(figsize=(7, 7))
    lc = LineCollection(segs, array=speed.mean(0), cmap="RdYlGn", linewidths=1.5)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect(1 / np.cos(np.deg2rad(35.7)))
    fig.colorbar(lc, ax=ax, shrink=0.7, label="Mean speed (km/h)")
    ax.set(xlabel="Longitude", ylabel="Latitude",
           title=f"Tokyo Metropolitan Expressway: {speed.shape[1]} road links")
    fig.savefig(f"{OUT}/fig_network_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Average daily profile: weekday vs weekend/holiday
    off = (times.weekday >= 5) | np.array([jpholiday.is_holiday(t.date()) for t in times])
    tod = times.hour + times.minute / 60
    net = speed.mean(1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for mask, label in [(~off, "Weekday"), (off, "Weekend / holiday")]:
        prof = pd.Series(net[mask]).groupby(tod[mask]).mean()
        ax.plot(prof.index, prof.values, label=label)
    ax.set(xlabel="Hour of day", ylabel="Network mean speed (km/h)", xticks=range(0, 25, 3),
           title="Average daily speed pattern (Oct-Dec 2021)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/fig_daily_profile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) One week on the most variable link, skipping the few links whose gaps were
    #    template-filled by the dataset creators (long runs of unchanged values).
    week = (times >= "2021-10-04") & (times < "2021-10-11")
    unchanged = (np.diff(speed, axis=0) == 0).mean(0)
    std = np.where(unchanged < 0.05, speed[week].std(0), -1)
    link = std.argmax()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(times[week], speed[week, link], lw=0.8)
    ax.set(ylabel="Speed (km/h)",
           title=f"One week of speed on a single road link (ID {d['link_ids'][link]})")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(f"{OUT}/fig_week_series.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4) Distribution of observed speeds
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(speed[obs], bins=100, color="steelblue")
    ax.set(xlabel="Speed (km/h)", ylabel="Count", title="Distribution of observed speeds")
    fig.savefig(f"{OUT}/fig_speed_hist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("saved figures to", OUT, "->", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
