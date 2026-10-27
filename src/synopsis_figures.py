"""Self-designed figures for the synopsis (Figures 1-6).

Usage:  python -m src.synopsis_figures   -> results/figures/synopsis/fig1..fig6.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUT = "results/figures/synopsis"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})

# Muted academic palette (prints well in greyscale)
BLUE, GREEN, ORANGE, PURPLE, GREY, RED = "#dbe8f6", "#dff0df", "#fbe5cc", "#e8e0f3", "#eeeeee", "#f6d5d5"
EDGE = "#333333"


def box(ax, x, y, w, h, text, fc=BLUE, fs=10.5, bold=False, round_=0.02):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0.005,rounding_size={round_}",
                                fc=fc, ec=EDGE, lw=1.1))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, text=None, rad=0.0, fs=9.5, color=EDGE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.2, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, text, fontsize=fs, ha="left", va="center",
                style="italic", bbox=dict(fc="white", ec="none", pad=1))


def diamond(ax, x, y, w, h, text, fc=ORANGE):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], closed=True,
                         fc=fc, ec=EDGE, lw=1.1))
    ax.text(x, y, text, ha="center", va="center", fontsize=10)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def fig1_congestion_cycle():
    fig, ax = canvas(8, 6.2)
    cx, cy, r = 4, 3.1, 2.35
    steps = [("Urban population\ngrowth", BLUE),
             ("Rising vehicle\nownership and use", BLUE),
             ("Traffic demand\nexceeds road capacity", ORANGE),
             ("Recurring and incident\ncongestion (speed drops)", RED),
             ("Lost time, extra fuel,\nemissions, economic cost", RED),
             ("Slow road expansion\n(cannot keep pace)", GREY)]
    angles = np.deg2rad(90 - np.arange(len(steps)) * 360 / len(steps))
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    for (x, y), (t, c) in zip(pts, steps):
        box(ax, x, y, 2.25, 0.78, t, fc=c)
    def edge(p, q, hw=1.2, hh=0.47):
        # point where the line p->q leaves the box centred at p
        dx, dy = q[0] - p[0], q[1] - p[1]
        t = min(hw / abs(dx) if dx else np.inf, hh / abs(dy) if dy else np.inf)
        return (p[0] + dx * t, p[1] + dy * t)
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        arrow(ax, edge(p, q), edge(q, p), rad=-0.2)
    ax.add_patch(plt.Circle((cx, cy), 1.05, fc=GREEN, ec=EDGE, lw=1.1))
    ax.text(cx, cy, "Short-term speed\nforecasting\n(proactive traffic\nmanagement)", ha="center", va="center",
            fontsize=10.5, weight="bold")
    save(fig, "fig1_congestion_cycle.png")


# ---------------------------------------------------------------- Figure 2
def fig2_its_pipeline():
    fig, ax = canvas(11.6, 4.7)
    cols = [(1.1, "1. Data sources", BLUE, ["GPS probe vehicles", "Loop detectors", "ETC toll gantries"]),
            (3.5, "2. Pre-processing", GREY, ["Cleaning and\nmissing values", "Normalisation", "Sliding windows\n+ road graph"]),
            (8.25, "4. Forecasts", ORANGE, ["Speed at\nt + 10 min", "Speed at\nt + 30 min", "Speed at\nt + 60 min"]),
            (10.55, "5. ITS applications", GREEN, ["Navigation / ETA", "Signal and\nramp control", "Congestion alerts"])]
    for x, title, c, items in cols:
        ax.text(x, 4.35, title, ha="center", weight="bold", fontsize=11)
        for k, it in enumerate(items):
            box(ax, x, 3.4 - k * 1.2, 1.9, 0.88, it, fc=c, fs=10)
    x = 5.9
    ax.text(x, 4.35, "3. Prediction models", ha="center", weight="bold", fontsize=11)
    ax.add_patch(FancyBboxPatch((x - 1.05, 2.5), 2.1, 1.45, boxstyle="round,pad=0.01", fc="white", ec=EDGE, ls="--"))
    ax.text(x, 3.75, "Machine learning", ha="center", style="italic", fontsize=10)
    box(ax, x, 3.32, 1.8, 0.4, "Linear Regression", fc=PURPLE, fs=9.5)
    box(ax, x, 2.82, 1.8, 0.4, "XGBoost", fc=PURPLE, fs=9.5)
    ax.add_patch(FancyBboxPatch((x - 1.05, 0.5), 2.1, 1.8, boxstyle="round,pad=0.01", fc="white", ec=EDGE, ls="--"))
    ax.text(x, 2.1, "Deep learning", ha="center", style="italic", fontsize=10)
    box(ax, x, 1.6, 1.8, 0.62, "LSTM\n(temporal)", fc=PURPLE, fs=9.5)
    box(ax, x, 0.88, 1.8, 0.62, "STGCN\n(spatial + temporal)", fc=PURPLE, fs=9.5)
    for x0, x1 in [(2.05, 2.55), (4.45, 4.85), (6.95, 7.3), (9.2, 9.6)]:
        arrow(ax, (x0, 2.2), (x1, 2.2))
    save(fig, "fig2_its_pipeline.png")


# ---------------------------------------------------------------- Figure 3
def fig3_spatiotemporal(link_id=83589957):
    d = np.load("data/processed/speed.npz")
    speed, times, ids = d["speed"], pd.to_datetime(d["timestamps"]), d["link_ids"]
    j = int(np.where(ids == link_id)[0][0])
    info = pd.read_csv("data/raw/expy-tky_link_info_add_end_latlon.csv", index_col=0).set_index("link_id").loc[ids]

    fig = plt.figure(figsize=(11, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.1], wspace=0.22)
    ax = fig.add_subplot(gs[0])
    segs = np.stack([info[["start_lon", "start_lat"]].values, info[["end_lon", "end_lat"]].values], 1)
    lc = LineCollection(segs, array=speed.mean(0), cmap="RdYlGn", linewidths=1.2)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect(1 / np.cos(np.deg2rad(35.7)))
    lon, lat = segs[j].mean(0)
    ax.plot(lon, lat, "o", ms=11, mfc="none", mec="black", mew=2)
    ax.annotate("link shown in (b)", (lon, lat), (lon + 0.06, lat - 0.055), fontsize=9.5,
                arrowprops=dict(arrowstyle="-", lw=0.8),
                bbox=dict(fc="white", ec="none", pad=1))
    fig.colorbar(lc, ax=ax, orientation="horizontal", shrink=0.85, pad=0.2, label="Mean speed (km/h)")
    ax.set(xlabel="Longitude", ylabel="Latitude")
    ax.set_title("(a) Shuto Expressway network (1,843 links)", fontsize=11)

    ax = fig.add_subplot(gs[1])
    wk = (times >= "2021-10-04") & (times < "2021-10-11")
    ax.plot(times[wk], speed[wk, j], lw=0.9, color="#1f4e8c")
    ax.axhspan(80, 100, color=GREEN, alpha=0.6, lw=0, label="typical free-flow band (80-100 km/h)")
    ax.set(ylabel="Speed (km/h)", ylim=(0, 125))
    ax.set_title(f"(b) One week of 10-minute speeds on link {link_id} (4-10 Oct 2021)", fontsize=11)
    ax.legend(loc="lower left", fontsize=9.5)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a\n%d %b"))
    save(fig, "fig3_spatiotemporal_data.png")


# ---------------------------------------------------------------- Figure 4
def fig4_block_diagram():
    fig, ax = canvas(11, 6.6)
    y0 = 5.9
    top = [(1.15, "EXPY-TKY raw CSVs\n(Oct-Dec 2021,\n2,841 links)", BLUE),
           (3.6, "Pivot + subset\n(13,248 x 1,843)", GREY),
           (6.0, "Normalise\n(train statistics)", GREY),
           (8.55, "Sliding window\nW in {6,12,18}\nH in {1,3,6}", GREY)]
    for x, t, c in top:
        box(ax, x, y0, 2.0, 1.0, t, fc=c)
    for (xa, _, _), (xb, _, _) in zip(top, top[1:]):
        arrow(ax, (xa + 1.0, y0), (xb - 1.0, y0))
    box(ax, 10.3, 4.35, 1.3, 0.8, "Adjacency\nmatrix\n(1,843 x 1,843)", fc=BLUE, fs=9)
    models = [(1.3, "Historical\nAverage"), (3.35, "Linear\nRegression"), (5.4, "XGBoost"),
              (7.45, "LSTM"), (9.5, "STGCN")]
    ym = 3.35
    for x, t in models:
        box(ax, x, ym, 1.7, 0.85, t, fc=PURPLE, bold=True)
        arrow(ax, (8.55, y0 - 0.5), (x, ym + 0.43), rad=0)
    arrow(ax, (10.3, 3.95), (9.9, ym + 0.2))
    ax.text(0.25, 4.6, "Same windows, same split\n(Oct-Nov train/val, Dec test)", fontsize=9.5, style="italic")
    ye = 1.95
    box(ax, 4.2, ye, 4.6, 0.75, "Evaluation: MAE, RMSE, MAPE, R$^2$  (45 experiments)", fc=ORANGE)
    for x, _ in models:
        arrow(ax, (x, ym - 0.43), (min(max(x, 2.2), 6.3), ye + 0.38))
    box(ax, 8.6, ye, 2.2, 0.75, "SQLite\nresults store", fc=GREY)
    arrow(ax, (6.5, ye), (7.5, ye))
    yb = 0.55
    box(ax, 2.2, yb, 2.4, 0.7, "Best model", fc=GREEN, bold=True)
    box(ax, 5.4, yb, 2.4, 0.7, "FastAPI\n/predict endpoint", fc=GREEN)
    box(ax, 8.6, yb, 2.4, 0.7, "Streamlit map\ndashboard", fc=GREEN)
    arrow(ax, (4.2, ye - 0.38), (2.2, yb + 0.35))
    arrow(ax, (3.4, yb), (4.2, yb))
    arrow(ax, (6.6, yb), (7.4, yb))
    save(fig, "fig4_block_diagram.png")


# ---------------------------------------------------------------- Figure 5
def fig5_flowchart():
    fig, ax = canvas(8.4, 11.2)
    x = 3.4
    def term(y, t):
        ax.add_patch(FancyBboxPatch((x - 0.9, y - 0.27), 1.8, 0.54, boxstyle="round,pad=0.02,rounding_size=0.27",
                                    fc=GREEN, ec=EDGE, lw=1.1))
        ax.text(x, y, t, ha="center", va="center", weight="bold")
    ys = {}
    term(10.8, "START"); ys["start"] = 10.8
    steps = [("load", 10.0, "Load 3 monthly CSV files"),
             ("pivot", 9.05, "Pivot to (time x link) matrix,\nselect 1,843 links + adjacency")]
    for k, y, t in steps:
        box(ax, x, y, 3.6, 0.62 if "\n" not in t else 0.75, t)
        ys[k] = y
    arrow(ax, (x, 10.53), (x, 10.31))
    arrow(ax, (x, 9.69), (x, 9.43))
    diamond(ax, x, 8.05, 2.6, 1.0, "Missing\nvalues?")
    arrow(ax, (x, 8.67), (x, 8.55))
    box(ax, 6.6, 8.05, 1.9, 0.6, "Interpolate\n(found: 0)", fc=GREY)
    arrow(ax, (x + 1.3, 8.05), (6.6 - 0.95, 8.05), text=" Yes")
    rest = [("feat", 6.85, "Add time features\n(time of day, day of week, holiday)"),
            ("norm", 5.95, "Normalise with training statistics;\nsplit Oct-Nov (train/val) | Dec (test)"),
            ("loop", 5.0, "For each W in {6,12,18}, H in {1,3,6},\nModel in {HA, LR, XGB, LSTM, STGCN}"),
            ("train", 4.05, "Train  >  Validate (early stopping)\n>  Test  >  Save metrics")]
    for k, y, t in rest:
        box(ax, x, y, 4.2, 0.72, t, fc=PURPLE if k in ("loop", "train") else BLUE)
        ys[k] = y
    arrow(ax, (x, 7.55), (x, 7.21))
    ax.text(x + 0.12, 7.38, "No", fontsize=9.5, style="italic", va="center")
    ax.add_patch(FancyArrowPatch((6.6, 7.75), (x + 2.1, 6.85), arrowstyle="-|>", mutation_scale=13, lw=1.2,
                                 color=EDGE, connectionstyle="angle,angleA=-90,angleB=0"))
    for a, b in [("feat", "norm"), ("norm", "loop"), ("loop", "train")]:
        arrow(ax, (x, ys[a] - 0.36), (x, ys[b] + 0.36))
    diamond(ax, x, 2.85, 2.6, 1.0, "All 45\nruns done?")
    arrow(ax, (x, 3.69), (x, 3.35))
    ax.add_patch(FancyArrowPatch((x - 1.3, 2.85), (x - 2.1, 5.0), arrowstyle="-|>", mutation_scale=13, lw=1.2,
                                 color=EDGE, connectionstyle="angle,angleA=180,angleB=-90"))
    ax.text(x - 2.3, 2.35, "No: next\nconfiguration", fontsize=9.5, style="italic", ha="center")
    box(ax, x, 1.65, 3.6, 0.62, "Compare results, deploy best model", fc=ORANGE)
    arrow(ax, (x, 2.35), (x, 1.96))
    ax.text(x + 0.12, 2.15, "Yes", fontsize=9.5, style="italic", va="center")
    term(0.65, "END")
    arrow(ax, (x, 1.34), (x, 0.92))
    ax.text(6.85, 2.85, "Checkpoint: if STGCN is\nnot training by Day 33,\nreport LSTM as best\nand STGCN as partial",
            fontsize=9.5, style="italic", ha="center", bbox=dict(fc="#fff8e1", ec=EDGE, lw=0.8, ls="--"))
    save(fig, "fig5_flowchart.png")


# ---------------------------------------------------------------- Figure 6
def fig6_model_framework():
    fig, ax = canvas(11, 5.6)
    ax.text(0.1, 5.35, "(a) STGCN", weight="bold", fontsize=12)
    y = 4.2
    box(ax, 1.0, y, 1.6, 0.9, "Input\n(B x W x N x F)", fc=BLUE)
    ax.add_patch(FancyBboxPatch((2.15, 3.35), 5.3, 1.7, boxstyle="round,pad=0.02", fc="white", ec=EDGE, ls="--"))
    ax.text(4.8, 4.88, "ST-Block  (x 2)", ha="center", style="italic")
    parts = [(3.0, "Temporal\ngated conv\n(GLU)"), (4.8, "Spatial\ngraph conv\n(Chebyshev)"), (6.6, "Temporal\ngated conv\n(GLU)")]
    for x, t in parts:
        box(ax, x, y, 1.45, 0.95, t, fc=PURPLE, fs=10)
    for (xa, _), (xb, _) in zip(parts, parts[1:]):
        arrow(ax, (xa + 0.73, y), (xb - 0.73, y))
    box(ax, 4.8, 2.55, 2.7, 0.55, "Adjacency  Â  (road graph)", fc=GREY, fs=10)
    arrow(ax, (4.8, 2.83), (4.8, 3.7))
    box(ax, 8.55, y, 1.5, 0.9, "Output\nlayer", fc=ORANGE)
    box(ax, 10.2, y, 1.4, 0.9, "Speed at t+H\n(all N links)", fc=GREEN, fs=10)
    arrow(ax, (1.8, y), (2.27, y))
    arrow(ax, (7.33, y), (7.8, y))
    arrow(ax, (9.3, y), (9.5, y))

    ax.text(0.1, 1.75, "(b) LSTM", weight="bold", fontsize=12)
    y = 0.75
    chain = [(1.0, "Input\n(B x W x F)", BLUE), (2.9, "LSTM layer 1\n(64 units)", PURPLE),
             (4.8, "LSTM layer 2\n(64 units)", PURPLE), (6.6, "Dropout\n(0.2)", GREY),
             (8.4, "Linear\nlayer", ORANGE), (10.2, "Speed at t+H\n(per link)", GREEN)]
    for x, t, c in chain:
        box(ax, x, y, 1.55, 0.85, t, fc=c, fs=10)
    for (xa, _, _), (xb, _, _) in zip(chain, chain[1:]):
        arrow(ax, (xa + 0.78, y), (xb - 0.78, y))
    ax.text(5.5, -0.05, "B = batch size, W = input window (time steps), N = road links (1,843), "
            "F = input features, H = prediction horizon", ha="center", fontsize=9.5, style="italic")
    save(fig, "fig6_model_framework.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in [fig1_congestion_cycle, fig2_its_pipeline, fig3_spatiotemporal, fig4_block_diagram,
              fig5_flowchart, fig6_model_framework]:
        f()
    print("saved:", sorted(os.listdir(OUT)))
