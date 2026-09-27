"""Summarise results/all_results.csv into tables and research plots.

Usage:  python -m src.report            -> prints test tables, saves results/figures/fig_*_vs_*.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["persistence", "linear", "xgboost", "lstm", "stgcn"]
OUT = "results/figures"


def main():
    df = pd.read_csv("results/all_results.csv").query("split == 'test'")
    df = df.drop_duplicates(["model", "window", "horizon"], keep="last")
    models = [m for m in ORDER if m in set(df.model)]

    for metric in ["MAE", "RMSE", "MAPE", "R2"]:
        t = df.pivot_table(index=["window", "horizon"], columns="model", values=metric)[models]
        print(f"\n--- test {metric} (rows: window, horizon in 10-min steps) ---")
        print(t.round(3).to_string())

    # Error vs horizon (window fixed at its middle value) and vs window (horizon fixed)
    w_mid = sorted(df.window.unique())[len(df.window.unique()) // 2]
    h_mid = sorted(df.horizon.unique())[len(df.horizon.unique()) // 2]
    for x, fixed, fval, fname in [("horizon", "window", w_mid, "fig_mae_vs_horizon.png"),
                                  ("window", "horizon", h_mid, "fig_mae_vs_window.png")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in models:
            s = df[(df.model == m) & (df[fixed] == fval)].sort_values(x)
            ax.plot(s[x] * 10, s.MAE, marker="o", label=m)
        ax.set(xlabel=f"{x.capitalize()} (minutes)", ylabel="Test MAE (km/h)",
               title=f"MAE vs {x} ({fixed} = {fval * 10} min)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.savefig(f"{OUT}/{fname}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"\nplots saved to {OUT}/fig_mae_vs_horizon.png, fig_mae_vs_window.png")


if __name__ == "__main__":
    main()
