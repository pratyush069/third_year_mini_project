# Short-Term Traffic Speed Prediction on Tokyo Expressways

Comparing traditional ML (Linear Regression, XGBoost), temporal deep learning (LSTM) and
spatiotemporal graph deep learning (STGCN) for short-term traffic speed prediction, and
studying how the **input window** and **prediction horizon** affect each model.

## Dataset
**EXPY-TKY**: speeds on 1,843 Tokyo Metropolitan Expressway road links, every 10 minutes,
Oct-Dec 2021 (from vehicle GPS probes). Source: MegaCRN repository, MIT license.

> R. Jiang et al., "Spatio-Temporal Meta-Graph Learning for Traffic Forecasting," AAAI 2023.

## Setup
The project is intended to run in **Kaggle Notebooks** so the LSTM and STGCN can use
Kaggle's available GPU. Install `requirements.txt` only if you also want to run the
pipeline locally.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Pipeline
| Step | Command | Output |
|---|---|---|
| 1. Download data (~414 MB) | `python -m src.data.download` | `data/raw/` |
| 2. Clean + build speed matrix | `python -m src.data.preprocess` | `data/processed/speed.npz` [13248 x 1843] |
| 3. Build road graph | `python -m src.data.graph` | `data/processed/adj.npy` [1843 x 1843] |
| 4. Exploratory figures | `python -m src.eda` | `results/figures/*.png` |
| 5. One model, one setting | `python -m src.train --model xgboost --window 6 --horizon 3` | row in `results/comparison_results.csv` |
| 6. Full window x horizon grid | `python -m src.experiments` | 45 runs in `results/comparison_results.csv` |
| 7. Tables + plots | `python -m src.report` | `results/figures/fig_mae_vs_*.png` |

Start with one setting of each model before launching the full grid, for example:
`python -m src.train --model stgcn --window 6 --horizon 3`.
The earlier `results/all_results.csv` is retained as an archive; new runs use
`results/comparison_results.csv` so results from different settings are not mixed.

All settings are in `configs/default.yaml`.

## Data-processing decisions
- Speeds < 0 set to 0, speeds > 200 set to 100 (same as MegaCRN).
- Speed 0 = no probe vehicle = **missing** (0.25%): filled by time interpolation for model
  inputs, and excluded from error metrics (masked MAE/RMSE/MAPE).
- Graph: Gaussian kernel on road distance, `w = exp(-(d/500 m)^2)`, weights < 0.1 dropped,
  plus direct road connections; 7 otherwise-isolated links joined to their nearest link.
- Split: Oct-Nov = train/validation (last 25% for validation), Dec = test (chronological).
- Known limitation: ~21-36 of 1,843 links contain template-filled (repeated) days in the
  source data; kept for comparability with published results.
