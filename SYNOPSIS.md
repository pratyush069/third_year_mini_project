# Synopsis — Traffic Speed Forecasting on Tokyo Expressways (EXPY-TKY) using ML and Graph Deep Learning

## Related Previous Work

**1. MegaCRN — (Jiang, R., et al., 2023) [1]**
MegaCRN (AAAI 2023) learns a road graph that changes over time instead of using a fixed one. A memory bank stores typical traffic patterns, and each road uses it to build its own adjacency matrix. This is fed into a GCRN (graph convolution + GRU) encoder–decoder. The model was tested on METR-LA, PEMS-BAY and the new Tokyo dataset **EXPY-TKY**, which the authors released. It beat 10+ baselines (STGCN, DCRNN, Graph WaveNet) at every horizon. **Gaps:** the model is complex, uses a fixed 12-step window, and doesn't compare simple baselines (LR, XGBoost) or use holiday features.

**2. PDFormer — (Jiang, J., et al., 2023) [2]**
PDFormer (AAAI 2023) is a Transformer that models the *delay* with which traffic spreads between sensors. Geographic and semantic (DTW-based) masks guide its spatial attention. On six datasets (PeMS04/07/08, NYCTaxi, CHIBike, T-Drive) it achieved state-of-the-art results (PeMS04 MAE ≈ 18.3). **Gaps:** attention costs O(N²), DTW is expensive to compute, and no Japanese or speed dataset is tested.

**3. STAEformer — (Liu, H., et al., 2023) [3]**
STAEformer (CIKM 2023) shows that a plain Transformer plus a learnable *spatio-temporal adaptive embedding* reaches state-of-the-art accuracy without a predefined graph. On METR-LA its 60-min MAE ≈ 3.34 across six benchmarks. **Gaps:** it can't handle unseen roads, attention costs O(N²), and it is evaluated only on old US/China datasets.

**4. STID — (Shao, Z., et al., 2022) [4]**
STID (CIKM 2022) replaces graph networks with a simple MLP plus spatial, time-of-day and day-of-week identity embeddings. It matched heavy spatio-temporal GNNs while being much faster (METR-LA 60-min MAE ≈ 3.55). **Gaps:** it ignores road topology, and it shows the field needs fair comparisons between simple and complex models.

**5. GNN Traffic Survey — (Jiang, W., et al., 2022) [5]**
This survey in *Expert Systems with Applications* reviews 200+ GNN traffic papers by task, graph type and model family. It finds that GNNs beat ARIMA/LSTM and that STGCN-type models train fastest. **Gaps it lists:** reliance on old US datasets, no unified benchmarks, little use of external factors (holidays, weather), and few real deployments.

### Literature Table

| # | Paper / Author | Intro | Dataset / Algorithm | Gaps | Future Scope / Conclusion | Accuracy |
|---|---|---|---|---|---|---|
| 1 | **(Jiang, R., et al., 2023) [1]** | Dynamic meta-graph traffic model | EXPY-TKY, METR-LA / GCRN + memory | Complex; no simple baselines | Adaptive graphs handle incidents | Best vs 10+ baselines |
| 2 | **(Jiang, J., et al., 2023) [2]** | Delay-aware traffic Transformer | PeMS04/08 etc. / Transformer | O(N²); no Japan data | Efficient attention needed | PeMS04 MAE ≈ 18.3 |
| 3 | **(Liu, H., et al., 2023) [3]** | Embeddings make Transformer SOTA | METR-LA, PEMS / Transformer | Not inductive; old data | Embeddings matter most | METR-LA MAE ≈ 3.34 |
| 4 | **(Shao, Z., et al., 2022) [4]** | Simple MLP baseline for traffic | METR-LA, PEMS / MLP + IDs | Ignores road graph | Simple models competitive | METR-LA MAE ≈ 3.55 |
| 5 | **(Jiang, W., et al., 2022) [5]** | Survey of GNN traffic models | 200+ papers / GNNs | Old data; no deployment | New datasets, deployment | N/A (survey) |

---

## Research Gaps

1. **Old US benchmarks dominate:** most models are tested only on METR-LA/PEMS, and few use newer non-US data like EXPY-TKY [3], [5].
2. **No fair comparison:** simple baselines (LR, XGBoost) are rarely compared with graph models under the same setup [1], [4].
3. **Window × horizon effect not studied:** how much history is needed for each prediction horizon is rarely reported [1], [2].
4. **No deployment and no holiday features:** most work stays in notebooks, and Japanese calendar effects are ignored [5].

---

## 1. Software & Hardware Requirements

| Category | Requirement | Specification | Purpose |
|---|---|---|---|
| **Network** | Internet / localhost | ≥ 10 Mbps; ports 8000, 8501 | Dataset download (~400 MB), Kaggle, API & dashboard |
| **Operating System** | macOS / Windows; Kaggle Notebook VM (Linux) | macOS 14+, Win 10+ | Development and GPU training |
| **Database** | SQLite 3 + CSV/NumPy files | `traffic.db`, `.npy` | Store links, readings, predictions, results |
| **System Config** | CPU / RAM / Storage | i5 or M1, 8–16 GB RAM, 10 GB SSD | Preprocessing, LR & XGBoost |
| **System Config** | GPU | NVIDIA T4/P100 16 GB (Kaggle free tier) | LSTM & STGCN training |
| Language | Python | 3.10+ | Implementation |
| Libraries | NumPy, Pandas, PyYAML, jpholiday | latest | Data handling, holiday features |
| ML / DL | scikit-learn, XGBoost, PyTorch | 1.4+, 2.x, 2.x | Four models |
| Visualisation | Matplotlib | 3.8+ | Plots |
| Deployment | FastAPI + Uvicorn, Streamlit | latest | REST API, map dashboard |
| Tools | Kaggle Notebooks, Git/GitHub, VS Code | — | Coding, version control |

---

## 2. Proposed Method

The project predicts future speed (km/h) on **1,843 Tokyo expressway links** using EXPY-TKY (real GPS-probe data, Oct–Dec 2021, 10-min readings, 13,248 time steps). The raw long-format files (`timestamp, linkid, speed`) are pivoted into a time × link matrix, which has 0 missing values after pivoting, and reordered to match the adjacency matrix. Time-of-day, day-of-week and Japanese-holiday features are added. Speeds are Z-score normalised. A **sliding window** uses W past steps (6/12/18 = 1–3 h) to predict H steps ahead (1/3/6 = 10/30/60 min). Following the MegaCRN split, Oct–Nov is used for training (last 25 % as validation: 6,588 / 2,196 steps) and **December (4,464 steps) for testing**. Five models of increasing complexity are trained: a **Historical-Average baseline, Linear Regression, XGBoost, LSTM and STGCN** (STGCN is adapted from an open-source implementation and uses the road adjacency matrix). That gives **45 experiments** (5 models × 3 windows × 3 horizons), evaluated with **MAE, RMSE, MAPE and R²**. The best model is served through a **FastAPI** API and a **Streamlit** map dashboard, and results are stored in **SQLite**.

**Steps of Execution:** ① Download data → ② Pivot, subset to 1,843 links, align with adjacency → ③ Add features & normalise → ④ Sliding windows & split → ⑤ Historical Avg, LR, XGBoost → ⑥ Train LSTM, STGCN (Kaggle GPU) → ⑦ Evaluate on December → ⑧ Save to DB → ⑨ Deploy API + dashboard → ⑩ Report & paper.

**Block Diagram**
```
EXPY-TKY ─▶ Pivot + Subset ─▶ Features + Normalise ─▶ Sliding Window
                                                        │
        ┌──────────┬──────────┬──────────┬─────────────┤
        ▼          ▼          ▼          ▼             ▼
    Hist. Avg     LR       XGBoost     LSTM      STGCN (+ Adjacency)
        └──────────┴──────────┴────┬─────┴─────────────┘
                                   ▼
                 Evaluation (MAE, RMSE, MAPE, R²) ─▶ SQLite
                                    ▼
                    Best model ─▶ FastAPI ─▶ Streamlit Map
```

**Flow Chart**
```
START → Load 3 monthly CSVs → Pivot → Subset 1,843 links → Missing? ─Yes→ Interpolate (found: 0)
                       │No
          Add features → Normalise (train stats only)
          Loop W ∈ {6,12,18}, H ∈ {1,3,6}, Model ∈ {HA,LR,XGB,LSTM,STGCN}
              Train → Validate (early stop) → Test → Save metrics
          Compare 45 results → Deploy best → END
          (Checkpoint: if STGCN not training by Day 33 → report LSTM as best, STGCN as partial)
```

**Model Framework (STGCN)**
```
Input (B×W×N×F) → [ST-Block: Temporal GLU → Graph Conv (Â) → Temporal GLU] ×2 → Output Layer → Speed at t+H
```
**LSTM:** Input → 2 LSTM layers (64) → Dropout 0.2 → Linear → Speed.

**Structural Diagram**
```
Streamlit Dashboard  ⇄ (HTTP/JSON) ⇄  FastAPI (/predict, /links, /metrics)  ⇄  SQLite + Saved Models
                                        ▲ trained on Kaggle Notebooks (GPU), code on GitHub
```

**Key Numbers**

| Item | Value |
|---|---|
| Nodes / time steps | 1,843 links (of 2,841) / 13,248 steps; adjacency 1843 × 1843 |
| Variables | 4 inputs (speed, time-of-day, day-of-week, holiday) + 1 target |
| Models / Experiments | 5 / 45 |
| Metrics | 4 (MAE, RMSE, MAPE, R²) |
| Formulas | 8 (below) |
| Hyper-parameters | Adam lr 1e-3, batch 64, ≤100 epochs, patience 10; XGB 300 trees, depth 6; LSTM 64×2 |
| Parameters (approx.) | LSTM ≈ 51 K; STGCN ≈ 0.3–0.5 M |
| LLMs used | **0** |
| Cloud services | **2** (Kaggle Notebooks + Kaggle Datasets, GitHub) |
| DB tables | **4** |
| Software tools | **14** |

**Formulas**

| Name | Formula |
|---|---|
| Z-score | x' = (x − μ) / σ |
| Linear Regression | ŷ = β₀ + Σ βᵢxᵢ |
| XGBoost objective | 𝓛 = Σ l(y, ŷ) + Σ (γT + ½λ‖w‖²) |
| Normalised adjacency | Â = D̃^(−½)(A + I)D̃^(−½) |
| MAE | (1/n) Σ \|y − ŷ\| |
| RMSE | √[(1/n) Σ (y − ŷ)²] |
| MAPE | (100/n) Σ \|(y − ŷ)/y\| |
| R² | 1 − Σ(y − ŷ)² / Σ(y − ȳ)² |

**Database Tables (4)**

| Table | Columns |
|---|---|
| `road_links` | link_id (PK), roadname, start_lon, start_lat, end_lon, end_lat |
| `speed_readings` | reading_id (PK), link_id (FK), timestamp, speed_kmh, is_holiday |
| `model_runs` | run_id (PK), model_name, window, horizon, mae, rmse, mape, train_time_s |
| `predictions` | pred_id (PK), run_id (FK), link_id (FK), target_time, predicted_kmh, actual_kmh |

**Other preparation:** dataset licence check (MIT); data uploaded as a private Kaggle Dataset; fixed seed (42); outputs persisted via Kaggle "Save Version"; backup dataset PEMS-BAY if EXPY-TKY becomes unavailable.

**Preliminary result:** Linear Regression (single link, 1-h input, 30-min horizon): MAE 6.145, RMSE 9.008, MAPE 9.35 %.

---

## 3. Deliverables

- **Dataset overview document:** EXPY-TKY_Dataset_Overview with raw samples, link metadata, adjacency shapes and sanity plots, for the professor's review.
- **Data pipeline:** scripts that download, clean and feature-engineer EXPY-TKY and build sliding windows from one config file.
- **5 trained models:** Historical Average, LR, XGBoost, LSTM and STGCN, trained on Oct–Nov and tested on December 2021.
- **Comparative study:** 45 experiments with MAE/RMSE/MAPE/R² tables and plots showing when graph models win.
- **REST API + dashboard:** FastAPI `/predict` endpoint and a Streamlit map of predicted speeds, backed by SQLite.
- **Documentation:** synopsis, report, GitHub repository and a research-paper draft.

---

## 4. Stakeholders

1. **Traffic authorities** (e.g. Metropolitan Expressway Co.) — congestion planning and control.
2. **Commuters** — better travel-time estimates and route choice.
3. **Logistics / taxi / ride-hailing firms** — efficient routing and dispatch.
4. **Navigation app developers** — plug the API into ETA engines.
5. **Urban planners** — study recurring congestion patterns.
6. **Emergency services** — faster routes at peak times.
7. **ITS / AI researchers and students** — reproducible benchmark on new Tokyo data.
8. **Environmental agencies** — less congestion means lower emissions.
9. **Academic institution** — publishable research output.

---

## 5. Gantt Charts (week-wise) — ■ = active

**Chart 1: July – November 2026**

| Task | Jul W1 | Jul W2 | Jul W3 | Jul W4 | Aug W1 | Aug W2 | Aug W3 | Aug W4 | Sep W1 | Sep W2 | Sep W3 | Sep W4 | Oct W1 | Oct W2 | Oct W3 | Oct W4 | Nov W1 | Nov W2 | Nov W3 | Nov W4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Topic selection & ML basics | ■ | ■ | | | | | | | | | | | | | | | | | | |
| Literature review | | ■ | ■ | ■ | ■ | ■ | | | | | | | | | | | | | | |
| Research gaps | | | | | ■ | ■ | | | | | | | | | | | | | | |
| Dataset finalisation | | | | | | ■ | ■ | ■ | ■ | ■ | ■ | ■ | | | | | | | | |
| Requirements & setup | | | | | | | ■ | ■ | | | | | | | | | | | | |
| System design | | | | | | | | ■ | ■ | | | | | | | | | | | |
| Synopsis submission | | | | | | | | | | | ■ | ■ | ■ | | | | | | | |
| Data download & EDA | | | | | | | | | | | | ■ | ■ | ■ | | | | | | |
| Preprocessing & features | | | | | | | | | | | | | | ■ | ■ | ■ | | | | |
| Sliding windows | | | | | | | | | | | | | | | | ■ | ■ | | | |
| Hist. Avg, LR & XGBoost | | | | | | | | | | | | | | | | | ■ | ■ | ■ | |
| Mid-term review | | | | | | | | | | | | | | | | | | | ■ | ■ |

**Chart 2: December 2026 – April 2027**

| Task | Dec W1 | Dec W2 | Dec W3 | Dec W4 | Jan W1 | Jan W2 | Jan W3 | Jan W4 | Feb W1 | Feb W2 | Feb W3 | Feb W4 | Mar W1 | Mar W2 | Mar W3 | Mar W4 | Apr W1 | Apr W2 | Apr W3 | Apr W4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LSTM training | ■ | ■ | ■ | | | | | | | | | | | | | | | | | |
| STGCN training | | | ■ | ■ | ■ | ■ | | | | | | | | | | | | | | |
| Hyper-parameter tuning | | | | | | ■ | ■ | | | | | | | | | | | | | |
| 45-run experiments | | | | | | | ■ | ■ | ■ | | | | | | | | | | | |
| Result analysis | | | | | | | | | ■ | ■ | | | | | | | | | | |
| SQLite + FastAPI | | | | | | | | | | ■ | ■ | ■ | | | | | | | | |
| Streamlit dashboard | | | | | | | | | | | | ■ | ■ | ■ | | | | | | |
| Testing | | | | | | | | | | | | | | ■ | ■ | | | | | |
| Research paper | | | | | | | | | | | ■ | ■ | ■ | ■ | ■ | ■ | | | | |
| Final report | | | | | | | | | | | | | | | ■ | ■ | ■ | ■ | | |
| Final presentation | | | | | | | | | | | | | | | | | | ■ | ■ | ■ |

---

## 1.8 References

[1] R. Jiang, Z. Wang, J. Yong, P. Jeph, Q. Chen, Y. Kobayashi, X. Song, S. Fukushima, and T. Suzumura (2023). Spatio-Temporal Meta-Graph Learning for Traffic Forecasting, in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 7, Washington, DC, USA, Feb. 2023, pp. 8078–8086. doi: 10.1609/aaai.v37i7.25976.

[2] J. Jiang, C. Han, W. X. Zhao, and J. Wang (2023). PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction, in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 4, Washington, DC, USA, Feb. 2023, pp. 4365–4373. doi: 10.1609/aaai.v37i4.25556.

[3] H. Liu, Z. Dong, R. Jiang, J. Deng, J. Deng, Q. Chen, and X. Song (2023). STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting, in Proceedings of the 32nd ACM International Conference on Information and Knowledge Management (CIKM '23), Birmingham, UK, Oct. 2023, pp. 4125–4129. doi: 10.1145/3583780.3615160.

[4] Z. Shao, Z. Zhang, F. Wang, W. Wei, and Y. Xu (2022). Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting, in Proceedings of the 31st ACM International Conference on Information and Knowledge Management (CIKM '22), Atlanta, GA, USA, Oct. 2022, pp. 4454–4458. doi: 10.1145/3511808.3557702.

[5] W. Jiang and J. Luo (2022). Graph Neural Network for Traffic Forecasting: A Survey, Expert Systems with Applications, vol. 207, Art. no. 117921. doi: 10.1016/j.eswa.2022.117921.

[6] B. Yu, H. Yin, and Z. Zhu (2018). Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting, in Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI-18), Stockholm, Sweden, Jul. 2018, pp. 3634–3640. doi: 10.24963/ijcai.2018/505.

[7] T. Chen and C. Guestrin (2016). XGBoost: A Scalable Tree Boosting System, in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, Aug. 2016, pp. 785–794. doi: 10.1145/2939672.2939785.

[8] S. Hochreiter and J. Schmidhuber (1997). Long Short-Term Memory, Neural Computation, vol. 9, no. 8, pp. 1735–1780. doi: 10.1162/neco.1997.9.8.1735.
