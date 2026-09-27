# Synopsis — Spatio-Temporal Traffic Speed Forecasting on Tokyo Expressways (EXPY-TKY) using ML and Graph Deep Learning

---

## Research Gaps

1. **Over-reliance on old, saturated US benchmarks.** Most traffic-forecasting studies still evaluate on METR-LA and PEMS-BAY (2012–2017 data from Los Angeles / Bay Area). Very few studies benchmark classical *and* graph models on newer, larger, non-US road networks such as the Tokyo Metropolitan Expressway (EXPY-TKY, 2021), so it is unclear whether published gains generalise.
2. **No fair, controlled comparison across model families.** Papers usually compare a new deep model against other deep models. Simple baselines (Linear Regression, gradient-boosted trees) are often omitted or tuned poorly, so the real benefit of spatial (graph) modelling over a strong tabular baseline is not quantified under the same data split, windows and horizons.
3. **Limited study of input-window × prediction-horizon sensitivity.** Most works report fixed settings (12 steps in → 3/6/12 out). A systematic grid of how much history (1 h, 2 h, 3 h) is needed for each horizon (10, 30, 60 min), and how this differs per model, is rarely reported.
4. **Research stays in notebooks — no deployable pipeline.** Few academic works deliver an end-to-end system (data → model → REST API → map dashboard) that operators or commuters can actually use, and calendar effects specific to Japan (public holidays, weekday/weekend) are seldom encoded explicitly.

---

## 1. Software & Hardware Requirements

### 1.1 Must-mention requirements

| Category | Requirement | Specification / Version | Purpose in Project |
|---|---|---|---|
| **Network** | Internet connection | ≥ 10 Mbps broadband | Download EXPY-TKY (~3 CSV.gz + adjacency files) from GitHub, use Google Colab GPU, push code to GitHub |
| **Network** | Local network (localhost) | HTTP on ports 8000 (API) & 8501 (dashboard) | FastAPI server and Streamlit dashboard communication |
| **Operating System** | Development OS | macOS 14+ / Windows 10+ / Ubuntu 22.04 | Local coding, preprocessing, API & dashboard |
| **Operating System** | Training OS | Google Colab (Ubuntu 22.04 VM) | GPU training of LSTM and STGCN |
| **Database** | SQLite 3 | Embedded, file-based (`traffic.db`) | Store road links, speed readings, predictions and experiment results (4 tables) |
| **Database** | File storage | CSV / NumPy `.npy` / `.npz` | Raw and processed tensors (speed matrix, adjacency) |
| **System Configuration** | Processor | Intel i5 / Apple M1 or higher (4+ cores) | Preprocessing, LR & XGBoost training |
| **System Configuration** | RAM | 8 GB minimum, 16 GB recommended | Holding 1843-link × ~13,000-step speed matrix in memory |
| **System Configuration** | Storage | 10 GB free (SSD) | Dataset, processed windows, saved models, results |
| **System Configuration** | GPU | NVIDIA T4 16 GB (Colab free tier) | Deep learning training (LSTM, STGCN) |

### 1.2 Additional project-specific requirements

| Category | Tool / Library | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.10+ | Complete implementation |
| Data handling | NumPy, Pandas | 1.26+, 2.x | Arrays, time-series tables |
| Configuration | PyYAML | 6.x | Read `configs/default.yaml` |
| Calendar features | jpholiday | 0.1+ | Japanese public-holiday flag |
| Classical ML | scikit-learn | 1.4+ | Linear Regression, scaling, metrics |
| Boosting | XGBoost | 2.x | Gradient-boosted tree model |
| Deep learning | PyTorch | 2.x (CUDA) | LSTM and STGCN models |
| Visualisation | Matplotlib | 3.8+ | Loss curves, error vs horizon plots |
| Notebook | Jupyter / Google Colab | — | Experiments and training |
| Backend API | FastAPI + Uvicorn | 0.110+ | REST endpoint `/predict` |
| Frontend | Streamlit (+ map layer) | 1.3x | Interactive dashboard with Tokyo road map |
| Version control | Git + GitHub | — | Code hosting, collaboration |
| IDE | VS Code | latest | Development |
| Documentation | MS Word / LaTeX | — | Synopsis, report, research paper |

---

## 2. Proposed Method

### 2.1 Description (process)

The project forecasts the future average speed (km/h) on **1,843 road links** of the Tokyo Metropolitan Expressway using the public **EXPY-TKY** dataset (Oct–Dec 2021, one reading every 10 min). First, the raw monthly CSVs and the road adjacency matrix are downloaded and cleaned (missing values filled by forward/linear interpolation). Time features — time-of-day, day-of-week and a Japanese-holiday flag — are added. Speeds are Z-score normalised using training statistics only. A **sliding window** turns the series into supervised samples: *W* past steps (W = 6, 12, 18 → 1, 2, 3 hours) predict the speed *H* steps ahead (H = 1, 3, 6 → 10, 30, 60 min). The split is chronological: Oct + Nov for training (last 10 % as validation) and **December as the unseen test month**. Four models of increasing complexity are trained: **Linear Regression** (baseline), **XGBoost** (non-linear tabular), **LSTM** (temporal deep model) and **STGCN** (spatio-temporal graph model that uses the road adjacency matrix). Each model is run on the full 3 × 3 grid of windows × horizons (**9 settings × 4 models = 36 experiments**) and evaluated with **MAE, RMSE and MAPE**. The best model is saved and served through a **FastAPI REST API**, and a **Streamlit dashboard** shows predicted vs. actual speeds on a Tokyo map. Results are stored in a SQLite database and summarised for a research paper.

### 2.2 Steps of Execution

| Step | Activity | Output |
|---|---|---|
| 1 | Download EXPY-TKY (3 monthly CSVs, adjacency, link info) | `data/raw/` |
| 2 | Select 1,843-link subset, clean & interpolate missing values | Clean speed matrix T × N |
| 3 | Feature engineering (time-of-day, day-of-week, holiday) | Feature tensor T × N × F |
| 4 | Z-score normalisation (train stats) + chronological split | train / val / test |
| 5 | Sliding-window sample generation for each (W, H) | X, Y arrays |
| 6 | Train LR and XGBoost (CPU) | Baseline models |
| 7 | Train LSTM and STGCN (Colab GPU, early stopping) | Deep models `.pt` |
| 8 | Evaluate on December test month (MAE, RMSE, MAPE) | Results table, plots |
| 9 | Store runs & predictions in SQLite | `traffic.db` |
| 10 | Deploy best model via FastAPI + Streamlit dashboard | Working web app |
| 11 | Write report and research paper | Final documents |

### 2.3 Block Diagram

```
 ┌──────────────┐   ┌────────────────┐   ┌─────────────────┐   ┌────────────────────┐
 │ EXPY-TKY Raw │──▶│ Preprocessing  │──▶│ Feature Eng. +  │──▶│ Sliding Window     │
 │ CSV + Adj.   │   │ clean/interp.  │   │ Normalisation   │   │ (W in → H ahead)   │
 └──────────────┘   └────────────────┘   └─────────────────┘   └─────────┬──────────┘
                                                                          │
              ┌───────────────┬───────────────┬───────────────┬──────────┘
              ▼               ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
        │ Linear   │    │ XGBoost  │    │  LSTM    │    │ STGCN (uses  │
        │ Regress. │    │          │    │          │    │ adjacency A) │
        └────┬─────┘    └────┬─────┘    └────┬─────┘    └──────┬───────┘
             └───────────────┴───────┬───────┴─────────────────┘
                                     ▼
                     ┌──────────────────────────────┐
                     │ Evaluation: MAE, RMSE, MAPE  │──▶ SQLite DB
                     └──────────────┬───────────────┘
                                    ▼
                  ┌──────────────────────────────────────┐
                  │ Best model → FastAPI → Streamlit map │
                  └──────────────────────────────────────┘
```

### 2.4 Flow Chart

```
        START
          │
   Download dataset
          │
   Missing values? ──Yes──▶ Interpolate
          │No                   │
          ◀─────────────────────┘
   Add time features, normalise
          │
   For each W ∈ {6,12,18}:
     For each H ∈ {1,3,6}:
       For each model ∈ {LR, XGB, LSTM, STGCN}:
          Train on Oct–Nov ─▶ Validate ─▶ Early stop?
          Test on December ─▶ Save MAE/RMSE/MAPE
          │
   Compare 36 results ─▶ select best model
          │
   Deploy (FastAPI + Streamlit)
          │
         END
```

### 2.5 Model Framework (STGCN — main model)

```
Input X (B × W × N × F)
   │
   ▼
[ST-Conv Block 1]  Temporal Gated Conv (GLU) ─▶ Spatial Graph Conv (Chebyshev, K=3, uses Â) ─▶ Temporal Gated Conv ─▶ LayerNorm
   │
   ▼
[ST-Conv Block 2]  (same structure)
   │
   ▼
[Output Layer]  Temporal Conv ─▶ Fully Connected ─▶ Ŷ (B × N) speed at t+H
```
LSTM framework: Input (W × F) ─▶ 2 stacked LSTM layers (64 hidden) ─▶ Dropout 0.2 ─▶ Linear ─▶ speed at t+H (weights shared across links).

### 2.6 Structural Diagram (system architecture)

```
┌───────────────── Presentation Layer ─────────────────┐
│  Streamlit Dashboard (map, link selector, charts)     │
└───────────────────────────┬───────────────────────────┘
                            │ HTTP/JSON
┌───────────────── Application Layer ──────────────────┐
│  FastAPI:  /predict   /links   /metrics   /health     │
│  Model loader (PyTorch / XGBoost / sklearn)           │
└───────────────────────────┬───────────────────────────┘
                            │
┌───────────────── Data Layer ─────────────────────────┐
│  SQLite (4 tables) │ .npy adjacency │ saved models   │
└──────────────────────────────────────────────────────┘
          ▲ trained on Google Colab (GPU) — code on GitHub
```

### 2.7 Quantitative Summary

| Item | Count / Value |
|---|---|
| Road links (nodes) | 1,843 |
| Time steps | ~13,248 (92 days × 144 per day) |
| Input variables / features | 4 (speed, time-of-day, day-of-week, holiday flag) |
| Target variable | 1 (speed in km/h at t + H) |
| Models | 4 (LR, XGBoost, LSTM, STGCN) |
| Input windows | 3 (6, 12, 18 steps) |
| Horizons | 3 (1, 3, 6 steps) |
| Total experiments | 36 |
| Evaluation metrics | 3 (MAE, RMSE, MAPE) |
| Key hyper-parameters | LR: none • XGB: n_estimators=300, max_depth=6, lr=0.1 • LSTM: hidden 64, layers 2, dropout 0.2 • STGCN: Kt=3, Ks=3, channels 64-16-64 • Adam lr=1e-3, batch 64, epochs ≤ 100, early-stop patience 10 |
| Trainable parameters (approx.) | LR ≈ W×F+1 per model; LSTM ≈ 51 K; STGCN ≈ 0.3–0.5 M (exact count reported after training) |
| LLMs used | **0** (no LLM is part of the model or system) |
| Cloud services | **2** — Google Colab (GPU training), GitHub (code hosting/version control) |
| Database tables | **4** (see below) |
| Software tools & services | 16 (Python, NumPy, Pandas, PyYAML, jpholiday, scikit-learn, XGBoost, PyTorch, Matplotlib, Jupyter/Colab, FastAPI, Uvicorn, Streamlit, SQLite, Git/GitHub, VS Code) |

### 2.8 Formulas Used

| # | Name | Formula |
|---|---|---|
| 1 | Z-score normalisation | x' = (x − μ) / σ |
| 2 | Linear Regression | ŷ = β₀ + Σ βᵢ xᵢ |
| 3 | XGBoost objective | 𝓛 = Σ l(yᵢ, ŷᵢ) + Σₖ [γT + ½ λ‖w‖²] |
| 4 | LSTM cell | fₜ=σ(W_f[hₜ₋₁,xₜ]+b_f), iₜ=σ(W_i[·]+b_i), c̃ₜ=tanh(W_c[·]+b_c), cₜ=fₜ⊙cₜ₋₁+iₜ⊙c̃ₜ, hₜ=oₜ⊙tanh(cₜ) |
| 5 | Normalised adjacency | Â = D̃^(−½) (A + I) D̃^(−½) |
| 6 | Graph convolution (Chebyshev) | Θ *𝒢 x = Σₖ₌₀^(K−1) θₖ Tₖ(L̃) x,  L̃ = 2L/λ_max − I |
| 7 | Gated temporal conv (GLU) | Γ *𝒯 Y = P ⊙ σ(Q) |
| 8 | MAE | (1/n) Σ \|yᵢ − ŷᵢ\| |
| 9 | RMSE | √[(1/n) Σ (yᵢ − ŷᵢ)²] |
| 10 | MAPE | (100/n) Σ \|(yᵢ − ŷᵢ)/yᵢ\| |

### 2.9 Database Table Structure (SQLite — 4 tables)

**Table 1: `road_links`**

| Column | Type | Key | Description |
|---|---|---|---|
| link_id | INTEGER | PK | Link index (0–1842) |
| road_name | TEXT | | Expressway route name |
| start_lat | REAL | | Start latitude |
| start_lon | REAL | | Start longitude |
| end_lat | REAL | | End latitude |
| end_lon | REAL | | End longitude |

**Table 2: `speed_readings`**

| Column | Type | Key | Description |
|---|---|---|---|
| reading_id | INTEGER | PK | Auto-increment |
| link_id | INTEGER | FK → road_links | Road link |
| timestamp | DATETIME | | 10-minute slot |
| speed_kmh | REAL | | Observed speed |
| is_holiday | BOOLEAN | | Japanese holiday flag |

**Table 3: `model_runs`**

| Column | Type | Key | Description |
|---|---|---|---|
| run_id | INTEGER | PK | Experiment ID |
| model_name | TEXT | | LR / XGB / LSTM / STGCN |
| window | INTEGER | | Input steps (6/12/18) |
| horizon | INTEGER | | Output step (1/3/6) |
| mae | REAL | | Test MAE |
| rmse | REAL | | Test RMSE |
| mape | REAL | | Test MAPE (%) |
| train_time_s | REAL | | Training time |
| created_at | DATETIME | | Run date |

**Table 4: `predictions`**

| Column | Type | Key | Description |
|---|---|---|---|
| pred_id | INTEGER | PK | Auto-increment |
| run_id | INTEGER | FK → model_runs | Which model produced it |
| link_id | INTEGER | FK → road_links | Road link |
| target_time | DATETIME | | Time being predicted |
| predicted_kmh | REAL | | Model output |
| actual_kmh | REAL | | Ground truth (if known) |

### 2.10 Other Preparation for the Project

- Dataset licence check (MegaCRN repo, MIT) and citation.
- Colab GPU session planning (checkpoints saved to Google Drive to survive 12-h disconnects).
- Fixed random seeds (42) for reproducibility; all settings in `configs/default.yaml`.
- Literature review of 15+ papers; results table template prepared in advance.

---

## 3. Deliverables

- **Cleaned & processed EXPY-TKY dataset pipeline** — reproducible scripts that download, clean, interpolate, add Japanese calendar features and create sliding-window samples for any window/horizon, driven by a single YAML config.
- **Four trained forecasting models** — Linear Regression, XGBoost, LSTM and STGCN, each saved with weights and hyper-parameters, trained on Oct–Nov 2021 and tested on December 2021.
- **Comparative experiment study (36 runs)** — results table and plots of MAE / RMSE / MAPE across 3 windows × 3 horizons × 4 models, showing when graph (spatial) modelling beats simpler models.
- **REST API (FastAPI)** — `/predict`, `/links`, `/metrics` endpoints that return predicted speeds for any road link, with SQLite storage of predictions and runs.
- **Interactive dashboard (Streamlit)** — map of Tokyo expressways coloured by predicted speed, per-link predicted-vs-actual chart and model comparison view.
- **Documentation** — synopsis, final project report, GitHub repository with README, and a research-paper draft suitable for a conference/journal submission.

---

## 4. Stakeholders

1. **Traffic management authorities** (e.g., Metropolitan Expressway Co., city traffic control centres) — use short-term speed forecasts to anticipate congestion and plan signal/ramp control.
2. **Daily commuters and drivers** — benefit from better travel-time estimates and route choice via the dashboard or navigation apps.
3. **Logistics and fleet operators** (delivery, taxis, ride-hailing) — plan dispatch and routes to reduce delays and fuel cost.
4. **Navigation / mobility app developers** — can plug the API into ETA and routing engines.
5. **Urban and transport planners** — study recurring congestion patterns (weekday vs holiday) for infrastructure decisions.
6. **Emergency services** (ambulance, fire, police) — choose faster routes during peak congestion.
7. **Researchers and students in ITS / AI** — get a reproducible benchmark comparing classical ML and graph deep learning on a modern non-US dataset.
8. **Environmental agencies** — reduced congestion means lower idling emissions and fuel use.
9. **Faculty / academic institution** — project output leads to a publishable research paper.
10. **Smart-city solution providers** — can extend the pipeline to other cities' sensor networks.

---

## 5. Gantt Charts (week-wise)

Legend: ■ = active week

### 5.1 Gantt Chart 1 — July 2026 to November 2026

| Task | Jul W1 | Jul W2 | Jul W3 | Jul W4 | Aug W1 | Aug W2 | Aug W3 | Aug W4 | Sep W1 | Sep W2 | Sep W3 | Sep W4 | Oct W1 | Oct W2 | Oct W3 | Oct W4 | Nov W1 | Nov W2 | Nov W3 | Nov W4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Topic selection & problem definition | ■ | ■ | | | | | | | | | | | | | | | | | | |
| Literature review | | ■ | ■ | ■ | ■ | ■ | | | | | | | | | | | | | | |
| Identify research gaps | | | | | ■ | ■ | | | | | | | | | | | | | | |
| Dataset search & finalisation (EXPY-TKY) | | | | | | ■ | ■ | ■ | ■ | ■ | ■ | ■ | | | | | | | | |
| Requirement analysis & tools setup | | | | | | | ■ | ■ | | | | | | | | | | | | |
| System design (block, flow, DB schema) | | | | | | | | ■ | ■ | | | | | | | | | | | |
| Synopsis writing & submission | | | | | | | | | | | ■ | ■ | ■ | | | | | | | |
| Data download & exploratory analysis | | | | | | | | | | | | ■ | ■ | ■ | | | | | | |
| Preprocessing & feature engineering | | | | | | | | | | | | | | ■ | ■ | ■ | | | | |
| Sliding-window dataset generation | | | | | | | | | | | | | | | | ■ | ■ | | | |
| Baseline models (LR, XGBoost) | | | | | | | | | | | | | | | | | ■ | ■ | ■ | |
| Mid-term review / presentation | | | | | | | | | | | | | | | | | | | ■ | ■ |

### 5.2 Gantt Chart 2 — December 2026 to April 2027

| Task | Dec W1 | Dec W2 | Dec W3 | Dec W4 | Jan W1 | Jan W2 | Jan W3 | Jan W4 | Feb W1 | Feb W2 | Feb W3 | Feb W4 | Mar W1 | Mar W2 | Mar W3 | Mar W4 | Apr W1 | Apr W2 | Apr W3 | Apr W4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LSTM implementation & training (Colab) | ■ | ■ | ■ | | | | | | | | | | | | | | | | | |
| STGCN implementation & training | | | ■ | ■ | ■ | ■ | | | | | | | | | | | | | | |
| Hyper-parameter tuning | | | | | | ■ | ■ | | | | | | | | | | | | | |
| 36-run experiment grid (W × H × model) | | | | | | | ■ | ■ | ■ | | | | | | | | | | | |
| Result analysis & visualisation | | | | | | | | | ■ | ■ | | | | | | | | | | |
| SQLite DB + FastAPI backend | | | | | | | | | | ■ | ■ | ■ | | | | | | | | |
| Streamlit dashboard (map UI) | | | | | | | | | | | | ■ | ■ | ■ | | | | | | |
| Integration & system testing | | | | | | | | | | | | | | ■ | ■ | | | | | |
| Research paper writing | | | | | | | | | | | ■ | ■ | ■ | ■ | ■ | ■ | | | | |
| Final report preparation | | | | | | | | | | | | | | | ■ | ■ | ■ | ■ | | |
| Final presentation & viva | | | | | | | | | | | | | | | | | | ■ | ■ | ■ |

---

## 1.8 References

[1] B. Yu, H. Yin, and Z. Zhu (2018). Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting, in Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI-18), Stockholm, Sweden, Jul. 2018, pp. 3634–3640. doi: 10.24963/ijcai.2018/505.

[2] R. Jiang, Z. Wang, J. Yong, P. Jeph, Q. Chen, Y. Kobayashi, X. Song, S. Fukushima, and T. Suzumura (2023). Spatio-Temporal Meta-Graph Learning for Traffic Forecasting, in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 7, Washington, DC, USA, Feb. 2023, pp. 8078–8086. doi: 10.1609/aaai.v37i7.25976.

[3] T. Chen and C. Guestrin (2016). XGBoost: A Scalable Tree Boosting System, in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, Aug. 2016, pp. 785–794. doi: 10.1145/2939672.2939785.

[4] S. Hochreiter and J. Schmidhuber (1997). Long Short-Term Memory, Neural Computation, vol. 9, no. 8, pp. 1735–1780. doi: 10.1162/neco.1997.9.8.1735.

[5] X. Ma, Z. Tao, Y. Wang, H. Yu, and Y. Wang (2015). Long Short-Term Memory Neural Network for Traffic Speed Prediction Using Remote Microwave Sensor Data, Transportation Research Part C: Emerging Technologies, vol. 54, pp. 187–197. doi: 10.1016/j.trc.2015.03.014.

[6] Y. Lv, Y. Duan, W. Kang, Z. Li, and F.-Y. Wang (2015). Traffic Flow Prediction With Big Data: A Deep Learning Approach, IEEE Transactions on Intelligent Transportation Systems, vol. 16, no. 2, pp. 865–873. doi: 10.1109/TITS.2014.2345663.

[7] L. Zhao, Y. Song, C. Zhang, Y. Liu, P. Wang, T. Lin, M. Deng, and H. Li (2020). T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction, IEEE Transactions on Intelligent Transportation Systems, vol. 21, no. 9, pp. 3848–3858. doi: 10.1109/TITS.2019.2935152.

[8] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang (2019). Graph WaveNet for Deep Spatial-Temporal Graph Modeling, in Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI-19), Macao, China, Aug. 2019, pp. 1907–1913. doi: 10.24963/ijcai.2019/264.

[9] S. Guo, Y. Lin, N. Feng, C. Song, and H. Wan (2019). Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting, in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 1, Honolulu, HI, USA, Jan. 2019, pp. 922–929. doi: 10.1609/aaai.v33i01.3301922.

[10] W. Jiang and J. Luo (2022). Graph Neural Network for Traffic Forecasting: A Survey, Expert Systems with Applications, vol. 207, Art. no. 117921. doi: 10.1016/j.eswa.2022.117921.

[11] E. I. Vlahogianni, M. G. Karlaftis, and J. C. Golias (2014). Short-Term Traffic Forecasting: Where We Are and Where We're Going, Transportation Research Part C: Emerging Technologies, vol. 43, pp. 3–19. doi: 10.1016/j.trc.2014.01.005.

[12] B. M. Williams and L. A. Hoel (2003). Modeling and Forecasting Vehicular Traffic Flow as a Seasonal ARIMA Process: Theoretical Basis and Empirical Results, Journal of Transportation Engineering, vol. 129, no. 6, pp. 664–672. doi: 10.1061/(ASCE)0733-947X(2003)129:6(664).

[13] J. Jiang, C. Han, W. X. Zhao, and J. Wang (2023). PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction, in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 4, Washington, DC, USA, Feb. 2023, pp. 4365–4373. doi: 10.1609/aaai.v37i4.25556.

[14] Y. Li, R. Yu, C. Shahabi, and Y. Liu (2018). Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting, in Proceedings of the 6th International Conference on Learning Representations (ICLR 2018), Vancouver, Canada, Apr.–May 2018. URL: https://openreview.net/forum?id=SJiHXGWAZ

[15] L. Bai, L. Yao, C. Li, X. Wang, and C. Wang (2020). Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting, in Advances in Neural Information Processing Systems 33 (NeurIPS 2020), pp. 17804–17815. URL: https://proceedings.neurips.cc/paper/2020/hash/ce1aad92b939420fc17005e5461e6f48-Abstract.html

> ⚠️ Before submission, verify each DOI at https://doi.org/ and page numbers on Google Scholar.

---

## Annexure

### A. Additional Readings (Articles / Blogs)

| # | Title | URL |
|---|---|---|
| 1 | Understanding LSTM Networks — Christopher Olah | https://colah.github.io/posts/2015-08-Understanding-LSTMs/ |
| 2 | A Gentle Introduction to Graph Neural Networks — Distill | https://distill.pub/2021/gnn-intro/ |
| 3 | Understanding Convolutions on Graphs — Distill | https://distill.pub/2021/understanding-gnns/ |
| 4 | Traffic prediction with advanced Graph Neural Networks — Google DeepMind Blog | https://deepmind.google/discover/blog/traffic-prediction-with-advanced-graph-neural-networks/ |
| 5 | Introduction to Boosted Trees — XGBoost Documentation | https://xgboost.readthedocs.io/en/stable/tutorials/model.html |
| 6 | MegaCRN / EXPY-TKY dataset repository — GitHub | https://github.com/deepkashiwa20/MegaCRN |
| 7 | PyTorch Geometric Temporal Documentation | https://pytorch-geometric-temporal.readthedocs.io/ |
| 8 | Google Colaboratory FAQ | https://research.google.com/colaboratory/faq.html |
| 9 | FastAPI Tutorial — User Guide | https://fastapi.tiangolo.com/tutorial/ |
| 10 | Streamlit Documentation | https://docs.streamlit.io/ |

### B. Key Terms Definitions

- **Traffic Speed Forecasting** — Predicting the average speed of vehicles on a road segment at a future time using past observations. Short-term forecasting (minutes to an hour ahead) supports congestion warnings, route guidance and traffic control in Intelligent Transportation Systems.
- **Intelligent Transportation System (ITS)** — The application of sensing, communication, computing and control technologies to transportation networks to improve safety, efficiency and sustainability. Examples include adaptive signals, electronic toll collection and real-time traveller information.
- **Spatio-Temporal Data** — Data that varies across both space (location) and time. In traffic, each road link (space) has a speed series (time), and a link's speed depends on its own history and on neighbouring links.
- **Graph Neural Network (GNN) / Graph Convolution** — Neural networks that operate on graph-structured data by passing and aggregating information between connected nodes. Graph convolution generalises the CNN convolution to irregular structures such as road networks.
- **STGCN (Spatio-Temporal Graph Convolutional Network)** — A deep-learning model proposed by Yu et al. (2018) that stacks gated temporal convolutions with spectral graph convolutions to capture time patterns and road-network dependencies together, using only convolutions (no recurrence) for fast training.
- **LSTM (Long Short-Term Memory)** — A recurrent neural network with input, forget and output gates and a memory cell, designed to learn long-range dependencies in sequences while avoiding the vanishing-gradient problem. Widely used for time-series prediction.
- **XGBoost** — An optimised, regularised implementation of gradient-boosted decision trees. It builds trees sequentially, each correcting the errors of the previous ones, and is known for strong accuracy on tabular data.
- **Linear Regression** — A statistical model that predicts a target as a weighted sum of input variables plus a bias, fitted by minimising squared error. Used here as the simplest baseline.
- **Adjacency Matrix** — An N × N matrix describing which nodes of a graph are connected (and optionally by what weight/distance). In this project it encodes which expressway links are physically connected.
- **Sliding Window** — A technique that converts a time series into supervised samples by taking a fixed-length window of past values as input and a future value as the target, then shifting the window forward one step at a time.
- **Prediction Horizon** — How far into the future the model predicts, e.g., H = 3 steps = 30 minutes ahead with 10-minute data.
- **MAE / RMSE / MAPE** — Error metrics: Mean Absolute Error (average absolute difference), Root Mean Squared Error (penalises large errors more), and Mean Absolute Percentage Error (error relative to actual value, in %).
- **EXPY-TKY** — A public dataset of 10-minute average speeds on the Tokyo Metropolitan Expressway (Oct–Dec 2021), released with the MegaCRN paper (AAAI 2023).

### C. Code Snippets

**C.1 Dataset download** (`src/data/download.py`)
```python
def download(config_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(config_path))["data"]
    os.makedirs(cfg["raw_dir"], exist_ok=True)
    for name in FILES:
        dest = os.path.join(cfg["raw_dir"], name)
        if os.path.exists(dest):
            continue
        url = f"{cfg['base_url']}/{name}"
        tmp = dest + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.rename(tmp, dest)
```

**C.2 Time features with Japanese holidays**
```python
import jpholiday

def add_time_features(index):
    tod = (index.hour * 60 + index.minute) / 1440.0          # 0..1 time of day
    dow = index.dayofweek / 6.0                               # 0..1 day of week
    hol = np.array([jpholiday.is_holiday(d.date()) or d.dayofweek >= 5
                    for d in index], dtype=float)
    return np.stack([tod, dow, hol], axis=-1)                 # T x 3
```

**C.3 Sliding window**
```python
def make_windows(data, window, horizon):
    """data: T x N x F  ->  X: S x W x N x F,  Y: S x N (speed at t+H)"""
    X, Y = [], []
    for t in range(window, len(data) - horizon + 1):
        X.append(data[t - window:t])
        Y.append(data[t + horizon - 1, :, 0])
    return np.array(X), np.array(Y)
```

**C.4 Metrics**
```python
def metrics(y_true, y_pred, eps=1e-5):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = y_true > eps
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
```

**C.5 XGBoost baseline**
```python
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                   tree_method="hist", device="cuda")
xgb.fit(X_train.reshape(len(X_train), -1), y_train)
```

**C.6 LSTM model**
```python
class LSTMModel(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                  # x: B x W x F
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze(-1)
```

**C.7 Normalised adjacency + STGCN block**
```python
def normalize_adj(A):
    A = A + np.eye(A.shape[0])
    d = np.power(A.sum(1), -0.5)
    return d[:, None] * A * d[None, :]      # D^-1/2 (A+I) D^-1/2

class TemporalGLU(nn.Module):
    def __init__(self, c_in, c_out, kt=3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, 2 * c_out, (kt, 1))
    def forward(self, x):                   # B x C x T x N
        p, q = self.conv(x).chunk(2, dim=1)
        return p * torch.sigmoid(q)

class STConvBlock(nn.Module):
    def __init__(self, c_in, c_hid, c_out, A_hat):
        super().__init__()
        self.t1 = TemporalGLU(c_in, c_hid)
        self.theta = nn.Linear(c_hid, c_hid)
        self.t2 = TemporalGLU(c_hid, c_out)
        self.register_buffer("A", torch.tensor(A_hat, dtype=torch.float32))
        self.norm = nn.LayerNorm(c_out)
    def forward(self, x):
        x = self.t1(x)
        x = torch.einsum("nm,bctm->bctn", self.A, x)            # spatial mixing
        x = torch.relu(self.theta(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        x = self.t2(x)
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
```

**C.8 Training loop with early stopping**
```python
def train(model, train_dl, val_dl, epochs=100, patience=10, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, wait = float("inf"), 0
    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = F.l1_loss(model(xb), yb)
            loss.backward(); opt.step()
        val = evaluate(model, val_dl)
        if val < best:
            best, wait = val, 0
            torch.save(model.state_dict(), "best.pt")
        else:
            wait += 1
            if wait >= patience:
                break
```

**C.9 FastAPI prediction endpoint**
```python
app = FastAPI(title="Tokyo Traffic Speed API")

@app.get("/predict")
def predict(link_id: int, horizon: int = 3):
    x = latest_window(link_id)                 # last W readings from SQLite
    y = model(torch.tensor(x)[None]).item()
    y = y * sigma + mu                         # de-normalise to km/h
    return {"link_id": link_id, "horizon_min": horizon * 10, "speed_kmh": round(y, 1)}
```

### D. Sample Data Set

**D.1 Speed matrix (`expy-tky_2021xx.csv`) — rows = 10-min timestamps, columns = road links, values = km/h** *(illustrative values)*

| timestamp | link_0 | link_1 | link_2 | … | link_1842 |
|---|---|---|---|---|---|
| 2021-10-01 08:00 | 62.0 | 48.5 | 71.0 | … | 55.2 |
| 2021-10-01 08:10 | 58.3 | 41.0 | 70.4 | … | 53.9 |
| 2021-10-01 08:20 | 51.7 | 35.2 | 69.8 | … | 50.1 |

**D.2 Road link information (`expy-tky_link_info_add_end_latlon.csv`)** *(illustrative values)*

| link_id | road_name | start_lat | start_lon | end_lat | end_lon |
|---|---|---|---|---|---|
| 0 | Inner Circular Route (C1) | 35.6812 | 139.7671 | 35.6835 | 139.7702 |
| 1 | Shuto Route 3 Shibuya | 35.6580 | 139.7016 | 35.6552 | 139.6958 |
| 2 | Bayshore Route (B) | 35.6190 | 139.7790 | 35.6150 | 139.7880 |

**D.3 One model-ready sample after feature engineering (W = 6, H = 3, one link)**

| step | speed (norm.) | time_of_day | day_of_week | holiday | → target speed at t+30 min |
|---|---|---|---|---|---|
| t−5 | 0.42 | 0.333 | 0.667 | 0 | |
| t−4 | 0.35 | 0.340 | 0.667 | 0 | |
| t−3 | 0.21 | 0.347 | 0.667 | 0 | |
| t−2 | 0.05 | 0.354 | 0.667 | 0 | |
| t−1 | −0.12 | 0.361 | 0.667 | 0 | |
| t | −0.30 | 0.368 | 0.667 | 0 | **38.6 km/h** |
