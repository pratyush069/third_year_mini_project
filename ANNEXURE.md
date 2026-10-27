# Annexure

## A. Additional Readings (Articles / Blogs)

| S.No | Title | Source | URL |
|---|---|---|---|
| 1 | Understanding LSTM Networks | Christopher Olah (colah's blog) | https://colah.github.io/posts/2015-08-Understanding-LSTMs/ |
| 2 | A Gentle Introduction to Graph Neural Networks | Distill | https://distill.pub/2021/gnn-intro/ |
| 3 | Understanding Convolutions on Graphs | Distill | https://distill.pub/2021/understanding-gnns/ |
| 4 | Traffic Prediction with Advanced Graph Neural Networks | Google DeepMind Blog | https://deepmind.google/discover/blog/traffic-prediction-with-advanced-graph-neural-networks/ |
| 5 | Introduction to Boosted Trees | XGBoost Documentation | https://xgboost.readthedocs.io/en/stable/tutorials/model.html |
| 6 | Linear Models (Ordinary Least Squares) | scikit-learn User Guide | https://scikit-learn.org/stable/modules/linear_model.html |
| 7 | torch.nn.LSTM | PyTorch Documentation | https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html |
| 8 | STGCN: Official Implementation (IJCAI-18) | GitHub – VeritasYin | https://github.com/VeritasYin/STGCN_IJCAI-18 |
| 9 | MegaCRN and the EXPY-TKY Dataset | GitHub – deepkashiwa20 | https://github.com/deepkashiwa20/MegaCRN |
| 10 | Intelligent Transportation Systems Joint Program Office | U.S. Department of Transportation | https://www.its.dot.gov/ |

---

## B. Key Terms Definitions

**1. Traffic Speed Prediction**
Traffic speed prediction estimates the average speed of vehicles on a road segment at a future time, using past and current traffic observations. Short-term prediction looks a few minutes to an hour ahead. It supports travel-time estimation, congestion warnings and route guidance, and it is a core task in modern traffic management.

**2. Intelligent Transportation Systems (ITS)**
An Intelligent Transportation System applies sensing, communication, computing and control technologies to road networks to make travel safer, more efficient and more sustainable. Examples include adaptive traffic signals, electronic toll collection, variable message signs and real-time traveller information services. Accurate traffic forecasts are a key input to ITS.

**3. Machine Learning (ML)**
Machine learning is a branch of artificial intelligence in which computer systems learn patterns from data and improve their performance without being explicitly programmed for every rule. In supervised learning, which this project uses, a model is trained on input–output pairs (past speeds → future speed) and then predicts outputs for new inputs.

**4. Deep Learning (DL)**
Deep learning is a subset of machine learning that uses artificial neural networks with many layers to learn complex, non-linear representations directly from raw data. It powers modern speech recognition, computer vision and time-series forecasting. In this project, LSTM and STGCN are the deep learning models.

**5. Spatio-Temporal Graph Convolutional Network (STGCN)**
STGCN is a deep learning model proposed by Yu et al. (2018) for traffic forecasting. It treats the road network as a graph and stacks gated temporal convolutions (to learn patterns over time) with graph convolutions (to learn how neighbouring roads influence each other). Because it uses only convolutions rather than recurrence, it trains faster than RNN-based models.

**6. Long Short-Term Memory (LSTM)**
LSTM is a type of recurrent neural network introduced by Hochreiter and Schmidhuber (1997). It has a memory cell controlled by input, forget and output gates, which let it remember useful information over long sequences and avoid the vanishing-gradient problem. It is widely used for time-series forecasting, such as predicting speed from past readings.

**7. XGBoost (Extreme Gradient Boosting)**
XGBoost is an optimised, regularised implementation of gradient-boosted decision trees developed by Chen and Guestrin (2016). It builds trees one after another, with each new tree correcting the errors of the previous ones. It is fast, handles non-linear relationships well and often gives strong accuracy on tabular data.

**8. Linear Regression**
Linear regression is a statistical method that models the relationship between a target variable and one or more input variables as a straight-line (weighted sum) equation. The weights are found by minimising the sum of squared errors. It is simple and easy to interpret, so it serves as the baseline model in this study.

**9. Time-Series Forecasting**
Time-series forecasting predicts future values of a variable from its values recorded at successive, equally spaced points in time. Methods range from statistical models (ARIMA) to machine learning and deep learning. Traffic speed recorded every 10 minutes is a typical time series with daily and weekly patterns.

**10. Graph Neural Networks (GNN)**
Graph neural networks are deep learning models designed for data organised as a graph of nodes connected by edges. They learn by passing and aggregating information between connected nodes. In traffic forecasting, road links or sensors are the nodes and road connectivity forms the edges, which lets the model capture spatial dependencies.

**11. Urban Traffic Management**
Urban traffic management is the planning, monitoring and control of traffic movement in cities to reduce congestion, improve safety and lower emissions. It includes signal timing, incident management, route guidance and demand management. Data-driven forecasts allow authorities to act before congestion builds up instead of reacting to it.

**12. Multi-Horizon Forecasting**
Multi-horizon forecasting means predicting a variable at several future time points, for example 10, 30 and 60 minutes ahead. Accuracy usually falls as the horizon grows, because the future becomes more uncertain. Comparing models across horizons shows which model stays reliable for longer-term predictions.

**13. Expressway Traffic Data**
Expressway traffic data are measurements collected on high-speed, access-controlled roads. Sources include loop detectors, toll gantries (ETC) and GPS probe vehicles, and typical measures are speed, flow and travel time. The EXPY-TKY dataset used here contains 10-minute GPS-probe speeds for 1,843 links of the Tokyo Shuto Expressway (Oct–Dec 2021).

---

## C. Code Snippets

**C.1 Loading the raw data and building the speed matrix (Kaggle)**
```python
import pandas as pd, numpy as np
DATA_DIR = '/kaggle/input/datasets/omkargupta459/expy-tky-traffic/'

frames = []
for m in ['202110', '202111', '202112']:
    df = pd.read_csv(DATA_DIR + f'expy-tky_{m}.csv')          # columns: timestamp, linkid, speed
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    frames.append(df.pivot(index='timestamp', columns='linkid', values='speed'))
full_speed = pd.concat(frames).sort_index()                     # (13248, 2841)

link_info = pd.read_csv(DATA_DIR + 'expy-tky_link_info.csv')
full_speed = full_speed[link_info['link_id'].tolist()]          # match adjacency order
```

**C.2 Selecting the 1,843-link subset and its adjacency matrices**
```python
subset_pos = pd.read_csv(DATA_DIR + 'tokyo_link_idx.csv', header=None)[0].tolist()
speed_1843 = full_speed.iloc[:, subset_pos]                     # (13248, 1843)

adj_dist = np.load(DATA_DIR + 'expy-tky_adjdis.npy')
adj_01   = np.load(DATA_DIR + 'expy-tky_adj01.npy')
adj_dist_1843 = adj_dist[np.ix_(subset_pos, subset_pos)]        # (1843, 1843)
adj_01_1843   = adj_01[np.ix_(subset_pos, subset_pos)]
print('Missing values:', speed_1843.isna().sum().sum())         # 0
```

**C.3 Chronological train / validation / test split**
```python
train_val = speed_1843.loc[:'2021-11-30 23:50']                 # Oct–Nov
test      = speed_1843.loc['2021-12-01':]                       # Dec (4,464 steps)
n_val     = int(len(train_val) * 0.25)
train, val = train_val.iloc[:-n_val], train_val.iloc[-n_val:]   # 6,588 / 2,196 steps

mu, sigma = train.values.mean(), train.values.std()             # fit on train only
normalize = lambda x: (x - mu) / sigma
```

**C.4 Lag features (sliding window) for LR and XGBoost**
```python
def make_lag_features(series_df, window, horizon):
    """Each row: speeds at t-window+1..t -> target speed at t+horizon (stacked over links)."""
    X, y = [], []
    vals = series_df.values                                     # T x N
    for t in range(window - 1, len(vals) - horizon):
        X.append(vals[t - window + 1:t + 1].T)                  # N x window
        y.append(vals[t + horizon])                             # N
    return np.concatenate(X), np.concatenate(y)

# windows 6/12/24 steps (1/2/4 h), horizons 1/3/6 steps (10/30/60 min)
```

**C.5 Evaluation metrics (MAE, RMSE, MAPE, R²)**
```python
from sklearn.metrics import r2_score

def evaluate(y_true, y_pred):
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2_score(y_true, y_pred)}
```

**C.6 Historical-average and Linear Regression baselines**
```python
from sklearn.linear_model import LinearRegression

# Historical average: mean speed of each link at the same time-of-day in training data
ha_table = train.groupby(train.index.time).mean()

X_tr, y_tr = make_lag_features(train, window=6, horizon=3)
X_te, y_te = make_lag_features(test,  window=6, horizon=3)
lr = LinearRegression().fit(X_tr, y_tr)
print(evaluate(y_te, lr.predict(X_te)))
```

**C.7 XGBoost model**
```python
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                   tree_method='hist', device='cuda')
xgb.fit(X_tr, y_tr)
print(evaluate(y_te, xgb.predict(X_te)))
```

**C.8 LSTM model (PyTorch)**
```python
import torch, torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, n_features=1, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                    # x: batch x window x features
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze(-1)
```

**C.9 Normalised adjacency for STGCN**
```python
def normalize_adj(A):
    A_hat = A + np.eye(A.shape[0])                       # add self-loops
    d_inv_sqrt = np.power(A_hat.sum(axis=1), -0.5)
    return d_inv_sqrt[:, None] * A_hat * d_inv_sqrt[None, :]   # D^-1/2 (A+I) D^-1/2

A_norm = normalize_adj(adj_01_1843)
```

**C.10 Training loop with early stopping (LSTM / STGCN)**
```python
def train_model(model, train_dl, val_dl, epochs=100, patience=10, lr=1e-3):
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=lr), nn.L1Loss()
    best, wait = float('inf'), 0
    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_fn(model(xb), yb).item() for xb, yb in val_dl) / len(val_dl)
        if val_loss < best:
            best, wait = val_loss, 0; torch.save(model.state_dict(), 'best_model.pt')
        else:
            wait += 1
            if wait >= patience: break
```

---

## D. Sample Data Set

**D.1 Raw speed file (`expy-tky_202110.csv`): long format, one row per link per 10 minutes**

| timestamp | linkid | speed |
|---|---|---|
| 2021-10-01 00:00:00 | 5000001 | 87.0 |
| 2021-10-01 00:10:00 | 5000001 | 85.0 |
| 2021-10-01 00:20:00 | 5000001 | 88.0 |

*(The October file alone has 12,682,224 rows = 4,464 time steps × 2,841 links.)*

**D.2 Speed matrix after pivoting (`speed_1843`): rows = time steps, columns = road links, values = km/h**

| timestamp | link 1 | link 2 | link 3 | … | link 1843 |
|---|---|---|---|---|---|
| 2021-10-01 08:00 | 82.0 | 64.0 | 91.0 | … | 76.0 |
| 2021-10-01 08:10 | 79.0 | 41.0 | 90.0 | … | 74.0 |
| 2021-10-01 08:20 | 81.0 | 28.0 | 92.0 | … | 75.0 |

*(Shape 13,248 × 1,843; 0 missing values.)*

**D.3 Link metadata (`expy-tky_link_info.csv`)**

| link_id | roadname | start_lon | start_lat |
|---|---|---|---|
| 5000001 | Shuto Expressway Inner Circular Route (C1) | 139.7671 | 35.6812 |
| 5000002 | Shuto Expressway Route 3 Shibuya Line | 139.7016 | 35.6580 |
| 5000003 | Shuto Expressway Bayshore Route (B) | 139.7790 | 35.6190 |

> Note: the values in D.1–D.3 are representative. Before submission, replace them with the actual first rows printed by `df.head()` and `link_info.head()` in your Kaggle notebook.
