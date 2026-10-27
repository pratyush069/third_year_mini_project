"""A pooled LSTM speed regressor for the EXPY-TKY road links.

Each training example represents one road link at one target time. Its lagged
speeds form a sequence; known calendar features and that link's training mean
are added after the LSTM so the same network can learn across all links.
"""
import copy

import numpy as np
import torch
from torch import nn


class _Network(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, dropout=0.2, context_size=5):
        super().__init__()
        self.sequence = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # Context = time-of-day (sin/cos), day-of-week, off-day, link mean speed.
        self.output = nn.Sequential(
            nn.Linear(hidden_size + context_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, past_speeds, context):
        sequence_output, _ = self.sequence(past_speeds)
        latest_state = sequence_output[:, -1, :]
        return self.output(torch.cat([latest_state, context], dim=1)).squeeze(1)


class LSTMRegressor:
    """Sklearn-like fit/predict wrapper used by ``src.train``."""

    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = None
        self.window = None
        self.speed_mean = None
        self.speed_std = None

    def _split_features(self, X):
        # The shared tabular builder stores latest lag first. Reverse it so the
        # LSTM reads the sequence from oldest to newest.
        past = X[:, :self.window][:, ::-1].copy()
        calendar = X[:, self.window:self.window + 4]
        link_mean = X[:, self.window + 4:self.window + 5]
        context = np.concatenate([calendar, link_mean], axis=1).copy()
        return past, context

    def _normalise(self, past, context):
        past = (past - self.speed_mean) / self.speed_std
        context[:, 4] = (context[:, 4] - self.speed_mean) / self.speed_std
        return past, context

    def fit(self, X, y, X_val, y_val):
        torch.manual_seed(int(self.config.get("seed", 42)))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.config.get("seed", 42)))

        self.window = X.shape[1] - 5
        past_train, context_train = self._split_features(X)
        # Fit scaling from training inputs only, then reuse it for validation/test.
        self.speed_mean = float(past_train.mean())
        self.speed_std = float(max(past_train.std(), 1e-6))
        past_train, context_train = self._normalise(past_train, context_train)
        past_val, context_val = self._split_features(X_val)
        past_val, context_val = self._normalise(past_val, context_val)
        y_mean = self.speed_mean
        y_std = self.speed_std
        target_train = (y.astype(np.float32) - y_mean) / y_std
        target_val = (y_val.astype(np.float32) - y_mean) / y_std

        # Keep training examples as CPU tensors. We shuffle indices with NumPy
        # instead of PyTorch's RandomSampler, which can crash in some macOS builds.
        train_sequence = torch.from_numpy(past_train[..., None].astype(np.float32))
        train_context = torch.from_numpy(context_train.astype(np.float32))
        train_target = torch.from_numpy(target_train)
        # Keep the full validation split in CPU memory; move only each batch to
        # the accelerator so large validation sets cannot exhaust MPS/GPU memory.
        val_sequence = torch.from_numpy(past_val[..., None].astype(np.float32))
        val_context = torch.from_numpy(context_val.astype(np.float32))
        val_target = torch.from_numpy(target_val)

        self.model = _Network(
            hidden_size=int(self.config.get("hidden_size", 64)),
            num_layers=int(self.config.get("num_layers", 2)),
            dropout=float(self.config.get("dropout", 0.2)),
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.config.get("learning_rate", 1e-3))
        )
        loss_fn = nn.MSELoss()
        batch_size = int(self.config.get("batch_size", 1024))
        rng = np.random.default_rng(int(self.config.get("seed", 42)))
        best_loss = float("inf")
        best_weights = None
        stale_epochs = 0
        patience = int(self.config.get("patience", 10))
        for epoch in range(int(self.config.get("epochs", 100))):
            self.model.train()
            total_loss = 0.0
            order = rng.permutation(len(train_target))
            for start in range(0, len(order), batch_size):
                batch_ids = torch.from_numpy(order[start:start + batch_size])
                seq_batch = train_sequence.index_select(0, batch_ids)
                ctx_batch = train_context.index_select(0, batch_ids)
                target_batch = train_target.index_select(0, batch_ids)
                seq_batch = seq_batch.to(self.device, non_blocking=True)
                ctx_batch = ctx_batch.to(self.device, non_blocking=True)
                target_batch = target_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(seq_batch, ctx_batch)
                loss = loss_fn(prediction, target_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(target_batch)

            self.model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for start in range(0, len(val_target), batch_size):
                    end = start + batch_size
                    seq_batch = val_sequence[start:end].to(self.device)
                    ctx_batch = val_context[start:end].to(self.device)
                    target_batch = val_target[start:end].to(self.device)
                    batch_loss = loss_fn(self.model(seq_batch, ctx_batch), target_batch)
                    val_loss_sum += batch_loss.item() * len(target_batch)
            val_loss = val_loss_sum / len(val_target)
            print(f"  LSTM epoch {epoch + 1:03d}: train MSE {total_loss / len(train_target):.5f}; "
                  f"validation MSE {val_loss:.5f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"  Early stopping after {epoch + 1} epochs")
                    break

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        return self

    def predict(self, X):
        past, context = self._split_features(X)
        past, context = self._normalise(past, context)
        sequence = torch.from_numpy(past[..., None].astype(np.float32))
        context = torch.from_numpy(context.astype(np.float32))
        self.model.eval()
        batch_size = int(self.config.get("inference_batch_size", 1024))
        outputs = []
        with torch.no_grad():
            for start in range(0, len(sequence), batch_size):
                end = start + batch_size
                seq_batch = sequence[start:end].to(self.device)
                ctx_batch = context[start:end].to(self.device)
                outputs.append(self.model(seq_batch, ctx_batch).cpu().numpy())
        prediction = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
        return (prediction * self.speed_std + self.speed_mean).astype(np.float32)
