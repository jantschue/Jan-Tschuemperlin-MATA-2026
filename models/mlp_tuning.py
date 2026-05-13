"""
Hyperparameter-Tuning für das MLP-Modell mittels Optuna.
Tuning läuft ausschliesslich auf 050_Brunnen_Mositunnel_R1.
Die besten Hyperparameter werden in results/model_results/mlp/best_params.json gespeichert.
"""

import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Konfiguration ────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/v5_engineered")
RESULTS_DIR = Path("results/model_results/mlp")

TUNING_DATASET = "050_Brunnen_Mositunnel_R1_engineered.csv"
N_TRIALS       = 50
TUNING_EPOCHS  = 150  # Reduziert gegenüber Final-Training für schnellere Trials
ES_PATIENCE    = 25

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10

FEATURES = [
    "Year", "Hour_sin", "Hour_cos", "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos",
    "is_weekend", "is_holiday", "temp", "rain_1h", "sun_1h", "snow_1h", "weather_cat",
]


class VerkehrsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def objective(trial, X_train_s, y_train_s, X_val_s, y_val_s, input_dim):
    n_layers     = trial.suggest_int("n_layers", 2, 5)
    n_units      = trial.suggest_categorical("n_units", [32, 64, 128, 256, 512])
    hidden_dims  = [n_units] * n_layers
    dropout      = trial.suggest_float("dropout", 0.1, 0.5)
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [128, 256, 512])

    train_loader = DataLoader(
        VerkehrsDataset(X_train_s, y_train_s), batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        VerkehrsDataset(X_val_s, y_val_s), batch_size=batch_size, shuffle=False
    )

    model     = MLP(input_dim, hidden_dims, dropout).to(device)
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(TUNING_EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss   = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            epoch_val_loss = 0.0
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                epoch_val_loss += loss_fn(model(X_batch), y_batch).item() * len(X_batch)
            epoch_val_loss /= len(val_loader.dataset)

        # Pruning: schlechte Trials früh abbrechen
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                break

    return best_val_loss


def main():
    print(f"Optuna Hyperparameter-Tuning – Device: {device}")
    print(f"Tuning-Datensatz: {TUNING_DATASET}")
    print(f"Anzahl Trials: {N_TRIALS}\n")

    file_path = DATA_DIR / TUNING_DATASET
    if not file_path.exists():
        raise FileNotFoundError(f"Tuning-Datensatz nicht gefunden: {file_path}")

    df = pd.read_csv(file_path).dropna()

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

    if "weather_cat" in df.columns and df["weather_cat"].dtype == "object":
        df["weather_cat"] = LabelEncoder().fit_transform(df["weather_cat"])

    feature_cols = [c for c in FEATURES if c in df.columns]

    X = df[feature_cols].values
    y = df["volume"].values.reshape(-1, 1)

    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, X_val = X[:train_end], X[train_end:val_end]
    y_train, y_val = y[:train_end], y[train_end:val_end]

    x_scaler  = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s   = x_scaler.transform(X_val)

    y_scaler  = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s   = y_scaler.transform(y_val)

    input_dim = X_train_s.shape[1]

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )

    def wrapped_objective(trial):
        return objective(trial, X_train_s, y_train_s, X_val_s, y_val_s, input_dim)

    print("Starte Optimierung...")
    start_time = time.time()
    study.optimize(wrapped_objective, n_trials=N_TRIALS, show_progress_bar=True)
    duration = time.time() - start_time

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\nTuning abgeschlossen in {duration:.1f}s")
    print(f"Abgeschlossene Trials: {len(completed)} | Geprunte Trials: {len(pruned)}")

    best = study.best_trial
    print(f"\nBester Trial: #{best.number}")
    print(f"Bester Validation Loss: {best.value:.6f}")

    hidden_dims   = [best.params["n_units"]] * best.params["n_layers"]
    dropout       = best.params["dropout"]
    learning_rate = best.params["lr"]
    weight_decay  = best.params["weight_decay"]
    batch_size    = best.params["batch_size"]

    sep = "─" * 48
    print(f"\n{sep}")
    print("Beste Hyperparameter – manuell in mlp.py eintragen:\n")
    print(f"  HIDDEN_DIMS   = {hidden_dims}")
    print(f"  DROPOUT       = {dropout:.2f}")
    print(f"  LEARNING_RATE = {learning_rate:.4f}")
    print(f"  WEIGHT_DECAY  = {weight_decay:.4f}")
    print(f"  BATCH_SIZE    = {batch_size}")
    print(sep)


if __name__ == "__main__":
    main()
