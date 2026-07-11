"""
Hyperparameter-Tuning für das MLP-Modell mittels Optuna.
Tuning läuft ausschliesslich auf 050_Brunnen_Mositunnel_R1.
Die besten Hyperparameter werden in results/model_results/mlp/best_params.json gespeichert.

GPU-Preload: Der gesamte (kleine) Datensatz wird einmalig auf die GPU geladen,
statt jeden Batch pro Epoche einzeln von der CPU zu transferieren. Das spart den
DataLoader- und Transfer-Overhead und erhöht die GPU-Auslastung. Die Batch-Grenzen
bleiben identisch (kein Shuffle, gleiche batch_size-Slices), die Resultate damit gleich.
"""

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Konfiguration ────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/v5")
RESULTS_DIR = Path("results/model_results/mlp_v5")

TUNING_DATASET = "050_Brunnen_Mositunnel_R1_v5.csv"
N_TRIALS       = 150
TUNING_EPOCHS  = 150  # Reduziert gegenüber Final-Training für schnellere Trials
ES_PATIENCE    = 25

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10

FEATURES = [
    "Year", "Hour_sin", "Hour_cos", "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos",
    "is_weekend", "is_holiday", "temp", "rain_1h", "sun_1h", "snow_1h", "weather_cat",
]


def iterate_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    """Liefert aufeinanderfolgende Batches direkt aus bereits auf der GPU liegenden
    Tensoren. Kein Shuffle, identische Batch-Grenzen wie der bisherige DataLoader
    (shuffle=False, drop_last=False) – damit bleiben die Resultate gleich."""
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]


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


def objective(trial, X_train_g, y_train_g, X_val_g, y_val_g, input_dim):
    n_layers     = trial.suggest_int("n_layers", 2, 4)
    hidden_dims  = [trial.suggest_categorical(f"n_units_l{i}", [32, 64, 128, 256, 512]) for i in range(n_layers)]
    dropout      = trial.suggest_float("dropout", 0.0, 0.3)
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [128, 256, 512])

    # Daten liegen bereits komplett auf der GPU (X_*_g/y_*_g) – die Batches werden
    # direkt per Slicing gebildet, kein DataLoader und kein .to(device) pro Batch.
    n_val = X_val_g.shape[0]

    model     = MLP(input_dim, hidden_dims, dropout).to(device)
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(TUNING_EPOCHS):
        model.train()
        for X_batch, y_batch in iterate_batches(X_train_g, y_train_g, batch_size):
            y_pred = model(X_batch)
            loss   = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            epoch_val_loss = 0.0
            for X_batch, y_batch in iterate_batches(X_val_g, y_val_g, batch_size):
                epoch_val_loss += loss_fn(model(X_batch), y_batch).item() * len(X_batch)
            epoch_val_loss /= n_val

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

    # GPU-Preload: gesamten Datensatz einmalig auf die GPU laden (statt pro Batch)
    X_train_g = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y_train_g = torch.tensor(y_train_s, dtype=torch.float32, device=device)
    X_val_g   = torch.tensor(X_val_s,   dtype=torch.float32, device=device)
    y_val_g   = torch.tensor(y_val_s,   dtype=torch.float32, device=device)

    input_dim = X_train_g.shape[1]

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Optuna-Studie samt SQLite-Datenbank im Modell-Ordner ablegen (neben best_params.json),
    # damit der Projekt-Root sauber bleibt. Ordner vor der Study-Erstellung sicherstellen.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{(RESULTS_DIR / 'optuna_mlp_v5.db').as_posix()}"
    study = optuna.create_study(
        study_name="mlp_v5_tuning",
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=None),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )

    def wrapped_objective(trial):
        return objective(trial, X_train_g, y_train_g, X_val_g, y_val_g, input_dim)

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

    hidden_dims   = [best.params[f"n_units_l{i}"] for i in range(best.params["n_layers"])]
    dropout       = best.params["dropout"]
    learning_rate = best.params["lr"]
    weight_decay  = best.params["weight_decay"]
    batch_size    = best.params["batch_size"]

    sep = "-" * 48
    print(f"\n{sep}")
    print("Beste Hyperparameter – manuell in mlp.py eintragen:\n")
    print(f"  HIDDEN_DIMS   = {hidden_dims}")
    print(f"  DROPOUT       = {dropout:.2f}")
    print(f"  LEARNING_RATE = {learning_rate:.4f}")
    print(f"  WEIGHT_DECAY  = {weight_decay:.4f}")
    print(f"  BATCH_SIZE    = {batch_size}")
    print(sep)

    # Beste Hyperparameter als JSON speichern
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best_params = {
        "HIDDEN_DIMS":   hidden_dims,
        "DROPOUT":       dropout,
        "LEARNING_RATE": learning_rate,
        "WEIGHT_DECAY":  weight_decay,
        "BATCH_SIZE":    batch_size,
        "best_val_loss": best.value,
        "trial_number":  best.number,
        "n_trials_completed": len(completed),
        "n_trials_pruned":    len(pruned),
        "raw_params":    best.params,
    }
    best_params_path = RESULTS_DIR / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"\nGespeichert: {best_params_path}")


if __name__ == "__main__":
    main()
