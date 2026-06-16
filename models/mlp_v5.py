"""
Dieses Skript trainiert ein MLP (Multi-Layer Perceptron) für die Verkehrsvorhersage.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Seed setzen
torch.manual_seed(42)
np.random.seed(42)

# Device konfigurieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameter ──────────────────────────────────────────────────────────
HIDDEN_DIMS   = [256, 128, 128, 64]        # Grösse der Hidden Layers
DROPOUT       = 0.2582               # Dropout-Rate (nicht nach letzter Hidden Layer)
BATCH_SIZE    = 128
MAX_EPOCHS    = 500
LEARNING_RATE = 0.0003352
WEIGHT_DECAY  = 0.0016735
LR_PATIENCE   = 15    # ReduceLROnPlateau: Epochen ohne Verbesserung bis LR sinkt
LR_FACTOR     = 0.5   # ReduceLROnPlateau: Faktor um den LR reduziert wird
ES_PATIENCE   = 10    # Early Stopping: Epochen ohne Verbesserung bis Abbruch
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.10
# TEST_RATIO  = 0.20 (implizit: 1 - TRAIN_RATIO - VAL_RATIO)

# Pfade konfigurieren
DATA_DIR    = Path("data/v5_engineered")
RESULTS_DIR = Path("results/model_results/mlp")

DATASETS = [
    "050_Brunnen_Mositunnel_R1_engineered.csv", "050_Brunnen_Mositunnel_R2_engineered.csv",
    "171_Sattel_R1_engineered.csv",              "171_Sattel_R2_engineered.csv",
    "216_Wangen_SZ_R1_engineered.csv",           "216_Wangen_SZ_R2_engineered.csv",
    "299_Wollerau_Blatttunnel_R1_engineered.csv","299_Wollerau_Blatttunnel_R2_engineered.csv",
    "720_Schwyz_R1_engineered.csv",              "720_Schwyz_R2_engineered.csv",
]

FEATURES = [
    "Year", "Hour_sin", "Hour_cos", "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos",
    "is_weekend", "is_holiday", "temp", "rain_1h", "sun_1h", "snow_1h", "weather_cat",
]

# Unterordner automatisch erstellen
for sub in ["metrics", "model_weights", "plots/loss_curves", "plots/scatter",
            "plots/timeseries", "plots/residuals", "plots/tagesverlauf", "summary", "predictions"]:
    (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)


# Dataset-Klasse für den DataLoader (wandelt numpy arrays in torch Tensoren um)
class VerkehrsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# MLP-Modell durch Subclassing von nn.Module
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))  # Batch Normalization vor ReLU
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:            # kein Dropout nach letzter Hidden Layer
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))         # Output: 1 Neuron, keine Aktivierungsfunktion (Regression)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main():
    all_metrics = []

    for i, filename in enumerate(DATASETS):
        name      = filename.replace(".csv", "")
        file_path = DATA_DIR / filename

        if not file_path.exists():
            print(f"Warnung: Datei {file_path} nicht gefunden, wird übersprungen.")
            continue

        # Daten laden, NaN droppen
        df = pd.read_csv(file_path).dropna()

        # datetime als Index setzen (nur als Index, nicht als Feature)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)

        # Label Encoding für weather_cat falls object
        if "weather_cat" in df.columns and df["weather_cat"].dtype == "object":
            df["weather_cat"] = LabelEncoder().fit_transform(df["weather_cat"])

        # Nur vorhandene Features nutzen
        feature_cols = [c for c in FEATURES if c in df.columns]

        X = df[feature_cols].values
        y = df["volume"].values.reshape(-1, 1)

        # Chronologischer 70/10/20-Split, kein Shuffle
        n         = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
        df_test = df.iloc[val_end:].copy()  # Für Plotting

        # Split-Info als JSON speichern (Reproduzierbarkeit/Dokumentation)
        train_idx = df.index[:train_end]
        val_idx   = df.index[train_end:val_end]
        test_idx  = df.index[val_end:]
        split_info = {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "train_pct": round(len(X_train) / len(df) * 100, 2),
            "val_samples": len(X_val),
            "val_pct": round(len(X_val) / len(df) * 100, 2),
            "test_samples": len(X_test),
            "test_pct": round(len(X_test) / len(df) * 100, 2),
            "split_type": "chronological 70/15/15",
            "date_range_train": {"start": str(train_idx[0]), "end": str(train_idx[-1])},
            "date_range_val":   {"start": str(val_idx[0]),   "end": str(val_idx[-1])},
            "date_range_test":  {"start": str(test_idx[0]),  "end": str(test_idx[-1])},
        }
        with open(RESULTS_DIR / "summary" / f"split_info_{name}.json", "w") as f:
            json.dump(split_info, f, indent=4)

        # StandardScaler NUR auf Trainingsdaten fitten
        x_scaler  = StandardScaler()
        X_train_s = x_scaler.fit_transform(X_train)
        X_val_s   = x_scaler.transform(X_val)
        X_test_s  = x_scaler.transform(X_test)

        # y ebenfalls skalieren
        y_scaler  = StandardScaler()
        y_train_s = y_scaler.fit_transform(y_train)
        y_val_s   = y_scaler.transform(y_val)

        # DataLoader erstellen (kein shuffle – Daten sind bereits chronologisch gesplittet)
        train_loader = DataLoader(
            VerkehrsDataset(X_train_s, y_train_s), batch_size=BATCH_SIZE, shuffle=False
        )
        val_loader = DataLoader(
            VerkehrsDataset(X_val_s, y_val_s), batch_size=BATCH_SIZE, shuffle=False
        )

        # Pro Datensatz neue Modellinstanz erstellen
        model     = MLP(len(feature_cols), HIDDEN_DIMS, DROPOUT).to(device)
        loss_fn   = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
        )

        # Early Stopping Zustand
        best_val_loss = float("inf")
        best_weights  = None
        no_improve    = 0

        # Track different values
        epoch_count  = []
        train_losses = []
        val_losses   = []

        print(f"\nTrainiere Modell {i + 1}/10: {name}")
        print(f"LR: {LEARNING_RATE}, Max. Epochen: {MAX_EPOCHS}, Batch Size: {BATCH_SIZE}")

        start_time = time.time()

        # 0. Loop through the data
        for epoch in range(MAX_EPOCHS):

            ### Training
            # Set the model to training mode
            model.train()

            epoch_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # 1. Forward pass
                y_pred = model(X_batch)

                # 2. Calculate the loss
                loss = loss_fn(y_pred, y_batch)

                # 3. Optimizer zero grad
                optimizer.zero_grad()

                # 4. Perform backpropagation on the loss
                loss.backward()

                # 5. Step the optimizer (gradient descent)
                optimizer.step()

                epoch_train_loss += loss.item() * len(X_batch)

            epoch_train_loss /= len(train_loader.dataset)

            ### Validation
            model.eval()  # turns off dropout and batchnorm training behaviour
            with torch.inference_mode():
                epoch_val_loss = 0.0
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    epoch_val_loss += loss_fn(model(X_batch), y_batch).item() * len(X_batch)
                epoch_val_loss /= len(val_loader.dataset)

            # LR-Scheduler auf Validation Loss anwenden
            scheduler.step(epoch_val_loss)

            # Loss-Werte jede Epoche speichern (für hochaufgelöste Loss-Kurven)
            epoch_count.append(epoch)
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

            # Konsolenausgabe nur alle 10 Epochen, um die Logs übersichtlich zu halten
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

            # Early Stopping: bestes Modell anhand Validation Loss merken
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= ES_PATIENCE:
                    print(f"  Early Stop bei Epoche {epoch} (bester Val-Loss: {best_val_loss:.6f})")
                    break

        train_duration = time.time() - start_time
        print(f"Trainingszeit: {train_duration:.1f}s")

        # Bestes Modell laden für Evaluation (nicht letztes!)
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

        # Evaluation auf Testset (torch.inference_mode)
        model.eval()
        with torch.inference_mode():
            preds_scaled = model(
                torch.tensor(X_test_s, dtype=torch.float32).to(device)
            ).cpu().numpy()

        # Zurück in Fahrzeuge/h
        preds = y_scaler.inverse_transform(preds_scaled)

        # Metriken (kein MSE in den Resultaten – nur MAE, RMSE, R²)
        # y_test bleibt original (nicht skaliert) für Metriken und Plots
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

        # In all_metrics anhängen
        metric_dict = {"station": name, "MAE": mae, "RMSE": rmse, "R2": r2}
        all_metrics.append(metric_dict)

        # Metriken sofort in CSV speichern (beim 1. Modell überschreiben, dann anhängen)
        metrics_file = RESULTS_DIR / "metrics" / "all_metrics.csv"
        pd.DataFrame([metric_dict]).to_csv(
            metrics_file, mode="w" if i == 0 else "a", header=(i == 0), index=False
        )

        # Modell speichern (bestes Modell, nicht letztes)
        torch.save(model.state_dict(), RESULTS_DIR / "model_weights" / f"mlp_{name}.pt")

        # Vorhersage-Vergleich speichern (Ist- vs. Prognosewerte)
        pd.DataFrame({
            "datetime":         df_test.index,
            "actual_volume":    y_test.flatten(),
            "predicted_volume": preds.flatten(),
        }).to_csv(RESULTS_DIR / "predictions" / f"predictions_{name}.csv", index=False)

        # --- PLOTS PRO MODELL ---

        # Plot A – Loss-Kurve (Training and validation loss curves)
        plt.figure(figsize=(10, 7))
        plt.plot(epoch_count, train_losses, label="Train Loss")
        plt.plot(epoch_count, val_losses,   label="Validation Loss")
        plt.title(f"Training and validation loss curves - {name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "plots" / "loss_curves" / f"loss_{name}.png")
        plt.close()

        # Plot B – Predicted vs. Actual Scatter
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, preds, alpha=0.3, s=5)
        max_val = max(y_test.max(), preds.max())
        plt.plot([0, max_val], [0, max_val], "r-")
        plt.title(f"Predicted vs. Actual - {name}")
        plt.xlabel("Tatsächliches Volumen (Ground Truth)")
        plt.ylabel("Vorhergesagtes Volumen (Predicted)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "plots" / "scatter" / f"scatter_{name}.png")
        plt.close()

        # Plot C – Zeitreihe (erste 2 zusammenhängende Wochen im Testset)
        # Manche Stationen (z.B. Sattel) haben grosse Lücken im Testset, daher
        # suchen wir den ersten Block ohne grosse Datenlücken (> 2 Stunden).
        plot_start, plot_end = 0, min(336, len(df_test))
        if isinstance(df_test.index, pd.DatetimeIndex) and len(df_test) > 1:
            time_diffs = df_test.index.to_series().diff()
            gap_positions = np.where(time_diffs > pd.Timedelta(hours=2))[0]
            # Block-Grenzen: zwischen aufeinanderfolgenden Lücken
            block_starts = np.concatenate(([0], gap_positions))
            block_ends   = np.concatenate((gap_positions, [len(df_test)]))
            # Ersten Block mit mind. 336 Stunden wählen, sonst längsten Block
            block_lengths = block_ends - block_starts
            valid_blocks  = np.where(block_lengths >= 336)[0]
            if len(valid_blocks) > 0:
                b = valid_blocks[0]
            else:
                b = int(np.argmax(block_lengths))
            plot_start = int(block_starts[b])
            plot_end   = min(plot_start + 336, int(block_ends[b]))

        plt.figure(figsize=(15, 5))
        time_x = df_test.index[plot_start:plot_end]
        plt.plot(time_x, y_test[plot_start:plot_end], color="black",  label="Ground Truth")
        plt.plot(time_x, preds[plot_start:plot_end],  color="orange", label="Vorhersage")
        if isinstance(df_test.index, pd.DatetimeIndex):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.title(f"Zeitreihe (erste 2 zusammenhängende Wochen Testset) - {name}")
        plt.xlabel("Datum/Zeit")
        plt.ylabel("Fahrzeuge/h")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "plots" / "timeseries" / f"timeseries_{name}.png")
        plt.close()

        # Residuals berechnen für Plot D und E
        residuals = preds.flatten() - y_test.flatten()
        df_test["residual"] = residuals
        df_test["pred"]     = preds.flatten()

        # Für Plot D und E brauchen wir die Stunde. Falls datetime der Index ist, daraus holen.
        if isinstance(df_test.index, pd.DatetimeIndex):
            df_test["Hour_for_plot"] = df_test.index.hour
        else:
            df_test["Hour_for_plot"] = 0  # Fallback

        # Plot D – Residuals nach Tagesstunde
        res_by_hour = df_test.groupby("Hour_for_plot")["residual"].mean()
        plt.figure(figsize=(10, 5))
        plt.bar(res_by_hour.index, res_by_hour.values)
        plt.axhline(0, color="black", linewidth=1)
        plt.title(f"Ø Residuals nach Tagesstunde - {name}")
        plt.xlabel("Stunde (0-23)")
        plt.ylabel("Ø Residual (Predicted - Actual)")
        plt.xticks(range(24))
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "plots" / "residuals" / f"residuals_{name}.png")
        plt.close()

        # Plot E – Durchschnittlicher Tagesverlauf
        avg_actual = df_test.groupby("Hour_for_plot")["volume"].mean()
        avg_pred   = df_test.groupby("Hour_for_plot")["pred"].mean()
        plt.figure(figsize=(10, 5))
        plt.plot(avg_actual.index, avg_actual.values, color="black", label="Ø Ground Truth")
        plt.plot(avg_pred.index,   avg_pred.values,   "r--",         label="Ø Vorhersage")
        plt.title(f"Durchschnittlicher Tagesverlauf - {name}")
        plt.xlabel("Stunde (0-23)")
        plt.ylabel("Fahrzeuge/h")
        plt.xticks(range(24))
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "plots" / "tagesverlauf" / f"tagesverlauf_{name}.png")
        plt.close()

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)

        # Summary Plot: R² Scores über alle Stationen
        plt.figure(figsize=(12, 6))
        colors = ["red" if r2 < 0 else "green" for r2 in metrics_df["R2"]]
        bars = plt.bar(metrics_df["station"], metrics_df["R2"], color=colors)
        plt.axhline(0, color="black", linewidth=1)
        plt.title("R² Scores über alle Stationen – MLP")
        plt.xlabel("Stationsname")
        plt.ylabel("R² Score")
        plt.xticks(rotation=45, ha="right")

        # R²-Werte als Label über/unter jedem Balken
        for bar in bars:
            yval = bar.get_height()
            # Positionierung abhängig davon, ob Balken nach oben oder unten geht
            if yval >= 0:
                plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02,
                         f"{yval:.2f}", ha="center", va="bottom", fontsize=9)
            else:
                plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.02,
                         f"{yval:.2f}", ha="center", va="top", fontsize=9)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "summary" / "r2_summary.png")
        plt.close()

        print("\nAlle Modelle trainiert und Ergebnisse gespeichert.")


if __name__ == "__main__":
    main()
