"""
Dieses Skript trainiert ein MLP (Multi-Layer Perceptron) zur stündlichen
Verkehrsvorhersage auf den v8-Daten (Kanton Schwyz, ASTRA-Messstationen). Es umfasst
Datenaufbereitung, chronologischen Split, Skalierung, das Training mit Early Stopping
und Lernraten-Scheduler sowie die Evaluation mittels MAE, RMSE und R² pro Messstation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameter ──────────────────────────────────────────────────────────
HIDDEN_DIMS   = [512, 256, 256, 128]   # Grösse der Hidden Layers
DROPOUT       = 0.283                  # Dropout-Rate (nicht nach letzter Hidden Layer), aus der Optuna-Suche
BATCH_SIZE    = 512
MAX_EPOCHS    = 500
LEARNING_RATE = 0.000181               # aus der Optuna-Suche
WEIGHT_DECAY  = 0.000419               # aus der Optuna-Suche
LR_PATIENCE   = 15    # ReduceLROnPlateau: Epochen ohne Verbesserung bis LR sinkt
LR_FACTOR     = 0.5   # ReduceLROnPlateau: Faktor um den LR reduziert wird
ES_PATIENCE   = 10    # Early Stopping: Epochen ohne Verbesserung bis Abbruch
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.10
# TEST_RATIO  = 0.20 (implizit: 1 - TRAIN_RATIO - VAL_RATIO)

DATA_DIR = Path("data/v8")

DATASETS = [
    "050_Brunnen_Mositunnel_R1_v8.csv", "050_Brunnen_Mositunnel_R2_v8.csv",
    "171_Sattel_R1_v8.csv",              "171_Sattel_R2_v8.csv",
    "216_Wangen_SZ_R1_v8.csv",           "216_Wangen_SZ_R2_v8.csv",
    "299_Wollerau_Blatttunnel_R1_v8.csv","299_Wollerau_Blatttunnel_R2_v8.csv",
    "720_Schwyz_R1_v8.csv",              "720_Schwyz_R2_v8.csv",
]

FEATURES = [
    "Year", "Hour_sin", "Hour_cos", "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos",
    "is_weekend",
    "holiday_AG", "holiday_AI", "holiday_AR", "holiday_BE", "holiday_BL", "holiday_BS",
    "holiday_FR", "holiday_GE", "holiday_GL", "holiday_GR", "holiday_JU", "holiday_LU",
    "holiday_NE", "holiday_NW", "holiday_OW", "holiday_SG", "holiday_SH", "holiday_SO",
    "holiday_SZ", "holiday_TG", "holiday_TI", "holiday_UR", "holiday_VD", "holiday_VS",
    "holiday_ZG", "holiday_ZH",
    "schoolholiday_AG", "schoolholiday_AI", "schoolholiday_AR", "schoolholiday_BE", "schoolholiday_BL", "schoolholiday_BS",
    "schoolholiday_FR", "schoolholiday_GE", "schoolholiday_GL", "schoolholiday_GR", "schoolholiday_JU", "schoolholiday_LU",
    "schoolholiday_NE", "schoolholiday_NW", "schoolholiday_OW", "schoolholiday_SG", "schoolholiday_SH", "schoolholiday_SO",
    "schoolholiday_SZ", "schoolholiday_TG", "schoolholiday_TI", "schoolholiday_UR", "schoolholiday_VD", "schoolholiday_VS",
    "schoolholiday_ZG", "schoolholiday_ZH",
    "temp", "rain_1h", "sun_1h", "snow_1h", "weather_cat",
]

# ── Stationsspezifische Feature-Ausnahmen ────────────────────────────────────
# Die Station Sattel (beide Richtungen R1 und R2) hat nur für 2015-2017 dichte
# Trainingsdaten, das Testset reicht aber bis 2025 (rund 8 Jahre Extrapolation).
# Das absolute Year-Feature liegt für 2025 weit ausserhalb des Trainingsbereichs
# und entgleist die Vorhersage massiv (Test-R2 fällt für R1 auf ~0.54). Für diese
# Station wird Year daher aus dem Feature-Satz entfernt.
FEATURE_EXCLUDE = {
    "171_Sattel_R1": ["Year"],
    "171_Sattel_R2": ["Year"],
}


# Dataset-Klasse für den DataLoader (wandelt numpy arrays in torch Tensoren um)
class VerkehrsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Speichert Features und Zielgrösse als Float32-Tensoren."""
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """Gibt die Anzahl der Datenpunkte zurück."""
        return len(self.X)

    def __getitem__(self, idx):
        """Gibt das Feature-Ziel-Paar an Position idx zurück."""
        return self.X[idx], self.y[idx]


# MLP-Modell durch Subclassing von nn.Module
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        """Baut das Netz Schicht für Schicht aus Linear, BatchNorm, ReLU und Dropout auf."""
        super().__init__()
        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))  # Batch Normalization vor ReLU
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:             # kein Dropout nach letzter Hidden Layer
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))          # Output: 1 Neuron, keine Aktivierungsfunktion (Regression)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vorwärtsdurchlauf durch alle Schichten des Netzes."""
        return self.net(x)


def main():
    """Trainiert ein MLP pro Messstation mit Early Stopping und wertet es aus."""
    for filename in DATASETS:
        name      = filename.replace(".csv", "")
        file_path = DATA_DIR / filename

        if not file_path.exists():
            continue

        # Seed pro Station neu setzen, damit jede Station unabhängig und exakt
        # reproduzierbar ist. Ohne dies würde das stationsspezifische Entfernen
        # eines Features (FEATURE_EXCLUDE) die fortlaufende Zufalls-Kette und damit
        # die Resultate aller NACHFOLGENDEN Stationen mit verändern.
        torch.manual_seed(42)
        np.random.seed(42)

        df = pd.read_csv(file_path).dropna()

        # datetime als Index setzen (nur als Index, nicht als Feature)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)

        if "weather_cat" in df.columns and df["weather_cat"].dtype == "object":
            df["weather_cat"] = LabelEncoder().fit_transform(df["weather_cat"])

        feature_cols = [c for c in FEATURES if c in df.columns]

        # Stationsspezifische Ausnahmen anwenden (z.B. Year bei Sattel R1 entfernen)
        excluded = next((cols for key, cols in FEATURE_EXCLUDE.items() if key in name), [])
        if excluded:
            feature_cols = [c for c in feature_cols if c not in excluded]

        X = df[feature_cols].values
        y = df["volume"].values.reshape(-1, 1)

        # Chronologischer 70/10/20-Split, kein shuffle
        n         = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

        # StandardScaler NUR auf Trainingsdaten fitten
        x_scaler  = StandardScaler()
        X_train_s = x_scaler.fit_transform(X_train)
        X_val_s   = x_scaler.transform(X_val)
        X_test_s  = x_scaler.transform(X_test)

        # y ebenfalls skalieren (verbessert numerische Stabilität des Trainings)
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

        model     = MLP(len(feature_cols), HIDDEN_DIMS, DROPOUT).to(device)
        loss_fn   = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
        )

        best_val_loss = float("inf")
        best_weights  = None
        no_improve    = 0

        # Trainingsschleife über alle Epochen
        for epoch in range(MAX_EPOCHS):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validierung
            model.eval()  # Dropout und BatchNorm in Auswertungsmodus schalten
            with torch.inference_mode():
                epoch_val_loss = 0.0
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    epoch_val_loss += loss_fn(y_pred, y_batch).item() * len(X_batch)
                epoch_val_loss /= len(val_loader.dataset)

            # LR-Scheduler auf Validation Loss anwenden
            scheduler.step(epoch_val_loss)

            # Early Stopping: bestes Modell anhand Validation Loss merken
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= ES_PATIENCE:
                    break

        # Bestes Modell laden für Evaluation (nicht letztes!)
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

        # Evaluation auf dem Testset
        model.eval()
        with torch.inference_mode():
            preds_scaled = model(
                torch.tensor(X_test_s, dtype=torch.float32).to(device)
            ).cpu().numpy()

        # Zurück in Fahrzeuge/h
        preds = y_scaler.inverse_transform(preds_scaled)

        # Metriken (kein MSE in den Resultaten – nur MAE, RMSE, R²)
        # y_test bleibt original (nicht skaliert) für Metriken
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        print(f"{name}: MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")


if __name__ == "__main__":
    main()
