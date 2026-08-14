"""
Dieses Skript trainiert ein lineares Regressionsmodell als Baseline zur stündlichen
Verkehrsvorhersage auf den v8-Daten (Kanton Schwyz, ASTRA-Messstationen). Es umfasst
Datenaufbereitung, chronologischen Split, Skalierung, Training und die Evaluation
mittels MAE, RMSE und R² pro Messstation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data/v8")

DATASETS = [
    "050_Brunnen_Mositunnel_R1_v8.csv", "050_Brunnen_Mositunnel_R2_v8.csv",
    "171_Sattel_R1_v8.csv", "171_Sattel_R2_v8.csv",
    "216_Wangen_SZ_R1_v8.csv", "216_Wangen_SZ_R2_v8.csv",
    "299_Wollerau_Blatttunnel_R1_v8.csv", "299_Wollerau_Blatttunnel_R2_v8.csv",
    "720_Schwyz_R1_v8.csv", "720_Schwyz_R2_v8.csv"
]

# v8-Feature-Satz: 26 kantonsspezifische Feiertagsspalten plus 26 kantonsspezifische
# Schulferienspalten (identisch zu anhang_mlp.py).
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

# ── Stationsspezifische Feature-Ausnahmen (identisch zu anhang_mlp.py) ────────
# Die Station Sattel (beide Richtungen R1 und R2) hat nur für 2015-2017 dichte
# Trainingsdaten, das Testset reicht aber bis 2025 (rund 8 Jahre Extrapolation).
# Das absolute Year-Feature liegt für 2025 weit ausserhalb des Trainingsbereichs
# und entgleist die Vorhersage. Damit die LR-Baseline konsistent zum v8-MLP ist,
# wird Year für diese Station ebenfalls entfernt.
FEATURE_EXCLUDE = {
    "171_Sattel_R1": ["Year"],
    "171_Sattel_R2": ["Year"],
}


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        """Initialisiert das Modell mit einer einzigen linearen Schicht."""
        super().__init__()
        self.linear_layer = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vorwärtsdurchlauf: gibt die lineare Transformation der Eingabe zurück."""
        return self.linear_layer(x)


def main():
    """Trainiert ein lineares Regressionsmodell pro Messstation und wertet es aus."""
    for filename in DATASETS:
        name = filename.replace(".csv", "")
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

        # Stationsspezifische Ausnahmen anwenden (z.B. Year bei Sattel entfernen)
        excluded = next((cols for key, cols in FEATURE_EXCLUDE.items() if key in name), [])
        if excluded:
            feature_cols = [c for c in feature_cols if c not in excluded]

        X = df[feature_cols].values
        y = df["volume"].values.reshape(-1, 1)

        # Chronologischer Train/Test-Split: 80/20, kein shuffle
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # StandardScaler NUR auf Trainingsdaten fitten
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # y ebenfalls skalieren (verbessert numerische Stabilität des Trainings)
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

        model = LinearRegressionModel(input_dim=len(feature_cols)).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        epochs = 200

        # Trainingsschleife über alle Epochen
        for epoch in range(epochs):
            model.train()

            y_pred = model(X_train_t)
            loss = loss_fn(y_pred, y_train_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation auf dem Testset
        model.eval()
        with torch.inference_mode():
            preds_t = model(X_test_t)
            preds_scaled = preds_t.cpu().numpy()

        # Zurück in Fahrzeuge/h
        preds = y_scaler.inverse_transform(preds_scaled)

        # Metriken (kein MSE in den Resultaten – nur MAE, RMSE, R²)
        # y_test bleibt original (nicht skaliert) für Metriken
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"{name}: MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}")


if __name__ == "__main__":
    main()
