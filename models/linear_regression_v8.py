"""
Dieses Skript trainiert ein lineares Regressionsmodell fuer die Verkehrsvorhersage
auf den v8-Daten als Baseline zum v8-MLP. v8 basiert auf v7 (26 kantonsspezifische
Feiertagsspalten holiday_AG … holiday_ZH) und fuegt zusaetzlich 26 kantonsspezifische
Schulferienspalten (schoolholiday_AG … schoolholiday_ZH) hinzu. Die Resultate werden
in einem separaten Verzeichnis abgelegt, damit die bestehenden v5/v6/v7-Resultate
unveraendert bleiben.
"""

import os
import json
from pathlib import Path
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Seed setzen (zusaetzlich pro Station, siehe Schleife)
torch.manual_seed(42)
np.random.seed(42)

# Device konfigurieren
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pfade konfigurieren
DATA_DIR = Path("data/v8")
RESULTS_DIR = Path("results/model_results/linear_regression_v8")

DATASETS = [
    "050_Brunnen_Mositunnel_R1_v8.csv", "050_Brunnen_Mositunnel_R2_v8.csv",
    "171_Sattel_R1_v8.csv", "171_Sattel_R2_v8.csv",
    "216_Wangen_SZ_R1_v8.csv", "216_Wangen_SZ_R2_v8.csv",
    "299_Wollerau_Blatttunnel_R1_v8.csv", "299_Wollerau_Blatttunnel_R2_v8.csv",
    "720_Schwyz_R1_v8.csv", "720_Schwyz_R2_v8.csv"
]

# v8-Feature-Satz: 26 kantonsspezifische Feiertagsspalten plus 26 kantonsspezifische
# Schulferienspalten (identisch zu mlp_v8.py).
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

# ── Stationsspezifische Feature-Ausnahmen (identisch zu mlp_v8.py) ────────────
# Die Station Sattel (beide Richtungen R1 und R2) hat nur fuer 2015-2017 dichte
# Trainingsdaten, das Testset reicht aber bis 2025 (rund 8 Jahre Extrapolation).
# Das absolute Year-Feature liegt fuer 2025 weit ausserhalb des Trainingsbereichs
# und entgleist die Vorhersage. Damit die LR-Baseline konsistent zum v8-MLP ist,
# wird Year fuer diese Station ebenfalls entfernt. Details siehe README
# ("Stationsspezifische Ausnahme").
FEATURE_EXCLUDE = {
    "171_Sattel_R1": ["Year"],
    "171_Sattel_R2": ["Year"],
}

# Unterordner automatisch erstellen
(RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "model_weights").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "plots" / "loss_curves").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "plots" / "scatter").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "plots" / "timeseries").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "plots" / "residuals").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "plots" / "tagesverlauf").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "summary").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)


# Create a linear model by subclassing nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Use nn.Linear() for creating the model parameters / also called: linear transform
        self.linear_layer = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


def main():
    all_metrics = []

    for i, filename in enumerate(DATASETS):
        name = filename.replace('.csv', '')
        file_path = DATA_DIR / filename

        if not file_path.exists():
            print(f"Warnung: Datei {file_path} nicht gefunden, wird übersprungen.")
            continue

        # Seed pro Station neu setzen, damit jede Station unabhaengig und exakt
        # reproduzierbar ist. Ohne dies wuerde das stationsspezifische Entfernen
        # eines Features (FEATURE_EXCLUDE) die fortlaufende Zufalls-Kette und damit
        # die Resultate aller NACHFOLGENDEN Stationen mit veraendern.
        torch.manual_seed(42)
        np.random.seed(42)

        # Daten laden, NaN droppen
        df = pd.read_csv(file_path).dropna()

        # datetime als Index setzen (nur als Index, nicht als Feature)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)

        # Label Encoding für weather_cat falls object
        if 'weather_cat' in df.columns and df['weather_cat'].dtype == 'object':
            df['weather_cat'] = LabelEncoder().fit_transform(df['weather_cat'])

        # Nur vorhandene Features nutzen
        feature_cols = [c for c in FEATURES if c in df.columns]

        # Stationsspezifische Ausnahmen anwenden (z.B. Year bei Sattel entfernen)
        excluded = next((cols for key, cols in FEATURE_EXCLUDE.items() if key in name), [])
        if excluded:
            feature_cols = [c for c in feature_cols if c not in excluded]
            print(f"  Hinweis: Fuer {name} aus dem Feature-Satz entfernt: {excluded}")

        X = df[feature_cols].values
        y = df['volume'].values.reshape(-1, 1)

        # Chronologischer Train/Test-Split: 80/20, kein shuffle
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        df_test = df.iloc[split_idx:].copy() # Für Plotting

        # Split-Info als JSON speichern (Reproduzierbarkeit/Dokumentation)
        train_idx = df.index[:split_idx]
        test_idx = df.index[split_idx:]
        split_info = {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "train_pct": round(len(X_train) / len(df) * 100, 2),
            "val_samples": None,
            "val_pct": None,
            "test_samples": len(X_test),
            "test_pct": round(len(X_test) / len(df) * 100, 2),
            "split_type": "chronological 80/20",
            "date_range_train": {"start": str(train_idx[0]), "end": str(train_idx[-1])},
            "date_range_val": None,
            "date_range_test": {"start": str(test_idx[0]), "end": str(test_idx[-1])},
        }
        with open(RESULTS_DIR / "summary" / f"split_info_{name}.json", "w") as f:
            json.dump(split_info, f, indent=4)

        # StandardScaler NUR auf Trainingsdaten fitten
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # NEU: y ebenfalls skalieren
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)

        # Konvertierung zu torch.float32 Tensoren, auf device verschieben
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

        # Pro Datensatz neue Modellinstanz erstellen
        model = LinearRegressionModel(input_dim=len(feature_cols)).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        epochs = 200

        print(f"\nTrainiere Modell {i+1}/10: {name}")
        print(f"LR: {optimizer.param_groups[0]['lr']}, Epochen: {epochs}")

        start_time = time.time()

        # Track different values (wie im PyTorch Tutorial)
        epoch_count = []
        loss_values = []
        test_loss_values = []

        # 0. Loop through the data
        for epoch in range(epochs):
            # Set the model to training mode
            model.train()

            # 1. Forward pass
            y_pred = model(X_train_t)

            # 2. Calculate the loss
            loss = loss_fn(y_pred, y_train_t)

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Perform backpropagation on the loss
            loss.backward()

            # 5. Step the optimizer (gradient descent)
            optimizer.step()

            ### Testing
            model.eval()
            with torch.inference_mode():
                # 1. Do the forward pass
                test_pred = model(X_test_t)

                # 2. Calculate the loss
                test_loss = loss_fn(test_pred, y_test_t)

            if epoch % 10 == 0:
                epoch_count.append(epoch)
                loss_values.append(loss.item())
                test_loss_values.append(test_loss.item())

        train_duration = time.time() - start_time

        print(f"Finale Loss: {loss.item():.2f}")
        print(f"Trainingszeit: {train_duration:.1f}s")

        # Evaluation (torch.inference_mode)
        model.eval()
        with torch.inference_mode():
            preds_t = model(X_test_t)
            preds_scaled = preds_t.cpu().numpy()

        # Zurück in Fahrzeuge/h
        preds = y_scaler.inverse_transform(preds_scaled)

        # Metriken (KEIN MSE in den Resultaten – nur MAE, RMSE, R²)
        # y_test bleibt original (nicht skaliert) für Metriken und Plots
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

        # In all_metrics anhängen
        metric_dict = {
            'station': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        all_metrics.append(metric_dict)

        # Metriken sofort in CSV speichern (beim 1. Modell überschreiben, dann anhängen)
        metrics_file = RESULTS_DIR / "metrics" / "all_metrics.csv"
        pd.DataFrame([metric_dict]).to_csv(metrics_file, mode='w' if i == 0 else 'a', header=(i == 0), index=False)

        # Modell speichern
        torch.save(model.state_dict(), RESULTS_DIR / "model_weights" / f"lr_{name}.pt")

        # Vorhersage-Vergleich speichern (Ist- vs. Prognosewerte)
        pd.DataFrame({
            "datetime":         df_test.index,
            "actual_volume":    y_test.flatten(),
            "predicted_volume": preds.flatten(),
        }).to_csv(RESULTS_DIR / "predictions" / f"predictions_{name}.csv", index=False)

        # --- PLOTS PRO MODELL ---

        # Plot A – Loss-Kurve (Training and test loss curves)
        plt.figure(figsize=(10, 7))
        plt.plot(epoch_count, loss_values, label="Train loss")
        plt.plot(epoch_count, test_loss_values, label="Test loss")
        plt.title(f"Training and test loss curves - {name}")
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
        plt.plot([0, max_val], [0, max_val], 'r-')
        plt.title(f"Predicted vs. Actual - {name}")
        plt.xlabel("Tatsächliches Volumen (Ground Truth)")
        plt.ylabel("Vorhergesagtes Volumen (Predicted)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "plots" / "scatter" / f"scatter_{name}.png")
        plt.close()

        # Plot C – Zeitreihe (erste 2 Wochen = 336 Stunden)
        plt.figure(figsize=(15, 5))
        # Da datetime nun der Index ist:
        time_x = df_test.index[:336]
        plt.plot(time_x, y_test[:336], color='black', label='Ground Truth')
        plt.plot(time_x, preds[:336], color='orange', label='Vorhersage')
        # Formatierung der x-Achse falls datetime
        if isinstance(df_test.index, pd.DatetimeIndex):
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.title(f"Zeitreihe (erste 2 Wochen Testset) - {name}")
        plt.xlabel("Datum/Zeit")
        plt.ylabel("Fahrzeuge/h")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "plots" / "timeseries" / f"timeseries_{name}.png")
        plt.close()

        # Residuals berechnen für Plot D
        residuals = preds.flatten() - y_test.flatten()
        df_test['residual'] = residuals
        df_test['pred'] = preds.flatten()

        # Für Plot D und E brauchen wir die Stunde. Falls Hour als Spalte existiert gut, ansonsten aus dem Index holen.
        if isinstance(df_test.index, pd.DatetimeIndex):
            df_test['Hour_for_plot'] = df_test.index.hour
        else:
            df_test['Hour_for_plot'] = 0 # Fallback

        # Plot D – Residuals nach Tagesstunde
        res_by_hour = df_test.groupby('Hour_for_plot')['residual'].mean()
        plt.figure(figsize=(10, 5))
        plt.bar(res_by_hour.index, res_by_hour.values)
        plt.axhline(0, color='black', linewidth=1)
        plt.title(f"Ø Residuals nach Tagesstunde - {name}")
        plt.xlabel("Stunde (0-23)")
        plt.ylabel("Ø Residual (Predicted - Actual)")
        plt.xticks(range(24))
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "plots" / "residuals" / f"residuals_{name}.png")
        plt.close()

        # Plot E – Durchschnittlicher Tagesverlauf
        avg_actual = df_test.groupby('Hour_for_plot')['volume'].mean()
        avg_pred = df_test.groupby('Hour_for_plot')['pred'].mean()
        plt.figure(figsize=(10, 5))
        plt.plot(avg_actual.index, avg_actual.values, color='black', label='Ø Ground Truth')
        plt.plot(avg_pred.index, avg_pred.values, 'r--', label='Ø Vorhersage')
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

        # Summary Plot: R² Scores
        plt.figure(figsize=(12, 6))
        colors = ['red' if r2 < 0 else 'green' for r2 in metrics_df['R2']]
        bars = plt.bar(metrics_df['station'], metrics_df['R2'], color=colors)
        plt.axhline(0, color='black', linewidth=1)
        plt.title("R² Scores über alle Stationen")
        plt.xlabel("Stationsname")
        plt.ylabel("R² Score")
        plt.xticks(rotation=45, ha='right')

        # R²-Werte als Label über/unter jedem Balken
        for bar in bars:
            yval = bar.get_height()
            # Positionierung abhängig davon, ob Balken nach oben oder unten geht
            if yval >= 0:
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
            else:
                plt.text(bar.get_x() + bar.get_width()/2, yval - 0.02, f"{yval:.2f}", ha='center', va='top', fontsize=9)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "summary" / "r2_summary.png")
        plt.close()

        print("\nAlle Modelle trainiert und Ergebnisse gespeichert.")

if __name__ == "__main__":
    main()
