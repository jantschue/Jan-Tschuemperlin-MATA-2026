"""
Dieses Skript exportiert die trainierten PyTorch-Modelle (MLP + Lineare Regression)
und die Test-Vorhersagen aller Stationen in das JSON-Format, das von der Webapp
unter webapp/public/data/ erwartet wird.

Lokal ausführen mit:  python export_weights.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Pfade ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data" / "v5_engineered"

MLP_WEIGHTS_DIR = PROJECT_ROOT / "results" / "model_results" / "mlp"     / "model_weights"
LR_WEIGHTS_DIR  = PROJECT_ROOT / "results" / "model_results" / "linear_regression" / "model_weights"
MLP_PREDS_DIR   = PROJECT_ROOT / "results" / "model_results" / "mlp"     / "predictions"
LR_PREDS_DIR    = PROJECT_ROOT / "results" / "model_results" / "linear_regression" / "predictions"

OUT_DATA    = PROJECT_ROOT / "webapp" / "public" / "data"
OUT_WEIGHTS = OUT_DATA / "weights"
OUT_RESULTS = OUT_DATA / "results"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_WEIGHTS.mkdir(parents=True, exist_ok=True)
OUT_RESULTS.mkdir(parents=True, exist_ok=True)

# ── Modellklassen aus den vorhandenen Python-Files importieren ──────────────
sys.path.insert(0, str(PROJECT_ROOT / "models"))
# noqa: import order – MLP und LinearRegressionModel werden zur Laufzeit gebraucht
from mlp import MLP, HIDDEN_DIMS, DROPOUT, FEATURES, TRAIN_RATIO, VAL_RATIO  # type: ignore
from linear_regression import LinearRegressionModel  # type: ignore

# ── Konfiguration ───────────────────────────────────────────────────────────
# Linear: chronologischer 80/20 Split, MLP: 70/10/20 Split.
LR_TRAIN_RATIO  = 0.8
MLP_TRAIN_RATIO = TRAIN_RATIO  # 0.70 aus mlp.py
MLP_VAL_RATIO   = VAL_RATIO    # 0.10 aus mlp.py

# Metadaten pro Stations-Nummer (LV95 → WGS84 wird unten umgerechnet)
STATION_META = {
    "050": {"display_name": "Brunnen Mositunnel",
            "dirs": {"R1": "Nord", "R2": "Süd"},
            "E": 2689129, "N": 1205058},
    "171": {"display_name": "Sattel",
            "dirs": {"R1": "Nord", "R2": "Süd"},
            "E": 2691106, "N": 1215398},
    "216": {"display_name": "Wangen SZ",
            "dirs": {"R1": "Ost",  "R2": "West"},
            "E": 2710351, "N": 1227960},
    "299": {"display_name": "Wollerau Blatttunnel",
            "dirs": {"R1": "Nord", "R2": "Süd"},
            "E": 2695600, "N": 1228200},
    "720": {"display_name": "Schwyz",
            "dirs": {"R1": "Ost",  "R2": "West"},
            "E": 2690230, "N": 1207580},
}


def lv95_to_wgs84(E: float, N: float) -> tuple[float, float]:
    """LV95 → WGS84 mit offizieller swisstopo-Näherungsformel."""
    yp = (E - 2_600_000) / 1e6
    xp = (N - 1_200_000) / 1e6
    lam = (2.6779094 + 4.728982 * yp + 0.791484 * yp * xp
           + 0.1306 * yp * xp * xp - 0.0436 * yp ** 3)
    phi = (16.9023892 + 3.238272 * xp - 0.270978 * yp * yp
           - 0.002528 * xp * xp - 0.0447 * yp * yp * xp - 0.0140 * xp ** 3)
    return phi * 100 / 36, lam * 100 / 36


def station_id_from_filename(name: str) -> str:
    """`mlp_050_Brunnen_Mositunnel_R1_engineered.pt` → `050_Brunnen_Mositunnel_R1`."""
    base = name
    for pfx in ("mlp_", "lr_", "predictions_"):
        if base.startswith(pfx):
            base = base[len(pfx):]
    if base.endswith(".pt"):
        base = base[:-3]
    if base.endswith(".csv"):
        base = base[:-4]
    if base.endswith("_engineered"):
        base = base[: -len("_engineered")]
    return base


def parse_station_id(station_id: str) -> tuple[str, str]:
    """`050_Brunnen_Mositunnel_R1` → (`050`, `R1`)."""
    parts = station_id.split("_")
    return parts[0], parts[-1]


def load_csv_with_split(csv_path: Path, train_ratio: float, val_ratio: float = 0.0):
    """
    Lädt einen Datensatz und reproduziert die chronologische Aufteilung der
    Trainings-Skripte, damit Scaler und Test-Set identisch sind.
    """
    df = pd.read_csv(csv_path).dropna()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

    if "weather_cat" in df.columns and df["weather_cat"].dtype == "object":
        df["weather_cat"] = LabelEncoder().fit_transform(df["weather_cat"])

    feature_cols = [c for c in FEATURES if c in df.columns]
    X = df[feature_cols].values
    y = df["volume"].values.reshape(-1, 1)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    test_start = max(train_end, val_end)

    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test  = X[test_start:]
    y_test  = y[test_start:]
    test_index = df.index[test_start:]

    return feature_cols, X_train, y_train, X_test, y_test, test_index


def fit_scalers(X_train: np.ndarray, y_train: np.ndarray):
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    return x_scaler, y_scaler


def export_linear(station_id: str) -> dict | None:
    """Exportiert ein lineares Modell als JSON. Gibt {weight, bias, scaler} zurück."""
    pt_path = LR_WEIGHTS_DIR / f"lr_{station_id}_engineered.pt"
    csv_path = DATA_DIR / f"{station_id}_engineered.csv"
    if not pt_path.exists() or not csv_path.exists():
        print(f"  !Lineare Regression {station_id}: Datei fehlt – übersprungen")
        return None

    feature_cols, X_train, y_train, *_ = load_csv_with_split(csv_path, LR_TRAIN_RATIO)
    x_scaler, y_scaler = fit_scalers(X_train, y_train)

    model = LinearRegressionModel(input_dim=len(feature_cols))
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()

    weight = model.linear_layer.weight.detach().cpu().numpy().flatten().tolist()
    bias   = float(model.linear_layer.bias.detach().cpu().numpy().flatten()[0])

    payload = {
        "features": feature_cols,
        "weight": weight,
        "bias": bias,
        "input_scaler":  {"mean": x_scaler.mean_.tolist(),
                          "std":  x_scaler.scale_.tolist()},
        "output_scaler": {"mean": y_scaler.mean_.tolist(),
                          "std":  y_scaler.scale_.tolist()},
    }
    out = OUT_WEIGHTS / f"{station_id}_linear.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def export_mlp(station_id: str) -> dict | None:
    """
    Exportiert ein MLP-Modell als JSON. Architektur (mlp.py):
    Pro Hidden-Block: Linear → BatchNorm1d → ReLU → (Dropout)
    Output: Linear (keine Aktivierung). Dropout ist im Eval-Modus identity
    und wird daher nicht exportiert.
    """
    pt_path  = MLP_WEIGHTS_DIR / f"mlp_{station_id}_engineered.pt"
    csv_path = DATA_DIR / f"{station_id}_engineered.csv"
    if not pt_path.exists() or not csv_path.exists():
        print(f"  !MLP {station_id}: Datei fehlt – übersprungen")
        return None

    feature_cols, X_train, y_train, *_ = load_csv_with_split(
        csv_path, MLP_TRAIN_RATIO, MLP_VAL_RATIO
    )
    x_scaler, y_scaler = fit_scalers(X_train, y_train)

    model = MLP(len(feature_cols), HIDDEN_DIMS, DROPOUT)
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    model.eval()

    # Die net.X-Indexierung in Python: pro Hidden-Block 4 Module
    # (Linear, BatchNorm, ReLU, Dropout), letzter Block hat keinen Dropout,
    # dann Output-Linear. Wir laufen über `model.net` und exportieren nur
    # Linear- und BatchNorm-Schichten (+ ReLU als Marker).
    layers_out: list[dict] = []
    import torch.nn as nn
    for module in model.net:
        if isinstance(module, nn.Linear):
            layers_out.append({
                "type": "linear",
                "weight": module.weight.detach().cpu().tolist(),
                "bias":   module.bias.detach().cpu().tolist(),
            })
        elif isinstance(module, nn.BatchNorm1d):
            layers_out.append({
                "type": "batchnorm",
                "weight":       module.weight.detach().cpu().tolist(),
                "bias":         module.bias.detach().cpu().tolist(),
                "running_mean": module.running_mean.detach().cpu().tolist(),
                "running_var":  module.running_var.detach().cpu().tolist(),
                "eps":          module.eps,
            })
        elif isinstance(module, nn.ReLU):
            layers_out.append({"type": "relu"})
        elif isinstance(module, nn.Dropout):
            # Dropout in eval = identity → wird nicht exportiert.
            continue
        else:
            raise RuntimeError(f"Unerwartetes Modul: {type(module).__name__}")

    payload = {
        "features": feature_cols,
        "layers": layers_out,
        "activation": "relu",
        "hidden_dims": list(HIDDEN_DIMS),
        "input_scaler":  {"mean": x_scaler.mean_.tolist(),
                          "std":  x_scaler.scale_.tolist()},
        "output_scaler": {"mean": y_scaler.mean_.tolist(),
                          "std":  y_scaler.scale_.tolist()},
    }
    out = OUT_WEIGHTS / f"{station_id}_mlp.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def export_daily_results(station_id: str) -> list[dict]:
    """Merged stündliche Test-Vorhersagen MLP + Linear → daily-results JSON."""
    mlp_csv = MLP_PREDS_DIR / f"predictions_{station_id}_engineered.csv"
    lr_csv  = LR_PREDS_DIR  / f"predictions_{station_id}_engineered.csv"

    if not mlp_csv.exists() or not lr_csv.exists():
        print(f"  !Predictions {station_id}: fehlen – übersprungen")
        return []

    mlp_df = pd.read_csv(mlp_csv, parse_dates=["datetime"]).rename(
        columns={"predicted_volume": "pred_mlp"}
    )
    lr_df = pd.read_csv(lr_csv, parse_dates=["datetime"]).rename(
        columns={"predicted_volume": "pred_linear"}
    )[["datetime", "pred_linear"]]

    merged = mlp_df.merge(lr_df, on="datetime", how="inner")
    merged = merged.sort_values("datetime")

    records = [{
        "datetime": ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "actual":   int(round(float(actual))),
        "pred_mlp": int(round(float(pm))),
        "pred_linear": int(round(float(pl))),
    } for ts, actual, pm, pl in zip(
        merged["datetime"], merged["actual_volume"],
        merged["pred_mlp"], merged["pred_linear"]
    )]

    out = OUT_RESULTS / f"{station_id}_daily.json"
    out.write_text(json.dumps(records), encoding="utf-8")
    return records


def load_metrics_table(path: Path) -> dict[str, dict]:
    """Liest all_metrics.csv und keyed nach Station-ID (ohne `_engineered`)."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        sid = str(row["station"])
        if sid.endswith("_engineered"):
            sid = sid[: -len("_engineered")]
        out[sid] = {
            "mae":  float(row["MAE"]),
            "rmse": float(row["RMSE"]),
            "r2":   float(row["R2"]),
        }
    return out


def collect_station_ids() -> list[str]:
    """Ermittelt alle Station-IDs aus den Daten-CSV-Dateinamen."""
    ids = []
    for csv in sorted(DATA_DIR.glob("*_engineered.csv")):
        ids.append(station_id_from_filename(csv.name))
    return ids


def main():
    print(f"Projekt-Wurzel: {PROJECT_ROOT}")
    print(f"Schreibe nach: {OUT_DATA}\n")

    lr_metrics  = load_metrics_table(LR_WEIGHTS_DIR.parent / "metrics" / "all_metrics.csv")
    mlp_metrics = load_metrics_table(MLP_WEIGHTS_DIR.parent / "metrics" / "all_metrics.csv")

    station_ids = collect_station_ids()
    print(f"{len(station_ids)} Stationen gefunden.\n")

    stations_out = []
    for sid in station_ids:
        number, direction_key = parse_station_id(sid)
        meta = STATION_META.get(number)
        if meta is None:
            print(f"  !Keine Metadaten für Stationsnummer {number} – übersprungen")
            continue

        print(f"-> {sid}")
        export_mlp(sid)
        export_linear(sid)
        export_daily_results(sid)

        lat, lng = lv95_to_wgs84(meta["E"], meta["N"])
        # R1/R2 minimal versetzen, damit beide Marker auf der Karte sichtbar sind
        if direction_key == "R1":
            lng -= 0.0003
        else:
            lng += 0.0003

        stations_out.append({
            "id": sid,
            "name": meta["display_name"],
            "direction": meta["dirs"].get(direction_key, direction_key),
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "metrics": {
                "linear": lr_metrics.get(sid, {"mae": 0.0, "rmse": 0.0, "r2": 0.0}),
                "mlp":    mlp_metrics.get(sid, {"mae": 0.0, "rmse": 0.0, "r2": 0.0}),
            },
        })

    (OUT_DATA / "stations.json").write_text(
        json.dumps(stations_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nOK Export abgeschlossen: {len(stations_out)} Stationen in stations.json")
    print(f"  Weights:        {OUT_WEIGHTS}")
    print(f"  Daily Results:  {OUT_RESULTS}")


if __name__ == "__main__":
    main()
