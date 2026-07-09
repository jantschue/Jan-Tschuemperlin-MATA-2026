"""
Dieses Skript berechnet die gruppierte Permutations-Wichtigkeit der Eingabemerkmale
fuer die beiden v8-Modelle (lineare Regression und MLP). Bei der Permutations-
Wichtigkeit wird ein Merkmal (bzw. eine Merkmalsgruppe) im Testset zufaellig
durchmischt und gemessen, wie stark das Bestimmtheitsmass R2 dadurch faellt. Je
groesser der Abfall, desto wichtiger ist die Gruppe fuer die Vorhersage.

Die 67 Einzelmerkmale werden dabei zu inhaltlichen Gruppen zusammengefasst
(Tageszeit, Wochentag, Jahreszeit, Feiertage, Schulferien, Wetter usw.), damit die
Auswertung lesbar bleibt. Zusammengehoerende Spalten (z.B. Hour_sin und Hour_cos)
werden mit derselben Zufallsreihenfolge permutiert, damit ihre gemeinsame Struktur
erhalten bleibt.

Das Skript laedt die bereits trainierten Modellgewichte aus results/model_results/
und rekonstruiert Split und Skalierung exakt wie in den Trainingsskripten
(linear_regression_v8.py, mlp_v8.py). Es trainiert also nichts neu.

Ergebnisse:
    results/analysis/feature_importance/permutation_importance_linear_regression_v8.csv
    results/analysis/feature_importance/permutation_importance_mlp_v8.csv
    results/analysis/feature_importance/feature_importance_linear_regression_v8.png
    results/analysis/feature_importance/feature_importance_mlp_v8.png
    results/analysis/feature_importance/feature_importance_vergleich.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

# Seed setzen (zusaetzlich pro Station/Modell, siehe Schleife)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Konfiguration ────────────────────────────────────────────────────────────
N_REPEATS = 10                 # Anzahl Durchmischungen pro Gruppe (Mittelwert daraus)
DATA_DIR  = Path("data/v8")
OUT_DIR   = Path("results/analysis/feature_importance")

# Split-Verhaeltnisse identisch zu den Trainingsskripten
LR_TEST_SPLIT = 0.80           # linear_regression_v8.py: 80/20
MLP_TRAIN_RATIO = 0.70         # mlp_v8.py: 70/10/20
MLP_VAL_RATIO   = 0.10

# MLP-Architektur identisch zu mlp_v8.py (damit die Gewichte passen)
HIDDEN_DIMS = [512, 256, 256, 128]
DROPOUT     = 0.28262030710606983   # im Eval-Modus ohne Wirkung, nur fuer identischen Aufbau

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

# Stationsspezifische Feature-Ausnahmen (identisch zu den Trainingsskripten)
FEATURE_EXCLUDE = {
    "171_Sattel_R1": ["Year"],
    "171_Sattel_R2": ["Year"],
}

# ── Merkmalsgruppen ──────────────────────────────────────────────────────────
# Reihenfolge = Reihenfolge in der Ausgabe. Kantons-Spalten werden zu je einer
# Gruppe (Feiertage / Schulferien) zusammengefasst.
KANTONE = ["AG","AI","AR","BE","BL","BS","FR","GE","GL","GR","JU","LU","NE","NW",
           "OW","SG","SH","SO","SZ","TG","TI","UR","VD","VS","ZG","ZH"]

GROUPS = {
    "Jahr":            ["Year"],
    "Tageszeit":       ["Hour_sin", "Hour_cos"],
    "Wochentag":       ["DayOfWeek_sin", "DayOfWeek_cos"],
    "Jahreszeit":      ["Month_sin", "Month_cos", "DayOfYear_sin", "DayOfYear_cos"],
    "Wochenende":      ["is_weekend"],
    "Feiertage":       [f"holiday_{k}" for k in KANTONE],
    "Schulferien":     [f"schoolholiday_{k}" for k in KANTONE],
    "Temperatur":      ["temp"],
    "Regen":           ["rain_1h"],
    "Sonne":           ["sun_1h"],
    "Schnee":          ["snow_1h"],
    "Wetterkategorie": ["weather_cat"],
}


# ── Modellklassen (Architektur identisch zu den Trainingsskripten) ───────────
class LinearRegressionModel(nn.Module):
    """Lineares Modell mit einer einzigen linearen Schicht (wie linear_regression_v8.py)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


class MLP(nn.Module):
    """Mehrschichtiges Perzeptron, exakt wie in mlp_v8.py aufgebaut."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
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


def load_and_prepare(filename: str):
    """Laedt eine v8-Datei und rekonstruiert Features, Ziel und Feature-Namen."""
    name = filename.replace(".csv", "")
    df = pd.read_csv(DATA_DIR / filename).dropna()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    if df["weather_cat"].dtype == "object":
        df["weather_cat"] = LabelEncoder().fit_transform(df["weather_cat"])

    feature_cols = [c for c in FEATURES if c in df.columns]
    excluded = next((cols for key, cols in FEATURE_EXCLUDE.items() if key in name), [])
    feature_cols = [c for c in feature_cols if c not in excluded]

    X = df[feature_cols].values
    y = df["volume"].values.reshape(-1, 1)
    return name, X, y, feature_cols


def split_scale(X, y, model_kind: str):
    """Rekonstruiert Train/Test-Split und Skalierung exakt wie im jeweiligen Trainingsskript."""
    n = len(X)
    if model_kind == "lr":
        test_start = int(n * LR_TEST_SPLIT)
    else:
        # Achtung: identischer Ausdruck wie in mlp_v8.py, inkl. Float-Verhalten.
        test_start = int(n * (MLP_TRAIN_RATIO + MLP_VAL_RATIO))
    train_end = int(n * MLP_TRAIN_RATIO) if model_kind == "mlp" else test_start

    X_train, X_test = X[:train_end], X[test_start:]
    y_train, y_test = y[:train_end], y[test_start:]

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_test_s = x_scaler.transform(X_test)
    return X_test_s, y_test, y_scaler


def predict(model, X_test_s, y_scaler):
    """Berechnet die Vorhersagen in Fahrzeugen pro Stunde (Ruecktransformation)."""
    model.eval()
    with torch.inference_mode():
        preds_s = model(torch.tensor(X_test_s, dtype=torch.float32).to(device)).cpu().numpy()
    return y_scaler.inverse_transform(preds_s)


def permutation_importance(model, X_test_s, y_test, y_scaler, feature_cols, rng):
    """Berechnet pro Merkmalsgruppe den mittleren R2-Abfall bei Permutation."""
    baseline_r2 = r2_score(y_test, predict(model, X_test_s, y_scaler))
    n = len(X_test_s)
    result = {}
    for group_name, cols in GROUPS.items():
        idx = [feature_cols.index(c) for c in cols if c in feature_cols]
        if not idx:
            result[group_name] = np.nan          # Gruppe fehlt fuer diese Station (z.B. Jahr bei Sattel)
            continue
        drops = []
        for _ in range(N_REPEATS):
            X_perm = X_test_s.copy()
            order = rng.permutation(n)            # eine gemeinsame Reihenfolge fuer die ganze Gruppe
            X_perm[:, idx] = X_perm[order][:, idx]
            r2_perm = r2_score(y_test, predict(model, X_perm, y_scaler))
            drops.append(baseline_r2 - r2_perm)
        result[group_name] = float(np.mean(drops))
    return baseline_r2, result


def build_model(model_kind: str, input_dim: int, name: str):
    """Erstellt die passende Modellinstanz und laedt die gespeicherten Gewichte."""
    if model_kind == "lr":
        model = LinearRegressionModel(input_dim)
        weights = Path("results/model_results/linear_regression_v8/model_weights") / f"lr_{name}.pt"
    else:
        model = MLP(input_dim, HIDDEN_DIMS, DROPOUT)
        weights = Path("results/model_results/mlp_v8/model_weights") / f"mlp_{name}.pt"
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device)


def run_model(model_kind: str):
    """Fuehrt die Permutations-Wichtigkeit fuer ein Modell ueber alle Stationen aus."""
    per_station = {}
    for filename in DATASETS:
        name, X, y, feature_cols = load_and_prepare(filename)

        # Seed pro Station/Modell zuruecksetzen (analog zu den Trainingsskripten),
        # damit die Permutationen unabhaengig und exakt reproduzierbar sind.
        rng = np.random.default_rng(SEED)

        X_test_s, y_test, y_scaler = split_scale(X, y, model_kind)
        model = build_model(model_kind, len(feature_cols), name)
        baseline_r2, importances = permutation_importance(
            model, X_test_s, y_test, y_scaler, feature_cols, rng
        )
        print(f"  {model_kind.upper()} {name}: Baseline-R2 = {baseline_r2:.4f}")
        per_station[name] = importances

    table = pd.DataFrame(per_station)          # Zeilen = Gruppen, Spalten = Stationen
    table = table.reindex(list(GROUPS.keys()))
    table["Mittelwert"] = table.mean(axis=1, skipna=True)
    table["Std"] = table.drop(columns=["Mittelwert"]).std(axis=1, skipna=True)
    return table


# ── Beschriftung der Balken ─────────────────────────────────────────────────
LABEL_DECIMALS = 3          # Nachkommastellen des Zahlenwerts neben dem Balken
LABEL_PAD = 0.005           # horizontaler Abstand der Beschriftung (Anteil der x-Achse)


def annotate_bars(ax, ypos, values):
    """Schreibt den gerundeten Zahlenwert neben jeden horizontalen Balken."""
    x_min, x_max = ax.get_xlim()
    pad = (x_max - x_min) * LABEL_PAD
    for y, v in zip(ypos, values):
        if np.isnan(v):
            continue
        # Positive Balken: Text rechts vom Balkenende, negative: links davon.
        if v >= 0:
            ax.text(v + pad, y, f"{v:.{LABEL_DECIMALS}f}", va="center", ha="left", fontsize=8)
        else:
            ax.text(v - pad, y, f"{v:.{LABEL_DECIMALS}f}", va="center", ha="right", fontsize=8)


def plot_single(table: pd.DataFrame, model_label: str, out_path: Path):
    """Zeichnet ein horizontales Balkendiagramm der mittleren Wichtigkeit pro Gruppe."""
    means = table["Mittelwert"].sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(means.index, means.values, color="#4C72B0")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"Permutations-Wichtigkeit der Merkmalsgruppen ({model_label})")
    ax.set_xlabel("Mittlerer R2-Abfall bei Permutation (ueber alle Datenreihen)")
    # Etwas Rand rechts/links schaffen, damit die Zahlenwerte nicht abgeschnitten werden.
    ax.margins(x=0.12)
    annotate_bars(ax, np.arange(len(means)), means.values)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_compare(table_lr, table_mlp, out_path: Path):
    """Vergleicht die mittlere Wichtigkeit pro Gruppe fuer lineare Regression und MLP."""
    order = table_mlp["Mittelwert"].sort_values(ascending=False).index
    y = np.arange(len(order))
    h = 0.4
    mlp_values = table_mlp.loc[order, "Mittelwert"].values
    lr_values  = table_lr.loc[order, "Mittelwert"].values
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(y + h / 2, mlp_values, height=h, label="MLP", color="#4C72B0")
    ax.barh(y - h / 2, lr_values,  height=h, label="Lineare Regression", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Permutations-Wichtigkeit der Merkmalsgruppen: MLP vs. lineare Regression")
    ax.set_xlabel("Mittlerer R2-Abfall bei Permutation (ueber alle Datenreihen)")
    ax.legend()
    # Etwas Rand schaffen, damit die Zahlenwerte neben den Balken Platz haben.
    ax.margins(x=0.14)
    annotate_bars(ax, y + h / 2, mlp_values)
    annotate_bars(ax, y - h / 2, lr_values)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    """Berechnet und speichert die gruppierte Permutations-Wichtigkeit beider Modelle."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Lineare Regression (v8):")
    table_lr = run_model("lr")
    table_lr.to_csv(OUT_DIR / "permutation_importance_linear_regression_v8.csv")
    plot_single(table_lr, "Lineare Regression", OUT_DIR / "feature_importance_linear_regression_v8.png")

    print("MLP (v8):")
    table_mlp = run_model("mlp")
    table_mlp.to_csv(OUT_DIR / "permutation_importance_mlp_v8.csv")
    plot_single(table_mlp, "MLP", OUT_DIR / "feature_importance_mlp_v8.png")

    plot_compare(table_lr, table_mlp, OUT_DIR / "feature_importance_vergleich.png")

    print(f"\nFertig. Ergebnisse in {OUT_DIR}/ gespeichert.")


if __name__ == "__main__":
    main()
