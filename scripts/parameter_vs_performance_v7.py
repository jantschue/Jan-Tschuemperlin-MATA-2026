"""
Dieses Skript erstellt eine Parameter-vs-Performance-Kurve fuer EINE repraesentative
Station aus den v7-Daten (Corona-bereinigt + 26 kantonsspezifische Feiertagsspalten).
Es variiert ausschliesslich die Groesse des MLP (ueber einen gemeinsamen Multiplikator
auf die Schichtproportionen) und misst, wie sich Trainings-/Test-RMSE und Test-R2 mit
der Parameterzahl entwickeln. So laesst sich die fuer das v7-Modell gewaehlte
Architektur [128, 512, 128, 64] begruenden.

Die gesamte Trainingslogik (Optimizer, LR-Scheduler, Early Stopping, Bestes-Modell-
Wiederherstellung, Skalierung) sowie alle Hyperparameter werden 1:1 aus dem
v7-Trainingsskript uebernommen, damit die Kurve mit den dort berichteten Metriken
vergleichbar ist. Aendert sich mlp_v7.py (z.B. neue Hyperparameter), zieht diese
Analyse automatisch mit.

Ausfuehren aus dem Projekt-Stammverzeichnis:

    python scripts/parameter_vs_performance_v7.py

(Das Skript wechselt zur Sicherheit selbst ins Projekt-Stammverzeichnis, damit die
relativen Daten- und Resultatpfade aus dem Trainingsmodul korrekt aufgeloest werden.)
"""

import os
import sys
from pathlib import Path

# ── Projektpfade -------------------------------------------------------------
# Damit die relativen Pfade (DATA_DIR, RESULTS_DIR) aus dem Trainingsmodul
# zuverlaessig aufgeloest werden, wird unabhaengig vom Aufrufort ins
# Projekt-Stammverzeichnis gewechselt und der models/-Ordner auf den Importpfad
# gelegt.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "models"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Import aus `mlp_v7`: Damit verweisen DATA_DIR (data/v7), RESULTS_DIR
# (results/.../mlp_v7), DATASETS (_v7.csv) und saemtliche Hyperparameter
# tatsaechlich auf die v7-Welt. Der Import ist gefahrlos, weil das eigentliche
# Training dort nur unter dem __main__-Guard laeuft.
from mlp_v7 import (  # noqa: E402  (Import bewusst nach os.chdir/sys.path)
    MLP, VerkehrsDataset, FEATURES, DATASETS, DATA_DIR, RESULTS_DIR,
    DROPOUT, BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_PATIENCE, LR_FACTOR, ES_PATIENCE, TRAIN_RATIO, VAL_RATIO,
)

# ── Konfiguration ────────────────────────────────────────────────────────────
# Repraesentative Station fuer die Kurve. MUSS einer der Eintraege in DATASETS
# sein (die ..._v7.csv-Dateien). Wird beim Start validiert.
# Gewaehlt: dieselbe Station wie beim Optuna-Tuning (050 Brunnen Mositunnel R1),
# hier in der v7-Variante. Brunnen ist stabil – fuer eine saubere Begruendung der
# Architektur besser geeignet als z.B. Sattel (kleine, lueckenhafte Daten).
STATION = "050_Brunnen_Mositunnel_R1_v7.csv"

# Proportionen der gewaehlten v7-Architektur [128, 512, 128, 64] = 64 * [2, 8, 2, 1].
BASE_SHAPE = [2, 8, 2, 1]

# Multiplikatoren auf BASE_SHAPE. mult=64 reproduziert die gewaehlte Architektur exakt.
MULTIPLIERS = [2, 4, 8, 16, 32, 48, 64, 96, 128]

# Anzahl Seeds pro Punkt (Seeds 0..N_SEEDS-1). Fuer einen schnellen Test reduzieren.
N_SEEDS = 5

# Optionaler Test-RMSE der linearen Regression fuer diese Station. Wenn gesetzt,
# wird er als gestrichelte horizontale Baseline-Linie eingezeichnet.
LR_TEST_RMSE = None

# Multiplikator, der die gewaehlte Architektur markiert (offener roter Kreis im Plot).
HIGHLIGHT_MULT = 64

# Ausgabeverzeichnis im v7-Resultatbaum.
OUT_DIR = RESULTS_DIR / "analysen" / "parameter_vs_performance"

# Toleranz fuer den Sanity-Check (relativer RMSE-Abstand zum gespeicherten Wert).
SANITY_TOL = 0.10

# Device wie im Trainingsskript.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Datenvorbereitung (einmalig, seed-unabhaengig) ───────────────────────────
def prepare_data():
    """Laedt die Stationsdaten und bereitet sie exakt wie im v7-Training auf:
    NaN droppen, datetime als sortierten Index, weather_cat encoden, chronologischer
    70/10/20-Split ohne Shuffle, StandardScaler nur auf dem Train-Split gefittet.
    y_train und y_test werden zusaetzlich in Originaleinheiten zurueckgegeben."""
    df = pd.read_csv(DATA_DIR / STATION).dropna()

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)

    if "weather_cat" in df.columns and df["weather_cat"].dtype == "object":
        df["weather_cat"] = LabelEncoder().fit_transform(df["weather_cat"])

    feature_cols = [c for c in FEATURES if c in df.columns]
    X = df[feature_cols].values
    y = df["volume"].values.reshape(-1, 1)

    # Chronologischer 70/10/20-Split, kein Shuffle
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    # StandardScaler NUR auf den Trainingsdaten fitten
    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)
    # y_train, y_test bleiben in Originaleinheiten (Fahrzeuge/h) fuer die Metriken

    # GPU-Preload: gesamten Datensatz einmalig auf die GPU laden, um den DataLoader-Overhead zu vermeiden
    X_train_g = torch.tensor(X_train_s, dtype=torch.float32, device=device)
    y_train_g = torch.tensor(y_train_s, dtype=torch.float32, device=device)
    X_val_g = torch.tensor(X_val_s, dtype=torch.float32, device=device)
    y_val_g = torch.tensor(y_val_s, dtype=torch.float32, device=device)
    X_test_g = torch.tensor(X_test_s, dtype=torch.float32, device=device)

    return {
        "feature_cols": feature_cols,
        "X_train_s": X_train_s, "X_val_s": X_val_s, "X_test_s": X_test_s,
        "y_train_s": y_train_s, "y_val_s": y_val_s,
        "y_train": y_train, "y_test": y_test,
        "X_train_g": X_train_g, "y_train_g": y_train_g,
        "X_val_g": X_val_g, "y_val_g": y_val_g,
        "X_test_g": X_test_g,
        "y_scaler": y_scaler,
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
    }

def iterate_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    """Liefert aufeinanderfolgende Batches direkt aus bereits auf der GPU liegenden Tensoren."""
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        yield X[start:end], y[start:end]


# ── Training (exakt wie in mlp_v7) ───────────────────────────────────────────
def train_model(hidden_dims, seed, data):
    """Trainiert ein MLP mit gegebener Architektur und Seed und gibt das Modell mit
    den nach Validierungs-Loss besten Gewichten zurueck. Nutzt GPU-Preload statt DataLoader."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(len(data["feature_cols"]), hidden_dims, DROPOUT).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
    )

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        for X_batch, y_batch in iterate_batches(data["X_train_g"], data["y_train_g"], BATCH_SIZE):
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.inference_mode():
            epoch_val_loss = 0.0
            for X_batch, y_batch in iterate_batches(data["X_val_g"], data["y_val_g"], BATCH_SIZE):
                epoch_val_loss += loss_fn(model(X_batch), y_batch).item() * len(X_batch)
            epoch_val_loss /= data["n_val"]

        scheduler.step(epoch_val_loss)

        # Early Stopping: bestes Modell anhand Validation Loss merken
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                break

    # Bestes Modell laden (nicht letztes)
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    return model


def predict_original(model, X_scaled_g, y_scaler):
    """Vorhersage in Originaleinheiten (Fahrzeuge/h) mit Tensoren, die schon auf der GPU sind."""
    model.eval()
    with torch.inference_mode():
        preds_scaled = model(X_scaled_g).cpu().numpy()
    return y_scaler.inverse_transform(preds_scaled)


def count_parameters(n_features, hidden_dims):
    """Anzahl trainierbarer Parameter fuer eine Architektur (seed-unabhaengig)."""
    model = MLP(n_features, hidden_dims, DROPOUT)
    return sum(p.numel() for p in model.parameters())


# ── Sanity-Check vor dem vollen Sweep ────────────────────────────────────────
def sanity_check(data):
    """Trainiert die gewaehlte Architektur (mult=HIGHLIGHT_MULT) mit Seed 0 und
    vergleicht den Test-RMSE mit dem im v7-Resultatbaum gespeicherten Wert. Weicht
    er um mehr als SANITY_TOL ab, wird abgebrochen – dann stimmt die Replikation der
    Pipeline nicht und ein Sweep waere wertlos."""
    hidden = [b * HIGHLIGHT_MULT for b in BASE_SHAPE]
    model = train_model(hidden, 0, data)
    preds_test = predict_original(model, data["X_test_g"], data["y_scaler"])
    rmse = float(np.sqrt(mean_squared_error(data["y_test"], preds_test)))

    metrics_csv = RESULTS_DIR / "metrics" / "all_metrics.csv"
    stored = pd.read_csv(metrics_csv)
    station_key = STATION.replace(".csv", "")
    row = stored.loc[stored["station"] == station_key]
    if row.empty:
        raise SystemExit(
            f"Sanity-Check fehlgeschlagen: Station '{station_key}' nicht in {metrics_csv} gefunden."
        )
    stored_rmse = float(row["RMSE"].iloc[0])

    rel = abs(rmse - stored_rmse) / stored_rmse
    print(f"  Sanity-Check  mult={HIGHLIGHT_MULT}, seed 0")
    print(f"    Test-RMSE (repliziert) : {rmse:.2f} Fahrzeuge/h")
    print(f"    Test-RMSE (gespeichert): {stored_rmse:.2f} Fahrzeuge/h")
    print(f"    Relative Abweichung    : {rel * 100:.1f} %")

    if rel > SANITY_TOL:
        raise SystemExit(
            f"Sanity-Check fehlgeschlagen: Abweichung {rel * 100:.1f} % > "
            f"{SANITY_TOL * 100:.0f} %. Die Pipeline-Replikation stimmt nicht – "
            f"Sweep wird NICHT gestartet."
        )
    print("    -> OK, Replikation innerhalb der Toleranz.\n")


# ── Sweep ────────────────────────────────────────────────────────────────────
def run_sweep(data):
    """Trainiert fuer jeden Multiplikator N_SEEDS Modelle und aggregiert Train-/Test-
    RMSE und Test-R2 (Mittelwert und Standardabweichung) in Originaleinheiten."""
    n_features = len(data["feature_cols"])
    rows = []

    for mult in MULTIPLIERS:
        hidden = [b * mult for b in BASE_SHAPE]
        params = count_parameters(n_features, hidden)

        train_rmses, test_rmses, test_r2s = [], [], []
        for seed in range(N_SEEDS):
            model = train_model(hidden, seed, data)
            preds_train = predict_original(model, data["X_train_g"], data["y_scaler"])
            preds_test = predict_original(model, data["X_test_g"], data["y_scaler"])
            train_rmses.append(np.sqrt(mean_squared_error(data["y_train"], preds_train)))
            test_rmses.append(np.sqrt(mean_squared_error(data["y_test"], preds_test)))
            test_r2s.append(r2_score(data["y_test"], preds_test))

        row = {
            "multiplier": mult,
            "hidden_dims": str(hidden),
            "params": params,
            "train_rmse_mean": float(np.mean(train_rmses)),
            "train_rmse_std": float(np.std(train_rmses)),
            "test_rmse_mean": float(np.mean(test_rmses)),
            "test_rmse_std": float(np.std(test_rmses)),
            "test_r2_mean": float(np.mean(test_r2s)),
            "test_r2_std": float(np.std(test_r2s)),
        }
        rows.append(row)
        print(
            f"  mult={mult:>3d} | hidden={str(hidden):<22} | params={params:>9,} | "
            f"Test-RMSE {row['test_rmse_mean']:7.2f} +/- {row['test_rmse_std']:5.2f}"
        )

    return pd.DataFrame(rows).sort_values("params").reset_index(drop=True)


# ── Plots ────────────────────────────────────────────────────────────────────
def _highlight_row(df):
    """Zeile der gewaehlten Architektur (mult=HIGHLIGHT_MULT), falls vorhanden."""
    match = df.loc[df["multiplier"] == HIGHLIGHT_MULT]
    return match.iloc[0] if not match.empty else None


def plot_rmse_curve(df, station_key):
    """RMSE-Kurve: Parameterzahl (log) gegen Train-/Test-RMSE mit Test-Streuband,
    Markierung der gewaehlten Architektur und optionaler LR-Baseline."""
    x = df["params"].values
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, df["train_rmse_mean"], marker="o", color="#4861c9", label="Train-RMSE (Ø)")
    ax.plot(x, df["test_rmse_mean"], marker="o", color="#c14b3c", label="Test-RMSE (Ø)")
    ax.fill_between(
        x,
        df["test_rmse_mean"] - df["test_rmse_std"],
        df["test_rmse_mean"] + df["test_rmse_std"],
        color="#c14b3c", alpha=0.18, label="Test-RMSE ± 1 Std",
    )

    if LR_TEST_RMSE is not None:
        ax.axhline(
            LR_TEST_RMSE, linestyle="--", color="#6b6b6b", linewidth=1.5,
            label="Linear Regression (Baseline)",
        )

    hl = _highlight_row(df)
    if hl is not None:
        ax.scatter(
            [hl["params"]], [hl["test_rmse_mean"]], s=170, facecolors="none",
            edgecolors="red", linewidths=2.2, zorder=5,
            label=f"Gewählte Architektur (mult={HIGHLIGHT_MULT})",
        )
        ax.annotate(
            f"{int(hl['params']):,} Parameter",
            (hl["params"], hl["test_rmse_mean"]),
            textcoords="offset points", xytext=(10, 12),
            color="red", fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Anzahl Modellparameter (logarithmisch)")
    ax.set_ylabel("RMSE (Fahrzeuge/h)")
    ax.set_title(f"Parameter vs. Performance (RMSE) – {station_key}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / f"rmse_curve_{station_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_r2_curve(df, station_key):
    """R2-Kurve: Parameterzahl (log) gegen Test-R2 mit Streuband und Markierung der
    gewaehlten Architektur."""
    x = df["params"].values
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, df["test_r2_mean"], marker="o", color="#3c8553", label="Test-R² (Ø)")
    ax.fill_between(
        x,
        df["test_r2_mean"] - df["test_r2_std"],
        df["test_r2_mean"] + df["test_r2_std"],
        color="#3c8553", alpha=0.18, label="Test-R² ± 1 Std",
    )

    hl = _highlight_row(df)
    if hl is not None:
        ax.scatter(
            [hl["params"]], [hl["test_r2_mean"]], s=170, facecolors="none",
            edgecolors="red", linewidths=2.2, zorder=5,
            label=f"Gewählte Architektur (mult={HIGHLIGHT_MULT})",
        )
        ax.annotate(
            f"{int(hl['params']):,} Parameter",
            (hl["params"], hl["test_r2_mean"]),
            textcoords="offset points", xytext=(10, -16),
            color="red", fontsize=9,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Anzahl Modellparameter (logarithmisch)")
    ax.set_ylabel("Test-R²")
    ax.set_title(f"Parameter vs. Performance (R²) – {station_key}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / f"r2_curve_{station_key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ── Ablauf ───────────────────────────────────────────────────────────────────
def main():
    # STATION validieren
    if STATION not in DATASETS:
        raise SystemExit(
            "STATION ist ungueltig. Bitte oben im Skript eine der _v7.csv "
            f"aus DATASETS setzen.\nAktuell: STATION = {STATION!r}\n"
            f"Gueltige Werte:\n  " + "\n  ".join(DATASETS)
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    station_key = STATION.replace(".csv", "")

    print(f"Station            : {station_key}")
    print(f"Device             : {device}")
    print(f"Multiplikatoren    : {MULTIPLIERS}")
    print(f"Seeds pro Punkt    : {N_SEEDS}")
    print(f"Ausgabeverzeichnis : {OUT_DIR}\n")

    data = prepare_data()
    print(
        f"Features ({len(data['feature_cols'])}): {data['feature_cols']}\n"
        f"Split: Train {data['n_train']} | Val {data['n_val']} | Test {data['n_test']}\n"
    )

    print("=== Sanity-Check (Pipeline-Replikation) ===")
    sanity_check(data)

    print("=== Sweep ===")
    df = run_sweep(data)

    # Aggregierte Werte als CSV speichern
    csv_path = OUT_DIR / f"parameter_vs_performance_{station_key}.csv"
    df.to_csv(csv_path, index=False)

    # Plots erzeugen
    rmse_path = plot_rmse_curve(df, station_key)
    r2_path = plot_r2_curve(df, station_key)

    print("\nGespeichert:")
    print(f"  {csv_path}")
    print(f"  {rmse_path}")
    print(f"  {r2_path}")


if __name__ == "__main__":
    main()
