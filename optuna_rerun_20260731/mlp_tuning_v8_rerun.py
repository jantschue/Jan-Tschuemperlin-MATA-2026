"""
Isolierter Re-Run des Hyperparameter-Tunings für das MLP-v8-Modell mittels Optuna.

Dieses Skript ist eine 1:1-Kopie von models/mlp_tuning_v8.py – es wurde AUSSCHLIESSLICH
in den Pfaden angepasst, sodass alle Ausgaben ausschliesslich in diesem isolierten
Ordner (optuna_rerun_20260731/) landen und keine bestehende Datei überschrieben wird.
Tuning-Logik, Sampler (seed=None), N_TRIALS (150), TUNING_EPOCHS (150) und sämtliche
Suchräume sind unverändert.

ZWECK dieses Re-Runs: ausschliesslich die Dokumentation der Tuning-DAUER und der
Ressourcen-AUSLASTUNG (CPU/RAM/GPU). Da der Sampler seed=None nutzt, ist die Suche
nicht-deterministisch – die vom Re-Run gefundene "beste" Architektur weicht daher
erwartungsgemäss ab und ist NUR ein Nebenprodukt. Die für die Arbeit gültige
Architektur bleibt die ursprüngliche aus
results/model_results/mlp_v8/best_params_v8.json ([512, 256, 256, 128]) und wird
durch diesen Re-Run NICHT ersetzt (siehe HINWEIS.txt).

Zusätzlich wird die reine Wanduhr-Tuning-Dauer dokumentiert:
- in die Datei tuning_dauer.txt (Sekunden und h:min:s),
- und in die Ergebnis-JSON (Felder "tuning_seconds" und "tuning_hms").

GPU-Preload: Der gesamte (kleine) Datensatz wird einmalig auf die GPU geladen,
statt jeden Batch pro Epoche einzeln von der CPU zu transferieren. Das spart den
DataLoader- und Transfer-Overhead und erhöht die GPU-Auslastung. Die Batch-Grenzen
bleiben identisch (kein Shuffle, gleiche batch_size-Slices), die Resultate damit gleich.
"""

import csv
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Konfiguration (nur Pfade gegenüber dem Original angepasst) ────────────────
# Alle Ein- und Ausgaben liegen relativ zu diesem Skript im isolierten Ordner.
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR            # lokale Kopie der CSV im isolierten Ordner
RESULTS_DIR = BASE_DIR            # sämtliche Ausgaben landen hier

TUNING_DATASET = "050_Brunnen_Mositunnel_R1_v8.csv"
N_TRIALS       = 150
TUNING_EPOCHS  = 150  # Reduziert gegenüber Final-Training für schnellere Trials
ES_PATIENCE    = 25

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10

# Abtastintervall der Ressourcen-Überwachung (Sekunden)
MONITOR_INTERVAL = 2.0

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


def _query_gpu():
    """Liest GPU-Auslastung (%) und Speicher (belegt/total, MiB) via nvidia-smi aus.
    Gibt (gpu_util, gpu_mem_used, gpu_mem_total) der ersten GPU zurück oder
    (None, None, None), falls nvidia-smi nicht verfügbar ist."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        first = out.stdout.strip().splitlines()[0]
        util, mem_used, mem_total = (float(v) for v in first.split(","))
        return util, mem_used, mem_total
    except Exception:
        return None, None, None


class ResourceMonitor:
    """Protokolliert in einem Hintergrund-Thread periodisch CPU-, RAM- und
    GPU-Auslastung und schreibt jede Messung fortlaufend in eine CSV-Datei.
    Am Ende liefert summary() die Kennzahlen (Ø / Max / Min) je Ressource."""

    FIELDS = [
        "timestamp", "elapsed_s",
        "cpu_percent", "ram_percent", "ram_used_gb",
        "gpu_util_percent", "gpu_mem_used_mb", "gpu_mem_total_mb",
    ]

    def __init__(self, csv_path: Path, interval: float = MONITOR_INTERVAL):
        self.csv_path = csv_path
        self.interval = interval
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._samples = []
        self._start   = None

    def _run(self):
        # CPU-Messung einmal "anwerfen" (erster Aufruf liefert sonst 0.0)
        psutil.cpu_percent(interval=None)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            while not self._stop.is_set():
                cpu = psutil.cpu_percent(interval=None)
                vm  = psutil.virtual_memory()
                gpu_util, gpu_used, gpu_total = _query_gpu()
                row = {
                    "timestamp":        datetime.now().isoformat(timespec="seconds"),
                    "elapsed_s":        round(time.time() - self._start, 1),
                    "cpu_percent":      round(cpu, 1),
                    "ram_percent":      round(vm.percent, 1),
                    "ram_used_gb":      round(vm.used / 1024**3, 2),
                    "gpu_util_percent": gpu_util,
                    "gpu_mem_used_mb":  gpu_used,
                    "gpu_mem_total_mb": gpu_total,
                }
                writer.writerow(row)
                f.flush()
                self._samples.append(row)
                self._stop.wait(self.interval)

    def start(self):
        self._start = time.time()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=self.interval + 5)

    def summary(self) -> dict:
        """Berechnet Ø / Max / Min je Kennzahl über alle Messungen."""
        def stats(key):
            vals = [s[key] for s in self._samples if s[key] is not None]
            if not vals:
                return None
            return {"avg": round(sum(vals) / len(vals), 1),
                    "max": round(max(vals), 1),
                    "min": round(min(vals), 1)}
        return {
            "n_samples":        len(self._samples),
            "interval_seconds": self.interval,
            "cpu_percent":      stats("cpu_percent"),
            "ram_percent":      stats("ram_percent"),
            "ram_used_gb":      stats("ram_used_gb"),
            "gpu_util_percent": stats("gpu_util_percent"),
            "gpu_mem_used_mb":  stats("gpu_mem_used_mb"),
        }


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
    print(f"Optuna Hyperparameter-Tuning (isolierter Re-Run) – Device: {device}")
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

    # Optuna-Studie samt SQLite-Datenbank im ISOLIERTEN Ordner ablegen (neuer DB-Name),
    # damit keine bestehende optuna_mlp_v8.db berührt wird.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = RESULTS_DIR / "optuna_rerun.db"
    # Sicherung: jeder Lauf startet frisch. Eine bereits vorhandene DB aus diesem
    # isolierten Ordner wird entfernt, damit die Studie NICHT fortgesetzt wird
    # (sonst würden sich Trial-Zahlen über mehrere Läufe aufsummieren, während die
    # gemessene Dauer nur den aktuellen Lauf abdeckt -> inkonsistente Doku).
    if db_path.exists():
        db_path.unlink()
        print("Hinweis: vorhandene optuna_rerun.db entfernt – Lauf startet frisch bei 0 Trials.")
    storage_url = f"sqlite:///{db_path.as_posix()}"
    study = optuna.create_study(
        study_name="mlp_v8_tuning",
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=None),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )

    def wrapped_objective(trial):
        return objective(trial, X_train_g, y_train_g, X_val_g, y_val_g, input_dim)

    # Ressourcen-Überwachung (CPU/RAM/GPU) im Hintergrund starten
    monitor = ResourceMonitor(RESULTS_DIR / "resource_usage.csv")
    print(f"Ressourcen-Überwachung aktiv (alle {MONITOR_INTERVAL:.0f}s -> resource_usage.csv)")

    print("Starte Optimierung...")
    monitor.start()
    start_time = time.time()
    try:
        study.optimize(wrapped_objective, n_trials=N_TRIALS, show_progress_bar=True)
    finally:
        duration = time.time() - start_time
        monitor.stop()
    resource_summary = monitor.summary()

    # Wanduhr-Dauer als h:min:s formatieren
    total_seconds = int(round(duration))
    hours, rem    = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    duration_hms  = f"{hours}h {minutes}min {secs}s"

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\nTuning abgeschlossen in {duration:.1f}s ({duration_hms})")
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
    print("Beste Hyperparameter (nur zur Dokumentation dieses Re-Runs):\n")
    print(f"  HIDDEN_DIMS   = {hidden_dims}")
    print(f"  DROPOUT       = {dropout:.2f}")
    print(f"  LEARNING_RATE = {learning_rate:.4f}")
    print(f"  WEIGHT_DECAY  = {weight_decay:.4f}")
    print(f"  BATCH_SIZE    = {batch_size}")
    print(sep)

    # Beste Hyperparameter als JSON speichern (gleiche Struktur wie best_params_v8.json,
    # zusätzlich Dauer-Felder).
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
        "tuning_seconds": round(duration, 1),
        "tuning_hms":     duration_hms,
        "resource_usage": resource_summary,
    }
    best_params_path = RESULTS_DIR / "best_params_v8_rerun.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"\nGespeichert: {best_params_path}")

    # Reine Dauer zusätzlich in tuning_dauer.txt dokumentieren
    dauer_path = RESULTS_DIR / "tuning_dauer.txt"
    with open(dauer_path, "w", encoding="utf-8") as f:
        f.write(f"Tuning-Dauer (Wanduhr): {duration:.1f} s\n")
        f.write(f"Tuning-Dauer (h:min:s): {duration_hms}\n")
        f.write(f"Abgeschlossene Trials: {len(completed)}\n")
        f.write(f"Geprunte Trials: {len(pruned)}\n")
    print(f"Gespeichert: {dauer_path}")

    # Ressourcen-Auslastung (CPU/RAM/GPU) menschenlesbar zusammenfassen
    res_path = RESULTS_DIR / "resource_summary.txt"

    def fmt(stat, unit):
        if stat is None:
            return "keine Daten"
        return f"Ø {stat['avg']}{unit}  |  Max {stat['max']}{unit}  |  Min {stat['min']}{unit}"

    rs = resource_summary
    with open(res_path, "w", encoding="utf-8") as f:
        f.write("Ressourcen-Auslastung während des Tunings\n")
        f.write(f"(Abtastung alle {rs['interval_seconds']:.0f}s, {rs['n_samples']} Messungen)\n\n")
        f.write(f"CPU-Auslastung : {fmt(rs['cpu_percent'], ' %')}\n")
        f.write(f"RAM-Auslastung : {fmt(rs['ram_percent'], ' %')}\n")
        f.write(f"RAM belegt     : {fmt(rs['ram_used_gb'], ' GB')}\n")
        f.write(f"GPU-Auslastung : {fmt(rs['gpu_util_percent'], ' %')}\n")
        f.write(f"GPU-Speicher   : {fmt(rs['gpu_mem_used_mb'], ' MiB')}\n")
        f.write("\nRohdaten je Messung: resource_usage.csv\n")
    print(f"Gespeichert: {res_path}")


if __name__ == "__main__":
    main()
