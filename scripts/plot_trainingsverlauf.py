"""
Dieses Skript liest die pro-Epoche gespeicherten Trainingsverlaeufe aller MLP-v7-
Stationen ein und erstellt je Station eine Kombination aus R²- und Verlustkurve
(im Stil von Abbildung 13 der Maturaarbeit) als hochaufloesende PNG-Abbildung.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Pfade aus Projektstruktur ableiten
SCRIPT_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT   = SCRIPT_DIR.parent
RESULTS_DIR    = PROJECT_ROOT / "results" / "model_results" / "mlp_v7"
HISTORY_DIR    = RESULTS_DIR / "training_history"
PLOT_DIR       = RESULTS_DIR / "plots" / "trainingsverlauf"


def lade_historien():
    """Laedt alle training_history_*.csv-Dateien aus dem Ergebnisordner."""
    csv_dateien = sorted(HISTORY_DIR.glob("training_history_*.csv"))
    if not csv_dateien:
        print(f"FEHLER: Keine 'training_history_*.csv'-Dateien in {HISTORY_DIR} gefunden.")
        print("Hinweis: Zuerst 'python models/mlp_v7.py' ausfuehren, um die Verlaufsdaten zu erzeugen.")
        sys.exit(1)

    historien = []
    for pfad in csv_dateien:
        df = pd.read_csv(pfad)
        station = pfad.stem.replace("training_history_", "")
        historien.append((station, df))
    return historien


def validierungs_ausgabe(station, df):
    """Gibt Anzahl Epochen sowie erste und letzte 3 Zeilen der Station aus."""
    print(f"\n{'=' * 60}")
    print(f"Validierung: {station}")
    print(f"{'=' * 60}")
    print(f"Epochen gesamt: {len(df)}")
    print("Erste 3 Zeilen:")
    print(df.head(3).to_string(index=False))
    print("Letzte 3 Zeilen:")
    print(df.tail(3).to_string(index=False))


def plot_station(station, df, index):
    """Erstellt die R²/Verlust-Kombinationsgrafik fuer eine Station."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / f"abbildung_trainingsverlauf_v7_{station}.png"

    fig, ax_r2 = plt.subplots(figsize=(10, 6))
    ax_loss = ax_r2.twinx()

    # R²-Kurven (links, blau, in Prozent)
    ax_r2.plot(df["epoch"], df["train_r2"] * 100,
               linestyle="-", color="blue", linewidth=1.8, label="Trainings-R²")
    ax_r2.plot(df["epoch"], df["val_r2"] * 100,
               linestyle=":", color="blue", linewidth=1.8, label="Validierungs-R²")
    ax_r2.set_ylim(0, 100)
    ax_r2.set_ylabel("Bestimmtheitsmass R² (%)")

    # Verlustkurven (rechts, rot, MSE)
    ax_loss.plot(df["epoch"], df["train_mse"],
                 linestyle="-", color="red", linewidth=1.8, label="Trainings-Verlust")
    ax_loss.plot(df["epoch"], df["val_mse"],
                 linestyle=":", color="red", linewidth=1.8, label="Validierungs-Verlust")
    ax_loss.set_ylabel("Verlust (MSE)")

    ax_r2.set_xlabel("Epoche")
    ax_r2.set_xlim(left=0)
    ax_r2.set_title(f"Trainingsverlauf MLP v7 – {station}", fontsize=14)
    ax_r2.grid(True, linestyle="--", alpha=0.45)

    # Gemeinsame Legende fuer alle vier Linien, unten platziert um Kurven nicht zu ueberdecken
    handles_r2, labels_r2     = ax_r2.get_legend_handles_labels()
    handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
    ax_r2.legend(handles_r2 + handles_loss, labels_r2 + labels_loss,
                 loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=4, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot gespeichert: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Bildunterschrift: Abbildung {index}: Kombination der R²- und Verlustkurven "
          f"des MLP v7-Modells (Station {station})")


def main():
    historien = lade_historien()

    for station, df in historien:
        validierungs_ausgabe(station, df)

    for i, (station, df) in enumerate(historien, start=1):
        plot_station(station, df, i)


if __name__ == "__main__":
    main()
