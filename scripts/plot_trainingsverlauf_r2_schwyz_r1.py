"""
Dieses Skript plottet den Trainingsverlauf des MLP (v8) für die Reihe Schwyz R1
im Stil der übrigen Trainingsverlauf-Abbildungen (analog zu plot_trainingsverlauf.py
für v7): eine Kombination aus R²- und Verlustkurve mit zwei Y-Achsen. Blau zeigt das
Bestimmtheitsmass R² (0 bis 1), rot den Verlust (MSE), jeweils für Training (durchgezogen)
und Validierung (gepunktet).
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Pfade relativ zur Projektstruktur ableiten
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "model_results" / "mlp_v8"
HISTORY_PATH = RESULTS_DIR / "training_history" / "training_history_720_Schwyz_R1_v8.csv"
PLOT_DIR     = RESULTS_DIR / "plots" / "trainingsverlauf"
OUTPUT_PATH  = PLOT_DIR / "trainingsverlauf_r2_720_Schwyz_R1_v8.png"

# Titeltext der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Trainingsverlauf des DNN (Schwyz R1)"


def main():
    """Liest die Verlaufsdaten, erstellt die R²-/Verlust-Abbildung und speichert sie als PNG."""
    if not HISTORY_PATH.exists():
        print(f"FEHLER: Verlaufsdatei nicht gefunden: {HISTORY_PATH}")
        print("Hinweis: Zuerst 'python models/mlp_v8.py' ausfuehren, um die Verlaufsdaten zu erzeugen.")
        return

    df = pd.read_csv(HISTORY_PATH)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax_r2 = plt.subplots(figsize=(10, 6))
    ax_loss = ax_r2.twinx()

    # R²-Kurven (links, blau, als Anteil 0 bis 1 – konsistent mit dem Fliesstext)
    ax_r2.plot(df["epoch"], df["train_r2"],
               linestyle="-", color="blue", linewidth=1.8, label="Trainings-R²")
    ax_r2.plot(df["epoch"], df["val_r2"],
               linestyle=":", color="blue", linewidth=1.8, label="Validierungs-R²")
    ax_r2.set_ylim(0, 1)
    ax_r2.set_ylabel("Bestimmtheitsmass R²")

    # Verlustkurven (rechts, rot, MSE)
    ax_loss.plot(df["epoch"], df["train_mse"],
                 linestyle="-", color="red", linewidth=1.8, label="Trainings-Verlust")
    ax_loss.plot(df["epoch"], df["val_mse"],
                 linestyle=":", color="red", linewidth=1.8, label="Validierungs-Verlust")
    ax_loss.set_ylabel("Verlust (MSE)")

    ax_r2.set_xlabel("Epoche")
    ax_r2.set_xlim(left=0)
    ax_r2.set_title(TITLE, fontsize=13, fontweight="bold", pad=12)
    ax_r2.grid(True, linestyle="--", alpha=0.45)

    # Gemeinsame Legende fuer alle vier Linien, unten platziert
    handles_r2, labels_r2     = ax_r2.get_legend_handles_labels()
    handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
    ax_r2.legend(handles_r2 + handles_loss, labels_r2 + labels_loss,
                 loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=4, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot gespeichert: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
