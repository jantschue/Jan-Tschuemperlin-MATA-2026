"""
Dieses Skript erzeugt ein aufgeraeumtes, thesenreifes R2-Balkendiagramm der
linearen Regression (v8) fuer Kapitel 4.1. Es benoetigt kein PyTorch, sondern
liest ausschliesslich die bereits berechnete Metrik-Datei
results/model_results/linear_regression_v8/metrics/all_metrics.csv und stellt
pro Datenreihe das Bestimmtheitsmass R2 dar. Die x-Achse zeigt lesbare
Stationsnamen statt der Dateinamen, die Balkenwerte werden in Schweizer
Schreibweise (Komma) beschriftet.

Ergebnis:
    results/model_results/linear_regression_v8/summary/r2_summary_clean.png
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Konfiguration ────────────────────────────────────────────────────────────
METRICS_CSV = Path("results/model_results/linear_regression_v8/metrics/all_metrics.csv")
OUT_PATH    = Path("results/model_results/linear_regression_v8/summary/r2_summary_clean.png")

BAR_COLOR      = "#4C72B0"   # neutrale Einheitsfarbe (kein Gruen)
Y_MIN, Y_MAX   = 0.0, 0.8    # y-Achse startet bei 0
LABEL_ROTATION = 30          # Drehung der x-Achsenbeschriftungen in Grad
BAR_DECIMALS   = 2           # Nachkommastellen der Balkenbeschriftung

# Lesbare Bezeichnungen statt der Dateinamen (Suffix _v8 und Nummern entfernt)
STATION_LABELS = {
    "050_Brunnen_Mositunnel_R1":  "Brunnen R1",
    "050_Brunnen_Mositunnel_R2":  "Brunnen R2",
    "171_Sattel_R1":              "Sattel R1",
    "171_Sattel_R2":              "Sattel R2",
    "216_Wangen_SZ_R1":           "Wangen R1",
    "216_Wangen_SZ_R2":           "Wangen R2",
    "299_Wollerau_Blatttunnel_R1":"Wollerau R1",
    "299_Wollerau_Blatttunnel_R2":"Wollerau R2",
    "720_Schwyz_R1":              "Schwyz R1",
    "720_Schwyz_R2":              "Schwyz R2",
}


def pretty_label(station: str) -> str:
    """Wandelt einen Stationsnamen (mit _v8-Suffix) in eine lesbare Bezeichnung um."""
    key = station.replace("_v8", "")
    return STATION_LABELS.get(key, key)


def swiss_number(value: float) -> str:
    """Formatiert eine Zahl mit zwei Nachkommastellen in Schweizer Schreibweise (Komma)."""
    return f"{value:.{BAR_DECIMALS}f}".replace(".", ",")


def main():
    """Liest die Metrik-CSV und speichert das aufgeraeumte R2-Balkendiagramm."""
    df = pd.read_csv(METRICS_CSV)
    labels = [pretty_label(s) for s in df["station"]]
    values = df["R2"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=BAR_COLOR)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylabel("R² (Bestimmtheitsmass)")
    ax.set_xlabel("Datenreihe")
    plt.setp(ax.get_xticklabels(), rotation=LABEL_ROTATION, ha="right")

    # Balkenwerte im Schweizer Zahlenformat ueber die Balken schreiben
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, swiss_number(value),
                ha="center", va="bottom", fontsize=9)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
