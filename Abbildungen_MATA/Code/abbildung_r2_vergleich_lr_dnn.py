"""
Dieses Skript erstellt ein gruppiertes Balkendiagramm, das den R²-Wert je
Datenreihe zwischen der linearen Regression (Baseline) und dem DNN (MLP)
gegenüberstellt. Die Metriken werden aus den gespeicherten CSV-Dateien der
beiden Modelle geladen. Die Grafik wird als PNG gespeichert.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------
LR_METRICS_PATH = os.path.join(
    "results", "model_results", "linear_regression_v8", "metrics", "all_metrics.csv"
)
DNN_METRICS_PATH = os.path.join(
    "results", "model_results", "mlp_v8", "metrics", "all_metrics.csv"
)

OUTPUT_DIR = os.path.join("Abbildungen_MATA", "Abbildungen")
OUTPUT_FILENAME = "abbildung_r2_vergleich_lr_dnn_v8.png"

# Reihenfolge der Datenreihen (station-Kennung -> Kurzbeschriftung)
STATION_ORDER = [
    ("050_Brunnen_Mositunnel_R1_v8", "Brunnen R1"),
    ("050_Brunnen_Mositunnel_R2_v8", "Brunnen R2"),
    ("171_Sattel_R1_v8", "Sattel R1"),
    ("171_Sattel_R2_v8", "Sattel R2"),
    ("216_Wangen_SZ_R1_v8", "Wangen R1"),
    ("216_Wangen_SZ_R2_v8", "Wangen R2"),
    ("299_Wollerau_Blatttunnel_R1_v8", "Wollerau R1"),
    ("299_Wollerau_Blatttunnel_R2_v8", "Wollerau R2"),
    ("720_Schwyz_R1_v8", "Schwyz R1"),
    ("720_Schwyz_R2_v8", "Schwyz R2"),
]

COLOR_LR = '#9ecae1'   # Lineare Regression (hellblau)
COLOR_DNN = '#3182bd'  # DNN (dunkelblau)

BAR_WIDTH = 0.4


def load_r2(csv_path):
    """
    Lädt die Metrik-CSV eines Modells und gibt ein Dictionary
    von station-Kennung auf R²-Wert zurück.
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df["station"], df["R2"]))


def generate_r2_comparison_plot():
    """
    Erstellt das gruppierte Balkendiagramm des R² je Datenreihe
    (lineare Regression gegen DNN) und speichert es als PNG-Datei.
    """
    r2_lr = load_r2(LR_METRICS_PATH)
    r2_dnn = load_r2(DNN_METRICS_PATH)

    labels = [short for _, short in STATION_ORDER]
    lr_values = [r2_lr[station] for station, _ in STATION_ORDER]
    dnn_values = [r2_dnn[station] for station, _ in STATION_ORDER]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 5))

    # Zwei Balken pro Datenreihe nebeneinander
    ax.bar(x - BAR_WIDTH / 2, lr_values, BAR_WIDTH,
           label="Lineare Regression", color=COLOR_LR)
    bars_dnn = ax.bar(x + BAR_WIDTH / 2, dnn_values, BAR_WIDTH,
                      label="DNN", color=COLOR_DNN)

    # DNN-Balken mit Wert beschriften
    ax.bar_label(bars_dnn, fmt="%.2f", padding=3, fontsize=8)

    # Achsen-Styling
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    # Legende ohne Rahmen
    ax.legend(frameon=False)

    # Helles y-Gitter im Hintergrund
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Obere und rechte Achsenlinien ausblenden
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Ordner erstellen und speichern
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    generate_r2_comparison_plot()
