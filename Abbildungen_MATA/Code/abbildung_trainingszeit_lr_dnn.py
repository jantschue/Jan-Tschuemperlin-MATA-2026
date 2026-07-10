"""
Dieses Skript erstellt ein gruppiertes Balkendiagramm der Trainingszeit je
Datenreihe und stellt die lineare Regression (Baseline) dem DNN (MLP) gegenüber.
Die Zeiten werden aus der Spalte "train_seconds" der beiden Metrik-CSV-Dateien
gelesen. Die y-Achse ist logarithmisch skaliert, da die ~0,2 s der linearen
Regression neben den ~15–40 s des DNN sonst unsichtbar wären.
Die Grafik dient dem Kapitel 4.6 (Ressourcenverbrauch) und wird als PNG gespeichert.
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
OUTPUT_FILENAME = "abbildung_trainingszeit_lr_dnn_v8.png"

# Titeltext der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Trainingszeit von linearer Regression und DNN je Datenreihe"

TIME_COLUMN = "train_seconds"

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

# Grenzen der logarithmischen y-Achse (Sekunden)
Y_MIN = 0.1
Y_MAX = 100.0


def load_train_seconds(csv_path):
    """
    Lädt die Metrik-CSV eines Modells und gibt ein Dictionary
    von station-Kennung auf Trainingszeit (Sekunden) zurück.
    """
    df = pd.read_csv(csv_path)
    return dict(zip(df["station"], df[TIME_COLUMN]))


def generate_training_time_plot():
    """
    Erstellt das gruppierte Balkendiagramm der Trainingszeit je Datenreihe
    (lineare Regression gegen DNN, logarithmische y-Achse) und speichert es
    als PNG-Datei.
    """
    lr_times = load_train_seconds(LR_METRICS_PATH)
    dnn_times = load_train_seconds(DNN_METRICS_PATH)

    labels = [short for _, short in STATION_ORDER]
    lr_values = [lr_times[station] for station, _ in STATION_ORDER]
    dnn_values = [dnn_times[station] for station, _ in STATION_ORDER]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 5))

    # Logarithmische y-Achse (vor dem Zeichnen setzen, damit die Balken
    # sauber von der unteren Achsengrenze aus gezeichnet werden)
    ax.set_yscale("log")
    ax.set_ylim(Y_MIN, Y_MAX)

    # Zwei Balken pro Datenreihe nebeneinander
    bars_lr = ax.bar(x - BAR_WIDTH / 2, lr_values, BAR_WIDTH,
                     label="Lineare Regression", color=COLOR_LR)
    bars_dnn = ax.bar(x + BAR_WIDTH / 2, dnn_values, BAR_WIDTH,
                      label="DNN", color=COLOR_DNN)

    # Beide Balken mit ihrem Sekundenwert beschriften – auf der Log-Achse sind
    # die kurzen LR-Balken sonst kaum lesbar.
    ax.bar_label(bars_lr, fmt="%.1f", padding=3, fontsize=7)
    ax.bar_label(bars_dnn, fmt="%.1f", padding=3, fontsize=7)

    # Achsen-Styling
    ax.set_ylabel("Trainingszeit [s]")
    ax.set_title(TITLE, fontsize=13, fontweight="bold", pad=12)
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
    generate_training_time_plot()
