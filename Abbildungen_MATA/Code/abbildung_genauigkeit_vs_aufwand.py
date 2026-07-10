"""
Dieses Skript erstellt das Streudiagramm "Genauigkeit gegen Rechenaufwand" für
Kapitel 4.6 (Ressourcenverbrauch). Anders als die frühere Zwei-Punkte-Version zeigt
es je einen Punkt pro Datenreihe und Modell: x = Trainingszeit dieser Reihe in
Sekunden (logarithmisch), y = Test-R² dieser Reihe. Lineare Regression und DNN
bilden zwei getrennte Punktwolken; zusätzlich wird je Modell ein grösserer
Mittelwert-Punkt eingezeichnet. Datenreihen ohne gültiges R² (z.B. Schwyz R2)
werden übersprungen. Gelesen werden ausschliesslich die vorhandenen Metrik-CSVs
(Spalten R2 und train_seconds); es wird nicht neu trainiert.
Die Grafik wird als PNG gespeichert.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

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
OUTPUT_FILENAME = "abbildung_genauigkeit_vs_aufwand_v8.png"

# Titeltext der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Genauigkeit gegen Rechenaufwand beider Modelle"

R2_COLUMN = "R2"
TIME_COLUMN = "train_seconds"

# Reihenfolge der Datenreihen (station-Kennung); dient nur der konsistenten
# Sortierung, gezeichnet werden alle Reihen mit gültigem R².
STATION_ORDER = [
    "050_Brunnen_Mositunnel_R1_v8", "050_Brunnen_Mositunnel_R2_v8",
    "171_Sattel_R1_v8", "171_Sattel_R2_v8",
    "216_Wangen_SZ_R1_v8", "216_Wangen_SZ_R2_v8",
    "299_Wollerau_Blatttunnel_R1_v8", "299_Wollerau_Blatttunnel_R2_v8",
    "720_Schwyz_R1_v8", "720_Schwyz_R2_v8",
]

COLOR_LR = '#9ecae1'   # Lineare Regression (hellblau)
COLOR_DNN = '#3182bd'  # DNN (dunkelblau)
COLOR_LR_MEAN = '#4292c6'   # dunkleres Blau für den LR-Mittelwert-Punkt
COLOR_DNN_MEAN = '#08519c'  # dunkleres Blau für den DNN-Mittelwert-Punkt

POINT_SIZE = 55        # Einzelpunkte (klein)
MEAN_SIZE = 220        # Mittelwert-Punkte (gross)
Y_MIN = 0.5
Y_MAX = 1.0


def swiss_decimal_formatter(decimals):
    """
    Gibt einen FuncFormatter zurück, der Achsenbeschriftungen in Schweizer
    Schreibweise mit Dezimal-Komma formatiert (z.B. 0,80 statt 0.80).
    """
    return FuncFormatter(
        lambda value, _pos: f"{value:.{decimals}f}".replace(".", ",")
    )


def load_points(csv_path):
    """
    Lädt eine Metrik-CSV und gibt zwei Arrays (Trainingszeit, Test-R²) für alle
    Datenreihen mit gültigem R² zurück – in der Reihenfolge von STATION_ORDER.
    Reihen mit fehlendem (NaN) R² werden übersprungen.
    """
    df = pd.read_csv(csv_path).set_index("station")
    times, r2s = [], []
    for station in STATION_ORDER:
        if station not in df.index:
            continue
        r2 = df.loc[station, R2_COLUMN]
        if pd.isna(r2):
            continue
        times.append(df.loc[station, TIME_COLUMN])
        r2s.append(r2)
    return np.array(times), np.array(r2s)


def generate_accuracy_vs_effort_plot():
    """
    Zeichnet das Streudiagramm Trainingszeit (x, logarithmisch) gegen Test-R² (y)
    mit je einer Punktwolke pro Modell und den beiden Mittelwert-Punkten und
    speichert es als PNG. Gibt die verwendeten Punkte und Mittelwerte zurück.
    """
    lr_times, lr_r2 = load_points(LR_METRICS_PATH)
    dnn_times, dnn_r2 = load_points(DNN_METRICS_PATH)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Einzelpunkte je Datenreihe (klein, dünner weisser Rand)
    ax.scatter(lr_times, lr_r2, s=POINT_SIZE, color=COLOR_LR,
               edgecolor="white", linewidth=0.6, zorder=3,
               label="Lineare Regression")
    ax.scatter(dnn_times, dnn_r2, s=POINT_SIZE, color=COLOR_DNN,
               edgecolor="white", linewidth=0.6, zorder=3,
               label="DNN")

    # Mittelwert-Punkte je Modell (gross, dunkler Rand)
    ax.scatter(lr_times.mean(), lr_r2.mean(), s=MEAN_SIZE, color=COLOR_LR,
               edgecolor=COLOR_DNN_MEAN, linewidth=1.5, zorder=4,
               label="Mittelwert lineare Regression")
    ax.scatter(dnn_times.mean(), dnn_r2.mean(), s=MEAN_SIZE, color=COLOR_DNN,
               edgecolor=COLOR_DNN_MEAN, linewidth=1.5, zorder=4,
               label="Mittelwert DNN")

    # Achsen-Styling
    ax.set_xscale("log")
    ax.set_xlabel("Trainingszeit (s, logarithmisch)")
    ax.set_ylabel("Test-R²")
    ax.set_title(TITLE, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.yaxis.set_major_formatter(swiss_decimal_formatter(1))
    ax.xaxis.set_major_formatter(swiss_decimal_formatter(1))

    # Legende ohne Rahmen; die Marker-Grösse in der Legende wird vereinheitlicht,
    # damit die grossen Mittelwert-Punkte die Zeilen nicht überlappen (die Punkte
    # im Diagramm selbst behalten ihre Grösse).
    legend = ax.legend(frameon=False, labelspacing=1.0)
    for handle in legend.legend_handles:
        handle.set_sizes([60])

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

    return (lr_times, lr_r2), (dnn_times, dnn_r2)


if __name__ == '__main__':
    generate_accuracy_vs_effort_plot()
