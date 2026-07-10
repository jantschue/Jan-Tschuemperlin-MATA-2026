"""
Dieses Skript erstellt das horizontale Balkendiagramm "Veränderung des R² durch den
Corona-Ausschluss" je Datenreihe, getrennt für DNN (MLP) und lineare Regression.
Gezeigt wird die Differenz des Test-R² zwischen dem Corona-bereinigten Datensatz (v6,
"ohne Corona") und dem ursprünglichen Datensatz (v5, "mit Corona"), also
R²(v6) − R²(v5). Positive Werte bedeuten, dass eine Reihe vom Weglassen des
Corona-Zeitraums profitiert. Gelesen werden ausschliesslich die bereits berechneten
Metrik-Dateien der v5- und v6-Modelle; es wird nicht neu trainiert.
Die Grafik wird als PNG gespeichert.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------
# "mit Corona" = v5 (enthält den Corona-Zeitraum), "ohne Corona" = v6 (bereinigt)
LR_WITH_CORONA = os.path.join(
    "results", "model_results", "linear_regression_v5", "metrics", "all_metrics.csv"
)
LR_WITHOUT_CORONA = os.path.join(
    "results", "model_results", "linear_regression_v6", "metrics", "all_metrics.csv"
)
DNN_WITH_CORONA = os.path.join(
    "results", "model_results", "mlp_v5", "metrics", "all_metrics.csv"
)
DNN_WITHOUT_CORONA = os.path.join(
    "results", "model_results", "mlp_v6", "metrics", "all_metrics.csv"
)

OUTPUT_DIR = os.path.join("Abbildungen_MATA", "Abbildungen")
OUTPUT_FILENAME = "abbildung_r2_veraenderung_corona_v8.png"

# Titeltext der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Veränderung des R² durch den Corona-Ausschluss"

# Reihenfolge der Datenreihen (Basiskennung ohne Versionskürzel -> Kurzbeschriftung)
STATION_ORDER = [
    ("050_Brunnen_Mositunnel_R1", "Brunnen R1"),
    ("050_Brunnen_Mositunnel_R2", "Brunnen R2"),
    ("171_Sattel_R1", "Sattel R1"),
    ("171_Sattel_R2", "Sattel R2"),
    ("216_Wangen_SZ_R1", "Wangen R1"),
    ("216_Wangen_SZ_R2", "Wangen R2"),
    ("299_Wollerau_Blatttunnel_R1", "Wollerau R1"),
    ("299_Wollerau_Blatttunnel_R2", "Wollerau R2"),
    ("720_Schwyz_R1", "Schwyz R1"),
    ("720_Schwyz_R2", "Schwyz R2"),
]

COLOR_DNN = '#3182bd'   # DNN (dunkelblau)
COLOR_LR = '#9ecae1'    # Lineare Regression (hellblau)

BAR_HEIGHT = 0.4

X_LABEL = "Veränderung des R² durch den Corona-Ausschluss (ohne minus mit Corona)"


def load_r2_by_base(csv_path):
    """
    Lädt eine Metrik-CSV und gibt ein Dictionary von der Basis-Stationskennung
    (ohne Versionskürzel wie _v5/_v6) auf das Test-R² zurück.
    """
    df = pd.read_csv(csv_path)
    result = {}
    for station, r2 in zip(df["station"], df["R2"]):
        base = station.rsplit("_v", 1)[0]  # "..._R1_v6" -> "..._R1"
        result[base] = r2
    return result


def compute_delta(with_corona_path, without_corona_path):
    """
    Berechnet je Datenreihe die R²-Differenz R²(ohne Corona) − R²(mit Corona)
    in der Reihenfolge von STATION_ORDER.
    """
    with_corona = load_r2_by_base(with_corona_path)
    without_corona = load_r2_by_base(without_corona_path)
    return [without_corona[base] - with_corona[base] for base, _ in STATION_ORDER]


def generate_corona_delta_plot():
    """
    Zeichnet das horizontale, gruppierte Balkendiagramm der R²-Veränderung durch
    den Corona-Ausschluss (DNN und lineare Regression) und speichert es als PNG.
    """
    delta_dnn = compute_delta(DNN_WITH_CORONA, DNN_WITHOUT_CORONA)
    delta_lr = compute_delta(LR_WITH_CORONA, LR_WITHOUT_CORONA)

    labels = [short for _, short in STATION_ORDER]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 6))

    # Zwei Balken je Datenreihe: DNN oben, lineare Regression unten
    ax.barh(y - BAR_HEIGHT / 2, delta_dnn, BAR_HEIGHT,
            label="DNN", color=COLOR_DNN)
    ax.barh(y + BAR_HEIGHT / 2, delta_lr, BAR_HEIGHT,
            label="Lineare Regression", color=COLOR_LR)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Brunnen R1 oben

    ax.set_xlabel(X_LABEL)
    ax.set_title(TITLE, fontsize=13, fontweight="bold", pad=12)

    # Nulllinie hervorheben
    ax.axvline(0, color="black", linewidth=0.8)

    # Legende ohne Rahmen, unten rechts
    ax.legend(frameon=False, loc="lower right")

    # Helles x-Gitter im Hintergrund
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Obere und rechte Achsenlinien ausblenden
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # Ordner erstellen und speichern
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_corona_delta_plot()
