"""
Dieses Skript erstellt das Raster-Streudiagramm "Prognose gegen Messwert je
Zählstelle (Fahrtrichtung R1)" für das DNN (MLP v8). Für jede der fünf R1-Reihen
werden die auf dem Testset vorhergesagten Werte den tatsächlich gemessenen
gegenübergestellt; die rote Linie markiert die ideale Übereinstimmung (y = x).
Gelesen werden ausschliesslich die bereits erzeugten Vorhersagen aus
results/model_results/mlp_v8/predictions/ sowie das Test-R² aus der zugehörigen
Metrik-Datei; es wird nicht neu trainiert. Die Grafik wird als PNG gespeichert.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------
PRED_DIR = os.path.join("results", "model_results", "mlp_v8", "predictions")
METRICS_PATH = os.path.join(
    "results", "model_results", "mlp_v8", "metrics", "all_metrics.csv"
)

OUTPUT_DIR = os.path.join("Abbildungen_MATA", "Abbildungen")
OUTPUT_FILENAME = "abbildung_prognose_vs_messwert_r1_v8.png"

# Gesamttitel der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Prognose gegen Messwert je Zählstelle (Fahrtrichtung R1)"

# Fünf R1-Reihen (station-Kennung -> lesbarer Anzeigename), in Darstellungsreihenfolge
R1_STATIONS = [
    ("050_Brunnen_Mositunnel_R1_v8", "Brunnen (Mositunnel)"),
    ("171_Sattel_R1_v8", "Sattel"),
    ("216_Wangen_SZ_R1_v8", "Wangen SZ"),
    ("299_Wollerau_Blatttunnel_R1_v8", "Wollerau (Blatttunnel)"),
    ("720_Schwyz_R1_v8", "Schwyz"),
]

N_ROWS, N_COLS = 2, 3          # Raster: 2 Zeilen x 3 Spalten (6. Feld bleibt leer)

COLOR_POINTS = "#1f77b4"       # Streupunkte (Blau wie im Dokument)
COLOR_IDENTITY = "red"         # Ideale Übereinstimmung y = x
POINT_SIZE = 5
POINT_ALPHA = 0.3

X_LABEL = "Tatsächliches Volumen [Fahrzeuge/h]"
Y_LABEL = "Vorhergesagtes Volumen [Fahrzeuge/h]"


def load_r2():
    """
    Lädt die Metrik-CSV des MLP und gibt ein Dictionary
    von station-Kennung auf Test-R² zurück.
    """
    df = pd.read_csv(METRICS_PATH)
    return dict(zip(df["station"], df["R2"]))


def generate_prediction_raster():
    """
    Zeichnet für jede der fünf R1-Reihen ein Streudiagramm (Messwert gegen
    Vorhersage) mit Identitätslinie in ein 2x3-Raster und speichert die
    Gesamtabbildung als PNG-Datei.
    """
    r2_by_station = load_r2()

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (station, display_name) in zip(axes, R1_STATIONS):
        pred_path = os.path.join(PRED_DIR, f"predictions_{station}.csv")
        df = pd.read_csv(pred_path)

        actual = df["actual_volume"].values
        predicted = df["predicted_volume"].values

        ax.scatter(actual, predicted, s=POINT_SIZE, alpha=POINT_ALPHA,
                   color=COLOR_POINTS)

        # Identitätslinie y = x von 0 bis zum gemeinsamen Maximum
        max_val = max(actual.max(), predicted.max())
        ax.plot([0, max_val], [0, max_val], color=COLOR_IDENTITY, linewidth=1.2)

        r2 = r2_by_station.get(station, float("nan"))
        ax.set_title(f"{display_name}  (R² = {r2:.2f})")

    # Überzähliges 6. Teilfeld ausblenden
    for ax in axes[len(R1_STATIONS):]:
        ax.set_visible(False)

    # Gemeinsame Achsenbeschriftungen (nur einmal, mittig)
    fig.supxlabel(X_LABEL)
    fig.supylabel(Y_LABEL)

    # Aussagekräftiger Gesamttitel über dem Raster
    fig.suptitle(TITLE, fontsize=15, fontweight="bold")

    fig.tight_layout()

    # Ordner erstellen und speichern
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate_prediction_raster()
