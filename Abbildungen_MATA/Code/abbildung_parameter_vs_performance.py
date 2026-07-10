"""
Dieses Skript erstellt eine saubere, zum übrigen Abbildungsstil passende Version der
Parameter-vs-Performance-Kurve (mittleres Test-R² gegen die Anzahl Modellparameter)
für Kapitel 3.4.2. Die Kurve wurde bereits von scripts/parameter_vs_performance_v8.py
berechnet; dieses Skript liest ausschliesslich die vorhandene CSV und trainiert nicht neu.
Die gewählte Architektur (multiplier 128, rund 267'000 Parameter) wird hervorgehoben.
Die Grafik wird als PNG gespeichert.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------
DATA_CSV = os.path.join(
    "results", "model_results", "mlp_v8", "analysen", "parameter_vs_performance",
    "parameter_vs_performance_050_Brunnen_Mositunnel_R1_v8.csv"
)

OUTPUT_DIR = os.path.join("Abbildungen_MATA", "Abbildungen")
OUTPUT_FILENAME = "abbildung_parameter_vs_performance_v8.png"

# Titeltext der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Einfluss der Netzgrösse auf das Test-R² (Brunnen R1)"

# Gewählte Architektur, die in der Arbeit verwendet wird (zum Hervorheben)
HIGHLIGHT_PARAMS = 267265
HIGHLIGHT_LABEL = "rund 267'000 Parameter"

COLOR_DNN = '#3182bd'   # Linie (dunkelblau)
COLOR_BAND = '#9ecae1'  # Unsicherheitsband (hellblau)
COLOR_HIGHLIGHT = '#8b1a1a'  # dezentes Dunkelrot für die Markierung
BAND_ALPHA = 0.3


def swiss_decimal_formatter():
    """
    Gibt einen FuncFormatter zurück, der Achsenbeschriftungen in Schweizer
    Schreibweise mit Dezimal-Komma formatiert (z.B. 0,80 statt 0.80).
    """
    return FuncFormatter(lambda value, _pos: f"{value:.2f}".replace(".", ","))


def generate_parameter_vs_performance_plot():
    """
    Liest die vorberechnete Parameter-vs-Performance-CSV und zeichnet die
    Test-R²-Kurve mit Unsicherheitsband sowie der hervorgehobenen gewählten
    Architektur. Speichert die Grafik als PNG-Datei.
    """
    df = pd.read_csv(DATA_CSV).sort_values("params")

    params = df["params"].values
    r2_mean = df["test_r2_mean"].values
    r2_std = df["test_r2_std"].values

    fig, ax = plt.subplots(figsize=(11, 5))

    # Unsicherheitsband test_r2_mean ± test_r2_std
    ax.fill_between(params, r2_mean - r2_std, r2_mean + r2_std,
                    color=COLOR_BAND, alpha=BAND_ALPHA,
                    label="Streuung (± 1 Standardabweichung)")

    # Kurve mit Markern
    ax.plot(params, r2_mean, marker="o", color=COLOR_DNN,
            label="Mittleres Test-R²", zorder=3)

    # Gewählte Architektur hervorheben (offener Kreis auf dem Punkt)
    highlight = df[df["params"] == HIGHLIGHT_PARAMS]
    if not highlight.empty:
        hx = highlight["params"].values[0]
        hy = highlight["test_r2_mean"].values[0]
        ax.scatter(hx, hy, s=160, facecolors="none",
                   edgecolors=COLOR_HIGHLIGHT, linewidths=2, zorder=4,
                   label="Gewählte Architektur")
        ax.annotate(HIGHLIGHT_LABEL, xy=(hx, hy),
                    xytext=(8, -16), textcoords="offset points",
                    ha="left", va="top", fontsize=9, color=COLOR_HIGHLIGHT)

    # Achsen-Styling
    ax.set_xscale("log")
    ax.set_xlabel("Anzahl Modellparameter (logarithmisch)")
    ax.set_ylabel("Test-R²")
    ax.set_title(TITLE, fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(swiss_decimal_formatter())

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
    generate_parameter_vs_performance_plot()
