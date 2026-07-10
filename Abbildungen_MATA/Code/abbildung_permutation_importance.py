"""
Dieses Skript erstellt ein horizontales Balkendiagramm der Permutation Importance
je Merkmalsgruppe und stellt dabei das DNN (MLP) der linearen Regression gegenüber.
Als Wichtigkeit dient der über alle Datenreihen gemittelte R²-Abfall (Spalte
"Mittelwert") aus den beiden CSV-Dateien.

Die Abbildung besteht aus zwei Panels: Das linke Panel zeigt alle Merkmale und
verdeutlicht die Dominanz der Tageszeit. Das rechte Panel blendet die Tageszeit aus
und skaliert die x-Achse automatisch, damit die Rangfolge der übrigen Merkmale
sichtbar wird. Die Grafik wird als PNG gespeichert.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------
DNN_PATH = os.path.join(
    "results", "analysis", "feature_importance",
    "permutation_importance_mlp_v8.csv"
)
LR_PATH = os.path.join(
    "results", "analysis", "feature_importance",
    "permutation_importance_linear_regression_v8.csv"
)

OUTPUT_DIR = os.path.join("Abbildungen_MATA", "Abbildungen")
OUTPUT_FILENAME = "abbildung_permutation_importance_v8.png"

COLOR_DNN = '#3182bd'  # DNN (dunkelblau)
COLOR_LR = '#9ecae1'   # Lineare Regression (hellblau)

# Gesamttitel der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Wichtigkeit der Merkmalsgruppen: DNN vs. lineare Regression"

BAR_HEIGHT = 0.4
EXCLUDE_FEATURE = "Tageszeit"  # Merkmal, das im rechten Panel entfällt

X_LABEL = "Mittlerer R²-Abfall bei Permutation"


def load_mean_importance(csv_path):
    """
    Lädt eine Permutation-Importance-CSV und gibt eine Series zurück,
    die den Merkmalsnamen (erste Spalte) auf den gemittelten R²-Abfall
    (Spalte "Mittelwert") abbildet.
    """
    df = pd.read_csv(csv_path, index_col=0)
    return df["Mittelwert"]


def draw_panel(ax, features, dnn_values, lr_values, title, label_features=None):
    """
    Zeichnet ein horizontales, gruppiertes Balkendiagramm in die übergebene
    Achse: pro Merkmal je ein Balken für DNN und lineare Regression.
    Die Merkmale werden von oben nach unten in der übergebenen Reihenfolge
    dargestellt.

    Über ``label_features`` wird gesteuert, welche DNN-Balken mit ihrem Wert
    beschriftet werden. ``None`` beschriftet alle DNN-Balken, eine Menge von
    Merkmalsnamen beschränkt die Beschriftung auf die genannten Merkmale.
    Die Balken der linearen Regression werden nie beschriftet.
    """
    y = np.arange(len(features))

    # y invertieren, damit das wichtigste Merkmal oben steht
    dnn_bars = ax.barh(y - BAR_HEIGHT / 2, dnn_values, BAR_HEIGHT,
                       label="DNN", color=COLOR_DNN)
    ax.barh(y + BAR_HEIGHT / 2, lr_values, BAR_HEIGHT,
            label="Lineare Regression", color=COLOR_LR)

    # Nur die gewünschten DNN-Balken mit zwei Nachkommastellen beschriften.
    # Das Label wird rechts von max(Wert, 0) verankert, damit es bei negativen
    # oder verschwindend kleinen Werten (z. B. Jahr, Schnee) nicht über die
    # y-Achse bzw. den Balken ragt.
    for feature, value, bar in zip(features, dnn_values, dnn_bars):
        if label_features is not None and feature not in label_features:
            continue
        ax.annotate(
            f"{value:.2f}",
            xy=(max(value, 0.0), bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left", va="center", fontsize=7,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()

    ax.set_xlabel(X_LABEL)
    ax.set_title(title)

    # Nulllinie hervorheben
    ax.axvline(0, color='black', linewidth=0.8)

    # Helles x-Gitter im Hintergrund
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Obere und rechte Achsenlinien ausblenden
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def generate_permutation_importance_plot():
    """
    Erstellt die zweiteilige Abbildung der Permutation Importance
    (alle Merkmale sowie Ausschnitt ohne Tageszeit) und speichert sie als PNG.
    """
    dnn = load_mean_importance(DNN_PATH)
    lr = load_mean_importance(LR_PATH)

    # Reihenfolge: absteigend nach DNN-Wichtigkeit
    order = dnn.sort_values(ascending=False).index.tolist()

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 6))

    # Linkes Panel: alle Merkmale
    draw_panel(
        ax_left,
        order,
        [dnn[f] for f in order],
        [lr[f] for f in order],
        "alle Merkmale",
        label_features={EXCLUDE_FEATURE},  # nur Tageszeit anschreiben
    )

    # Rechtes Panel: dieselben Merkmale ohne Tageszeit, auto-skalierte x-Achse
    order_wo = [f for f in order if f != EXCLUDE_FEATURE]
    draw_panel(
        ax_right,
        order_wo,
        [dnn[f] for f in order_wo],
        [lr[f] for f in order_wo],
        "ohne Tageszeit (Ausschnitt)",
    )

    # Legende ohne Rahmen (einmal, aus dem linken Panel)
    ax_left.legend(frameon=False)

    # Gesamttitel ueber beide Teilplots (die Teil-Titel der Panels bleiben)
    fig.suptitle(TITLE, fontsize=15, fontweight="bold")

    plt.tight_layout()

    # Ordner erstellen und speichern
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    generate_permutation_importance_plot()
