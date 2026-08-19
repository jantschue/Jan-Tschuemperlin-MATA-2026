"""
Dieses Skript zeichnet die verwendete MLP-Architektur des v8-Modells
(67 Eingabemerkmale, verborgene Schichten 512/256/256/128, ein Ausgabeneuron)
als beschriftete Schemazeichnung mit Matplotlib und speichert sie als Bilddatei.
Pro Schicht werden stellvertretend einige Neuronen gezeichnet, die restlichen
durch Auslassungspunkte angedeutet.
Die Schichtgrössen stammen direkt aus models/mlp_v8.py (HIDDEN_DIMS, DROPOUT).
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Reproduzierbarkeit (gemäss globalen Projektvorgaben)
# ---------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------
# Architektur (identisch zu models/mlp_v8.py)
# ---------------------------------------------------------
INPUT_DIM = 67                          # Anzahl Merkmale in FEATURES
HIDDEN_DIMS = [512, 256, 256, 128]      # HIDDEN_DIMS aus mlp_v8.py
OUTPUT_DIM = 1                          # Regression: ein Neuron ohne Aktivierung
DROPOUT = 0.28                          # gerundet von 0.28262030710606983

LAYER_SIZES = [INPUT_DIM] + HIDDEN_DIMS + [OUTPUT_DIM]

# ---------------------------------------------------------
# Konfiguration der Zeichnung (Vermeidung von "Magic Numbers")
# ---------------------------------------------------------
FIGURE_SIZE = (13.0, 8.3)
DPI = 300

NODES_VISIBLE = 6           # dargestellte Kreise pro Schicht
NODES_TOP = 3               # davon oberhalb der Auslassung

V_SPACING = 1.05
H_SPACING = 2.7
CIRCLE_RADIUS = 0.28
ELLIPSIS_GAP = 1.2          # vertikaler Platzhalter für die Auslassung
DOT_RADIUS = 0.05
DOT_SPACING = 0.32

FONT_SIZE_LABELS = 10
FONT_SIZE_BOTTOM = 11
FONT_SIZE_TOP = 10
FONT_SIZE_TITLE = 15

LABEL_MARGIN = 0.22         # Abstand der Beschriftung vom Kreisrand
ANNOTATION_MARGIN = 1.0     # Abstand der Schichtbeschriftungen vom äussersten Kreis
TEXT_ROOM_TOP = 0.55        # Platz für die vierzeiligen Beschriftungen oben
TEXT_ROOM_BOTTOM = 1.3      # Platz für die dreizeiligen Beschriftungen unten
TITLE_GAP = 0.0             # zusätzlicher Leerraum zwischen Titel und Zeichnung
TITLE_Y = 0.995             # Titelposition in Figurkoordinaten

COLOR_BG = 'white'
COLOR_LINE = '#d9d9d9'
COLOR_NODE_FACE_DEFAULT = 'white'
COLOR_NODE_EDGE_DEFAULT = '#7a7a7a'
COLOR_NODE_FACE_INPUT = '#f0f0f0'
COLOR_NODE_EDGE_INPUT = '#6e6e6e'
COLOR_NODE_FACE_OUTPUT = '#dbeafe'
COLOR_NODE_EDGE_OUTPUT = '#1a73e8'
COLOR_DOTS = '#8a8a8a'
COLOR_TEXT_MUTED = '#555555'

# Beschriftung der Eingabeschicht (stellvertretend für die 67 Merkmale)
INPUT_LABELS_TOP = ['Year', 'Hour_sin', 'Hour_cos']
INPUT_LABELS_BOTTOM = ['schoolholiday_ZH', 'temp', 'weather_cat']
OUTPUT_LABEL = 'volume\n(Fahrzeuge pro Stunde)'

TITLE = 'Architektur des verwendeten MLP (Modell v8)'
SUBTITLE = ('67 Eingabemerkmale, vier verborgene Schichten, ein Ausgabeneuron, '
            '267 265 trainierbare Parameter')

SAVE_DIR = os.path.join('Abbildungen_MATA', 'Abbildungen')
SAVE_PATH = os.path.join(SAVE_DIR, 'mlp_architektur_v8.png')


def compute_node_positions() -> list:
    """Berechnet die Kreismittelpunkte aller dargestellten Neuronen je Schicht."""
    positions = []
    for layer_size in LAYER_SIZES:
        n_drawn = min(layer_size, NODES_VISIBLE)
        layer_x = len(positions) * H_SPACING

        # Gesamthöhe der Schicht inklusive Platzhalter für die Auslassung
        gap = ELLIPSIS_GAP if layer_size > NODES_VISIBLE else 0.0
        y_start = ((n_drawn - 1) * V_SPACING + gap) / 2.0

        layer_positions = []
        offset = 0.0
        for node_index in range(n_drawn):
            # Nach den oberen Knoten wird der Platzhalter eingefügt
            if layer_size > NODES_VISIBLE and node_index == NODES_TOP:
                offset += gap
            layer_positions.append((layer_x, y_start - node_index * V_SPACING - offset))
        positions.append(layer_positions)
    return positions


def draw_connections(ax, positions: list) -> None:
    """Zeichnet die Verbindungen zwischen den dargestellten Neuronen."""
    for layer_index in range(len(positions) - 1):
        for x1, y1 in positions[layer_index]:
            for x2, y2 in positions[layer_index + 1]:
                ax.plot([x1, x2], [y1, y2], '-', color=COLOR_LINE, lw=0.8, zorder=1)


def draw_nodes(ax, positions: list) -> None:
    """Zeichnet die Neuronenkreise, die Auslassungspunkte und die Randbeschriftungen."""
    last_index = len(LAYER_SIZES) - 1

    for layer_index, layer_positions in enumerate(positions):
        if layer_index == 0:
            face, edge = COLOR_NODE_FACE_INPUT, COLOR_NODE_EDGE_INPUT
        elif layer_index == last_index:
            face, edge = COLOR_NODE_FACE_OUTPUT, COLOR_NODE_EDGE_OUTPUT
        else:
            face, edge = COLOR_NODE_FACE_DEFAULT, COLOR_NODE_EDGE_DEFAULT

        for node_index, (x, y) in enumerate(layer_positions):
            ax.add_patch(plt.Circle((x, y), radius=CIRCLE_RADIUS, facecolor=face,
                                    edgecolor=edge, lw=1.5, zorder=2))

            # Merkmalsnamen links neben der Eingabeschicht
            if layer_index == 0:
                if node_index < NODES_TOP:
                    label = INPUT_LABELS_TOP[node_index]
                else:
                    label = INPUT_LABELS_BOTTOM[node_index - NODES_TOP]
                ax.text(x - CIRCLE_RADIUS - LABEL_MARGIN, y, label, ha='right',
                        va='center', fontsize=FONT_SIZE_LABELS, zorder=3)

            # Zielgrösse rechts neben dem Ausgabeneuron
            if layer_index == last_index:
                ax.text(x + CIRCLE_RADIUS + LABEL_MARGIN, y, OUTPUT_LABEL, ha='left',
                        va='center', fontsize=FONT_SIZE_LABELS, zorder=3)

        # Drei Punkte als Hinweis auf die nicht gezeichneten Neuronen
        if LAYER_SIZES[layer_index] > NODES_VISIBLE:
            x_center = layer_positions[0][0]
            y_center = (layer_positions[NODES_TOP - 1][1]
                        + layer_positions[NODES_TOP][1]) / 2.0
            for k in (-1, 0, 1):
                ax.add_patch(plt.Circle((x_center, y_center + k * DOT_SPACING),
                                        radius=DOT_RADIUS, facecolor=COLOR_DOTS,
                                        edgecolor='none', zorder=3))


def draw_annotations(ax, positions: list) -> tuple:
    """Beschriftet die Schichten unten und die Rechenschritte oben, gibt die Ränder zurück."""
    last_index = len(LAYER_SIZES) - 1
    y_values = [y for layer in positions for _, y in layer]
    bottom_y = min(y_values) - ANNOTATION_MARGIN
    top_y = max(y_values) + ANNOTATION_MARGIN

    # Untere Zeile: Rolle und Grösse jeder Schicht
    bottom_labels = [f'Eingabeschicht\n{INPUT_DIM} Merkmale']
    for i, dim in enumerate(HIDDEN_DIMS):
        bottom_labels.append(f'Verborgene\nSchicht {i + 1}\n{dim} Neuronen')
    bottom_labels.append(f'Ausgabeschicht\n{OUTPUT_DIM} Neuron')

    for layer_index, label in enumerate(bottom_labels):
        ax.text(layer_index * H_SPACING, bottom_y, label, ha='center', va='top',
                fontsize=FONT_SIZE_BOTTOM, linespacing=1.5)

    # Obere Zeile: Rechenschritte je verborgener Schicht
    for layer_index in range(1, last_index):
        text = 'Linear\nBatchNorm\nReLU'
        if layer_index != last_index - 1:   # kein Dropout nach der letzten Hidden Layer
            text += f'\nDropout {DROPOUT:.2f}'
        ax.text(layer_index * H_SPACING, top_y, text, ha='center', va='bottom',
                fontsize=FONT_SIZE_TOP, color=COLOR_TEXT_MUTED, linespacing=1.5)

    ax.text(last_index * H_SPACING, top_y, 'Linear\nkeine Aktivierung', ha='center',
            va='bottom', fontsize=FONT_SIZE_TOP, color=COLOR_TEXT_MUTED,
            linespacing=1.5)

    return bottom_y, top_y


def draw_mlp_architecture() -> None:
    """Erstellt die beschriftete Abbildung der v8-Architektur und speichert sie ab."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    positions = compute_node_positions()
    draw_connections(ax, positions)
    draw_nodes(ax, positions)
    bottom_y, top_y = draw_annotations(ax, positions)

    ax.axis('off')
    ax.set_aspect('equal')
    # Achsenbereich so erweitern, dass die Beschriftungen darin Platz haben und
    # oberhalb zusätzlich Luft für den Titel bleibt
    ax.set_xlim(-2.4, (len(LAYER_SIZES) - 1) * H_SPACING + 2.8)
    ax.set_ylim(bottom_y - TEXT_ROOM_BOTTOM, top_y + TEXT_ROOM_TOP + TITLE_GAP)

    # Titel auf Figurenebene, damit er die oberen Beschriftungen nicht überlagert
    fig.suptitle(f'{TITLE}\n{SUBTITLE}', fontsize=FONT_SIZE_TITLE, fontweight='bold',
                 y=TITLE_Y, va='top', linespacing=1.6)

    plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches='tight', facecolor=COLOR_BG)
    plt.close(fig)
    print(f'Abbildung gespeichert unter: {SAVE_PATH}')


if __name__ == '__main__':
    draw_mlp_architecture()
