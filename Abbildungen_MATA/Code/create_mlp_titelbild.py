"""
Dieses Skript zeichnet die vereinfachte MLP-Architektur des v8-Modells als
Titelbild für die Maturaarbeit: Neuronen, Verbindungen und deutsche
Merkmalsbezeichnungen links, ohne Titel und ohne Schichtbeschriftungen.
Die Anzahl gezeichneter Kreise pro Schicht folgt der Reihenfolge der echten
Schichtgrössen 67 < 512 > 256 = 256 > 128 > 1, ohne jedes Neuron zu zeichnen.
Gespeichert wird als PNG (Vorschau), SVG und PDF (Druck).
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

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

LAYER_SIZES = [INPUT_DIM] + HIDDEN_DIMS + [OUTPUT_DIM]

# Sichtbare Kreise pro Schicht. Die Reihenfolge der Grössen entspricht der
# echten Architektur (67 < 512 > 256 = 256 > 128 > 1), die Zahlen selbst sind
# stark gestaucht, da 512 Kreise nicht darstellbar sind.
NODES_PER_LAYER = [6, 12, 10, 10, 8, 1]

# ---------------------------------------------------------
# Konfiguration der Zeichnung (Vermeidung von "Magic Numbers")
# ---------------------------------------------------------
FIGURE_SIZE = (13.0, 6.4)
DPI = 300
TRANSPARENT_BG = False      # True, falls das Bild auf farbigem Untergrund liegt

V_SPACING = 0.85
H_SPACING = 2.7
CIRCLE_RADIUS = 0.28
ELLIPSIS_GAP = 1.1          # vertikaler Platzhalter für die Auslassung
DOT_RADIUS = 0.05
DOT_SPACING = 0.30

# Schrift: möglichst die Schrift der Maturaarbeit, sonst Rückfall auf Standard
FONT_FAMILY = ['Arial', 'Helvetica', 'DejaVu Sans']
FONT_SIZE_LABELS = 12

LABEL_MARGIN = 0.24         # Abstand der Beschriftung vom Kreisrand
PLOT_MARGIN = 0.5           # Rand über und unter der Zeichnung
LEFT_TEXT_ROOM = 4.2        # Platz für die Merkmalsnamen links
RIGHT_TEXT_ROOM = 4.4       # Platz für die Zielgrösse rechts

# Verbindungen: Grauton links, Akzentfarbe rechts (Leserichtung)
COLOR_LINE_START = '#c0c0c0'
COLOR_LINE_END = '#8fb8e8'
LINE_WIDTH = 1.0

COLOR_BG = 'white'
COLOR_NODE_FACE_DEFAULT = 'white'
COLOR_NODE_EDGE_DEFAULT = '#6f6f6f'
COLOR_NODE_FACE_INPUT = '#ececec'
COLOR_NODE_EDGE_INPUT = '#5f5f5f'
COLOR_NODE_FACE_OUTPUT = '#cfe2fb'
COLOR_NODE_EDGE_OUTPUT = '#1a73e8'
COLOR_DOTS = '#8a8a8a'
COLOR_TEXT = '#1a1a1a'
NODE_EDGE_WIDTH = 1.7

# Randverlauf: die Verbindungen laufen nach aussen weich ins Weiss aus
FADE_ENABLED = True
FADE_WIDTH = 0.15           # Anteil der Bildbreite, über den ausgeblendet wird
FADE_STRENGTH = 0.80        # maximale Deckkraft des Verlaufs am äusseren Rand
FADE_RESOLUTION = 512       # Auflösung des Verlaufsbildes

# Deutsche Bezeichnungen statt der Spaltennamen aus dem Code
INPUT_LABELS = ['Stunde', 'Wochentag', 'Monat',
                'Feiertag', 'Schulferien', 'Wetter']
OUTPUT_LABEL = 'Verkehrsaufkommen'

SAVE_DIR = os.path.join('Abbildungen_MATA', 'Abbildungen')
SAVE_PATH_PNG = os.path.join(SAVE_DIR, 'mlp_titelbild.png')
SAVE_PATH_SVG = os.path.join(SAVE_DIR, 'mlp_titelbild.svg')
SAVE_PATH_PDF = os.path.join(SAVE_DIR, 'mlp_titelbild.pdf')


def compute_node_positions() -> list:
    """Berechnet die Kreismittelpunkte aller dargestellten Neuronen je Schicht."""
    positions = []
    for layer_index, layer_size in enumerate(LAYER_SIZES):
        n_drawn = min(layer_size, NODES_PER_LAYER[layer_index])
        n_top = n_drawn // 2
        layer_x = layer_index * H_SPACING

        # Gesamthöhe der Schicht inklusive Platzhalter für die Auslassung
        gap = ELLIPSIS_GAP if layer_size > n_drawn else 0.0
        y_start = ((n_drawn - 1) * V_SPACING + gap) / 2.0

        layer_positions = []
        offset = 0.0
        for node_index in range(n_drawn):
            # Nach den oberen Knoten wird der Platzhalter eingefügt
            if gap and node_index == n_top:
                offset += gap
            layer_positions.append((layer_x, y_start - node_index * V_SPACING - offset))
        positions.append(layer_positions)
    return positions


def transition_color(layer_index: int, n_transitions: int) -> tuple:
    """Mischt die Verbindungsfarbe je nach Position zwischen Grau und Akzentfarbe."""
    ratio = layer_index / max(n_transitions - 1, 1)
    start = np.array(to_rgb(COLOR_LINE_START))
    end = np.array(to_rgb(COLOR_LINE_END))
    return tuple(start + (end - start) * ratio)


def draw_connections(ax, positions: list) -> None:
    """Zeichnet die Verbindungen mit von links nach rechts wanderndem Farbton."""
    n_transitions = len(positions) - 1
    for layer_index in range(n_transitions):
        color = transition_color(layer_index, n_transitions)
        for x1, y1 in positions[layer_index]:
            for x2, y2 in positions[layer_index + 1]:
                ax.plot([x1, x2], [y1, y2], '-', color=color, lw=LINE_WIDTH, zorder=1)


def draw_edge_fade(ax) -> None:
    """Legt einen weissen Verlauf über die linke und rechte Kante der Zeichnung."""
    if not FADE_ENABLED:
        return

    # Der Verlauf deckt nur den Bereich der Neuronenspalten ab, damit die
    # Beschriftungen am Rand nicht mit ausgeblendet werden.
    y0, y1 = ax.get_ylim()
    x0 = -CIRCLE_RADIUS
    x1 = (len(LAYER_SIZES) - 1) * H_SPACING + CIRCLE_RADIUS

    # Alphaprofil: aussen stark, innerhalb von FADE_WIDTH auf null abfallend
    t = np.linspace(0.0, 1.0, FADE_RESOLUTION)
    ramp = np.clip(1.0 - t / FADE_WIDTH, 0.0, 1.0) \
        + np.clip(1.0 - (1.0 - t) / FADE_WIDTH, 0.0, 1.0)
    alpha = np.clip(ramp, 0.0, 1.0) ** 1.5 * FADE_STRENGTH

    overlay = np.ones((1, FADE_RESOLUTION, 4))
    overlay[0, :, 3] = alpha

    # zorder zwischen Linien (1) und Neuronen (2): nur die Verbindungen verblassen
    ax.imshow(overlay, extent=(x0, x1, y0, y1), aspect='auto',
              interpolation='bilinear', zorder=1.5)

    # imshow setzt das Seitenverhältnis zurück, deshalb hier erneut fixieren,
    # sonst werden aus den Kreisen Ellipsen
    ax.set_aspect('equal')


def draw_nodes(ax, positions: list) -> None:
    """Zeichnet die Neuronenkreise, die Auslassungspunkte und die Beschriftungen."""
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
                                    edgecolor=edge, lw=NODE_EDGE_WIDTH, zorder=2))

            # Merkmalsnamen links neben der Eingabeschicht
            if layer_index == 0:
                ax.text(x - CIRCLE_RADIUS - LABEL_MARGIN, y, INPUT_LABELS[node_index],
                        ha='right', va='center', fontsize=FONT_SIZE_LABELS,
                        color=COLOR_TEXT, zorder=3)

            # Zielgrösse rechts neben dem Ausgabeneuron
            if layer_index == last_index:
                ax.text(x + CIRCLE_RADIUS + LABEL_MARGIN, y, OUTPUT_LABEL,
                        ha='left', va='center', fontsize=FONT_SIZE_LABELS,
                        color=COLOR_TEXT, zorder=3)

        # Drei Punkte als Hinweis auf die nicht gezeichneten Neuronen
        n_drawn = len(layer_positions)
        if LAYER_SIZES[layer_index] > n_drawn:
            n_top = n_drawn // 2
            x_center = layer_positions[0][0]
            y_center = (layer_positions[n_top - 1][1] + layer_positions[n_top][1]) / 2.0
            for k in (-1, 0, 1):
                ax.add_patch(plt.Circle((x_center, y_center + k * DOT_SPACING),
                                        radius=DOT_RADIUS, facecolor=COLOR_DOTS,
                                        edgecolor='none', zorder=2))


def draw_title_image() -> None:
    """Erstellt das Titelbild und speichert es als PNG, SVG und PDF."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['pdf.fonttype'] = 42    # Schrift im PDF einbetten, nicht als Typ 3

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    positions = compute_node_positions()
    draw_connections(ax, positions)

    y_values = [y for layer in positions for _, y in layer]
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xlim(-LEFT_TEXT_ROOM,
                (len(LAYER_SIZES) - 1) * H_SPACING + RIGHT_TEXT_ROOM)
    ax.set_ylim(min(y_values) - PLOT_MARGIN, max(y_values) + PLOT_MARGIN)

    draw_edge_fade(ax)
    draw_nodes(ax, positions)

    for path in (SAVE_PATH_PNG, SAVE_PATH_SVG, SAVE_PATH_PDF):
        plt.savefig(path, dpi=DPI, bbox_inches='tight',
                    facecolor='none' if TRANSPARENT_BG else COLOR_BG,
                    transparent=TRANSPARENT_BG)
    plt.close(fig)
    print(f'Titelbild gespeichert unter: {SAVE_PATH_PNG}, {SAVE_PATH_SVG}, {SAVE_PATH_PDF}')


if __name__ == '__main__':
    draw_title_image()
