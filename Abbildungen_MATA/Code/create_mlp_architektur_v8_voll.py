"""
Dieses Skript zeichnet die vollständige MLP-Architektur des v8-Modells mit jedem
einzelnen Neuron und jeder einzelnen Verbindung (67-512-256-256-128-1, also 1220
Neuronen und 263808 Gewichte) als unbeschriftete Schemazeichnung.
Die Schichtgrössen stammen direkt aus models/mlp_v8.py (HIDDEN_DIMS).
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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

# ---------------------------------------------------------
# Konfiguration der Zeichnung (Vermeidung von "Magic Numbers")
# ---------------------------------------------------------
FIGURE_SIZE = (16.0, 11.0)
DPI = 300

LAYER_HEIGHT = 10.0         # jede Schicht füllt dieselbe Höhe (Dichte zeigt die Grösse)
H_SPACING = 4.0             # horizontaler Abstand der Schichten

NODE_SIZE = 2.2             # Fläche der Neuronenpunkte
NODE_SIZE_OUTPUT = 60.0     # Ausgabeneuron etwas grösser, sonst unsichtbar
EDGE_WIDTH = 0.05           # Basis-Linienbreite der Verbindungen

# Deckkraft und Breite werden pro Schichtübergang an dessen Verbindungszahl
# angepasst. Sonst verschwindet ein dünner Übergang (128 Linien zur Ausgabe)
# neben einem dichten Übergang (131072 Linien) vollständig.
EDGE_INK_BUDGET = 2400      # Zielwert: Deckkraft mal Verbindungszahl
EDGE_ALPHA_MIN = 0.018
EDGE_ALPHA_MAX = 0.40
EDGE_WIDTH_REF = 20000      # Referenzzahl für die Verbreiterung dünner Übergänge
EDGE_WIDTH_MAX_FACTOR = 5.0

COLOR_BG = 'white'
COLOR_LINE = '#666666'
COLOR_NODE = '#222222'
COLOR_NODE_INPUT = '#555555'
COLOR_NODE_OUTPUT = '#1a73e8'

SAVE_DIR = os.path.join('Abbildungen_MATA', 'Abbildungen')
SAVE_PATH = os.path.join(SAVE_DIR, 'mlp_architektur_v8_voll.png')


def compute_node_positions() -> list:
    """Berechnet die Koordinaten jedes einzelnen Neurons, Schicht für Schicht."""
    positions = []
    for layer_index, layer_size in enumerate(LAYER_SIZES):
        x = layer_index * H_SPACING
        if layer_size == 1:
            y = np.array([0.0])
        else:
            # Neuronen gleichmässig über die einheitliche Schichthöhe verteilen
            y = np.linspace(LAYER_HEIGHT / 2.0, -LAYER_HEIGHT / 2.0, layer_size)
        positions.append(np.column_stack([np.full(layer_size, x), y]))
    return positions


def build_segments(positions: list) -> list:
    """Erzeugt die Verbindungssegmente je Schichtübergang als eigene Liste."""
    segments_per_gap = []
    for layer_index in range(len(positions) - 1):
        left = positions[layer_index]
        right = positions[layer_index + 1]
        # Kartesisches Produkt: jedes Neuron links mit jedem Neuron rechts
        starts = np.repeat(left, len(right), axis=0)
        ends = np.tile(right, (len(left), 1))
        segments_per_gap.append(np.stack([starts, ends], axis=1))
    return segments_per_gap


def edge_style(n_edges: int) -> tuple:
    """Bestimmt Deckkraft und Linienbreite passend zur Verbindungszahl eines Übergangs."""
    alpha = np.clip(EDGE_INK_BUDGET / n_edges, EDGE_ALPHA_MIN, EDGE_ALPHA_MAX)
    factor = np.clip((EDGE_WIDTH_REF / n_edges) ** 0.25, 1.0, EDGE_WIDTH_MAX_FACTOR)
    return float(alpha), float(EDGE_WIDTH * factor)


def draw_full_architecture() -> None:
    """Zeichnet alle Neuronen und Verbindungen und speichert die Abbildung ab."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    positions = compute_node_positions()
    segments_per_gap = build_segments(positions)
    n_weights = sum(len(seg) for seg in segments_per_gap)
    n_neurons = sum(LAYER_SIZES)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # Verbindungen als LineCollection (deutlich schneller als einzelne plot-Aufrufe),
    # eine Collection pro Übergang mit eigener Deckkraft und Linienbreite
    for segments in segments_per_gap:
        alpha, width = edge_style(len(segments))
        ax.add_collection(LineCollection(segments, colors=COLOR_LINE, linewidths=width,
                                         alpha=alpha, zorder=1))

    # Neuronen als Punkte
    last_index = len(LAYER_SIZES) - 1
    for layer_index, layer in enumerate(positions):
        if layer_index == 0:
            color, size = COLOR_NODE_INPUT, NODE_SIZE
        elif layer_index == last_index:
            color, size = COLOR_NODE_OUTPUT, NODE_SIZE_OUTPUT
        else:
            color, size = COLOR_NODE, NODE_SIZE
        ax.scatter(layer[:, 0], layer[:, 1], s=size, c=color, zorder=2,
                   linewidths=0, marker='o')

    ax.axis('off')
    ax.set_xlim(-0.6, last_index * H_SPACING + 0.6)
    ax.set_ylim(-LAYER_HEIGHT / 2.0 - 0.4, LAYER_HEIGHT / 2.0 + 0.4)

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=DPI, bbox_inches='tight', facecolor=COLOR_BG)
    plt.close(fig)

    print(f'Neuronen: {n_neurons}, Verbindungen: {n_weights}')
    print(f'Abbildung gespeichert unter: {SAVE_PATH}')


if __name__ == '__main__':
    draw_full_architecture()
