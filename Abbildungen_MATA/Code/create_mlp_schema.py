"""
Dieses Skript erstellt eine schlichte Schemazeichnung eines mehrschichtigen 
Perzeptrons (MLP) mit Matplotlib und speichert sie als Bilddatei.
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
# Konfiguration (Vermeidung von "Magic Numbers")
# ---------------------------------------------------------
FIGURE_SIZE = (10, 6)
LAYER_SIZES = [5, 4, 4, 1]
INPUT_LABELS = ['Stunde', 'Wochentag', 'Temperatur', 'Niederschlag', 'Feiertag']
OUTPUT_LABEL = 'Fahrzeuge/Stunde'

# Farben und Abstände
COLOR_BG = 'white'
COLOR_LINE = '#d3d3d3'
COLOR_NODE_FACE_DEFAULT = 'white'
COLOR_NODE_EDGE_DEFAULT = '#888888'
COLOR_NODE_FACE_OUTPUT = '#e8f4f8' # Leicht bläulich hervorgehoben
COLOR_NODE_EDGE_OUTPUT = '#8ab4f8'

V_SPACING = 1.0
H_SPACING = 2.5
CIRCLE_RADIUS = 0.25
FONT_SIZE_LABELS = 11
FONT_SIZE_BOTTOM = 12

# Titeltext der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = 'Aufbau des verwendeten MLP'

# Pfade relativ zum Projektroot (einheitlicher Abbildungsordner)
SAVE_DIR = os.path.join('Abbildungen_MATA', 'Abbildungen')
SAVE_PATH = os.path.join(SAVE_DIR, 'mlp_schema.png')


def draw_mlp_schema():
    """
    Zeichnet ein MLP-Schema mit Eingabe-, verborgenen und Ausgabeschichten 
    und speichert die Abbildung ab.
    """
    # Verzeichnis erstellen, falls es nicht existiert
    os.makedirs(SAVE_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # Berechne die Positionen der Knoten
    node_positions = []
    for i, size in enumerate(LAYER_SIZES):
        layer_x = i * H_SPACING
        
        # Zentriere die Schichten vertikal
        y_offset = (max(LAYER_SIZES) - size) * V_SPACING / 2.0
        
        layer_positions = []
        for j in range(size):
            # y-Koordinate invertieren, damit der erste Knoten (Stunde) oben ist
            layer_y = -(y_offset + j * V_SPACING)
            layer_positions.append((layer_x, layer_y))
            
        node_positions.append(layer_positions)

    # Zeichne die Verbindungen (dünne graue Linien)
    for i in range(len(LAYER_SIZES) - 1):
        for pos1 in node_positions[i]:
            for pos2 in node_positions[i + 1]:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], '-', 
                        color=COLOR_LINE, lw=1.0, zorder=1)

    # Zeichne die Knoten und Beschriftungen
    for i, positions in enumerate(node_positions):
        for j, (x, y) in enumerate(positions):
            # Ausgabeneuron farblich leicht hervorheben
            if i == len(LAYER_SIZES) - 1:
                face_color = COLOR_NODE_FACE_OUTPUT
                edge_color = COLOR_NODE_EDGE_OUTPUT
            else:
                face_color = COLOR_NODE_FACE_DEFAULT
                edge_color = COLOR_NODE_EDGE_DEFAULT

            # Knoten zeichnen
            circle = plt.Circle((x, y), radius=CIRCLE_RADIUS, facecolor=face_color, 
                                edgecolor=edge_color, lw=1.5, zorder=2)
            ax.add_patch(circle)

            # Eingabebeschriftungen (links neben den Knoten)
            if i == 0:
                ax.text(x - CIRCLE_RADIUS - 0.15, y, INPUT_LABELS[j], 
                        ha='right', va='center', fontsize=FONT_SIZE_LABELS, zorder=3)
            # Ausgabebeschriftung (rechts neben dem Knoten)
            elif i == len(LAYER_SIZES) - 1:
                ax.text(x + CIRCLE_RADIUS + 0.15, y, OUTPUT_LABEL, 
                        ha='left', va='center', fontsize=FONT_SIZE_LABELS, zorder=3)

    # Beschriftungen unter den Schichten
    bottom_y = -(max(LAYER_SIZES) * V_SPACING) - 0.5
    
    ax.text(0 * H_SPACING, bottom_y, 'Eingabeschicht', 
            ha='center', va='top', fontsize=FONT_SIZE_BOTTOM)
    
    # "verborgene Schichten" mittig zwischen den beiden Hidden Layers platzieren
    hidden_center_x = (1 * H_SPACING + 2 * H_SPACING) / 2.0
    ax.text(hidden_center_x, bottom_y, 'verborgene Schichten', 
            ha='center', va='top', fontsize=FONT_SIZE_BOTTOM)
            
    ax.text(3 * H_SPACING, bottom_y, 'Ausgabe', 
            ha='center', va='top', fontsize=FONT_SIZE_BOTTOM)

    # Achsen und Rahmen ausblenden
    ax.axis('off')
    ax.set_aspect('equal')

    # Aussagekraeftiger Titel direkt im Plot
    ax.set_title(TITLE, fontsize=16, fontweight='bold', pad=16)

    # Layout optimieren und speichern/anzeigen
    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor=COLOR_BG)
    plt.close(fig)

if __name__ == "__main__":
    draw_mlp_schema()
