"""
Dieses Skript generiert eine Abbildung zur visuellen Gegenüberstellung 
der linearen Aktivierungsfunktion und der ReLU-Aktivierungsfunktion.
Die Grafik wird als PNG gespeichert.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------
# Hyperparameter und Konfiguration
# ---------------------------------------------------------
LINE_WIDTH_AXES = 1.5
LINE_WIDTH_PLOT = 3

COLOR_LINE = '#1B4F82'      # Dunkelblau
COLOR_RELU = 'firebrick'    # Rot

# Gesamttitel der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Lineare Aktivierungsfunktion und ReLU im Vergleich"

def generate_activation_plot():
    """
    Erstellt die Abbildung mit zwei Subplots für die lineare und die ReLU-Aktivierungsfunktion
    und speichert diese als PNG-Datei im entsprechenden Verzeichnis.
    """
    # Setup Figure mit 2 Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # x-Werte für beide Plots
    x = np.linspace(-5, 5, 400)
    
    # =====================================================
    # Plot 1: Lineare Aktivierungsfunktion
    # =====================================================
    y_linear = x
    
    ax1.plot(x, y_linear, color=COLOR_LINE, linewidth=LINE_WIDTH_PLOT, zorder=3)
    ax1.text(1.5, 3.5, '$f(x)$', fontsize=20, color='black', zorder=4)
    
    # =====================================================
    # Plot 2: ReLU Aktivierungsfunktion
    # =====================================================
    y_relu = np.maximum(0, x)
    
    ax2.plot(x, y_relu, color=COLOR_RELU, linewidth=LINE_WIDTH_PLOT, zorder=3)
    ax2.text(1.5, 3.5, '$f(x)$', fontsize=20, color='black', zorder=4)
    
    # =====================================================
    # Achsen-Styling für beide Plots
    # =====================================================
    for ax in [ax1, ax2]:
        # Achsen durch den Ursprung
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        
        # Obere und rechte Achsenlinien ausblenden
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Achsendicke anpassen
        ax.spines['left'].set_linewidth(LINE_WIDTH_AXES)
        ax.spines['bottom'].set_linewidth(LINE_WIDTH_AXES)
        
        # Ticks ausblenden
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Grenzen setzen
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-5.5, 5.5)
        
        # Pfeile an den Achsenenden
        ax.plot(5.5, 0, ">k", clip_on=False, zorder=4)
        ax.plot(0, 5.5, "^k", clip_on=False, zorder=4)
    
    # Gesamttitel ueber beide Teilplots (die bestehenden f(x)-Beschriftungen bleiben)
    fig.suptitle(TITLE, fontsize=16, fontweight="bold")

    plt.tight_layout(w_pad=5.0)

    # Ordner erstellen und speichern
    output_dir = os.path.join("Abbildungen_MATA", "Abbildungen")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "abbildung_5_linear_vs_relu.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    generate_activation_plot()
