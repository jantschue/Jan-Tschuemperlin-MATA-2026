"""
Dieses Skript generiert eine Abbildung zur visuellen Gegenüberstellung von 
Regression und Klassifikation, ohne Achsenbeschriftungen oder Text.
Die Grafik wird als PNG gespeichert.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ---------------------------------------------------------
# Hyperparameter und Konfiguration
# ---------------------------------------------------------
RANDOM_SEED = 42
LINE_WIDTH_AXES = 2
LINE_WIDTH_PLOT = 3
LINE_WIDTH_SEP = 2
SCATTER_SIZE_REG = 80
SCATTER_SIZE_CLS = 200

COLOR_REG_LINE = 'firebrick'
COLOR_REG_POINTS = '#1B4F82'      # Dunkelblau
COLOR_CLS_CLASS1 = 'indigo'       # Lila
COLOR_CLS_CLASS2 = 'goldenrod'    # Gelb/Orange

# Reproduzierbarkeit sicherstellen
np.random.seed(RANDOM_SEED)

def generate_comparison_plot():
    """
    Erstellt die Abbildung mit zwei Subplots für Regression und Klassifikation 
    und speichert diese als PNG-Datei im aktuellen Verzeichnis.
    """
    # Setup Figure mit 2 Subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # =====================================================
    # Plot 1: Regression
    # =====================================================
    
    # Manuelle Punkte für ähnlichen Look wie im Vorbild
    x_reg = np.array([1, 1.5, 2.5, 4, 5, 6, 6.5, 7.5, 8.5, 8.5, 9.5])
    y_reg = np.array([0.5, 3.5, 1.5, 4, 4.5, 7, 8.5, 10.5, 8.5, 11.5, 11])
    
    # Regressionslinie, die direkt am Ursprung (0,0) beginnt
    x_line = np.array([0, 11])
    y_line = x_line * 1.2
    
    ax1.plot(x_line, y_line, color=COLOR_REG_LINE, linewidth=LINE_WIDTH_PLOT, zorder=1)
    ax1.scatter(x_reg, y_reg, s=SCATTER_SIZE_REG, facecolors='white', edgecolors=COLOR_REG_POINTS, linewidths=2.5, zorder=2)
    
    # =====================================================
    # Plot 2: Klassifikation
    # =====================================================
    
    # Trennlinie
    x_sep = np.array([0, 11])
    y_sep = x_sep * 0.9 + 1
    
    ax2.plot(x_sep, y_sep, color='black', linestyle='--', linewidth=LINE_WIDTH_SEP)
    
    # Klasse 1: Lila Kreise (oben links)
    x_cls1 = []
    y_cls1 = []
    while len(x_cls1) < 25:
        x = np.random.uniform(0.5, 9.5)
        y = np.random.uniform(2, 12)
        if y > x * 0.9 + 1.8:  # Abstand zur Trennlinie
            x_cls1.append(x)
            y_cls1.append(y)
            
    # Klasse 2: Gelbe Quadrate (unten rechts)
    x_cls2 = []
    y_cls2 = []
    while len(x_cls2) < 25:
        x = np.random.uniform(1.5, 10)
        y = np.random.uniform(0.5, 10)
        if y < x * 0.9 + 0.2:  # Abstand zur Trennlinie
            x_cls2.append(x)
            y_cls2.append(y)
            
    # Scatter Plots für Klassifikation
    ax2.scatter(x_cls1, y_cls1, s=SCATTER_SIZE_CLS, marker='o', facecolors='white', edgecolors=COLOR_CLS_CLASS1, linewidths=2.5)
    ax2.scatter(x_cls2, y_cls2, s=SCATTER_SIZE_CLS, marker='s', facecolors='white', edgecolors=COLOR_CLS_CLASS2, linewidths=2.5)
    
    # =====================================================
    # Achsen-Styling für beide Plots
    # =====================================================
    for ax in [ax1, ax2]:
        # Nur untere und linke Achsenlinien anzeigen
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(LINE_WIDTH_AXES)
        ax.spines['left'].set_linewidth(LINE_WIDTH_AXES)
        
        # Ticks komplett ausblenden
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Achsenursprung auf 0,0 setzen für den L-Shape Look
        ax.set_xlim(0, 10.5)
        ax.set_ylim(0, 12.5)
    
    plt.tight_layout(w_pad=5.0)
    
    # Ordner erstellen und speichern
    output_dir = os.path.join("Abbildungen_MATA", "Abbildungen")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "abbildung_1_regression_vs_klassifikation.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # Plot schließen, damit er bei automatischer Ausführung nicht hängen bleibt

if __name__ == '__main__':
    generate_comparison_plot()
