import matplotlib.pyplot as plt
import numpy as np
import os

# Verzeichnisse erstellen, falls sie nicht existieren
os.makedirs("Abbildungen_MATA/Abbildungen", exist_ok=True)
os.makedirs("Abbildungen_MATA/Code", exist_ok=True)

# Daten generieren (ca. 10 Datenpunkte)
np.random.seed(42)
x = np.array([5, 8, 12, 14, 18, 22, 24, 28, 30, 33])
# y ungefähr 8*x + 20 plus sichtbare Streuung
y = 8 * x + 20 + np.random.normal(0, 35, len(x))
y = np.clip(y, 0, 300) # Sicherstellen, dass keine negativen Werte entstehen

# Plot initialisieren, schlichtes Layout
fig, ax = plt.subplots(figsize=(8, 6))

x_line = np.linspace(0, 35, 100)

# Drei mögliche (schlechte) Geraden in dünn und grau
y_bad1 = 4 * x_line + 80
y_bad2 = 12 * x_line - 50
y_bad3 = 2 * x_line + 150

ax.plot(x_line, y_bad1, color='#E74C3C', linestyle='-', linewidth=1.5, alpha=0.6, label='mögliche Geraden', zorder=1)
ax.plot(x_line, y_bad2, color='#E74C3C', linestyle='-', linewidth=1.5, alpha=0.6, zorder=1)
ax.plot(x_line, y_bad3, color='#E74C3C', linestyle='-', linewidth=1.5, alpha=0.6, zorder=1)

# Beste Gerade (Regression)
coeffs = np.polyfit(x, y, 1)
y_best = np.polyval(coeffs, x_line)
ax.plot(x_line, y_best, color='#27AE60', linewidth=3.5, label='beste Gerade', zorder=2)

# Datenpunkte
ax.scatter(x, y, color='#4A6E82', s=70, label='Datenpunkte', zorder=3, edgecolors='white', linewidths=1.2)

# Achsenbeschriftung
ax.set_xlabel('Temperatur (°C)', fontsize=12)
ax.set_ylabel('Verkaufte Glace', fontsize=12)

# Achsenlimits
ax.set_xlim(0, 35)
ax.set_ylim(0, 300)

# Dezentes Gitter
ax.grid(True, linestyle='--', alpha=0.5, color='#E0E0E0', zorder=0)

# Wissenschaftliches Layout: Obere und rechte Linie ausblenden
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legende
ax.legend(loc='upper left', fontsize=11, frameon=True, edgecolor='#E0E0E0')

plt.tight_layout()

# Speichern als PNG und PDF
plt.savefig("Abbildungen_MATA/Abbildungen/temperatur_glace_regression.png", dpi=300, bbox_inches='tight')
plt.savefig("Abbildungen_MATA/Abbildungen/temperatur_glace_regression.pdf", bbox_inches='tight')

print("Abbildung erfolgreich in Abbildungen_MATA/Abbildungen gespeichert.")
