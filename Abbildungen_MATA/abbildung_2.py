import matplotlib.pyplot as plt
import numpy as np
import os

# --- Daten (erfundenes Beispiel) ---
xs = [15, 18, 20, 22, 28, 30, 32, 35]
ys = [124.2, 175.5, 171.6, 215.8, 234.2, 278.4, 264.5, 295.8]

# --- Weight und Bias mit der Methode der kleinsten Quadrate berechnen ---
n = len(xs)
xm = sum(xs) / n
ym = sum(ys) / n
w = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
b = ym - w * xm
print(f"Weight w = {w}, Bias b = {b}")     # 1.4 und 0.5

# --- Vorhersage aus dem Rechenbeispiel ---
x_new = 25                    # neues Eingabemerkmal
y_new = w * x_new + b         # f(2) = 3,3

# --- Diagramm ---
fig, ax = plt.subplots(figsize=(6, 3.8), dpi=200)
xx = np.linspace(10, 40, 100)
ax.plot(xx, w * xx + b, color="#1f6fb4", lw=2,
        label=f"f(x) = {w:.0f} x + {b:.0f}".replace(".", ","))
ax.scatter(xs, ys, color="#16324f", s=55, zorder=3, label="Datenpunkte")

# Vorhersage hervorheben
ax.plot([x_new, x_new], [80, y_new], color="#e67e22", ls="--", lw=1.2)
ax.plot([10, x_new], [y_new, y_new], color="#e67e22", ls="--", lw=1.2)
ax.scatter([x_new], [y_new], color="#e67e22", s=80, zorder=4,
           label=f"Vorhersage f({x_new}) = {y_new:.0f}".replace(".", ","))
ax.annotate(f"f({x_new}) = {y_new:.0f}".replace(".", ","),
            (x_new, y_new), xytext=(x_new + 2, y_new - 30),
            fontsize=9, color="#b5651d",
            arrowprops=dict(arrowstyle="->", color="#b5651d", lw=1, shrinkB=8))

ax.set_xlabel("x (Temperatur in °C)")
ax.set_ylabel("y (Verkaufte Glace)")
ax.set_xlim(10, 40)
ax.set_ylim(80, 350)
ax.grid(True, color="#e6e6e6")
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper left")

fig.tight_layout()

# Ordner erstellen und speichern
output_dir = os.path.join("Abbildungen_MATA", "Abbildungen")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "abbildung_2_regression_diagramm.png")

fig.savefig(output_path, facecolor="white")
print(f"Gespeichert: {output_path}")