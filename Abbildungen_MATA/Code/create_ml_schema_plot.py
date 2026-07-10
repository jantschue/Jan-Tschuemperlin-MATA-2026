import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Titeltext der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Klassische Programmierung vs. maschinelles Lernen"

# Zielordner der Abbildung (einheitlich mit den uebrigen Abbildungen)
OUTPUT_DIR = os.path.join("Abbildungen_MATA", "Abbildungen")
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 11.5)
ax.set_ylim(-0.5, 5.5)
ax.axis('off')

# Aussagekraeftiger Titel direkt im Plot
ax.set_title(TITLE, fontsize=17, fontweight="bold", pad=16)

# Colors - completely different palette (modern & muted)
color_rules = "#5D9B9B" # Muted Teal
color_data = "#4A6E82" # Steel Blue
color_answers = "#D37D52" # Muted Orange
color_box = "#F2F4F5" # Soft Gray
color_text_box = "#2C3E50" # Dark Slate

# Fonts
font_small_box = {'family': 'sans-serif', 'weight': 'bold', 'size': 13, 'color': 'white', 'ha': 'center', 'va': 'center'}
font_box = {'family': 'sans-serif', 'weight': 'bold', 'size': 16, 'color': color_text_box, 'ha': 'center', 'va': 'center'}

def draw_custom_box(x, y, width, height, text, bg_color, text_color, font_dict):
    # Shadow
    shadow = patches.FancyBboxPatch((x+0.05, y-0.05), width, height, boxstyle="round,pad=0.1,rounding_size=0.15", facecolor='black', alpha=0.1, edgecolor='none')
    ax.add_patch(shadow)
    # Box
    box = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1,rounding_size=0.15", facecolor=bg_color, edgecolor=text_color if bg_color == color_box else 'none', linewidth=1.2)
    ax.add_patch(box)
    # Text
    ax.text(x + width/2, y + height/2, text, fontdict=font_dict)

def draw_arrow(x_start, y_start, x_end, y_end):
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->,head_length=0.7,head_width=0.4", color='#A2B1B8', lw=2.5))

sb_w = 2.0
sb_h = 0.9
left_x = 0.5
right_x = 9.0

# Central box size
box_w = 4.4
box_h = 2.0
box_x = 3.7
box_y_top = 3.1
box_y_bot = 0.1

# Top part: Classical Programming
y_top1 = 4.9
y_top2 = 3.3
center_y_top = 4.1

draw_custom_box(left_x, y_top1 - sb_h/2, sb_w, sb_h, "Regeln", color_rules, 'white', font_small_box)
draw_custom_box(left_x, y_top2 - sb_h/2, sb_w, sb_h, "Daten", color_data, 'white', font_small_box)

draw_custom_box(box_x, box_y_top, box_w, box_h, "Klassische\nProgrammierung", color_box, color_text_box, font_box)

draw_arrow(left_x + sb_w + 0.15, y_top1, box_x - 0.1, y_top1)
draw_arrow(left_x + sb_w + 0.15, y_top2, box_x - 0.1, y_top2)

draw_custom_box(right_x, center_y_top - sb_h/2, sb_w, sb_h, "Antworten", color_answers, 'white', font_small_box)
draw_arrow(box_x + box_w + 0.15, center_y_top, right_x - 0.1, center_y_top)


# Bottom part: Machine Learning
y_bot1 = 1.9
y_bot2 = 0.3
center_y_bot = 1.1

draw_custom_box(left_x, y_bot1 - sb_h/2, sb_w, sb_h, "Daten", color_data, 'white', font_small_box)
draw_custom_box(left_x, y_bot2 - sb_h/2, sb_w, sb_h, "Antworten", color_answers, 'white', font_small_box)

draw_custom_box(box_x, box_y_bot, box_w, box_h, "Maschinelles\nLernen", color_box, color_text_box, font_box)

draw_arrow(left_x + sb_w + 0.15, y_bot1, box_x - 0.1, y_bot1)
draw_arrow(left_x + sb_w + 0.15, y_bot2, box_x - 0.1, y_bot2)

draw_custom_box(right_x, center_y_bot - sb_h/2, sb_w, sb_h, "Regeln", color_rules, 'white', font_small_box)
draw_arrow(box_x + box_w + 0.15, center_y_bot, right_x - 0.1, center_y_bot)

# Save high quality formats (in den Standard-Abbildungsordner)
plt.savefig(os.path.join(OUTPUT_DIR, "klassische_vs_ml_programmierung.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, "klassische_vs_ml_programmierung.pdf"), bbox_inches='tight')
print(f"Abbildung erfolgreich im Ordner '{OUTPUT_DIR}' erstellt!")
