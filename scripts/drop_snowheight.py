"""
"Entfernt die Spalte 'snowheight' aus den kategorisierten Wetterdaten."
"""

import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

files = [
    os.path.join(base_dir, "data", "external", "weather", "Luzern", "wetter_luzern_2010-2026_categorized.csv"),
    os.path.join(base_dir, "data", "external", "weather", "Waedenswil", "wetter_waedenswil_2010-2026_categorized.csv"),
]

for file in files:
    df = pd.read_csv(file)
    if 'snowheight' in df.columns:
        df = df.drop(columns=['snowheight'])
        df.to_csv(file, index=False)
        print(f"snowheight entfernt: {os.path.basename(file)}")
    else:
        print(f"snowheight nicht vorhanden: {os.path.basename(file)}")
