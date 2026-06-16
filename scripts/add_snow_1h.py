"""
"Fügt die Spalte 'snow_1h' zu den gefilterten Wetterdaten hinzu. snow_1h = max(0, snowheight[t] - snowheight[t-1]). Das heisst: Schneezuwachs pro Stunde in cm. Negative Differenzen (Schmelzen) werden als 0 gesetzt. Die Dateien werden direkt überschrieben."
"""

import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

files = [
    os.path.join(base_dir, "data", "weather", "processed", "weather_luzern_2010-2026_filtered.csv"),
    os.path.join(base_dir, "data", "weather", "processed", "weather_waedenswil_2010-2026_filtered.csv"),
]

for file in files:
    print(f"Verarbeite: {file}")

    df = pd.read_csv(file)

    # snowheight in numerisch umwandeln (fehlende Werte → NaN)
    df['snowheight'] = pd.to_numeric(df['snowheight'], errors='coerce')

    # Differenz zur vorherigen Stunde berechnen
    diff = df['snowheight'].diff()

    # Negative Werte (Schmelzen) auf 0 setzen, NaN bleibt NaN
    df['snow_1h'] = diff.clip(lower=0)

    # Erste Zeile hat keinen Vorgänger → auf 0 setzen
    df.loc[0, 'snow_1h'] = 0.0

    df.to_csv(file, index=False)
    print(f"  -> snow_1h Spalte hinzugefügt und gespeichert ({len(df)} Zeilen)")

print("\nFertig!")
