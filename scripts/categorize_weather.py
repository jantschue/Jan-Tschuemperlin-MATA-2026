"""
"Fügt die Spalte 'weather_cat' zu den gefilterten Wetterdaten hinzu. Kategorien (Prioritätsreihenfolge): 1. Snow, 2. Rain, 3. Night, 4. Clear, 5. Cloudy. Erstellt neue CSV-Dateien mit dem Suffix '_categorized.csv'."
"""

import pandas as pd
import os
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

files = {
    os.path.join(base_dir, "data", "weather", "processed", "weather_luzern_2010-2026_filtered.csv"):
        os.path.join(base_dir, "data", "weather", "processed", "weather_luzern_2010-2026_categorized.csv"),
    os.path.join(base_dir, "data", "weather", "processed", "weather_waedenswil_2010-2026_filtered.csv"):
        os.path.join(base_dir, "data", "weather", "processed", "weather_waedenswil_2010-2026_categorized.csv"),
}

for input_file, output_file in files.items():
    print(f"Verarbeite: {input_file}")

    df = pd.read_csv(input_file)

    # Stunde aus dem Zeitstempel extrahieren
    df['_hour'] = pd.to_datetime(df['time']).dt.hour

    # Numerische Spalten sicherstellen
    df['snow_1h'] = pd.to_numeric(df['snow_1h'], errors='coerce').fillna(0)
    df['rain_1h'] = pd.to_numeric(df['rain_1h'], errors='coerce').fillna(0)
    df['sun_1h'] = pd.to_numeric(df['sun_1h'], errors='coerce').fillna(0)

    # Kategorien nach Priorität zuweisen
    conditions = [
        df['snow_1h'] > 0,                                          # 1. Snow
        df['rain_1h'] > 0.1,                                        # 2. Rain
        (df['_hour'] >= 20) | (df['_hour'] <= 6),                   # 3. Night
        df['sun_1h'] >= 30,                                         # 4. Clear
    ]
    choices = ['Snow', 'Rain', 'Night', 'Clear']

    df['weather_cat'] = np.select(conditions, choices, default='Cloudy')  # 5. Cloudy

    # Hilfsspalte entfernen
    df = df.drop(columns=['_hour'])

    df.to_csv(output_file, index=False)
    os.remove(input_file)  # Clean up intermediate file
    print(f"  -> Erstellt: {os.path.basename(output_file)} und Original gelöscht ({len(df)} Zeilen)")

    # Verteilung anzeigen
    print(f"     Verteilung:")
    for cat, count in df['weather_cat'].value_counts().items():
        pct = count / len(df) * 100
        print(f"       {cat:8s}: {count:7d}  ({pct:.1f}%)")
    print()

print("Fertig!")
