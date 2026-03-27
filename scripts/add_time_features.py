"""
Liest die bereinigten Datensätze aus 'merged_gapless' ein und erweitert sie um
verschiedene zeitbezogene Merkmale (Tag des Jahres, Tag des Monats, Wochentag,
Jahr, Monat, Stunde), basierend auf der Spalte 'datetime'. Die neuen Dateien
werden im Ordner 'merged_gapless_time' gespeichert.
"""

import pandas as pd
import os

# Paths
input_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_gapless')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_gapless_time')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each merged_gapless CSV file
for filename in sorted(os.listdir(input_dir)):
    if not filename.endswith('_merged_gapless.csv'):
        continue

    filepath = os.path.join(input_dir, filename)
    df = pd.read_csv(filepath)

    # Parse datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract time features
    df.insert(1, 'DayOfYear', df['datetime'].dt.dayofyear)
    df.insert(2, 'DayOfMonth', df['datetime'].dt.day)
    df.insert(3, 'DayOfWeek', df['datetime'].dt.dayofweek)
    df.insert(4, 'Year', df['datetime'].dt.year)
    df.insert(5, 'Month', df['datetime'].dt.month)
    df.insert(6, 'Hour', df['datetime'].dt.hour)

    # Save to new folder
    output_filename = filename.replace('_merged_gapless.csv', '_merged_gapless_time.csv')
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    print(f'{filename}: {len(df)} Zeilen, 6 Zeit-Features hinzugefuegt -> {output_filename}')

print('\nAlle Dateien in data/merged_gapless_time gespeichert.')
