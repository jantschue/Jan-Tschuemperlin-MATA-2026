"""
Entfernt alle Zeilen mit fehlenden Messwerten (NaN) in der Spalte 'volume' aus
den zusammengeführten Datensätzen ('merged'). Die lückenlosen Datensätze werden
anschliessend im Ordner 'merged_gapless' gespeichert.
"""

import pandas as pd
import os

# Paths
merged_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'merged')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_gapless')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each merged CSV file
for filename in sorted(os.listdir(merged_dir)):
    if not filename.endswith('_merged.csv'):
        continue

    filepath = os.path.join(merged_dir, filename)
    df = pd.read_csv(filepath)

    rows_before = len(df)
    df_clean = df.dropna(subset=['volume'])
    rows_after = len(df_clean)
    rows_removed = rows_before - rows_after

    # Save cleaned file
    output_filename = filename.replace('_merged.csv', '_merged_gapless.csv')
    output_path = os.path.join(output_dir, output_filename)
    df_clean.to_csv(output_path, index=False)

    print(f'{filename}: {rows_before} -> {rows_after} Zeilen ({rows_removed} entfernt)')

print('\nAlle Dateien bereinigt und in data/merged_gapless gespeichert.')
