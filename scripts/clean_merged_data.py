"""
Entfernt alle Zeilen mit fehlenden Messwerten (NaN) in der Spalte 'volume' aus den
zusammengeführten Datensätzen. Die bereinigten Datensätze werden im Ordner
'data/v4' gespeichert.
"""

import pandas as pd
import os

# Paths
merged_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'v3')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'v4')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each merged CSV file
for filename in sorted(os.listdir(merged_dir)):
    if not filename.endswith('_v3.csv'):
        continue

    filepath = os.path.join(merged_dir, filename)
    df = pd.read_csv(filepath)

    rows_before = len(df)
    df_clean = df.dropna(subset=['volume'])
    rows_after = len(df_clean)
    rows_removed = rows_before - rows_after

    # Save cleaned file
    output_filename = filename.replace('_v3.csv', '_v4.csv')
    output_path = os.path.join(output_dir, output_filename)
    df_clean.to_csv(output_path, index=False)

    print(f'{filename}: {rows_before} -> {rows_after} Zeilen ({rows_removed} entfernt)')

print('\nAlle Dateien bereinigt und in data/merged_gapless gespeichert.')
