"""
"Überprüft alle zusammengeführten Datensätze ('_v3.csv') auf fehlende Verkehrsdaten (lückenhafte Stunden). Die gefundenen Lücken werden berechnet und in einer Übersichtstextdatei ('gaps.txt') im Ordner 'results' gespeichert."
"""

import pandas as pd
import os

merged_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'v3')
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)
output_file = os.path.join(results_dir, 'gaps.txt')

lines = []

for filename in sorted(os.listdir(merged_dir)):
    if not filename.endswith('_v3.csv'):
        continue

    df = pd.read_csv(os.path.join(merged_dir, filename))
    df['datetime'] = pd.to_datetime(df['datetime'])
    missing = df[df['volume'].isna()].copy()

    name = filename.replace('_v3.csv', '')

    if missing.empty:
        lines.append(f"{name}: Keine fehlenden Daten\n")
        continue

    # Find contiguous gaps
    missing['gap_group'] = (missing['datetime'].diff() > pd.Timedelta(hours=1)).cumsum()
    gaps = missing.groupby('gap_group')['datetime'].agg(['min', 'max', 'count'])

    lines.append(f"=== {name} ===\n")
    for _, row in gaps.iterrows():
        start = row['min'].strftime('%Y-%m-%d %H:%M')
        end = row['max'].strftime('%Y-%m-%d %H:%M')
        hours = int(row['count'])
        days = hours / 24
        lines.append(f"  {start}  bis  {end}  ({hours} Stunden / {days:.1f} Tage)\n")
    lines.append("\n")

with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"Gaps-Datei gespeichert: {output_file}")
