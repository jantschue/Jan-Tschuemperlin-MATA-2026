"""
"Überprüft alle zusammengeführten Datensätze ('_merged.csv') auf fehlende Verkehrsdaten (lückenhafte Stunden). Die gefundenen Lücken werden berechnet und in einer Übersichtstextdatei ('gaps.txt') im Ordner 'merged_gapless' gespeichert."
"""

import pandas as pd
import os

merged_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'merged')
output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_gapless', 'gaps.txt')

lines = []

for filename in sorted(os.listdir(merged_dir)):
    if not filename.endswith('_merged.csv'):
        continue

    df = pd.read_csv(os.path.join(merged_dir, filename))
    df['datetime'] = pd.to_datetime(df['datetime'])
    missing = df[df['volume'].isna()].copy()

    name = filename.replace('_merged.csv', '')

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
