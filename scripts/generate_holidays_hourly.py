"""
VERALTET – nicht mehr Teil der Pipeline.

Die zentrale Feiertags-Datenbank wird jetzt von scripts/generate_holidays.py
erzeugt (data/holidays/swiss_holidays_2015_2025.csv, alle 26 Kantone).
merge_datasets.py liest direkt aus dieser Datei und braucht keine vorprozessierte
stündliche Version mehr. Dieses Skript wird nicht mehr aufgerufen.
"""

import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
holiday_dir_in = os.path.join(base_dir, "data", "holidays", "raw")
holiday_dir_out = os.path.join(base_dir, "data", "holidays", "processed")
os.makedirs(holiday_dir_out, exist_ok=True)

# Feiertags-CSV laden
input_file = os.path.join(holiday_dir_in, "feiertage_SZ_2015_2026.csv")
output_file = os.path.join(holiday_dir_out, "feiertage_SZ_2015_2026_hourly.csv")

print(f"Lade Feiertage aus: {input_file}")
df_holidays = pd.read_csv(input_file)

# Feiertags-Daten als Set (nur Datum ohne Uhrzeit)
holiday_dates = set(pd.to_datetime(df_holidays['Datum']).dt.date)

# Stündlichen Zeitraum erzeugen: 01.01.2015 00:00 bis 31.12.2026 23:00
date_range = pd.date_range(start="2015-01-01", end="2026-12-31 23:00:00", freq="h")

print(f"Erzeuge stündliches Dataset mit {len(date_range)} Einträgen...")

# is_holiday: 1 wenn das Datum ein Feiertag ist, sonst 0
is_holiday = [1 if dt.date() in holiday_dates else 0 for dt in date_range]

df_out = pd.DataFrame({
    "datetime": date_range.strftime("%Y-%m-%dT%H:%M:%S"),
    "is_holiday": is_holiday
})

df_out.to_csv(output_file, index=False)

# Statistik ausgeben
total = len(df_out)
holidays_count = df_out['is_holiday'].sum()
print(f"\nGespeichert: {output_file}")
print(f"  Gesamte Stunden:    {total}")
print(f"  Feiertags-Stunden:  {holidays_count}  ({holidays_count / total * 100:.1f}%)")
print(f"  Normale Stunden:    {total - holidays_count}")
print("\nFertig!")
