"""
Erstellt ein stündliches Feiertags-Dataset für den gesamten Zeitraum 2015-2026.
Liest die Feiertage aus 'feiertage_SZ_2015_2026.csv' und erzeugt für jede Stunde
des Zeitraums einen Eintrag mit den Spalten:
  - datetime: Zeitstempel im ISO 8601 Format (z.B. 2015-01-01T00:00:00)
  - is_holiday: 1 wenn die Stunde auf einen Feiertag fällt, sonst 0
Das Ergebnis wird als 'feiertage_SZ_2015_2026_hourly.csv' gespeichert.
"""

import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
holiday_dir = os.path.join(base_dir, "data", "holidays")

# Feiertags-CSV laden
input_file = os.path.join(holiday_dir, "feiertage_SZ_2015_2026.csv")
output_file = os.path.join(holiday_dir, "feiertage_SZ_2015_2026_hourly.csv")

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
