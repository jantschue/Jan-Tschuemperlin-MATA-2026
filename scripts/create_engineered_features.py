"""
"Erstellt neue Datensätze mit zyklischen und binären Features (inklusive is_weekend und is_holiday) sowie relevanten Wetterdaten in einer logischen Reihenfolge für das anschliessende Machine Learning."
"""

import pandas as pd
import numpy as np
import os

# Pfade für In- und Output definieren
input_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'v5_engineered')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'v5_engineered')

# Output-Verzeichnis erstellen, falls es nicht existiert
os.makedirs(output_dir, exist_ok=True)

# Alle Dateien im Input-Verzeichnis verarbeiten
for filename in sorted(os.listdir(input_dir)):
    if not filename.endswith('_merged_gapless_time.csv'):
        continue

    filepath = os.path.join(input_dir, filename)
    df = pd.read_csv(filepath)
    
    # Datetime korrekt parsen für leapyear check
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Neuen DataFrame für die sortierten und engineered Features erstellen
    out_df = pd.DataFrame()

    # 1. Basis-Information
    out_df['datetime'] = df['datetime']
    out_df['volume'] = df['volume']
    out_df['Year'] = df['Year']

    # 2. Zyklische Zeit-Features (von klein nach gross)
    # Stunde (24h)
    out_df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    out_df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # Wochentag (7 Tage) - In Pandas ist DayOfWeek 0-6
    out_df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    out_df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    # Monat (12 Monate)
    out_df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    out_df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Tag des Jahres (365 / 366 Tage Dynamisch basierend auf Schaltjahr)
    days_in_year = df['datetime'].dt.is_leap_year.map({True: 366, False: 365})
    out_df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / days_in_year)
    out_df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / days_in_year)

    # 3. Binäre Kalender-Features
    # is_weekend: 1 für Sa (5) / So (6), sonst 0
    out_df['is_weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    # is_holiday (bereits vorhanden)
    out_df['is_holiday'] = df['is_holiday']

    # 4. Wetterdaten
    out_df['temp'] = df['temp']
    out_df['rain_1h'] = df['rain_1h']
    out_df['sun_1h'] = df['sun_1h']
    out_df['snow_1h'] = df['snow_1h']
    out_df['weather_cat'] = df['weather_cat']

    # Speichern
    output_filename = filename.replace('_merged_gapless_time.csv', '_engineered.csv')
    if '_engineered.csv' not in output_filename:
        output_filename = filename.replace('.csv', '_engineered.csv')
        
    output_path = os.path.join(output_dir, output_filename)
    
    out_df.to_csv(output_path, index=False)
    os.remove(filepath)  # Clean up intermediate file
    print(f'{filename} verarbeitet -> {output_filename} und Original gelöscht')

print(f'\nAlle neuen Datensätze im Ordner "data/v5_engineered" gespeichert.')
