"""
"Liest die Wetterdaten für Luzern (2010-2019 und 2020-2026) ein, fügt diese zusammen, filtert die relevanten Spalten (Zeit, Temperatur, Regen, Schnee, Sonne) heraus, formatiert die Zeitstempel nach ISO 8601 und speichert die bereinigten Daten in einer neuen CSV-Datei."
"""

import pandas as pd
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_path_in = os.path.join(base_dir, "data", "weather", "raw", "Luzern")
    base_path_out = os.path.join(base_dir, "data", "weather", "processed")
    os.makedirs(base_path_out, exist_ok=True)
    file1 = os.path.join(base_path_in, "wetter_luzern_2010-2019.csv")
    file2 = os.path.join(base_path_in, "wetter_luzern_2020-2026.csv")
    output_file = os.path.join(base_path_out, "wetter_luzern_2010-2026_filtered.csv")

    print(f"Lade Daten von {file1}...")
    df1 = pd.read_csv(file1, sep=';', low_memory=False)
    
    print(f"Lade Daten von {file2}...")
    df2 = pd.read_csv(file2, sep=';', low_memory=False)

    print("Füge Daten zusammen...")
    df = pd.concat([df1, df2], ignore_index=True)

    # Behalte nur die gewünschten Spalten
    columns_to_keep = ['reference_timestamp', 'tre200h0', 'rre150h0', 'sre000h0', 'htoauths']
    df_filtered = df[columns_to_keep].copy()

    print("Konvertiere Zeitstempel zu ISO 8601 und benenne Spalten um...")
    df_filtered = df_filtered.rename(columns={
        'reference_timestamp': 'time',
        'rre150h0': 'rain_1h',
        'tre200h0': 'temp',
        'htoauths': 'snowheight',
        'sre000h0': 'sun_1h'
    })
    
    # Date Format umwandeln
    df_filtered['time'] = pd.to_datetime(df_filtered['time'], format='%d.%m.%Y %H:%M')
    df_filtered['time'] = df_filtered['time'].dt.strftime('%Y-%m-%dT%H:%M:%S')

    print(f"Speichere gefilterte Daten unter {output_file}...")
    df_filtered.to_csv(output_file, index=False)
    
    print("Abgeschlossen!")

if __name__ == "__main__":
    main()
