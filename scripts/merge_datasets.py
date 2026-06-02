"""
"Erstellt 10 zusammengeführte Datensets (5 Zählstellen × 2 Richtungen). Für jede stündliche Verkehrsdatei wird: 1. Das Verkehrsvolumen geladen, 2. Die Feiertags-Daten angehängt, 3. Die kategorisierten Wetterdaten angehängt. Die Dateien werden im Ordner 'data/v3_merged' gespeichert."
"""

import pandas as pd
import os
import glob

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Quell-Dateien ---
weather_file = os.path.join(base_dir, "data", "external", "weather", "Waedenswil",
                            "wetter_waedenswil_2010-2026_categorized.csv")
holiday_file = os.path.join(base_dir, "data", "external", "holidays",
                            "feiertage_SZ_2015_2026_hourly.csv")
traffic_dir = os.path.join(base_dir, "data", "v2_intermediate")

# --- Ausgabe-Ordner ---
output_dir = os.path.join(base_dir, "data", "v3_merged")
os.makedirs(output_dir, exist_ok=True)

# --- Wetter laden ---
print(f"Lade Wetterdaten: {os.path.basename(weather_file)}")
df_weather = pd.read_csv(weather_file)
df_weather.rename(columns={'time': 'datetime'}, inplace=True)

# --- Feiertage laden ---
print(f"Lade Feiertagsdaten: {os.path.basename(holiday_file)}")
df_holidays = pd.read_csv(holiday_file)

# --- Alle stündlichen Verkehrsdateien finden ---
traffic_files = sorted(glob.glob(
    os.path.join(traffic_dir, "*_hourly.csv")
))

print(f"\n{len(traffic_files)} Verkehrsdateien gefunden.\n")

for traffic_file in traffic_files:
    basename = os.path.basename(traffic_file)
    print(f"Verarbeite: {basename}")

    # Verkehrsdaten laden
    df_traffic = pd.read_csv(traffic_file)

    # 1. Feiertage anhängen (inner join auf datetime)
    df_merged = df_traffic.merge(df_holidays, on='datetime', how='left')

    # 2. Wetter anhängen (inner join auf datetime)
    df_merged = df_merged.merge(df_weather, on='datetime', how='left')

    # Spaltenreihenfolge sicherstellen
    cols = ['datetime', 'volume', 'temp', 'rain_1h', 'sun_1h', 'snow_1h', 'weather_cat', 'is_holiday']
    df_merged = df_merged[cols]

    # Speichern
    out_name = basename.replace('_hourly.csv', '_merged.csv')
    out_path = os.path.join(output_dir, out_name)
    df_merged.to_csv(out_path, index=False)

    # Statistik
    total = len(df_merged)
    missing_weather = df_merged['temp'].isna().sum()
    missing_holiday = df_merged['is_holiday'].isna().sum()
    print(f"  -> {out_name}  ({total} Zeilen)")
    if missing_weather > 0:
        print(f"     [!] {missing_weather} Zeilen ohne Wetterdaten")
    if missing_holiday > 0:
        print(f"     [!] {missing_holiday} Zeilen ohne Feiertagsdaten")

print("\nFertig! Alle Datensets wurden in 'data/v3_merged/' gespeichert.")
