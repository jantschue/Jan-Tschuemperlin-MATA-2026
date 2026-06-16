"""
Erstellt 10 zusammengefuehrte Datensets (5 Zaehlstellen x 2 Richtungen). Fuer jede
stundliche Verkehrsdatei wird: 1. Das Verkehrsvolumen geladen, 2. Die is_holiday-Spalte
aus der zentralen Feiertags-Datenbank (Kanton SZ) angehaengt, 3. Die kategorisierten
Wetterdaten angehaengt. Die Dateien werden im Ordner 'data/v3' gespeichert.

Feiertags-Quelle: data/holidays/swiss_holidays_2015_2025.csv (zentrale Datenbank,
generiert von scripts/generate_holidays.py). Es wird die SZ-Spalte als is_holiday
verwendet, da das Projektgebiet im Kanton Schwyz liegt.
"""

import pandas as pd
import os
import glob

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Quell-Dateien ---
weather_file = os.path.join(base_dir, "data", "weather", "processed",
                            "weather_waedenswil_2010-2026_categorized.csv")
# Zentrale Feiertags-Datenbank (alle 26 Kantone, Datumsebene)
holiday_file = os.path.join(base_dir, "data", "holidays",
                            "swiss_holidays_2015_2025.csv")
traffic_dir = os.path.join(base_dir, "data", "v2")

# --- Ausgabe-Ordner ---
output_dir = os.path.join(base_dir, "data", "v3")
os.makedirs(output_dir, exist_ok=True)

# --- Wetter laden ---
print(f"Lade Wetterdaten: {os.path.basename(weather_file)}")
df_weather = pd.read_csv(weather_file)
df_weather.rename(columns={'time': 'datetime'}, inplace=True)

# --- Feiertage laden (SZ-Spalte als is_holiday) ---
# Die zentrale CSV arbeitet auf Datumsebene. Das is_holiday-Flag wird durch
# Abgleich des Datumsteils der datetime-Spalte mit der SZ-Spalte gesetzt.
print(f"Lade Feiertagsdaten: {os.path.basename(holiday_file)}")
df_hol = pd.read_csv(holiday_file, parse_dates=["date"])
sz_holiday_dates = set(df_hol.loc[df_hol["SZ"] == 1, "date"].dt.date)

# --- Alle stündlichen Verkehrsdateien finden ---
traffic_files = sorted(glob.glob(
    os.path.join(traffic_dir, "*_v2.csv")
))

print(f"\n{len(traffic_files)} Verkehrsdateien gefunden.\n")

for traffic_file in traffic_files:
    basename = os.path.basename(traffic_file)
    print(f"Verarbeite: {basename}")

    # Verkehrsdaten laden
    df_traffic = pd.read_csv(traffic_file)

    # 1. Feiertage anhängen: Datumsteil extrahieren, mit SZ-Feiertagen abgleichen
    df_traffic['is_holiday'] = (
        pd.to_datetime(df_traffic['datetime']).dt.date
        .isin(sz_holiday_dates).astype(int)
    )
    df_merged = df_traffic

    # 2. Wetter anhängen (left join auf datetime)
    df_merged = df_merged.merge(df_weather, on='datetime', how='left')

    # Spaltenreihenfolge sicherstellen
    cols = ['datetime', 'volume', 'temp', 'rain_1h', 'sun_1h', 'snow_1h', 'weather_cat', 'is_holiday']
    df_merged = df_merged[cols]

    # Speichern
    out_name = basename.replace('_v2.csv', '_v3.csv')
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

print("\nFertig! Alle Datensets wurden in 'data/v3/' gespeichert.")
