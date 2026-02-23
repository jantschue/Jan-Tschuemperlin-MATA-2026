"""
Transformiert die R1- und R2-CSV-Dateien in ein stündliches Format.
Für jeden der 5 Ordner werden zwei neue CSV-Dateien erstellt (*_R1_hourly.csv, *_R2_hourly.csv).
Jede Zeile enthält:
  - datetime: Zeitstempel im ISO 8601 Format (z.B. 2015-01-01T00:00:00)
  - volume:   Anzahl Fahrzeuge in dieser Stunde
Die Originaldateien bleiben unverändert.
"""

import os
import glob
import pandas as pd
from datetime import datetime, timedelta

base_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "traffic_volume"
)

hour_cols = [f"H{str(i).zfill(2)}" for i in range(24)]

for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    # R1/R2-Dateien liegen jetzt im Unterordner "Richtungsgetrennt"
    richtung_dir = os.path.join(folder_path, "Richtungsgetrennt")
    if not os.path.isdir(richtung_dir):
        continue

    for suffix in ("_R1.csv", "_R2.csv"):
        matches = glob.glob(os.path.join(richtung_dir, f"*{suffix}"))
        if not matches:
            continue

        src_file = matches[0]
        print(f"Verarbeite: {src_file}")

        df = pd.read_csv(src_file)
        df.columns = df.columns.str.strip()

        # Plausibilitätscheck
        if "JJMMTT" not in df.columns or "H00" not in df.columns:
            print(f"  -> Übersprungen (unerwartetes Format)")
            continue

        rows = []
        for _, row in df.iterrows():
            date_str = str(int(row["JJMMTT"])).zfill(6)       # z.B. "150101"
            year  = 2000 + int(date_str[0:2])                 # 15 -> 2015
            month = int(date_str[2:4])
            day   = int(date_str[4:6])

            try:
                base_dt = datetime(year, month, day)
            except ValueError:
                # Ungültiges Datum → Zeile überspringen
                continue

            for h in range(24):
                col = f"H{str(h).zfill(2)}"
                val = row[col]
                # Leere / ungültige Werte als leeren String speichern
                try:
                    vol = int(float(val))
                except (ValueError, TypeError):
                    vol = ""
                dt = base_dt + timedelta(hours=h)
                rows.append({"datetime": dt.strftime("%Y-%m-%dT%H:%M:%S"), "volume": vol})

        out_df = pd.DataFrame(rows)

        # Output in Richtungsgetrennt_stündlich-Unterordner
        out_dir = os.path.join(richtung_dir, "Richtungsgetrennt_stündlich")
        os.makedirs(out_dir, exist_ok=True)

        out_basename = os.path.basename(src_file).replace(suffix, suffix.replace(".csv", "_hourly.csv"))
        out_file = os.path.join(out_dir, out_basename)
        out_df.to_csv(out_file, index=False)
        print(f"  -> Erstellt: {os.path.basename(out_file)}  ({len(out_df)} Zeilen)")

print("\nFertig!")
