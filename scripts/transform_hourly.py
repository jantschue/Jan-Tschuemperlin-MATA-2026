"""
"Transformiert die R1- und R2-CSV-Dateien in ein stündliches Format. Die Originaldateien im Ordner 'data/v2_intermediate/traffic' werden gelesen und stündliche Versionen (*_hourly.csv) am selben Ort gespeichert."
"""

import os
import glob
import pandas as pd
from datetime import datetime, timedelta

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
traffic_dir = os.path.join(base_dir, "data", "v2_intermediate", "traffic")

hour_cols = [f"H{str(i).zfill(2)}" for i in range(24)]

files = glob.glob(os.path.join(traffic_dir, "*_R[12].csv"))

for src_file in files:
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

    out_basename = os.path.basename(src_file).replace(".csv", "_hourly.csv")
    out_file = os.path.join(traffic_dir, out_basename)
    out_df.to_csv(out_file, index=False)
    print(f"  -> Erstellt: {os.path.basename(out_file)}  ({len(out_df)} Zeilen)")

print("\nFertig!")
