"""
Liest rohe Verkehrsdaten-CSV-Dateien ein und teilt diese nach den beiden Fahrtrichtungen (R1 und R2) auf. 
Die aufgeteilten Daten werden in separaten CSV-Dateien im Unterordner 'Richtungsgetrennt' gespeichert.
"""

import pandas as pd
import os
import glob

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
traffic_dir = os.path.join(base_dir, "data", "traffic_volume")

# Raw-Dateien liegen jetzt in Unterordnern pro Ort: traffic_volume/<Ort>/*_raw.CSV
files = glob.glob(os.path.join(traffic_dir, "**", "*_raw.CSV"), recursive=True)

for file in files:
    try:
        # Some rows might have spaces as missing values, and columns might have leading/trailing whitespaces in names
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        
        common_cols = ['ZST', 'TAG', 'T', 'JJMMTT']
        r1_cols = [f'R1H{str(i).zfill(2)}' for i in range(24)]
        r2_cols = [f'R2H{str(i).zfill(2)}' for i in range(24)]
        
        if all(c in df.columns for c in common_cols + r1_cols + r2_cols):
            df_r1 = df[common_cols + r1_cols].copy()
            df_r1.columns = common_cols + [f'H{str(i).zfill(2)}' for i in range(24)]
            df_r1 = df_r1.sort_values(by=['JJMMTT']).reset_index(drop=True)
            
            df_r2 = df[common_cols + r2_cols].copy()
            df_r2.columns = common_cols + [f'H{str(i).zfill(2)}' for i in range(24)]
            df_r2 = df_r2.sort_values(by=['JJMMTT']).reset_index(drop=True)
            
            # Output in Richtungsgetrennt-Unterordner
            out_dir = os.path.join(os.path.dirname(file), "Richtungsgetrennt")
            os.makedirs(out_dir, exist_ok=True)
            
            basename = os.path.basename(file)
            out_file_r1 = os.path.join(out_dir, basename.replace('_raw.CSV', '_R1.csv'))
            df_r1.to_csv(out_file_r1, index=False)
            
            out_file_r2 = os.path.join(out_dir, basename.replace('_raw.CSV', '_R2.csv'))
            df_r2.to_csv(out_file_r2, index=False)
            print(f"Erstellt: {os.path.basename(out_file_r1)} und {os.path.basename(out_file_r2)}")
        else:
            print(f"Unerwartetes Format, übersprungen: {os.path.basename(file)}")
    except Exception as e:
        print(f"Fehler bei {os.path.basename(file)}: {e}")
