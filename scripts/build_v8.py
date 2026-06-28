"""
Dieses Skript lädt die Schulferiendaten, kombiniert sie mit den bestehenden
v7-Verkehrsdatensätzen und speichert das Ergebnis als v8-Datensätze ab.
Für jeden Kanton werden 26 neue 'schoolholiday_<Kanton>'-Spalten hinzugefügt.
"""

import os
import glob
import random
import pandas as pd
import numpy as np
import torch

# Fixierte Seeds für Reproduzierbarkeit (gemäss Konvention, auch wenn hier nicht zwingend nötig)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Konfiguration der Pfade (relativ zum Projektroot)
V7_DIR = "data/v7"
V8_DIR = "data/v8"
SCHULFERIEN_PATH = "data/holidays/raw/schulferien_kantone_2015_2026.csv"

def get_timestamp_col(df):
    """
    Erkennt automatisch die Zeitstempel-Spalte in einem DataFrame.
    """
    possible_names = ["datetime", "timestamp", "time", "date"]
    for col in df.columns:
        if col.lower() in possible_names:
            return col
    
    # Fallback: Suche nach datetime-Datentypen
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
            
    raise ValueError("Keine Zeitstempel-Spalte gefunden!")

def main():
    """
    Hauptfunktion: Führt den Build-Prozess für die v8-Daten durch.
    """
    # 1. Ordner für v8 anlegen, falls nicht existent
    os.makedirs(V8_DIR, exist_ok=True)
    
    # 2. Schulferiendaten laden und vorbereiten
    print(f"Lade Schulferiendaten aus {SCHULFERIEN_PATH}...")
    df_holidays = pd.read_csv(SCHULFERIEN_PATH)
    
    # Datum für den Join vorbereiten
    df_holidays['Datum'] = pd.to_datetime(df_holidays['Datum']).dt.date
    
    # Kantonsspalten umbenennen (Präfix hinzufügen)
    cantons = [col for col in df_holidays.columns if col != 'Datum']
    rename_dict = {c: f"schoolholiday_{c}" for c in cantons}
    df_holidays = df_holidays.rename(columns=rename_dict)
    
    new_holiday_cols = list(rename_dict.values())
    
    # 3. Über alle v7-Dateien iterieren
    v7_files = glob.glob(os.path.join(V7_DIR, "*_v7.csv"))
    
    if not v7_files:
        print(f"Keine *_v7.csv Dateien in {V7_DIR} gefunden.")
        return
        
    for file_path in v7_files:
        file_name = os.path.basename(file_path)
        print(f"\nVerarbeite {file_name}...")
        
        # Einlesen der v7-Datei
        df_v7 = pd.read_csv(file_path)
        original_rows = len(df_v7)
        original_cols = len(df_v7.columns)
        
        # Zeitstempel erkennen und Kalenderdatum für den Join extrahieren
        ts_col = get_timestamp_col(df_v7)
        df_v7['join_date'] = pd.to_datetime(df_v7[ts_col]).dt.date
        
        # Left-Join mit den Schulferiendaten durchführen
        df_v8 = pd.merge(df_v7, df_holidays, left_on='join_date', right_on='Datum', how='left')
        
        # Fehlende Join-Treffer zählen (sollte im Idealfall 0 sein)
        missing_joins = df_v8[new_holiday_cols[0]].isna().sum()
        
        # Fehlwerte (falls vorhanden) mit 0 füllen und sicherstellen, dass die Spalten Integer sind
        for col in new_holiday_cols:
            df_v8[col] = df_v8[col].fillna(0).astype(int)
            
        # Hilfsspalten für das Datum aus dem Join wieder entfernen
        df_v8 = df_v8.drop(columns=['join_date', 'Datum'], errors='ignore')
        
        # Neue Datei speichern
        new_file_name = file_name.replace('_v7', '_v8')
        out_path = os.path.join(V8_DIR, new_file_name)
        df_v8.to_csv(out_path, index=False)
        
        # Qualitätskontrolle für diese Datei ausgeben
        print(f"--- Qualitätskontrolle für {new_file_name} ---")
        
        # Zeilenvergleich
        is_rows_identical = (original_rows == len(df_v8))
        print(f"Zeilen: v7={original_rows} vs v8={len(df_v8)} (Identisch: {is_rows_identical})")
        
        # Spaltenvergleich
        is_cols_correct = (original_cols + 26 == len(df_v8.columns))
        print(f"Spaltenzahl: v7={original_cols} + 26 = {len(df_v8.columns)} (Korrekt: {is_cols_correct})")
        
        # Join-Treffer
        print(f"Zeilen ohne Join-Treffer (sollte 0 sein): {missing_joins}")
        
        # Bestätigung der Spaltenintegrität (alte und neue)
        holiday_cols_v7 = [c for c in df_v7.columns if c.startswith('holiday_')]
        holiday_cols_v8 = [c for c in df_v8.columns if c.startswith('holiday_')]
        has_all_old_holidays = (set(holiday_cols_v7) == set(holiday_cols_v8))
        schoolholidays_exist = all(c in df_v8.columns for c in new_holiday_cols)
        
        if has_all_old_holidays and schoolholidays_exist:
            print("Bestätigung: alle holiday_*-Spalten unverändert vorhanden und 26 neue schoolholiday_*-Spalten ergänzt.")
        else:
            print("WARNUNG: Fehlerhafte oder fehlende holiday_*/schoolholiday_*-Spalten!")

if __name__ == "__main__":
    main()
