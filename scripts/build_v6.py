"""
Dieses Skript entfernt den Corona-Zeitraum aus den Engineered-CSV-Dateien.
Alle Zeilen mit datetime zwischen 2020-03-16 und 2021-02-28 (inklusive) werden
verworfen, die uebrigen Zeilen unter gleichem Dateinamen nach
data/v6/ geschrieben. Pro Datei wird anschliessend eine kurze
Zusammenfassung ausgegeben. Bestehende Dateien und Ordner werden nicht
veraendert.
"""

import os
import glob
import pandas as pd


def find_project_root(marker_subpath):
    """Sucht aufwaerts vom Skript-Ort den ersten Ordner, der den Marker
    enthaelt. Macht die Pfadaufloesung unabhaengig von der Ablagetiefe."""
    start = os.path.dirname(os.path.abspath(__file__))
    current = start
    for _ in range(8):
        if os.path.isdir(os.path.join(current, marker_subpath)):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    # Fallback: aktuelles Arbeitsverzeichnis
    if os.path.isdir(os.path.join(os.getcwd(), marker_subpath)):
        return os.getcwd()
    return None


BASE_DIR = find_project_root(os.path.join("data", "v5"))
if BASE_DIR is None:
    raise FileNotFoundError("Konnte 'data/v5' weder vom Skript-Ort "
                            "noch vom Arbeitsverzeichnis aus finden.")
INPUT_DIR = os.path.join(BASE_DIR, "data", "v5")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "v6")

# Auszuschliessender Zeitraum (inklusive beider Grenzen)
CORONA_START = pd.Timestamp("2020-03-16 00:00:00")
CORONA_END = pd.Timestamp("2021-02-28 23:00:00")


def process_file(file_path):
    """Liest eine CSV-Datei, entfernt den Corona-Zeitraum und schreibt sie
    mit dem Suffix '_v6.csv' nach OUTPUT_DIR. Gibt eine Statistik
    fuer die Zusammenfassung zurueck."""
    filename = os.path.basename(file_path)
    # Suffix umbenennen, damit v6-Dateien sich klar von v5 unterscheiden
    out_filename = filename.replace("_v5.csv", "_v6.csv")

    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    original_rows = len(df)

    # Maske: alle Zeilen, die NICHT im Corona-Zeitraum liegen, bleiben erhalten
    corona_mask = (df["datetime"] >= CORONA_START) & (df["datetime"] <= CORONA_END)
    removed_rows = int(corona_mask.sum())
    df_filtered = df.loc[~corona_mask].copy()
    df_filtered = df_filtered.sort_values("datetime").reset_index(drop=True)

    new_rows = len(df_filtered)
    if new_rows:
        first_ts = df_filtered["datetime"].iloc[0]
        last_ts = df_filtered["datetime"].iloc[-1]
    else:
        first_ts = None
        last_ts = None

    out_path = os.path.join(OUTPUT_DIR, out_filename)
    df_filtered.to_csv(out_path, index=False)

    return {
        "filename": out_filename,
        "original_rows": original_rows,
        "removed_rows": removed_rows,
        "new_rows": new_rows,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def print_summary(results):
    """Gibt eine kompakte Tabelle mit den wichtigsten Kennzahlen aus."""
    header = (
        f"{'Datei':<42} | {'Original':>9} | {'Entfernt':>9} | "
        f"{'Neu':>9} | {'Neuer Zeitraum (von -> bis)':<43}"
    )
    print("=" * len(header))
    print("ZUSAMMENFASSUNG: Corona-Bereinigung (v5 -> v6)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        if r["first_ts"] is not None and r["last_ts"] is not None:
            range_str = (
                f"{r['first_ts'].strftime('%Y-%m-%d %H:%M')} -> "
                f"{r['last_ts'].strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            range_str = "leerer Datensatz"
        print(
            f"{r['filename']:<42} | "
            f"{r['original_rows']:>9} | "
            f"{r['removed_rows']:>9} | "
            f"{r['new_rows']:>9} | "
            f"{range_str:<43}"
        )
    print("=" * len(header))


def main():
    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_v5.csv")))
    if not csv_files:
        print(f"Keine CSV-Dateien in {INPUT_DIR} gefunden.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Bereinige {len(csv_files)} Datei(en) aus {INPUT_DIR}")
    print(f"Ausschluss-Zeitraum: {CORONA_START} bis {CORONA_END} (inklusive)\n")

    results = [process_file(fp) for fp in csv_files]

    print_summary(results)
    print(f"\nGespeichert unter: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
