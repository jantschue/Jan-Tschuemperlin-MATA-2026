"""
Dieses Skript erstellt eine Uebersicht ueber alle Engineered-CSV-Dateien.
Pro Datei werden Zeilenanzahl, Zeitraum, erwartete Zeilen bei stuendlicher
Frequenz, fehlende Zeilen, Luecken laenger als 24 Stunden und der COVID-
Anteil ausgegeben. Es handelt sich um eine reine Read-Only-Analyse;
bestehende Skripte werden nicht veraendert.
"""

import os
import glob
import pandas as pd

def find_project_root(marker_subpath):
    """Sucht aufwaerts vom Skript-Ort den ersten Ordner, der den Marker enthaelt.
    Faengt damit unterschiedliche Ablageorte des Skripts ab (z. B. scripts/ oder
    src/analysis/), damit die Pfadaufloesung nicht von der Verschachtelungstiefe
    abhaengt."""
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


BASE_DIR = find_project_root(os.path.join("data", "v5_engineered"))
if BASE_DIR is None:
    raise FileNotFoundError("Konnte 'data/v5_engineered' weder vom Skript-Ort "
                            "noch vom Arbeitsverzeichnis aus finden.")
INPUT_DIR = os.path.join(BASE_DIR, "data", "v5_engineered")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "analysis", "dataset_overview")

# COVID-Zeitraum
COVID_START = pd.Timestamp("2020-03-16")
COVID_END = pd.Timestamp("2021-06-30 23:00:00")

# Schwellwert fuer "lange" Luecken
GAP_THRESHOLD_HOURS = 24


def analyze_file(file_path):
    """Berechnet alle Kennzahlen fuer eine einzelne CSV-Datei."""
    df = pd.read_csv(file_path, usecols=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    total_rows = len(df)
    first_ts = df["datetime"].iloc[0]
    last_ts = df["datetime"].iloc[-1]

    # Erwartete Zeilen bei stuendlicher Frequenz (einschliesslich Endpunkt)
    expected_rows = int((last_ts - first_ts) / pd.Timedelta(hours=1)) + 1
    missing_rows = expected_rows - total_rows

    # Luecken: aufeinanderfolgende Zeitstempel mit Differenz > GAP_THRESHOLD_HOURS
    diffs = df["datetime"].diff()
    gap_mask = diffs > pd.Timedelta(hours=GAP_THRESHOLD_HOURS)
    long_gaps = []
    for idx in df.index[gap_mask]:
        gap_start = df["datetime"].iloc[idx - 1]
        gap_end = df["datetime"].iloc[idx]
        gap_hours = (gap_end - gap_start) / pd.Timedelta(hours=1) - 1
        long_gaps.append((gap_start, gap_end, int(gap_hours)))

    # COVID-Anteil
    covid_mask = (df["datetime"] >= COVID_START) & (df["datetime"] <= COVID_END)
    covid_rows = int(covid_mask.sum())
    covid_pct = covid_rows / total_rows * 100.0 if total_rows else 0.0

    return {
        "filename": os.path.basename(file_path),
        "total_rows": total_rows,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "expected_rows": expected_rows,
        "missing_rows": missing_rows,
        "long_gaps": long_gaps,
        "covid_rows": covid_rows,
        "covid_pct": covid_pct,
    }


def build_overview_table(results):
    """Baut die Hauptuebersicht als Liste von Strings (eine Zeile pro Element)."""
    header = (
        f"{'Datei':<42} | {'Zeilen':>8} | {'Erwartet':>9} | "
        f"{'Fehlend':>8} | {'Erster Zeitstempel':<19} | "
        f"{'Letzter Zeitstempel':<19} | {'COVID':>7} | {'COVID %':>8}"
    )
    lines = []
    lines.append("=" * len(header))
    lines.append("DATENSATZ-UEBERSICHT (Engineered Files)")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        lines.append(
            f"{r['filename']:<42} | "
            f"{r['total_rows']:>8} | "
            f"{r['expected_rows']:>9} | "
            f"{r['missing_rows']:>8} | "
            f"{r['first_ts'].strftime('%Y-%m-%d %H:%M'):<19} | "
            f"{r['last_ts'].strftime('%Y-%m-%d %H:%M'):<19} | "
            f"{r['covid_rows']:>7} | "
            f"{r['covid_pct']:>7.2f}%"
        )
    lines.append("=" * len(header))
    return lines


def build_gap_details(results):
    """Baut den Luecken-Bericht als Liste von Strings."""
    lines = []
    lines.append("")
    lines.append("=" * 100)
    lines.append(f"LUECKEN > {GAP_THRESHOLD_HOURS} STUNDEN")
    lines.append("=" * 100)
    for r in results:
        lines.append("")
        lines.append(f"{r['filename']}:")
        if not r["long_gaps"]:
            lines.append("  Keine Luecken laenger als der Schwellwert.")
            continue
        lines.append(f"  {'Luecken-Start':<19} -> {'Luecken-Ende':<19} | {'Dauer (h)':>10}")
        lines.append(f"  {'-' * 19}    {'-' * 19}   {'-' * 10}")
        for gap_start, gap_end, gap_hours in r["long_gaps"]:
            lines.append(
                f"  {gap_start.strftime('%Y-%m-%d %H:%M'):<19} -> "
                f"{gap_end.strftime('%Y-%m-%d %H:%M'):<19} | "
                f"{gap_hours:>10}"
            )
    lines.append("=" * 100)
    return lines


def write_outputs(results, report_lines):
    """Speichert den Textbericht und zwei CSV-Dateien im OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Textbericht (gleiche formatierte Ausgabe wie auf der Konsole)
    report_path = os.path.join(OUTPUT_DIR, "dataset_overview.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        f.write("\n")

    # Maschinenlesbare Uebersicht
    overview_rows = [
        {
            "filename": r["filename"],
            "total_rows": r["total_rows"],
            "first_timestamp": r["first_ts"].strftime("%Y-%m-%d %H:%M:%S"),
            "last_timestamp": r["last_ts"].strftime("%Y-%m-%d %H:%M:%S"),
            "expected_rows": r["expected_rows"],
            "missing_rows": r["missing_rows"],
            "covid_rows": r["covid_rows"],
            "covid_pct": round(r["covid_pct"], 4),
        }
        for r in results
    ]
    pd.DataFrame(overview_rows).to_csv(
        os.path.join(OUTPUT_DIR, "dataset_overview.csv"),
        index=False,
    )

    # Detail-CSV mit allen langen Luecken
    gap_rows = []
    for r in results:
        for gap_start, gap_end, gap_hours in r["long_gaps"]:
            gap_rows.append(
                {
                    "filename": r["filename"],
                    "gap_start": gap_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "gap_end": gap_end.strftime("%Y-%m-%d %H:%M:%S"),
                    "gap_hours": gap_hours,
                }
            )
    pd.DataFrame(
        gap_rows,
        columns=["filename", "gap_start", "gap_end", "gap_hours"],
    ).to_csv(os.path.join(OUTPUT_DIR, "long_gaps.csv"), index=False)


def main():
    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_engineered.csv")))
    if not csv_files:
        print(f"Keine CSV-Dateien in {INPUT_DIR} gefunden.")
        return

    print(f"Analysiere {len(csv_files)} Datei(en) aus {INPUT_DIR} ...\n")
    results = [analyze_file(fp) for fp in csv_files]

    report_lines = build_overview_table(results) + build_gap_details(results)
    for line in report_lines:
        print(line)

    write_outputs(results, report_lines)
    print(f"\nErgebnisse gespeichert unter: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
