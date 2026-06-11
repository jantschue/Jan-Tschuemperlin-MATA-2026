"""
Dieses Skript generiert die zentrale Feiertags-Datenbank des Projekts:
data/holidays/swiss_holidays_2015_2025.csv

Die Datei enthält alle gesetzlichen Feiertage aller 26 Schweizer Kantone von
2015 bis 2025 (inklusive). Jede Zeile ist ein Datum, an dem mindestens ein
Kanton Feiertag hat. Die 26 Kantonsspalten enthalten 1 (Feiertag) oder 0.

Dies ist die EINZIGE Feiertags-Quelle im Projekt. Alle anderen Skripte, die
Feiertagsinformationen benötigen, lesen ausschliesslich diese Datei.
"""

from pathlib import Path
import pandas as pd
import holidays

# ── Konfiguration ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
YEARS    = range(2015, 2026)   # 2015–2025 inkl. (Abdeckung der gesamten Datenbasis)
OUT_DIR  = BASE_DIR / "data" / "holidays"
OUT_FILE = OUT_DIR / "swiss_holidays_2015_2025.csv"

# 26 Schweizer Kantonskürzel (eindeutig, alphabetisch)
CANTONS = [
    "AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR",
    "JU", "LU", "NE", "NW", "OW", "SG", "SH", "SO", "SZ", "TG",
    "TI", "UR", "VD", "VS", "ZG", "ZH",
]


def generate():
    """Sammelt alle Feiertage je Kanton und Jahr, pivotiert in ein breites Format
    (eine Spalte pro Kanton) und speichert die CSV."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    canton_dates: dict[str, set] = {c: set() for c in CANTONS}
    for year in YEARS:
        for canton in CANTONS:
            h = holidays.Switzerland(years=year, prov=canton)
            canton_dates[canton].update(h.keys())

    all_dates = sorted(set().union(*canton_dates.values()))

    rows = []
    for date in all_dates:
        row = {"date": date.strftime("%Y-%m-%d")}
        for canton in CANTONS:
            row[canton] = 1 if date in canton_dates[canton] else 0
        rows.append(row)

    df = pd.DataFrame(rows, columns=["date"] + CANTONS)
    df.to_csv(OUT_FILE, index=False)
    print(f"{len(df)} Feiertagsdaten geschrieben nach {OUT_FILE}")
    return df


if __name__ == "__main__":
    generate()
