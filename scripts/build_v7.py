"""
Dieses Skript erstellt die v7-Datensätze: Es liest die 10 Corona-bereinigten
v6-Dateien (_withoutcorona.csv), ersetzt die binäre is_holiday-Spalte durch
26 kantonsspezifische Feiertagsspalten (holiday_AG … holiday_ZH) aus der
zentralen Feiertags-Datenbank (data/holidays/swiss_holidays_2015_2025.csv)
und schreibt die Ergebnisse als _v7.csv nach data/v7/.

Die zentrale Feiertags-Datenbank muss zuvor mit scripts/generate_holidays.py
erzeugt worden sein (Phase 1 der Pipeline).
"""

from pathlib import Path
import pandas as pd

# ── Konfiguration ────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = BASE_DIR / "data" / "v6_withoutcorona"
HOLIDAYS_CSV = BASE_DIR / "data" / "holidays" / "swiss_holidays_2015_2025.csv"
OUT_DIR      = BASE_DIR / "data" / "v7"

DATASETS = [
    "050_Brunnen_Mositunnel_R1_withoutcorona.csv",
    "050_Brunnen_Mositunnel_R2_withoutcorona.csv",
    "171_Sattel_R1_withoutcorona.csv",
    "171_Sattel_R2_withoutcorona.csv",
    "216_Wangen_SZ_R1_withoutcorona.csv",
    "216_Wangen_SZ_R2_withoutcorona.csv",
    "299_Wollerau_Blatttunnel_R1_withoutcorona.csv",
    "299_Wollerau_Blatttunnel_R2_withoutcorona.csv",
    "720_Schwyz_R1_withoutcorona.csv",
    "720_Schwyz_R2_withoutcorona.csv",
]

CANTONS = [
    "AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR",
    "JU", "LU", "NE", "NW", "OW", "SG", "SH", "SO", "SZ", "TG",
    "TI", "UR", "VD", "VS", "ZG", "ZH",
]
CANTON_COLS = [f"holiday_{c}" for c in CANTONS]


def load_holidays() -> pd.DataFrame:
    """Laedt die zentrale Feiertags-Datenbank und benennt Kantonsspalten mit holiday_-Praefix."""
    df = pd.read_csv(HOLIDAYS_CSV, parse_dates=["date"])
    df = df.rename(columns={c: f"holiday_{c}" for c in CANTONS})
    df["date"] = df["date"].dt.date
    return df


def build_v7(filename: str, holidays_df: pd.DataFrame) -> pd.DataFrame:
    """Liest eine v6-CSV, fuegt die 26 Feiertagsspalten hinzu und entfernt is_holiday."""
    df = pd.read_csv(DATA_DIR / filename)
    df["_date"] = pd.to_datetime(df["datetime"]).dt.date
    df = df.merge(holidays_df, left_on="_date", right_on="date", how="left")
    df[CANTON_COLS] = df[CANTON_COLS].fillna(0).astype(int)
    df = df.drop(columns=["_date", "date", "is_holiday"], errors="ignore")
    return df


def sanity_check(filename: str, df: pd.DataFrame):
    """Gibt Zeilenzahl, Feiertagszeilen und 3 SZ-Beispielzeilen aus."""
    total = len(df)
    any_holiday = (df[CANTON_COLS].sum(axis=1) > 0).sum()
    print(f"\n-- Sanity-Check: {filename.replace('_withoutcorona', '_v7')} --")
    print(f"  Zeilen gesamt              : {total}")
    print(f"  Zeilen mit mind. 1 Feiertag: {any_holiday}")
    sz_rows = df[df["holiday_SZ"] == 1][["datetime", "holiday_SZ", "holiday_ZH", "holiday_AG"]].head(3)
    print(f"  3 Beispielzeilen holiday_SZ=1:")
    print(sz_rows.to_string(index=False))


def main():
    """Erstellt alle 10 v7-Dateien und gibt einen Sanity-Check fuer die erste Datei aus."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    holidays_df = load_holidays()

    for filename in DATASETS:
        df = build_v7(filename, holidays_df)
        out_name = filename.replace("_withoutcorona", "_v7")
        out_path = OUT_DIR / out_name
        df.to_csv(out_path, index=False)
        print(f"Geschrieben: {out_path}  ({len(df)} Zeilen, {len(df.columns)} Spalten)")

    df_check = pd.read_csv(OUT_DIR / DATASETS[0].replace("_withoutcorona", "_v7"))
    sanity_check(DATASETS[0], df_check)


if __name__ == "__main__":
    main()
