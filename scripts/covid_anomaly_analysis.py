"""
Dieses Skript analysiert die COVID-19 bedingten Verkehrsanomalien fuer alle
zehn Messstationen. Es erzeugt pro Station drei Diagramme (monatliches
Durchschnittsvolumen mit COVID-Markierung, Jahresvergleich pro Kalenderwoche
und prozentuale Abweichung von 2020/2021 gegenueber dem Basisjahres-Mittel)
und gibt am Schluss eine Uebersichtstabelle in der Konsole aus. Es handelt
sich um eine reine Read-Only-Analyse; bestehende Skripte werden nicht
veraendert.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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


BASE_DIR = find_project_root(os.path.join("data", "v4_cleaned"))
if BASE_DIR is None:
    raise FileNotFoundError("Konnte 'data/v4_cleaned' weder vom Skript-Ort "
                            "noch vom Arbeitsverzeichnis aus finden.")
INPUT_DIR = os.path.join(BASE_DIR, "data", "v4_cleaned")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "analysis", "covid_anomaly")

# COVID-Zeitraum
COVID_START = pd.Timestamp("2020-03-16")
COVID_END = pd.Timestamp("2021-06-30")

# Vergleichsjahre fuer die Wochen-Analysen
YEAR_COMPARE = [2019, 2020, 2021, 2022]
BASELINE_YEARS = [2018, 2019, 2022, 2023]
ANOMALY_YEARS = [2020, 2021]


def load_station(file_path):
    """Liest eine Stations-CSV ein und konvertiert die datetime-Spalte."""
    df = pd.read_csv(file_path, usecols=["datetime", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def plot_monthly_average(df, station_name, out_path):
    """Plottet das monatliche Durchschnittsvolumen mit roter COVID-Markierung."""
    monthly = (
        df.set_index("datetime")["volume"].resample("MS").mean().dropna()
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly.index, monthly.values, color="steelblue", linewidth=2,
            marker="o", markersize=4, label="Monatliches Durchschnittsvolumen")
    ax.axvspan(COVID_START, COVID_END, color="red", alpha=0.2,
               label="COVID-Periode (16.03.2020 - 30.06.2021)")

    ax.set_title(f"Monatliches Durchschnittsvolumen: {station_name}")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Durchschnittliches Verkehrsvolumen (Fahrzeuge/Stunde)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_year_over_year(df, station_name, out_path):
    """Plottet 2019 vs 2020 vs 2021 vs 2022, jeweils pro Kalenderwoche."""
    df_local = df.copy()
    df_local["year"] = df_local["datetime"].dt.year
    df_local["week"] = df_local["datetime"].dt.isocalendar().week.astype(int)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {2019: "tab:green", 2020: "tab:red",
              2021: "tab:orange", 2022: "tab:blue"}

    for year in YEAR_COMPARE:
        subset = df_local[df_local["year"] == year]
        if subset.empty:
            continue
        weekly = subset.groupby("week")["volume"].mean()
        # Nur Wochen 1-52 anzeigen (Woche 53 ist Sonderfall)
        weekly = weekly[weekly.index <= 52]
        ax.plot(weekly.index, weekly.values, color=colors[year],
                linewidth=2, marker="o", markersize=3, label=str(year))

    ax.axvspan(11, 26, color="red", alpha=0.1,
               label="COVID-Hochphase 2020 (ca. KW 11-26)")
    ax.set_title(f"Jahresvergleich pro Kalenderwoche: {station_name}")
    ax.set_xlabel("Kalenderwoche")
    ax.set_ylabel("Durchschnittliches Verkehrsvolumen (Fahrzeuge/Stunde)")
    ax.set_xticks(range(1, 53, 2))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_percentage_deviation(df, station_name, out_path):
    """Plottet die prozentuale Abweichung von 2020 und 2021 gegenueber dem
    Mittel der Basisjahre (2018, 2019, 2022, 2023) pro Kalenderwoche."""
    df_local = df.copy()
    df_local["year"] = df_local["datetime"].dt.year
    df_local["week"] = df_local["datetime"].dt.isocalendar().week.astype(int)

    # Basismittel: durchschnittliches Wochenvolumen ueber die Basisjahre
    baseline_df = df_local[df_local["year"].isin(BASELINE_YEARS)]
    baseline = baseline_df.groupby("week")["volume"].mean()
    baseline = baseline[baseline.index <= 52]

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.4
    offset = {2020: -bar_width / 2, 2021: bar_width / 2}
    colors = {2020: "tab:red", 2021: "tab:orange"}

    for year in ANOMALY_YEARS:
        subset = df_local[df_local["year"] == year]
        if subset.empty:
            continue
        weekly = subset.groupby("week")["volume"].mean()
        weekly = weekly[weekly.index <= 52]
        # Auf gemeinsame Wochen ausrichten
        common = baseline.index.intersection(weekly.index)
        deviation = (weekly.loc[common] - baseline.loc[common]) / baseline.loc[common] * 100.0
        ax.bar(common + offset[year], deviation.values, width=bar_width,
               color=colors[year], label=str(year), alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Prozentuale Wochenabweichung gegenueber Basisjahren "
                 f"({', '.join(str(y) for y in BASELINE_YEARS)}): {station_name}")
    ax.set_xlabel("Kalenderwoche")
    ax.set_ylabel("Abweichung (%)")
    ax.set_xticks(range(1, 53, 2))
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_summary(df, station_name):
    """Berechnet Zusammenfassungsstatistiken fuer eine Station."""
    total_rows = len(df)
    covid_mask = (df["datetime"] >= COVID_START) & (df["datetime"] <= COVID_END)
    covid_rows = int(covid_mask.sum())
    covid_pct_rows = covid_rows / total_rows * 100.0 if total_rows else float("nan")

    covid_avg = df.loc[covid_mask, "volume"].mean()

    # Baseline: gleiche Kalendermonate aus den Basisjahren
    df_local = df.copy()
    df_local["year"] = df_local["datetime"].dt.year
    baseline_mask = df_local["year"].isin(BASELINE_YEARS)
    baseline_avg = df_local.loc[baseline_mask, "volume"].mean()

    if baseline_avg and not np.isnan(baseline_avg):
        avg_drop_pct = (covid_avg - baseline_avg) / baseline_avg * 100.0
    else:
        avg_drop_pct = float("nan")

    # Schlechtester Monat (kleinster Mittelwert)
    monthly = df.set_index("datetime")["volume"].resample("MS").mean().dropna()
    if not monthly.empty:
        worst_idx = monthly.idxmin()
        worst_month = worst_idx.strftime("%Y-%m")
        worst_value = monthly.loc[worst_idx]
    else:
        worst_month = "n/a"
        worst_value = float("nan")

    return {
        "station": station_name,
        "covid_pct_rows": covid_pct_rows,
        "covid_avg": covid_avg,
        "baseline_avg": baseline_avg,
        "avg_drop_pct": avg_drop_pct,
        "worst_month": worst_month,
        "worst_value": worst_value,
    }


def print_summary_table(rows):
    """Gibt die Zusammenfassung als formatierte Tabelle in der Konsole aus."""
    header = (
        f"{'Station':<45} | {'% Zeilen COVID':>15} | "
        f"{'Volumen-Drop COVID vs Basis (%)':>32} | "
        f"{'Schlechtester Monat':>20} | {'Vol.':>8}"
    )
    print()
    print("=" * len(header))
    print("ZUSAMMENFASSUNG COVID-ANOMALIE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['station']:<45} | "
            f"{r['covid_pct_rows']:>14.2f}% | "
            f"{r['avg_drop_pct']:>31.2f}% | "
            f"{r['worst_month']:>20} | "
            f"{r['worst_value']:>8.1f}"
        )
    print("=" * len(header))


def main():
    csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_merged_gapless.csv")))
    if not csv_files:
        print(f"Keine CSV-Dateien in {INPUT_DIR} gefunden.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_rows = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        station_name = os.path.splitext(filename)[0]
        station_out_dir = os.path.join(OUTPUT_DIR, station_name)
        os.makedirs(station_out_dir, exist_ok=True)

        print(f"Verarbeite {station_name} ...")
        df = load_station(file_path)

        plot_monthly_average(
            df, station_name,
            os.path.join(station_out_dir, "01_monthly_average.png"),
        )
        plot_year_over_year(
            df, station_name,
            os.path.join(station_out_dir, "02_year_over_year_weekly.png"),
        )
        plot_percentage_deviation(
            df, station_name,
            os.path.join(station_out_dir, "03_percentage_deviation.png"),
        )

        summary_rows.append(compute_summary(df, station_name))

    print_summary_table(summary_rows)
    print(f"\nAlle Plots gespeichert unter: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
