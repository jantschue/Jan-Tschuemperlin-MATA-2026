"""
Erstellt eine aufgeräumte Korrelationsmatrix auf den v8-Datensätzen für Kapitel 3.3.2.
Die 26 kantonalen Feiertagsspalten und die 26 Schulferienspalten werden je zu einer
Sammelspalte verdichtet (Anteil der Kantone mit Feiertag bzw. Schulferien pro Zeile),
sodass eine lesbare Matrix über die 17 zentralen Merkmale entsteht. Pro Datenreihe
werden die Pearson-Korrelationsmatrix als CSV und als Heatmap (PNG) gespeichert.
"""

import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Pfade relativ zum Skript-Speicherort festlegen
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_dir = os.path.join(base_dir, "data", "v8")
output_dir = os.path.join(base_dir, "results", "data_visualizations", "correlation_analysis")

# Titeltext der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Korrelationsmatrix der Eingabemerkmale"

# Spalten der reduzierten Matrix in fixer Reihenfolge (weather_cat ist kategorial
# und wird bewusst nicht einbezogen, da Pearson dafür nicht sinnvoll ist)
REDUCED_COLUMNS = [
    "volume", "Year",
    "Hour_sin", "Hour_cos",
    "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos",
    "DayOfYear_sin", "DayOfYear_cos",
    "is_weekend",
    "Feiertagsanteil", "Schulferienanteil",
    "temp", "rain_1h", "sun_1h", "snow_1h",
]


def build_reduced_df(df):
    """
    Baut den reduzierten DataFrame: verdichtet die 26 holiday_*- und die 26
    schoolholiday_*-Spalten zu je einer Anteils-Sammelspalte (0 bis 1) und wählt
    die zentralen Merkmale in der definierten Reihenfolge aus.
    """
    holiday_cols = [c for c in df.columns if c.startswith("holiday_")]
    schoolholiday_cols = [c for c in df.columns if c.startswith("schoolholiday_")]

    reduced = df.copy()
    # Mittelwert über alle Kantone = Anteil der Kantone mit Feiertag/Schulferien
    reduced["Feiertagsanteil"] = df[holiday_cols].mean(axis=1)
    reduced["Schulferienanteil"] = df[schoolholiday_cols].mean(axis=1)

    return reduced[REDUCED_COLUMNS]


def main():
    """Berechnet und speichert die reduzierten v8-Korrelationsmatrizen je Datenreihe."""
    # Zielordner erstellen, falls nicht vorhanden
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*_v8.csv")))

    if not csv_files:
        print(f"Keine v8-CSV-Dateien in {input_dir} gefunden.")
        return

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        # Basisname ohne Endung und ohne doppeltes '_v8' (z. B. 720_Schwyz_R1)
        base_name = os.path.splitext(filename)[0]
        if base_name.endswith("_v8"):
            base_name = base_name[: -len("_v8")]

        print(f"Verarbeite Datensatz: {filename} ...")
        df = pd.read_csv(file_path)

        reduced_df = build_reduced_df(df)

        # Pearson-Korrelationsmatrix der reduzierten Merkmale
        corr_matrix = reduced_df.corr()

        # CSV exportieren
        csv_output_path = os.path.join(output_dir, f"{base_name}_v8_correlation_clean.csv")
        corr_matrix.to_csv(csv_output_path)

        # Heatmap generieren (thesenreif: kein Titel, coolwarm von -1 bis 1)
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            annot_kws={"size": 7},
            cbar_kws={"label": "Pearson-Korrelationskoeffizient"},
        )
        # Aussagekraeftiger Titel direkt im Plot
        plt.title(TITLE, fontsize=15, fontweight="bold", pad=12)
        plt.tight_layout()

        # Plot exportieren
        plot_output_path = os.path.join(output_dir, f"{base_name}_v8_correlation_clean.png")
        plt.savefig(plot_output_path, dpi=200)
        plt.close()

    print(f"\nFertig! {len(csv_files)} aufgeräumte v8-Korrelationsmatrizen in '{output_dir}' gespeichert.")


if __name__ == "__main__":
    main()
