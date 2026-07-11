"""
Erstellt thesenreife (aufgeräumte) Versionen des durchschnittlichen Tagesverlaufs
je Datenreihe auf den aktuellen v8-Daten. Gezeigt wird der über alle Tage gemittelte
Verlauf von Temperatur, Sonnenscheindauer und Verkehrsvolumen über die 24 Stunden.
Im Gegensatz zur Diagnostik-Variante (plot_hourly_averages.py) trägt die Abbildung
keinen technischen Titel mit Stationscode/Versionskürzel, sondern einen aussagekräftigen
Titel ("Tagesverlauf von Wetter und Verkehrsvolumen"); die genaue Beschreibung liefert
die Bildunterschrift im Dokument.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Pfade relativ zum Skript-Speicherort festlegen
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_dir = os.path.join(base_dir, "data", "v8")
output_dir = os.path.join(base_dir, "results", "data_visualizations", "hourly_plots")

# Titeltext der Abbildung (Schweizer Schreibweise, kein Gedankenstrich)
TITLE = "Tagesverlauf von Wetter und Verkehrsvolumen"


def main():
    """Berechnet und speichert je v8-Datenreihe den aufgeräumten Tagesverlauf-Plot."""
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

        # Stunde aus der datetime-Spalte extrahieren
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["Hour"] = df["datetime"].dt.hour

        # Durchschnitt pro Stunde des Tages berechnen
        hourly_avg = df.groupby("Hour")[["temp", "sun_1h", "volume"]].mean()

        # Plot mit zwei Y-Achsen: Wetter links, Verkehr rechts
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Temperatur und Sonnenscheindauer auf der linken Y-Achse
        left_color = "tab:red"
        ax1.set_xlabel("Stunde des Tages")
        ax1.set_ylabel("Temperatur (°C) / Sonnenscheindauer (min)", color=left_color)
        ax1.plot(hourly_avg.index, hourly_avg["temp"], color="red",
                 label="Temperatur (°C)", linewidth=2, marker="o")
        ax1.plot(hourly_avg.index, hourly_avg["sun_1h"], color="orange",
                 label="Sonnenscheindauer (min)", linewidth=2, marker="s")
        ax1.tick_params(axis="y", labelcolor=left_color)
        ax1.grid(True, linestyle="--", alpha=0.7)
        ax1.set_xticks(range(0, 24))

        # Verkehrsvolumen auf der rechten Y-Achse
        ax2 = ax1.twinx()
        right_color = "tab:blue"
        ax2.set_ylabel("Verkehrsvolumen (Durchschnitt pro Stunde)", color=right_color)
        ax2.plot(hourly_avg.index, hourly_avg["volume"], color="blue",
                 label="Verkehrsvolumen", linewidth=2, marker="^", linestyle="--")
        ax2.tick_params(axis="y", labelcolor=right_color)

        # Legenden beider Achsen zusammenführen
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        # Aussagekraeftiger Titel direkt im Plot (ohne Stationscode/Versionskuerzel)
        ax1.set_title(TITLE, fontsize=14, fontweight="bold", pad=12)

        fig.tight_layout()

        # Plot exportieren
        plot_output_path = os.path.join(output_dir, f"{base_name}_v8_hourly_trend_clean.png")
        plt.savefig(plot_output_path, dpi=200)
        plt.close()

    print(f"\nFertig! {len(csv_files)} aufgeräumte Tagesverlauf-Plots in '{output_dir}' gespeichert.")


if __name__ == "__main__":
    main()
