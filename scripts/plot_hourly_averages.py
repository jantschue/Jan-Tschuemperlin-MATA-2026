"""
"Dieses Skript berechnet den durchschnittlichen Verlauf von Temperatur, Sonnenstunden und Verkehrsvolumen über die Stunden des Tages (0-23) für alle Datensätze in 'merged_gapless_time'. Es erstellt Liniendiagramme und speichert diese im Ordner 'hourly_plots'."
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# Definiere Pfade relativ zum Speicherort des Skripts
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_dir = os.path.join(base_dir, "data", "v5")
output_dir = os.path.join(base_dir, "results", "hourly_plots")

def main():
    # Erstelle den Ausgabeordner, falls er nicht existiert
    os.makedirs(output_dir, exist_ok=True)
    
    # Hole alle CSV-Dateien mit '_v5.csv'
    csv_files = glob.glob(os.path.join(input_dir, "*_v5.csv"))
    
    if not csv_files:
        print(f"Keine CSV-Dateien in {input_dir} gefunden.")
        return
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        print(f"Verarbeite Datensatz: {filename}...")
        df = pd.read_csv(file_path)
        # Extrahiere die Stunde aus der datetime-Spalte
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['Hour'] = df['datetime'].dt.hour
        
        # Berechne den Durchschnitt pro Stunde
        hourly_avg = df.groupby('Hour')[['temp', 'sun_1h', 'volume']].mean()
        
        # Erstelle einen Plot mit zwei Y-Achsen (eine für Wetter, eine für Verkehr)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Temperatur und Sonne auf der linken Y-Achse
        color = 'tab:red'
        ax1.set_xlabel('Stunde des Tages')
        ax1.set_ylabel('Temperatur (°C) / Sonnenscheindauer (min)', color=color)
        ax1.plot(hourly_avg.index, hourly_avg['temp'], color='red', label='Temp (°C)', linewidth=2, marker='o')
        ax1.plot(hourly_avg.index, hourly_avg['sun_1h'], color='orange', label='Sonne (min)', linewidth=2, marker='s')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xticks(range(0, 24))
        
        # Verkehrsvolumen auf der rechten Y-Achse
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('Verkehrsvolumen (Durchschnitt)', color=color)  
        ax2.plot(hourly_avg.index, hourly_avg['volume'], color='blue', label='Verkehrsvolumen', linewidth=2, marker='^', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Legenden zusammenführen
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title(f"Durchschnittlicher Tagesverlauf: {base_name}\n(Zeitigt deutlich einen nicht-linearen Kurvenverlauf)")
        fig.tight_layout()  
        
        # Plot speichern
        plot_output_path = os.path.join(output_dir, f"{base_name}_hourly_trend.png")
        plt.savefig(plot_output_path, dpi=300)
        plt.close()
        
    print(f"Fertig! Erfolgreich {len(csv_files)} Plots generiert und in '{output_dir}' gespeichert.")

if __name__ == "__main__":
    main()
