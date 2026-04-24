"""
"Liest die Feature-Engineered-Datensätze ein, berechnet die Pearson-Korrelationsmatrix über alle numerischen Variablen und visualisiert sie als Heatmap. Die Ergebnisse (CSV und Plot) werden im Ordner 'correlation_analysis_engineered' abgelegt."
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# Pfade relativ zum Skript-Speicherort festlegen
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_dir = os.path.join(base_dir, "data", "v5_engineered")
output_dir = os.path.join(base_dir, "results", "correlation_analysis")

def main():
    # Zielordner erstellen, falls nicht vorhanden
    os.makedirs(output_dir, exist_ok=True)
    
    # Alle .csv-Dateien laden
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"Keine CSV-Dateien in {input_dir} gefunden.")
        return
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        print(f"Verarbeite Datensatz: {filename}...")
        df = pd.read_csv(file_path)
        
        # Nur numerische Variablen nutzen (schliesst z.B. weather_cat aus)
        numeric_df = df.select_dtypes(include=['number'])
        
        # Pearson-Korrelationsmatrix berechnen
        corr_matrix = numeric_df.corr()
        
        # CSV exportieren
        csv_output_path = os.path.join(output_dir, f"{base_name}_correlation.csv")
        corr_matrix.to_csv(csv_output_path)
        
        # Visuelle Heatmap generieren (Grösseres Format, da mehr Features)
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
        plt.title(f"Korrelationsmatrix: {base_name}", fontsize=14)
        plt.tight_layout()
        
        # Plot exportieren
        plot_output_path = os.path.join(output_dir, f"{base_name}_correlation.png")
        plt.savefig(plot_output_path, dpi=300)
        plt.close()
        
    print(f"\nFertig! Erfolgreich {len(csv_files)} Matrizen generiert und in '{output_dir}' gespeichert.")

if __name__ == "__main__":
    main()
