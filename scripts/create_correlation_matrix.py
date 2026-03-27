"""
Liest die vorbereiteten Datensätze aus 'merged_gapless_time' ein und berechnet
für jede Datei die Pearson-Korrelationsmatrix über alle numerischen Variablen.
Speichert die berechneten Matrizen als CSV und generiert visuelle Heatmaps
(PNG-Plots) im Ordner 'correlation_analysis'.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob

# Define paths relative to the script location
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_dir = os.path.join(base_dir, "data", "merged_gapless_time")
output_dir = os.path.join(base_dir, "correlation_analysis")

def main():
    # Create the separate folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all csv files
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        
        print(f"Processing dataset: {filename}...")
        df = pd.read_csv(file_path)
        
        # Select only numeric data for the correlation matrix
        numeric_df = df.select_dtypes(include=['number'])
        
        # Calculate the Pearson correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Save the correlation matrix as a CSV file
        csv_output_path = os.path.join(output_dir, f"{base_name}_correlation.csv")
        corr_matrix.to_csv(csv_output_path)
        
        # Create a heatmap visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(f"Korrelationsmatrix: {base_name}")
        plt.tight_layout()
        
        # Save the plot
        plot_output_path = os.path.join(output_dir, f"{base_name}_correlation.png")
        plt.savefig(plot_output_path, dpi=300)
        plt.close()
        
    print(f"Fertig! Erfolgreich {len(csv_files)} Matrizen generiert und in '{output_dir}' gespeichert.")

if __name__ == "__main__":
    main()
