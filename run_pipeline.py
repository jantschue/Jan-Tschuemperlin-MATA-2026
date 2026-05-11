"""
"Master-Skript, um das gesamte Projekt vollständig reproduzieren zu können. Es führt nacheinander alle 17 Analyse- und Daten-Aufbereitungsskripte in der korrekten, chronologischen Reihenfolge aus."
"""

import subprocess
import os
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, "scripts")

    pipeline_scripts = [
        "split_directions.py",
        "transform_hourly.py",
        "export_feiertage.py",
        "generate_holidays_hourly.py",
        "filter_weather_luzern_2010_2026.py",
        "filter_weather_waedenswil_2010_2026.py",
        "add_snow_1h.py",
        "categorize_weather.py",
        "drop_snowheight.py",
        "merge_datasets.py",
        "show_gaps.py",
        "clean_merged_data.py",
        "add_time_features.py",
        "create_engineered_features.py",
        "plot_hourly_averages.py",
        "create_correlation_matrix_engineered.py"
    ]

    print("="*60)
    print("Starte vollständige Reproduzierbarkeits-Pipeline...")
    print("="*60)

    for i, script_name in enumerate(pipeline_scripts, start=1):
        script_path = os.path.join(scripts_dir, script_name)
        
        print(f"\n[{i}/{len(pipeline_scripts)}] Führe aus: {script_name} ...")
        
        try:
            # Verwendet denselben Python-Interpreter wie dieses Skript
            result = subprocess.run([sys.executable, script_path], check=True)
            print(f"-> {script_name} erfolgreich beendet.")
        except subprocess.CalledProcessError as e:
            print(f"\n[FEHLER] Pipeline abgebrochen. Das Skript {script_name} hat einen Fehler verursacht.")
            print(f"Exit Code: {e.returncode}")
            sys.exit(1)
        except Exception as e:
            print(f"\n[FEHLER] Konnte {script_name} nicht ausführen: {e}")
            sys.exit(1)

    print("\n" + "="*60)
    print("Pipeline erfolgreich und fehlerfrei abgeschlossen!")
    print("Alle Datensätze und Analysen wurden reproduziert.")
    print("="*60)

if __name__ == "__main__":
    main()
