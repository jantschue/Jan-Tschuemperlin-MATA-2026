"""
Master-Skript, um das gesamte Projekt vollstaendig reproduzieren zu koennen.
Die Pipeline laeuft in drei Phasen:
    Phase 1: Datenverarbeitung (Rohdaten -> v5_engineered)
    Phase 2: Analyse + v6_withoutcorona (Corona-bereinigter Datensatz) + v7
    Phase 3: Modelltraining (Lineare Regression & MLP auf v5 und v6, MLP auf v7)
Alle Schritte werden in der korrekten Reihenfolge ausgefuehrt.

Zentrale Feiertags-Datenbank: data/holidays/swiss_holidays_2015_2025.csv
    Erzeugt in Phase 1 durch generate_holidays.py (alle 26 Kantone, 2015-2025).
    merge_datasets.py und build_v7.py lesen ausschliesslich diese Datei.
"""

import subprocess
import os
import sys

def main():
    """Führt die gesamte Pipeline aus: Datenverarbeitung, Analyse und Modelltraining."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, "scripts")
    models_dir = os.path.join(base_dir, "models")

    # Phase 1: Datenverarbeitung (v1 -> v5_engineered)
    # generate_holidays.py erzeugt als erstes die zentrale Feiertags-Datenbank,
    # damit merge_datasets.py (Schritt 9) sie direkt verwenden kann.
    data_scripts = [
        "split_directions.py",
        "transform_hourly.py",
        "generate_holidays.py",          # zentrale Feiertags-DB (alle 26 Kantone)
        "filter_weather_luzern_2010_2026.py",
        "filter_weather_waedenswil_2010_2026.py",
        "add_snow_1h.py",
        "categorize_weather.py",
        "drop_snowheight.py",
        "merge_datasets.py",             # liest aus swiss_holidays_2015_2025.csv
        "show_gaps.py",
        "clean_merged_data.py",
        "add_time_features.py",
        "create_engineered_features.py",
        "plot_hourly_averages.py",
        "create_correlation_matrix_engineered.py",
    ]

    # Phase 2: Analyse + v6_withoutcorona + v7 (kantonsspezifische Feiertage)
    analysis_scripts = [
        "dataset_overview.py",
        "covid_anomaly_analysis.py",
        "create_v6_withoutcorona.py",
        "build_v7.py",                   # v6 + 26 Feiertagsspalten -> v7
    ]

    # Phase 3: Modelltraining (v5 + v6 + v7)
    # v7 hat nur ein MLP (die kantonsspezifische Feiertagskodierung ist eine
    # Verfeinerung des MLP-Hauptmodells; keine eigene lineare Baseline).
    training_scripts = [
        "linear_regression.py",
        "mlp.py",
        "linear_regression_v6.py",
        "mlp_v6.py",
        "mlp_v7.py",
    ]

    total_steps = len(data_scripts) + len(analysis_scripts) + len(training_scripts)
    step_counter = 0

    def run_step(script_path, label):
        """Fuehrt ein Skript aus und beendet die Pipeline bei Fehlern sauber."""
        nonlocal step_counter
        step_counter += 1
        print(f"\n[{step_counter}/{total_steps}] Fuehre aus: {label} ...")
        try:
            subprocess.run([sys.executable, script_path], check=True)
            print(f"-> {label} erfolgreich beendet.")
        except subprocess.CalledProcessError as e:
            print(f"\n[FEHLER] Pipeline abgebrochen. {label} ist mit Fehler beendet worden.")
            print(f"Exit Code: {e.returncode}")
            sys.exit(1)
        except Exception as e:
            print(f"\n[FEHLER] Konnte {label} nicht ausfuehren: {e}")
            sys.exit(1)

    print("=" * 60)
    print("Starte vollstaendige Reproduzierbarkeits-Pipeline...")
    print("=" * 60)

    # --- Phase 1: Datenverarbeitung ---
    print("\n--- Phase 1: Datenverarbeitung (Rohdaten -> v5_engineered) ---")
    for name in data_scripts:
        run_step(os.path.join(scripts_dir, name), name)

    # --- Phase 2: Analyse + v6_withoutcorona + v7 ---
    print("\n--- Phase 2: Analyse + Corona-Bereinigung (v6_withoutcorona) + v7-Aufbau ---")
    for name in analysis_scripts:
        run_step(os.path.join(scripts_dir, name), name)

    # --- Phase 3: Modelltraining (v5 + v6 + v7) ---
    print("\n--- Phase 3: Modelltraining (Lineare Regression & MLP auf v5/v6, MLP auf v7) ---")
    for name in training_scripts:
        run_step(os.path.join(models_dir, name), f"models/{name}")

    print("\n" + "=" * 60)
    print("Pipeline erfolgreich und fehlerfrei abgeschlossen!")
    print("Alle Datensaetze, Analysen und Modelle wurden reproduziert.")
    print("=" * 60)


if __name__ == "__main__":
    main()
