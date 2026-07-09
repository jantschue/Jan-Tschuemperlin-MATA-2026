# Maturaarbeit: Verkehrsvolumen-Prognose im Kanton Schwyz (2026)

**Autor:** Jan Tschümperlin, Klasse 3d

In meiner Maturaarbeit untersuche ich, ob sich das stündliche Verkehrsvolumen an Schweizer Nationalstrassen mithilfe von Machine Learning zuverlässig vorhersagen lässt. Dazu habe ich Verkehrsdaten des ASTRA, Wetterdaten von MeteoSchweiz und Feiertagsinformationen für den Kanton Schwyz zusammengeführt und zwei Modelle trainiert: ein lineares Regressionsmodell als Baseline und ein MLP (Multi-Layer Perceptron) als Hauptmodell.

Um den Einfluss der Corona-Anomalie auf die Modellgüte zu testen, werden beide Modelle zusätzlich auf einem Corona-bereinigten Datensatz (`v6`) trainiert und mit den Original-Resultaten verglichen.

Eine dritte Variante (`v7`) verfeinert die Feiertagskodierung: Statt eines einzigen `is_holiday`-Flags (Kanton Schwyz) erhält das MLP **26 kantonsspezifische Feiertagsspalten** (`holiday_AG … holiday_ZH`), um zu prüfen, ob diese zusätzliche Information die Vorhersage verbessert. Die Hyperparameter des v7-MLP wurden separat per Optuna getunt (das Feature-Set hat sich geändert).

Eine vierte Variante (`v8`) baut auf v7 auf und fügt zusätzlich **26 kantonsspezifische Schulferienspalten** (`schoolholiday_AG … schoolholiday_ZH`) hinzu. Auch dieses Modell wurde mit Optuna neu getunt, da der Feature-Satz auf 67 Spalten anwuchs.

## Interaktive Webapp

Die Ergebnisse sind als interaktive Web-Applikation verfügbar:

**[tschue.ch](https://tschue.ch)**

Die Webapp ermöglicht:
- **Stationskarte** – Übersicht aller 5 Messstationen mit MLP-Vorhersagegüte (R²) farbcodiert auf einer Leaflet-Karte
- **Live-Vorhersage** – MLP und lineare Regression rechnen Prognosen live im Browser (aktuelle Wetterdaten via Open-Meteo, automatische Feiertagserkennung für Kt. Schwyz, manueller Schulferien-Schalter)
- **Datums-Analyse** – Tagesverlauf eines beliebigen Datums im Testset: Ist-Werte vs. MLP- und LR-Vorhersagen
- **Feiertags-Analyse** – Zeigt für jeden Schwyzer Feiertag im Testset, wie stark der Verkehr vom Wochentags-Durchschnitt abweicht, wie gut das Modell diesen Tag vorhersagt und welchen isolierten Effekt die Feiertagskodierung hat (Counterfactual: selber Stunden-Vektor mit allen `holiday_*`-Spalten auf 0 vs. dem realen Feiertag)
- **Schulferien-Analyse** – Analog zur Feiertags-Analyse, aber für die mehrwöchigen Schwyzer Schulferien (Sport-, Frühlings-, Sommer-, Herbst-, Weihnachtsferien): durchschnittliches Tagesprofil je Ferienperiode, Abweichung von normalen Schulwochen, Modellgüte und der isolierte Effekt des Schulferien-Features (Counterfactual mit allen `schoolholiday_*`-Spalten auf 0)
- **Feature-Sensitivität** – Wie stark verändern Uhrzeit, Temperatur, Niederschlag und Sonnenstunden die Vorhersage (Sweep über den vollen Wertebereich, beide Modelle); Feiertag und Schulferien als Basislinien-Schalter
- **Ausreisseranalyse** – Durchsucht alle stündlichen Ergebnisse nach grossen Fehlern, Peak-Versagen und Wochentag-Mustern; Export als CSV

Der Quellcode der Webapp liegt unter `webapp/`. Die Webapp wird via Vercel deployed.

## Datenquellen & Danksagung

Die verwendeten Rohdaten stammen aus folgenden offiziellen Quellen:

- **Verkehrsdaten:** Bundesamt für Strassen (ASTRA). Die Veröffentlichung der aufbereiteten Daten erfolgt mit freundlicher Genehmigung des ASTRA.
- **Wetterdaten:** Bundesamt für Meteorologie und Klimatologie (MeteoSchweiz) über das [Daten-Portal](https://www.meteoschweiz.admin.ch/service-und-publikationen/applikationen/ext/daten-ohne-programmierkenntnisse-herunterladen.html)
- **Feiertage:** Automatisch generiert mit dem Python-Modul `holidays` (alle 26 Schweizer Kantone, 2015–2025). Zentrale Datenbank: `data/holidays/swiss_holidays_2015_2025.csv`.

*Jegliche Weiterverwendung der rohen Verkehrsdaten erfordert eine eigene Abklärung mit dem ASTRA.*

## Projektstruktur

### Datenpipeline

Die Rohdaten durchlaufen mehrere Verarbeitungsstufen, bevor sie für das Modelltraining verwendet werden:

| Ordner | Inhalt |
|---|---|
| `data/v1/` | Unveränderte ASTRA-Verkehrsrohdaten (`traffic/`) |
| `data/v2/` | Verarbeitete Verkehrsdaten direkt im Ordner: Richtungsaufspaltung (R1/R2) und Stundenaggregation (10 `*_v2.csv`-Dateien) |
| `data/weather/` | Externe Wetterdaten (MeteoSchweiz): `raw/` mit den Stations-Rohdateien (Luzern, Wädenswil), `processed/` mit den gefilterten und kategorisierten Datensätzen |
| `data/holidays/` | Zentrale Feiertags-Datenbank: `swiss_holidays_2015_2025.csv` (alle 26 Kantone, erzeugt von `scripts/generate_holidays.py`). Alle Skripte lesen ausschliesslich diese Datei. |
| `data/v3/` | Verkehr + Wetter + Feiertage zusammengeführt (10 Dateien, je Station × Richtung) |
| `data/v4/` | NaN-Zeilen entfernt |
| `data/v5/` | Fertige ML-Features: zyklische Zeitkodierung, Wetterklassen, alle 16 Feature-Spalten |
| `data/v6/` | `v5` ohne den Corona-Anomaliebereich (2020-03-16 bis 2021-02-28) |
| `data/v7/` | `v6` mit 26 kantonsspezifischen Feiertagsspalten (holiday_AG … holiday_ZH) statt binärem `is_holiday` |
| `data/v8/` | `v7` mit zusätzlichen 26 kantonsspezifischen Schulferienspalten (schoolholiday_AG … schoolholiday_ZH) |

**Warum `data/weather/` und `data/holidays/` separat?** Wetter- und Feiertagsdaten stammen aus externen Quellen (MeteoSchweiz, Python-`holidays`-Modul, kantonale Schulferien-CSVs) und sind unabhängig von der ASTRA-Verkehrs-Verarbeitungskette. Sie liegen daher in eigenen Ordnern. Die Trennung macht den Datenfluss klarer: Das `v1 → … → v8`-Schema enthält ausschliesslich ASTRA-Verkehrsdaten, `weather/` und `holidays/` die externen Datenquellen.

### Ordnerübersicht

| Ordner | Inhalt |
|---|---|
| `scripts/` | Python-Skripte für die Datenverarbeitung (v1 → v8), die Corona-Bereinigung (v6), den v7/v8-Aufbau, explorative Analysen sowie die Parameter-vs-Performance-Studien (v6, v7, v8), den Trainingsverlauf-Plot (v7) und die Permutations-Wichtigkeit (v8) |
| `models/` | Trainings-Skripte für die ML-Modelle (Varianten für v5, v6, v7, v8) sowie Optuna-Tuning (v5, v7, v8) |
| `results/model_results/` | Metriken, Plots und Vorhersage-Vergleiche pro Modell (separat für v5-, v6-, v7- und v8-Varianten) |
| `results/analysis/` | Output der explorativen Analyse-Skripte (Datensatz-Übersicht, COVID-Anomalie, Permutations-Wichtigkeit der Merkmalsgruppen) |
| `results/data_visualizations/` | Diagnostik der Datenpipeline: Korrelationsmatrizen, Tagesverlaufs-Plots und Lücken-Übersicht (`gaps.txt`) |
| `webapp/` | Interaktive React-Webapp (Vite + Tailwind + Recharts); live unter [tschue.ch](https://tschue.ch) |
| `Theorie/` | Lokale Erklärungsdateien zu den Modellskripten (Prüfungsvorbereitung; nicht im Repository) |
| `Tutorials/` | Lernmaterialien, die ich während der Einarbeitung in Python, NumPy, Pandas und PyTorch erstellt habe |
| `Abbildungen_MATA/` | Abbildungen für die schriftliche Arbeit samt erzeugendem Code (`Code/`) und den fertigen Grafiken (`Abbildungen/`); eigenständige Skripte, nicht Teil von `run_pipeline.py`. Enthält u. a. die Zählstellen-Karte (`Code/plot_zaehlstellen_karte.py`, benötigt `geopandas`/`contextily`, die Kantonsgrenze unter `data/geo/` sowie eine Internetverbindung für die Kartenkacheln von OpenStreetMap/CARTO) |

### Modell-Skripte

| Skript | Beschreibung |
|---|---|
| `run_pipeline.py` | Führt die gesamte Pipeline in einem Schritt aus (Datenverarbeitung + Analyse + Modelltraining v5, v6, v7 und v8) |
| `export_weights.py` | Exportiert trainierte Modellgewichte, Test-Vorhersagen und Feiertags-Feature-Vektoren als JSON für die Webapp |
| `models/linear_regression_v5.py` | Lineares Regressionsmodell als Baseline, trainiert auf v5-Daten (10 Datensätze = 5 Stationen × 2 Richtungen) |
| `models/mlp_v5.py` | MLP-Hauptmodell mit Early Stopping, Learning Rate Scheduler und Batch-Normalisierung, trainiert auf v5-Daten |
| `models/linear_regression_v6.py` | Gleiches lineares Modell, trainiert auf v6 zum direkten Vergleich |
| `models/mlp_v6.py` | Gleiches MLP (identische Hyperparameter), trainiert auf v6 zum direkten Vergleich |
| `models/linear_regression_v7.py` | Lineare Baseline für v7 (26 kantonsspezifische Feiertagsspalten, 41 Features) zum direkten Vergleich mit dem v7-MLP. Dieselbe stationsspezifische Ausnahme wie das v7-MLP (siehe unten). |
| `models/mlp_v7.py` | MLP auf v7-Daten (26 kantonsspezifische Feiertagsspalten statt `is_holiday`, 41 Features), mit separat getunten Hyperparametern. **Stationsspezifische Ausnahme:** Für `171_Sattel_R1` und `171_Sattel_R2` wird das `Year`-Feature entfernt (siehe Abschnitt [Stationsspezifische Ausnahme](#stationsspezifische-ausnahme-sattel)). |
| `models/linear_regression_v8.py` | Lineare Baseline für v8 (26 Feiertage + 26 Schulferien, 67 Features) zum direkten Vergleich mit dem v8-MLP. Dieselbe stationsspezifische Ausnahme wie das v8-MLP. |
| `models/mlp_v8.py` | MLP auf v8-Daten (26 Feiertage + 26 Schulferien, 67 Features), mit separat getunten Hyperparametern. Dieselbe stationsspezifische Ausnahme wie bei v7. |
| `models/mlp_tuning_v5.py` | Optuna-Hyperparameter-Tuning für das v5-MLP (optional, nicht Teil der Pipeline) |
| `models/mlp_tuning_v7.py` | Optuna-Hyperparameter-Tuning für das v7-MLP (optional, nicht Teil der Pipeline) |
| `models/mlp_tuning_v8.py` | Optuna-Hyperparameter-Tuning für das v8-MLP (optional, nicht Teil der Pipeline) |

### Analyse-Skripte

| Skript | Beschreibung | Output |
|---|---|---|
| `scripts/dataset_overview.py` | Übersicht aller v5-Datensätze: Zeilen, Zeitraum, fehlende Stunden, Lücken > 24 h, COVID-Anteil | `results/analysis/dataset_overview/` |
| `scripts/covid_anomaly_analysis.py` | Pro Station 3 Plots: monatliches Durchschnittsvolumen mit COVID-Markierung, KW-Vergleich 2019–2022, prozentuale Abweichung 2020/2021 vs. Basisjahre | `results/analysis/covid_anomaly/` |
| `scripts/build_v6.py` | Erzeugt aus `v5` den Corona-bereinigten Datensatz `v6` (entfernt 2020-03-16 bis 2021-02-28) | `data/v6/` |
| `scripts/build_v7.py` | Erzeugt aus `v6` den v7-Datensatz: ersetzt `is_holiday` durch 26 kantonsspezifische Feiertagsspalten aus der zentralen Feiertags-DB | `data/v7/` |
| `scripts/build_v8.py` | Erzeugt aus `v7` den v8-Datensatz: ergänzt 26 kantonsspezifische Schulferienspalten | `data/v8/` |
| `scripts/parameter_vs_performance_v6.py` | Parameter-vs-Performance-Kurve einer Station (v6): variiert die MLP-Grösse und misst Train-/Test-RMSE und Test-R² → begründet die gewählte Architektur | `results/model_results/mlp_v6/analysen/` |
| `scripts/parameter_vs_performance_v7.py` | Dasselbe für das v7-Modell (importiert Hyperparameter live aus `mlp_v7.py`) | `results/model_results/mlp_v7/analysen/` |
| `scripts/parameter_vs_performance_v8.py` | Dasselbe für das v8-Modell | `results/model_results/mlp_v8/analysen/` |
| `scripts/plot_trainingsverlauf.py` | Liest die pro Station von `mlp_v7.py` gespeicherten `training_history_*.csv`-Dateien und plottet je Station eine Kombination aus R²- und Verlustkurve (Trainings-/Validierungswerte) | `results/model_results/mlp_v7/plots/trainingsverlauf/abbildung_trainingsverlauf_v7_<station>.png` |
| `scripts/permutation_importance.py` | Lädt die trainierten v8-Gewichte (`lr_*.pt`, `mlp_*.pt`), rekonstruiert Split und Skalierung exakt wie in den Trainingsskripten und berechnet die gruppierte Permutations-Wichtigkeit der Merkmalsgruppen (mittlerer R²-Abfall bei Permutation) für lineare Regression und MLP | `results/analysis/feature_importance/` (CSV je Modell + Balkendiagramme, inkl. Vergleich) |
| `scripts/plot_r2_summary_clean.py` | Aufgeräumtes R²-Balkendiagramm für Kapitel 4.1 direkt aus `linear_regression_v8/metrics/all_metrics.csv` (kein PyTorch nötig): lesbare Stationsnamen, neutrale Einheitsfarbe, y-Bereich 0–0,8, Balkenwerte in Schweizer Schreibweise (Komma) | `results/model_results/linear_regression_v8/summary/r2_summary_clean.png` |
| `scripts/plot_timeseries_clean.py` | Aufgeräumte Zeitreihe (Kapitel 4.1) der ersten zwei zusammenhängenden Wochen des Testsets für Schwyz R1 aus den v8-LR-Vorhersagen: Messwert (schwarz) vs. Vorhersage (orange) | `results/model_results/linear_regression_v8/summary/timeseries_clean_Schwyz_R1.png` |
| `scripts/plot_r2_summary_clean_mlp.py` | Wie `plot_r2_summary_clean.py`, aber für das MLP-Hauptmodell aus `mlp_v8/metrics/all_metrics.csv`; y-Bereich bis 1,0 (höhere R²-Werte) | `results/model_results/mlp_v8/summary/r2_summary_clean.png` |
| `scripts/plot_timeseries_clean_mlp.py` | Wie `plot_timeseries_clean.py`, aber für das MLP (Schwyz R1) aus den v8-MLP-Vorhersagen | `results/model_results/mlp_v8/summary/timeseries_clean_Schwyz_R1.png` |

### Ergebnisse pro Modell

Nach dem Training werden die Resultate unter `results/model_results/<modell>/` gespeichert (mit `<modell>` ∈ `linear_regression_v5`, `mlp_v5`, `linear_regression_v6`, `mlp_v6`, `linear_regression_v7`, `mlp_v7`, `linear_regression_v8`, `mlp_v8`):

| Unterordner | Inhalt |
|---|---|
| `metrics/` | MAE, RMSE und R² für alle Stationen als CSV |
| `model_weights/` | Trainierte Modell-Gewichte (`.pt`, nicht in Git versioniert) |
| `predictions/` | Stündlicher Vergleich: tatsächliches vs. vorhergesagtes Verkehrsvolumen |
| `plots/loss_curves/` | Trainings- und Validierungsverlust über die Epochen |
| `plots/scatter/` | Predicted vs. Actual Streudiagramme |
| `plots/timeseries/` | Zeitreihendarstellung (erste 2 zusammenhängende Wochen des Testsets) |
| `plots/residuals/` | Durchschnittliche Residuals nach Tagesstunde |
| `plots/tagesverlauf/` | Durchschnittlicher Tagesverlauf: Ist-Werte vs. Vorhersage |
| `plots/trainingsverlauf/` | Nur beim v7-MLP: Kombination aus R²- und Verlustkurve pro Station (`abbildung_trainingsverlauf_v7_<station>.png`) |
| `summary/` | R²-Übersichtsplot über alle Stationen + Split-Infos pro Station |
| `training_history/` | Beim v7- und v8-MLP: `training_history_<station>.csv` mit Spalten `epoch,train_mse,val_mse,train_rmse,val_rmse,train_r2,val_r2` (MSE/RMSE in Fahrzeuge/h zurücktransformiert, R² pro Epoche auf Trainings- bzw. Validierungsbatches) |

Sowohl `mlp_v7.py` als auch `mlp_v8.py` speichern zusätzlich pro Station `training_history/training_history_<station>.csv`. Daraus erzeugt `scripts/plot_trainingsverlauf.py` (liest nur `mlp_v7`) je Station eine Trainingsverlauf-Abbildung in `plots/trainingsverlauf/` (R² links, MSE rechts, je Trainings- und Validierungskurve).

Bei getunten Modellen liegt zusätzlich `best_params.json` (bzw. `best_params_v7.json`) im Modell-Ordner (Output des Optuna-Tunings), und die Parameter-vs-Performance-Studien schreiben nach `<modell>/analysen/`.

### Webapp-Datenpfade

Die Webapp nutzt die **v8-Modelle** (MLP + lineare Baseline). Sie liest ausschliesslich aus `webapp/public/data/`. Diese Dateien werden von `export_weights.py` erzeugt und sind im Repository eingecheckt. Der Live-Forward-Pass im Browser baut den Feature-Vektor pro Station anhand der exportierten `features`-Liste auf (67 Features mit 26 Feiertags- und 26 Schulferienspalten; Sattel R1/R2 ohne `Year` → 66), sodass die [stationsspezifische Ausnahme](#stationsspezifische-ausnahme-sattel) automatisch berücksichtigt wird. Der „Feiertag"-Schalter der Live-Vorhersage wird als Schwyzer Feiertag interpretiert (`holiday_SZ`), der „Schulferien"-Schalter als Schwyzer Schulferien (`schoolholiday_SZ`):

| Pfad | Inhalt | Erzeugt von |
|---|---|---|
| `webapp/public/data/stations.json` | Stationsmetadaten (Name, GPS, R²-Metriken) | `export_weights.py` |
| `webapp/public/data/weights/{id}_mlp.json` | MLP-Gewichte als JSON (Forward Pass im Browser) | `export_weights.py` |
| `webapp/public/data/weights/{id}_linear.json` | LR-Gewichte als JSON | `export_weights.py` |
| `webapp/public/data/results/{id}_daily.json` | Stündliche Test-Vorhersagen (actual, pred_mlp, pred_linear) | `export_weights.py` |
| `webapp/public/data/features/{id}_holidays.json` | Rohe Feature-Vektoren (67 bzw. 66 bei Sattel) aller Feiertags-Stunden im Testset (für Counterfactual) | `export_weights.py` |
| `webapp/public/data/features/schoolholidays_sz.json` | Gemeinsamer SZ-Schulferien-Kalender als benannte Perioden (`[{name, start, end}]`) | `export_weights.py` |
| `webapp/public/data/features/{id}_schoolholidays.json` | Counterfactual-Vorhersage (`[{datetime, mlpNo}]`, Modell ohne Schulferien-Flag) je Schulferien-Stunde im Testset | `export_weights.py` |

## Installation

> Benötigt Python >= 3.11. Für das MLP wird eine NVIDIA-GPU mit CUDA 12.1 empfohlen (Fallback: CPU).

### Windows (mit GPU/CUDA)

Doppelklick auf `install_windows.bat` — erstellt automatisch eine virtuelle Umgebung (`.venv`) und installiert alle Pakete inklusive PyTorch mit CUDA 12.1.

Anschliessend Umgebung aktivieren:
```
.venv\Scripts\activate
```

### Mac / Linux (ohne GPU)

```bash
bash install_mac_linux.sh
source .venv/bin/activate
```

Da CUDA auf Mac und Linux ohne NVIDIA-GPU nicht verfügbar ist, wird hier automatisch die CPU-Version von PyTorch installiert.

## Vollständige Reproduktion

Das gesamte Projekt lässt sich in zwei Schritten vollständig reproduzieren:

### Schritt 1: Modelle trainieren

```bash
python run_pipeline.py
```

Das Skript läuft in drei Phasen ab:

**Phase 1 – Datenverarbeitung (15 Skripte):** Verarbeitet die Rohdaten schrittweise bis zu den fertigen Feature-Datensätzen in `data/v5/`. Die Reihenfolge:

| Schritt | Skript | Input → Output |
|---|---|---|
| 1 | `split_directions.py` | `v1/traffic/` → `v2/` (R1/R2 trennen) |
| 2 | `transform_hourly.py` | `v2/` → `v2/` (Stundenaggregation, ersetzt R1/R2 durch `*_v2.csv`) |
| 3 | `generate_holidays.py` | erzeugt `data/holidays/swiss_holidays_2015_2025.csv` (alle 26 Kantone — **zentrale Feiertags-Datenbank**) |
| 4 | `filter_weather_luzern.py` | `weather/raw/Luzern/` → `weather/processed/` (Spaltenfilter) |
| 5 | `filter_weather_waedenswil.py` | `weather/raw/Waedenswil/` → `weather/processed/` |
| 6 | `add_snow_1h.py` | `weather/processed/` (snow_1h-Spalte hinzufügen, in-place) |
| 7 | `categorize_weather.py` | `weather/processed/` (weather_cat-Spalte, erstellt `_categorized.csv`) |
| 8 | `drop_snowheight.py` | `weather/processed/` (snowheight-Spalte entfernen) |
| 9 | `merge_datasets.py` | `v2/` + `weather/processed/` + `holidays/swiss_holidays_2015_2025.csv` (SZ-Spalte) → `v3/` |
| 10 | `show_gaps.py` | `v3/` → `results/data_visualizations/gaps.txt` |
| 11 | `clean_merged_data.py` | `v3/` → `v4/` |
| 12 | `add_time_features.py` | `v4/` (zyklische Zeitfeatures, in-place) |
| 13 | `create_engineered_features.py` | `v4/` → `v5/` |
| 14 | `plot_hourly_averages.py` | `v5/` → `results/data_visualizations/` |
| 15 | `create_correlation_matrix_engineered.py` | `v5/` → `results/data_visualizations/` |

**Phase 2 – Analyse + Corona-Bereinigung + v7/v8 (5 Skripte):** `dataset_overview.py` und `covid_anomaly_analysis.py` erzeugen die Diagnostik unter `results/analysis/`; `build_v6.py` schreibt den Corona-bereinigten Datensatz nach `data/v6/`; `build_v7.py` erweitert v6 mit 26 kantonsspezifischen Feiertagsspalten aus der zentralen Feiertags-DB nach `data/v7/`; `build_v8.py` fügt 26 Schulferienspalten hinzu und speichert nach `data/v8/`.

**Phase 3 – Modelltraining (8 Skripte + 6 Analysen/Abbildungen):** `linear_regression_v5.py` und `mlp_v5.py` trainieren je 10 Modelle auf v5, `linear_regression_v6.py` und `mlp_v6.py` trainieren mit identischen Hyperparametern dieselben Modelle auf v6. `linear_regression_v7.py` und `mlp_v7.py` trainieren Baseline und MLP auf den v7-Daten, `linear_regression_v8.py` und `mlp_v8.py` dieselben auf den v8-Daten. Beide neuen MLPs (v7, v8) nutzen eigene getunte Hyperparameter. Direkt danach liest `scripts/plot_trainingsverlauf.py` die von `mlp_v7.py` erzeugten `training_history/training_history_*.csv`-Dateien ein und erstellt je Station eine R²-/Verlust-Trainingsverlauf-Abbildung unter `plots/trainingsverlauf/`. Danach berechnet `scripts/permutation_importance.py` aus den trainierten v8-Gewichten die gruppierte Permutations-Wichtigkeit der Merkmalsgruppen und speichert CSV und Balkendiagramme unter `results/analysis/feature_importance/`. Zum Schluss erzeugen `scripts/plot_r2_summary_clean.py`/`scripts/plot_timeseries_clean.py` (lineare Regression) sowie `scripts/plot_r2_summary_clean_mlp.py`/`scripts/plot_timeseries_clean_mlp.py` (MLP) aus den jeweiligen v8-Metriken bzw. -Vorhersagen die aufgeräumten Abbildungen für Kapitel 4.1 (`r2_summary_clean.png`, `timeseries_clean_Schwyz_R1.png`) unter `results/model_results/linear_regression_v8/summary/` bzw. `results/model_results/mlp_v8/summary/`.

### Schritt 2: Webapp-Daten exportieren

```bash
python export_weights.py
```

Dieser Schritt ist **nicht** Teil von `run_pipeline.py` und muss nach dem Training separat ausgeführt werden. Er liest die trainierten `.pt`-Gewichte und die Test-Vorhersagen der **v8-Modelle** (`mlp_v8`, `linear_regression_v8`) aus `results/` und schreibt alle für die Webapp nötigen JSON-Dateien nach `webapp/public/data/`. Dabei werden auch die rohen Feature-Vektoren aller Feiertags-Stunden (für das Counterfactual der Feiertags-Analyse), der SZ-Schulferien-Kalender und die Schulferien-Counterfactual-Werte (für die Schulferien-Analyse) exportiert. Die aktive Modellversion ist über die Konstante `VERSION` am Anfang von `export_weights.py` gesteuert.

### Optionales Hyperparameter-Tuning (MLP)

```bash
python models/mlp_tuning_v5.py       # für mlp_v5.py (v5)
python models/mlp_tuning_v7.py       # für mlp_v7.py (v7)
python models/mlp_tuning_v8.py       # für mlp_v8.py (v8)
```

Führt Optuna-Trials auf der Messstation Brunnen (R1) durch und gibt am Ende die besten Hyperparameter zur manuellen Übernahme in das jeweilige Trainingsskript (`mlp_v5.py` bzw. `mlp_v7.py` / `mlp_v8.py`) aus. Die Resultate werden zusätzlich als `best_params.json` / `best_params_v7.json` / `best_params_v8.json` im jeweiligen Modell-Ordner gespeichert. Beide Skripte laden den (kleinen) Datensatz einmalig komplett auf die GPU, um den Tuning-Durchlauf zu beschleunigen. Dieser Schritt ist nicht Teil von `run_pipeline.py`.

### Optionale Architektur-Analyse (Parameter vs. Performance)

```bash
python scripts/parameter_vs_performance_v6.py       # v6-Modell
python scripts/parameter_vs_performance_v7.py       # v7-Modell
python scripts/parameter_vs_performance_v8.py       # v8-Modell
```

Variiert systematisch die MLP-Grösse für eine repräsentative Station und zeichnet Train-/Test-RMSE und Test-R² gegen die Parameterzahl auf. So lässt sich die gewählte Architektur begründen. Output (CSV + Plots) landet unter `results/model_results/<modell>/analysen/`. Nicht Teil von `run_pipeline.py`.

## Reproduzierbarkeit

Damit die Ergebnisse reproduzierbar sind, habe ich auf folgende Punkte geachtet:

- **Feste Seeds:** Beide Modelle setzen `torch.manual_seed(42)` und `numpy.random.seed(42)`. Die v7-/v8-Skripte (`mlp_v7.py`, `linear_regression_v7.py`, `mlp_v8.py`, `linear_regression_v8.py`) setzen den Seed zusätzlich **vor jeder Station neu**, damit jede Station unabhängig reproduzierbar ist (siehe [Stationsspezifische Ausnahme](#stationsspezifische-ausnahme-sattel)).
- **Kein Datenleck:** Der `StandardScaler` wird ausschliesslich auf den Trainingsdaten gefittet und danach auf Validierungs- und Testdaten angewendet.
- **Chronologischer Split:** Die Daten werden immer zeitlich geordnet aufgeteilt — kein `shuffle` (Linear: 80/20, MLP: 70/10/20).
- **Kein `shuffle` im DataLoader:** Die zeitliche Reihenfolge bleibt auch während des Trainings erhalten.
- **Identische Hyperparameter zwischen v5- und v6-Modellen:** Damit der Vergleich wirklich nur den Daten-Cut (Corona-Bereinigung) misst und nicht ein verändertes Modell. Das v7-Modell wurde dagegen **separat getunt**, weil sich mit den 26 Feiertagsspalten das Feature-Set (16 → 41 Features) geändert hat – ein direkter Vergleich mit v6 wäre mit den v6-Hyperparametern sonst verzerrt.
- **Einheitliche Feiertags-Datenbank:** `data/holidays/swiss_holidays_2015_2025.csv` ist die einzige Feiertags-Quelle im Projekt. Sie wird in Phase 1 von `generate_holidays.py` erzeugt und von `merge_datasets.py` (SZ-Spalte als `is_holiday`) sowie `build_v7.py` (alle 26 Kantonsspalten) gelesen.
- **Konsistente Feiertags-Logik:** `export_weights.py` und `webapp/src/utils/swissHolidays.js` verwenden denselben Butcher-Algorithmus für das Osterdatum und dieselbe Liste der 13 Schwyzer Feiertage — die exportierten Feature-Vektoren und die in der Webapp angezeigten Feiertage stimmen dadurch exakt überein.

## Stationsspezifische Ausnahme: Sattel

Die Messstation **Sattel** (`171`, beide Richtungen R1 und R2) hat ein ungewöhnliches Datenprofil: dichte Messdaten nur für **2015–2017**, danach eine mehrjährige Lücke und ein isoliertes Stück 2025. Durch den chronologischen Split (kein Shuffle) liegt das Training dadurch fast vollständig in 2015–2017, während das Testset bis 2025 reicht — das Modell muss also bis zu **8 Jahre extrapolieren**.

Davon ist als einziges das absolute `Year`-Feature betroffen: Der `StandardScaler` kennt aus dem Training nur die Jahre 2015–2017, für 2025 liegt der skalierte Wert weit ausserhalb dieses Bereichs. Das MLP (und die lineare Baseline) extrapolieren `Year` dann unkontrolliert und überschätzen das Volumen massiv. Im v7-MLP — dessen Hyperparameter separat nur auf Station 050 (Brunnen, ohne dieses Extrapolationsproblem) getunt wurden — kollabiert dadurch besonders das 2025-Segment.

**Lösung:** Für `171_Sattel_R1` und `171_Sattel_R2` wird das `Year`-Feature in den v7- und v8-Modellen (`mlp_v7.py`, `linear_regression_v7.py`, `mlp_v8.py`, `linear_regression_v8.py`) über das `FEATURE_EXCLUDE`-Dict aus dem Feature-Satz entfernt. Alle übrigen, zeitlich stabilen Features (Tageszeit, Wochentag, Saison, Wetter, Feiertage, Schulferien) bleiben erhalten und müssen nicht extrapoliert werden. Damit diese stationsspezifische Änderung die übrigen Stationen nicht über die fortlaufende Zufalls-Kette beeinflusst, setzen die Skripte den Seed **vor jeder Station neu**.

Wirkung auf das v7-MLP:

| Station | R² mit `Year` | R² ohne `Year` |
|---|---|---|
| `171_Sattel_R1` | 0.54 | **0.92** |
| `171_Sattel_R2` | 0.74 | **0.92** |

(Bei `Sattel_R1` lag der R² allein des 2025-Segments mit `Year` bei ~0.03 und steigt ohne `Year` auf ~0.92.) Die Ausnahme gilt **nur für diese eine Station**; alle anderen Stationen behalten `Year`, da sie kein vergleichbares Extrapolationsproblem haben.
