# Maturaarbeit: Verkehrsvolumen-Prognose im Kanton Schwyz (2026)

**Autor:** Jan Tschümperlin, Klasse 3d

In meiner Maturaarbeit untersuche ich, ob sich das stündliche Verkehrsvolumen an Schweizer Nationalstrassen mithilfe von Machine Learning zuverlässig vorhersagen lässt. Dazu habe ich Verkehrsdaten des ASTRA, Wetterdaten von MeteoSchweiz und Feiertagsinformationen für den Kanton Schwyz zusammengeführt und zwei Modelle trainiert: ein lineares Regressionsmodell als Baseline und ein MLP (Multi-Layer Perceptron) als Hauptmodell.

Um den Einfluss der Corona-Anomalie auf die Modellgüte zu testen, werden beide Modelle zusätzlich auf einem Corona-bereinigten Datensatz (`v6_withoutcorona`) trainiert und mit den Original-Resultaten verglichen.

Eine dritte Variante (`v7`) verfeinert die Feiertagskodierung: Statt eines einzigen `is_holiday`-Flags (Kanton Schwyz) erhält das MLP **26 kantonsspezifische Feiertagsspalten** (`holiday_AG … holiday_ZH`), um zu prüfen, ob diese zusätzliche Information die Vorhersage verbessert. Die Hyperparameter des v7-MLP wurden separat per Optuna getunt (das Feature-Set hat sich geändert).

## Interaktive Webapp

Die Ergebnisse sind als interaktive Web-Applikation verfügbar:

**[tschue.ch](https://tschue.ch)**

Die Webapp ermöglicht:
- **Stationskarte** – Übersicht aller 5 Messstationen mit MLP-Vorhersagegüte (R²) farbcodiert auf einer Leaflet-Karte
- **Live-Vorhersage** – MLP und lineare Regression rechnen Prognosen live im Browser (aktuelle Wetterdaten via Open-Meteo, automatische Feiertagserkennung für Kt. Schwyz)
- **Datums-Analyse** – Tagesverlauf eines beliebigen Datums im Testset: Ist-Werte vs. MLP- und LR-Vorhersagen
- **Feiertags-Analyse** – Zeigt für jeden Schwyzer Feiertag im Testset, wie stark der Verkehr vom Wochentags-Durchschnitt abweicht, wie gut das Modell diesen Tag vorhersagt und welchen isolierten Effekt das `is_holiday`-Flag hat (Counterfactual: selber Stunden-Vektor mit `is_holiday=0` vs. `is_holiday=1`)
- **Feature-Sensitivität** – Wie stark verändern Uhrzeit, Temperatur, Niederschlag und Sonnenstunden die Vorhersage (Sweep über den vollen Wertebereich, beide Modelle)
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
| `data/v1_raw/` | Unveränderte ASTRA-Verkehrsrohdaten (`traffic/`) |
| `data/v2_intermediate/` | Verarbeitete Verkehrsdaten direkt im Ordner: Richtungsaufspaltung (R1/R2) und Stundenaggregation (10 `*_hourly.csv`-Dateien) |
| `data/weather/` | Externe Wetterdaten (MeteoSchweiz): `raw/` mit den Stations-Rohdateien (Luzern, Wädenswil), `processed/` mit den gefilterten und kategorisierten Datensätzen |
| `data/holidays/` | Zentrale Feiertags-Datenbank: `swiss_holidays_2015_2025.csv` (alle 26 Kantone, erzeugt von `scripts/generate_holidays.py`). Alle Skripte lesen ausschliesslich diese Datei. |
| `data/v3_merged/` | Verkehr + Wetter + Feiertage zusammengeführt (10 Dateien, je Station × Richtung) |
| `data/v4_cleaned/` | NaN-Zeilen entfernt |
| `data/v5_engineered/` | Fertige ML-Features: zyklische Zeitkodierung, Wetterklassen, alle 16 Feature-Spalten |
| `data/v6_withoutcorona/` | `v5_engineered` ohne den Corona-Anomaliebereich (2020-03-16 bis 2021-02-28) |
| `data/v7/` | `v6_withoutcorona` mit 26 kantonsspezifischen Feiertagsspalten (holiday_AG … holiday_ZH) statt binärem `is_holiday` |

**Warum `data/weather/` und `data/holidays/` separat?** Wetter- und Feiertagsdaten stammen aus externen Quellen (MeteoSchweiz, Python-`holidays`-Modul) und sind unabhängig von der ASTRA-Verkehrs-Verarbeitungskette. Sie liegen daher in eigenen Ordnern. Die Trennung macht den Datenfluss klarer: Das `v1_raw → … → v7`-Schema enthält ausschliesslich ASTRA-Verkehrsdaten, `weather/` und `holidays/` die externen Datenquellen.

### Ordnerübersicht

| Ordner | Inhalt |
|---|---|
| `scripts/` | Python-Skripte für die Datenverarbeitung (v1 → v7), die Corona-Bereinigung (v6), den v7-Aufbau und explorative Analysen |
| `models/` | Trainings-Skripte für die ML-Modelle (Varianten für v5, v6 und v7) sowie Optuna-Tuning (v5 und v7) |
| `analysen/` | Eigenständige Analyse-Skripte zur Begründung der Modellwahl (Parameter-vs-Performance-Studien für v6 und v7) |
| `results/model_results/` | Metriken, Plots und Vorhersage-Vergleiche pro Modell (separat für v5-, v6- und v7-Varianten) |
| `results/analysis/` | Output der explorativen Analyse-Skripte (Datensatz-Übersicht, COVID-Anomalie) |
| `results/data_visualizations/` | Diagnostik der Datenpipeline: Korrelationsmatrizen, Tagesverlaufs-Plots und Lücken-Übersicht (`gaps.txt`) |
| `webapp/` | Interaktive React-Webapp (Vite + Tailwind + Recharts); live unter [tschue.ch](https://tschue.ch) |
| `Theorie/` | Lokale Erklärungsdateien zu den Modellskripten (Prüfungsvorbereitung; nicht im Repository) |
| `Tutorials/` | Lernmaterialien, die ich während der Einarbeitung in Python, NumPy, Pandas und PyTorch erstellt habe |

### Modell-Skripte

| Skript | Beschreibung |
|---|---|
| `run_pipeline.py` | Führt die gesamte Pipeline in einem Schritt aus (Datenverarbeitung + Analyse + Modelltraining v5, v6 und v7) |
| `export_weights.py` | Exportiert trainierte Modellgewichte, Test-Vorhersagen und Feiertags-Feature-Vektoren als JSON für die Webapp |
| `models/linear_regression.py` | Lineares Regressionsmodell als Baseline, trainiert auf v5-Daten (10 Datensätze = 5 Stationen × 2 Richtungen) |
| `models/mlp.py` | MLP-Hauptmodell mit Early Stopping, Learning Rate Scheduler und Batch-Normalisierung, trainiert auf v5-Daten |
| `models/linear_regression_v6.py` | Gleiches lineares Modell, trainiert auf v6_withoutcorona zum direkten Vergleich |
| `models/mlp_v6.py` | Gleiches MLP (identische Hyperparameter), trainiert auf v6_withoutcorona zum direkten Vergleich |
| `models/mlp_v7.py` | MLP auf v7-Daten (26 kantonsspezifische Feiertagsspalten statt `is_holiday`, 41 Features), mit separat getunten Hyperparametern. Keine eigene lineare Baseline. |
| `models/mlp_tuning.py` | Optuna-Hyperparameter-Tuning für das v5-MLP (optional, nicht Teil der Pipeline) |
| `models/mlp_tuning_v7.py` | Optuna-Hyperparameter-Tuning für das v7-MLP (optional, nicht Teil der Pipeline) |

### Analyse-Skripte

| Skript | Beschreibung | Output |
|---|---|---|
| `scripts/dataset_overview.py` | Übersicht aller v5-Datensätze: Zeilen, Zeitraum, fehlende Stunden, Lücken > 24 h, COVID-Anteil | `results/analysis/dataset_overview/` |
| `scripts/covid_anomaly_analysis.py` | Pro Station 3 Plots: monatliches Durchschnittsvolumen mit COVID-Markierung, KW-Vergleich 2019–2022, prozentuale Abweichung 2020/2021 vs. Basisjahre | `results/analysis/covid_anomaly/` |
| `scripts/create_v6_withoutcorona.py` | Erzeugt aus `v5_engineered` den Corona-bereinigten Datensatz `v6_withoutcorona` (entfernt 2020-03-16 bis 2021-02-28) | `data/v6_withoutcorona/` |
| `scripts/build_v7.py` | Erzeugt aus `v6_withoutcorona` den v7-Datensatz: ersetzt `is_holiday` durch 26 kantonsspezifische Feiertagsspalten aus der zentralen Feiertags-DB | `data/v7/` |
| `analysen/parameter_vs_performance.py` | Parameter-vs-Performance-Kurve einer Station (v6): variiert die MLP-Grösse und misst Train-/Test-RMSE und Test-R² → begründet die gewählte Architektur | `results/model_results/mlp_v6/analysen/` |
| `analysen/parameter_vs_performance_v7.py` | Dasselbe für das v7-Modell (importiert Hyperparameter live aus `mlp_v7.py`) | `results/model_results/mlp_v7/analysen/` |

### Ergebnisse pro Modell

Nach dem Training werden die Resultate unter `results/model_results/<modell>/` gespeichert (mit `<modell>` ∈ `linear_regression`, `mlp`, `linear_regression_v6`, `mlp_v6`, `mlp_v7`):

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
| `summary/` | R²-Übersichtsplot über alle Stationen + Split-Infos pro Station |

Bei getunten Modellen liegt zusätzlich `best_params.json` (bzw. `best_params_v7.json`) im Modell-Ordner (Output des Optuna-Tunings), und die Parameter-vs-Performance-Studien schreiben nach `<modell>/analysen/`.

### Webapp-Datenpfade

Die Webapp liest ausschliesslich aus `webapp/public/data/`. Diese Dateien werden von `export_weights.py` erzeugt und sind im Repository eingecheckt:

| Pfad | Inhalt | Erzeugt von |
|---|---|---|
| `webapp/public/data/stations.json` | Stationsmetadaten (Name, GPS, R²-Metriken) | `export_weights.py` |
| `webapp/public/data/weights/{id}_mlp.json` | MLP-Gewichte als JSON (Forward Pass im Browser) | `export_weights.py` |
| `webapp/public/data/weights/{id}_linear.json` | LR-Gewichte als JSON | `export_weights.py` |
| `webapp/public/data/results/{id}_daily.json` | Stündliche Test-Vorhersagen (actual, pred_mlp, pred_linear) | `export_weights.py` |
| `webapp/public/data/features/{id}_holidays.json` | Rohe 16-Feature-Vektoren aller Feiertags-Stunden im Testset (für Counterfactual) | `export_weights.py` |

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

**Phase 1 – Datenverarbeitung (15 Skripte):** Verarbeitet die Rohdaten schrittweise bis zu den fertigen Feature-Datensätzen in `data/v5_engineered/`. Die Reihenfolge:

| Schritt | Skript | Input → Output |
|---|---|---|
| 1 | `split_directions.py` | `v1_raw/traffic/` → `v2_intermediate/` (R1/R2 trennen) |
| 2 | `transform_hourly.py` | `v2_intermediate/` → `v2_intermediate/` (Stundenaggregation, ersetzt R1/R2 durch `*_hourly.csv`) |
| 3 | `generate_holidays.py` | erzeugt `data/holidays/swiss_holidays_2015_2025.csv` (alle 26 Kantone — **zentrale Feiertags-Datenbank**) |
| 4 | `filter_weather_luzern_2010_2026.py` | `weather/raw/Luzern/` → `weather/processed/` (Spaltenfilter) |
| 5 | `filter_weather_waedenswil_2010_2026.py` | `weather/raw/Waedenswil/` → `weather/processed/` |
| 6 | `add_snow_1h.py` | `weather/processed/` (snow_1h-Spalte hinzufügen, in-place) |
| 7 | `categorize_weather.py` | `weather/processed/` (weather_cat-Spalte, erstellt `_categorized.csv`) |
| 8 | `drop_snowheight.py` | `weather/processed/` (snowheight-Spalte entfernen) |
| 9 | `merge_datasets.py` | `v2_intermediate/` + `weather/processed/` + `holidays/swiss_holidays_2015_2025.csv` (SZ-Spalte) → `v3_merged/` |
| 10 | `show_gaps.py` | `v3_merged/` → `results/data_visualizations/gaps.txt` |
| 11 | `clean_merged_data.py` | `v3_merged/` → `v4_cleaned/` |
| 12 | `add_time_features.py` | `v4_cleaned/` (zyklische Zeitfeatures, in-place) |
| 13 | `create_engineered_features.py` | `v4_cleaned/` → `v5_engineered/` |
| 14 | `plot_hourly_averages.py` | `v5_engineered/` → `results/data_visualizations/` |
| 15 | `create_correlation_matrix_engineered.py` | `v5_engineered/` → `results/data_visualizations/` |

**Phase 2 – Analyse + Corona-Bereinigung + v7 (4 Skripte):** `dataset_overview.py` und `covid_anomaly_analysis.py` erzeugen die Diagnostik unter `results/analysis/`; `create_v6_withoutcorona.py` schreibt den Corona-bereinigten Datensatz nach `data/v6_withoutcorona/`; `build_v7.py` erweitert v6 mit 26 kantonsspezifischen Feiertagsspalten aus der zentralen Feiertags-DB nach `data/v7/`.

**Phase 3 – Modelltraining (5 Skripte):** `linear_regression.py` und `mlp.py` trainieren je 10 Modelle auf v5, `linear_regression_v6.py` und `mlp_v6.py` trainieren mit identischen Hyperparametern dieselben Modelle auf v6, und `mlp_v7.py` trainiert das MLP auf den v7-Daten (26 kantonsspezifische Feiertagsspalten, eigene getunte Hyperparameter). Für v7 gibt es keine eigene lineare Baseline.

### Schritt 2: Webapp-Daten exportieren

```bash
python export_weights.py
```

Dieser Schritt ist **nicht** Teil von `run_pipeline.py` und muss nach dem Training separat ausgeführt werden. Er liest die trainierten `.pt`-Gewichte und die Test-Vorhersagen aus `results/` und schreibt alle für die Webapp nötigen JSON-Dateien nach `webapp/public/data/`. Dabei werden auch die rohen Feature-Vektoren aller Feiertags-Stunden im Testset exportiert (für das Counterfactual in der Feiertags-Analyse-Seite).

### Optionales Hyperparameter-Tuning (MLP)

```bash
python models/mlp_tuning.py       # für mlp.py (v5)
python models/mlp_tuning_v7.py    # für mlp_v7.py (v7)
```

Führt Optuna-Trials auf der Messstation Brunnen (R1) durch und gibt am Ende die besten Hyperparameter zur manuellen Übernahme in das jeweilige Trainingsskript (`mlp.py` bzw. `mlp_v7.py`) aus. Die Resultate werden zusätzlich als `best_params.json` / `best_params_v7.json` im jeweiligen Modell-Ordner gespeichert. Beide Skripte laden den (kleinen) Datensatz einmalig komplett auf die GPU, um den Tuning-Durchlauf zu beschleunigen. Dieser Schritt ist nicht Teil von `run_pipeline.py`.

### Optionale Architektur-Analyse (Parameter vs. Performance)

```bash
python analysen/parameter_vs_performance.py       # v6-Modell
python analysen/parameter_vs_performance_v7.py    # v7-Modell
```

Variiert systematisch die MLP-Grösse für eine repräsentative Station und zeichnet Train-/Test-RMSE und Test-R² gegen die Parameterzahl auf. So lässt sich die gewählte Architektur begründen. Output (CSV + Plots) landet unter `results/model_results/<modell>/analysen/`. Nicht Teil von `run_pipeline.py`.

## Reproduzierbarkeit

Damit die Ergebnisse reproduzierbar sind, habe ich auf folgende Punkte geachtet:

- **Feste Seeds:** Beide Modelle setzen `torch.manual_seed(42)` und `numpy.random.seed(42)`.
- **Kein Datenleck:** Der `StandardScaler` wird ausschliesslich auf den Trainingsdaten gefittet und danach auf Validierungs- und Testdaten angewendet.
- **Chronologischer Split:** Die Daten werden immer zeitlich geordnet aufgeteilt — kein `shuffle` (Linear: 80/20, MLP: 70/10/20).
- **Kein `shuffle` im DataLoader:** Die zeitliche Reihenfolge bleibt auch während des Trainings erhalten.
- **Identische Hyperparameter zwischen v5- und v6-Modellen:** Damit der Vergleich wirklich nur den Daten-Cut (Corona-Bereinigung) misst und nicht ein verändertes Modell. Das v7-Modell wurde dagegen **separat getunt**, weil sich mit den 26 Feiertagsspalten das Feature-Set (16 → 41 Features) geändert hat – ein direkter Vergleich mit v6 wäre mit den v6-Hyperparametern sonst verzerrt.
- **Einheitliche Feiertags-Datenbank:** `data/holidays/swiss_holidays_2015_2025.csv` ist die einzige Feiertags-Quelle im Projekt. Sie wird in Phase 1 von `generate_holidays.py` erzeugt und von `merge_datasets.py` (SZ-Spalte als `is_holiday`) sowie `build_v7.py` (alle 26 Kantonsspalten) gelesen.
- **Konsistente Feiertags-Logik:** `export_weights.py` und `webapp/src/utils/swissHolidays.js` verwenden denselben Butcher-Algorithmus für das Osterdatum und dieselbe Liste der 13 Schwyzer Feiertage — die exportierten Feature-Vektoren und die in der Webapp angezeigten Feiertage stimmen dadurch exakt überein.
