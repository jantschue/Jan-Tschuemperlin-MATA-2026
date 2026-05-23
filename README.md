# Maturaarbeit: Verkehrsvolumen-Prognose im Kanton Schwyz (2026)

**Autor:** Jan Tschümperlin, Klasse 3d

In meiner Maturaarbeit untersuche ich, ob sich das stündliche Verkehrsvolumen an Schweizer Nationalstrassen mithilfe von Machine Learning zuverlässig vorhersagen lässt. Dazu habe ich Verkehrsdaten des ASTRA, Wetterdaten von MeteoSchweiz und Feiertagsinformationen für den Kanton Schwyz zusammengeführt und zwei Modelle trainiert: ein lineares Regressionsmodell als Baseline und ein MLP (Multi-Layer Perceptron) als Hauptmodell.

Um den Einfluss der Corona-Anomalie auf die Modellgüte zu testen, werden beide Modelle zusätzlich auf einem Corona-bereinigten Datensatz (v6_withoutcorona) trainiert und mit den Original-Resultaten verglichen.

## Interaktive Webapp

Die Ergebnisse sind als interaktive Web-Applikation verfügbar:

**[tschue.ch](https://tschue.ch)**

Die Webapp ermöglicht:
- **Stationskarte** – Übersicht aller 5 Messstationen mit MLP-Vorhersagegüte (R²) farbcodiert
- **Live-Vorhersage** – MLP und lineare Regression rechnen Prognosen live im Browser (aktuelle Wetterdaten via Open-Meteo, automatische Feiertagserkennung)
- **Datums-Analyse** – Tagesverlauf eines beliebigen Datums: Ist-Werte vs. Modellvorhersagen
- **Feature-Sensitivität** – Zeigt, wie stark Uhrzeit, Temperatur, Niederschlag und Sonnenstunden die Vorhersage beeinflussen
- **Anomalie-Explorer** – Durchsucht alle stündlichen Ergebnisse nach grossen Fehlern, Peak-Versagen und Wochentag-Mustern; Export als CSV

Der Quellcode der Webapp liegt unter `webapp/`.

## Datenquellen & Danksagung

Die verwendeten Rohdaten stammen aus folgenden offiziellen Quellen:

- **Verkehrsdaten:** Bundesamt für Strassen (ASTRA). Die Veröffentlichung der aufbereiteten Daten erfolgt mit freundlicher Genehmigung des ASTRA.
- **Wetterdaten:** Bundesamt für Meteorologie und Klimatologie (MeteoSchweiz) über das [Daten-Portal](https://www.meteoschweiz.admin.ch/service-und-publikationen/applikationen/ext/daten-ohne-programmierkenntnisse-herunterladen.html)
- **Feiertage:** Automatisch generiert mit dem Python-Modul `holidays` (offizielle Feiertage Kanton Schwyz).

*Jegliche Weiterverwendung der rohen Verkehrsdaten erfordert eine eigene Abklärung mit dem ASTRA.*

## Projektstruktur

### Datenpipeline

Die Rohdaten durchlaufen sechs Verarbeitungsstufen, bevor sie für das Modelltraining verwendet werden:

| Stufe | Ordner | Inhalt |
|---|---|---|
| 1 | `data/v1_raw/` | Unveränderte Originaldaten (ASTRA, MeteoSchweiz, Feiertage) |
| 2 | `data/v2_intermediate/` | Richtungsaufspaltung (R1/R2), Stundenaggregation, gefilterte Wetterdaten |
| 3 | `data/v3_merged/` | Verkehr + Wetter + Feiertage zusammengeführt |
| 4 | `data/v4_cleaned/` | NaN-Zeilen entfernt |
| 5 | `data/v5_engineered/` | Fertige ML-Features (zyklische Zeitkodierung, Wetterklassen) |
| 6 | `data/v6_withoutcorona/` | v5 ohne den Corona-Anomaliebereich (2020-03-16 bis 2021-02-28) |

### Ordnerübersicht

| Ordner | Inhalt |
|---|---|
| `scripts/` | Python-Skripte für die Datenverarbeitung (v1 → v5), die Corona-Bereinigung (v6) und explorative Analysen |
| `models/` | Trainings-Skripte für die ML-Modelle (jeweils Variante für v5 und v6) |
| `results/model_results/` | Metriken, Plots und Vorhersage-Vergleiche pro Modell (separat für v5- und v6-Varianten) |
| `results/analysis/` | Output der explorativen Analyse-Skripte (Datensatz-Übersicht, COVID-Anomalie) |
| `webapp/` | Interaktive React-Webapp (Vite + Tailwind); live unter [tschue.ch](https://tschue.ch) |
| `Tutorials/` | Lernmaterialien, die ich während der Einarbeitung in Python, NumPy, Pandas und PyTorch erstellt habe |

### Modell-Skripte

| Skript | Beschreibung |
|---|---|
| `run_pipeline.py` | Führt die gesamte Pipeline in einem Schritt aus (Datenverarbeitung + Analyse + Modelltraining v5 und v6) |
| `models/linear_regression.py` | Lineares Regressionsmodell als Baseline, trainiert auf v5-Daten (10 Datensätze = 5 Stationen × 2 Richtungen) |
| `models/mlp.py` | MLP-Hauptmodell mit Early Stopping und Learning Rate Scheduler, trainiert auf v5-Daten |
| `models/linear_regression_v6.py` | Gleiches lineares Modell, trainiert auf v6_withoutcorona zum direkten Vergleich |
| `models/mlp_v6.py` | Gleiches MLP (identische Hyperparameter), trainiert auf v6_withoutcorona zum direkten Vergleich |
| `models/mlp_tuning.py` | Optuna-Hyperparameter-Tuning für das MLP (optional, nicht Teil der Pipeline) |

### Analyse-Skripte

| Skript | Beschreibung | Output |
|---|---|---|
| `scripts/dataset_overview.py` | Übersicht aller v5-Datensätze: Zeilen, Zeitraum, fehlende Stunden, Lücken > 24 h, COVID-Anteil | `results/analysis/dataset_overview/` |
| `scripts/covid_anomaly_analysis.py` | Pro Station 3 Plots: monatliches Durchschnittsvolumen mit COVID-Markierung, KW-Vergleich 2019-2022, prozentuale Abweichung 2020/2021 vs. Basisjahre | `results/analysis/covid_anomaly/` |
| `scripts/create_v6_withoutcorona.py` | Erzeugt aus v5_engineered den Corona-bereinigten Datensatz v6_withoutcorona (entfernt 2020-03-16 bis 2021-02-28) | `data/v6_withoutcorona/` |

### Ergebnisse pro Modell

Nach dem Training werden die Resultate unter `results/model_results/<modell>/` gespeichert (mit `<modell>` ∈ `linear_regression`, `mlp`, `linear_regression_v6`, `mlp_v6`):

| Unterordner | Inhalt |
|---|---|
| `metrics/` | MAE, RMSE und R² für alle Stationen als CSV |
| `model_weights/` | Trainierte Modell-Gewichte (`.pt`, nicht in Git versioniert) |
| `predictions/` | Stündlicher Vergleich: tatsächliches vs. vorhergesagtes Verkehrsvolumen |
| `plots/loss_curves/` | Trainings- und Validierungsverlust über die Epochen |
| `plots/scatter/` | Predicted vs. Actual Streudiagramme |
| `plots/timeseries/` | Zeitreihendarstellung (erste 2 Wochen des Testsets) |
| `plots/residuals/` | Durchschnittliche Residuals nach Tagesstunde |
| `plots/tagesverlauf/` | Durchschnittlicher Tagesverlauf: Ist-Werte vs. Vorhersage |
| `summary/` | R²-Übersichtsplot über alle Stationen + Split-Infos pro Station |

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

## Reproduktion

Die gesamte Pipeline lässt sich mit einem einzigen Befehl reproduzieren:

```bash
python run_pipeline.py
```

Das Skript läuft in drei Phasen ab:

**Phase 1 – Datenverarbeitung (16 Skripte):** verarbeitet die Rohdaten schrittweise bis zu den fertigen Feature-Datensätzen in `data/v5_engineered/`.

**Phase 2 – Analyse + Corona-Bereinigung (3 Skripte):** `dataset_overview.py` und `covid_anomaly_analysis.py` erzeugen die Diagnostik unter `results/analysis/`; `create_v6_withoutcorona.py` schreibt anschliessend den Corona-bereinigten Datensatz nach `data/v6_withoutcorona/`.

**Phase 3 – Modelltraining (4 Skripte):** `linear_regression.py` und `mlp.py` trainieren je 10 Modelle auf v5, `linear_regression_v6.py` und `mlp_v6.py` trainieren mit identischen Hyperparametern dieselben Modelle auf v6 — direkter Vergleichswert für die Corona-Hypothese.

### Optionales Hyperparameter-Tuning (MLP)

```bash
python models/mlp_tuning.py
```

Führt Optuna-Trials auf der Messstation Brunnen (R1) durch und gibt am Ende die besten Hyperparameter zur manuellen Übernahme in `mlp.py` aus. Dieser Schritt ist nicht Teil von `run_pipeline.py` und muss separat ausgeführt werden.

## Reproduzierbarkeit

Damit die Ergebnisse reproduzierbar sind, habe ich auf folgende Punkte geachtet:

- **Feste Seeds:** Beide Modelle setzen `torch.manual_seed(42)` und `numpy.random.seed(42)`.
- **Kein Datenleck:** Der `StandardScaler` wird ausschliesslich auf den Trainingsdaten gefittet und danach auf Validation- und Testdaten angewendet.
- **Chronologischer Split:** Die Daten werden immer zeitlich geordnet aufgeteilt — kein `shuffle` (Linear: 80/20, MLP: 70/10/20).
- **Kein `shuffle` im DataLoader:** Die zeitliche Reihenfolge bleibt auch während des Trainings erhalten.
- **Identische Hyperparameter zwischen v5- und v6-Modellen:** Damit der Vergleich wirklich nur den Daten-Cut misst und nicht ein verändertes Modell.
- **Robuste Pfadauflösung in Analyse-Skripten:** Die Skripte in `scripts/` finden den Projekt-Root unabhängig vom Ablage-Ort, indem sie aufwärts nach dem `data/`-Ordner suchen.
