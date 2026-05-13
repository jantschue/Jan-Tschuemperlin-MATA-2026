# Maturaarbeit: Verkehrsvolumen-Prognose im Kanton Schwyz (2026)

**Autor:** Jan Tschümperlin, Klasse 3d  

In meiner Maturaarbeit untersuche ich, ob sich das stündliche Verkehrsvolumen an Schweizer Nationalstrassen mithilfe von Machine Learning zuverlässig vorhersagen lässt. Dazu habe ich Verkehrsdaten des ASTRA, Wetterdaten von MeteoSchweiz und Feiertagsinformationen für den Kanton Schwyz zusammengeführt und zwei Modelle trainiert: ein lineares Regressionsmodell als Baseline und ein MLP (Multi-Layer Perceptron) als Hauptmodell.

## Datenquellen & Danksagung

Die verwendeten Rohdaten stammen aus folgenden offiziellen Quellen:

- **Verkehrsdaten:** Bundesamt für Strassen (ASTRA). Die Veröffentlichung der aufbereiteten Daten erfolgt mit freundlicher Genehmigung des ASTRA.
- **Wetterdaten:** Bundesamt für Meteorologie und Klimatologie (MeteoSchweiz) über das [Daten-Portal](https://www.meteoschweiz.admin.ch/service-und-publikationen/applikationen/ext/daten-ohne-programmierkenntnisse-herunterladen.html)
- **Feiertage:** Automatisch generiert mit dem Python-Modul `holidays` (offizielle Feiertage Kanton Schwyz).

*Jegliche Weiterverwendung der rohen Verkehrsdaten erfordert eine eigene Abklärung mit dem ASTRA.*

## Projektstruktur

### Datenpipeline

Die Rohdaten durchlaufen fünf Verarbeitungsstufen, bevor sie für das Modelltraining verwendet werden:

| Stufe | Ordner | Inhalt |
|---|---|---|
| 1 | `data/v1_raw/` | Unveränderte Originaldaten (ASTRA, MeteoSchweiz, Feiertage) |
| 2 | `data/v2_intermediate/` | Richtungsaufspaltung (R1/R2), Stundenaggregation, gefilterte Wetterdaten |
| 3 | `data/v3_merged/` | Verkehr + Wetter + Feiertage zusammengeführt |
| 4 | `data/v4_cleaned/` | NaN-Zeilen entfernt |
| 5 | `data/v5_engineered/` | Fertige ML-Features (zyklische Zeitkodierung, Wetterklassen) |

### Ordnerübersicht

| Ordner | Inhalt |
|---|---|
| `scripts/` | 16 Python-Skripte für die Datenverarbeitung (v1 → v5) und explorative Analysen |
| `models/` | Trainings-Skripte für die ML-Modelle |
| `results/model_results/` | Metriken, Plots und Vorhersage-Vergleiche pro Modell |
| `Tutorials/` | Lernmaterialien, die ich während der Einarbeitung in Python, NumPy, Pandas und PyTorch erstellt habe |

### Skripte

| Skript | Beschreibung |
|---|---|
| `run_pipeline.py` | Führt die gesamte Pipeline in einem Schritt aus (Datenverarbeitung + Modelltraining) |
| `models/linear_regression.py` | Lineares Regressionsmodell als Baseline, trainiert auf 10 Datensätzen (5 Stationen × 2 Richtungen) |
| `models/mlp.py` | MLP-Hauptmodell mit Early Stopping und Learning Rate Scheduler, ebenfalls 10 Datensätze |
| `models/mlp_tuning.py` | Optuna-Hyperparameter-Tuning für das MLP (optionaler Schritt, nicht Teil der Pipeline) |

### Ergebnisse pro Modell

Nach dem Training werden die Resultate unter `results/model_results/<modell>/` gespeichert:

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
| `summary/` | R²-Übersichtsplot über alle Stationen |

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

Das Skript läuft in zwei Phasen ab:

**Phase 1 – Datenverarbeitung:** 16 Skripte verarbeiten die Rohdaten schrittweise bis zu den fertigen Feature-Datensätzen in `data/v5_engineered/`.

**Phase 2 – Modelltraining:** `linear_regression.py` und `mlp.py` trainieren je 10 Modelle und speichern alle Resultate unter `results/model_results/`.

### Optionales Hyperparameter-Tuning (MLP)

```bash
python models/mlp_tuning.py
```

Führt 50 Optuna-Trials auf der Messstation Brunnen (R1) durch und gibt am Ende die besten Hyperparameter zur manuellen Übernahme in `mlp.py` aus. Dieser Schritt ist nicht Teil von `run_pipeline.py` und muss separat ausgeführt werden.

## Reproduzierbarkeit

Damit die Ergebnisse reproduzierbar sind, habe ich auf folgende Punkte geachtet:

- **Feste Seeds:** Beide Modelle setzen `torch.manual_seed(42)` und `numpy.random.seed(42)`.
- **Kein Datenleck:** Der `StandardScaler` wird ausschliesslich auf den Trainingsdaten gefittet und danach auf Validation- und Testdaten angewendet.
- **Chronologischer Split:** Die Daten werden immer zeitlich geordnet aufgeteilt — kein `shuffle` (Linear: 80/20, MLP: 70/10/20).
- **Kein `shuffle` im DataLoader:** Die zeitliche Reihenfolge bleibt auch während des Trainings erhalten.
