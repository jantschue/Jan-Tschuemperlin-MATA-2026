# Maturaarbeit: Verkehrsvolumen-Prognose im Kanton Schwyz (MATA 2026)

Dieses Repository enthält den vollständigen Code und die Datenpipeline der Maturaarbeit 2026 von Jan Tschuemperlin. Ziel ist die stundengenaue Vorhersage des Verkehrsvolumens an fünf Messstationen im Kanton Schwyz mithilfe von Machine-Learning-Modellen, trainiert auf Verkehrs-, Wetter- und Feiertagsdaten.

## Datenquellen & Danksagung

Die Rohdaten stammen aus folgenden offiziellen Quellen:

- **Verkehrsdaten:** Bundesamt für Strassen (ASTRA). Die Veröffentlichung der aufbereiteten Daten erfolgt mit freundlicher Genehmigung des ASTRA.
- **Wetterdaten:** Bundesamt für Meteorologie und Klimatologie (MeteoSchweiz) über das [Daten-Portal](https://www.meteoschweiz.admin.ch/service-und-publikationen/applikationen/ext/daten-ohne-programmierkenntnisse-herunterladen.html)
- **Feiertage:** Automatisch generiert via Python-Modul `holidays` (offizielle Feiertage Kanton Schwyz).

*Jegliche Weiterverwendung der rohen Verkehrsdaten erfordert eine eigene Abklärung mit dem ASTRA.*

## Projektstruktur

### 5-stufige Datenpipeline

Die Rohdaten durchlaufen fünf Verarbeitungsstufen im Ordner `data/`:

| Stufe | Ordner | Inhalt |
|---|---|---|
| 1 | `data/v1_raw/` | Unveränderte Originaldaten (ASTRA, MeteoSchweiz, Feiertage) |
| 2 | `data/v2_intermediate/` | Richtungsaufspaltung (R1/R2), Stundenaggregation, gefilterte Wetterdaten |
| 3 | `data/v3_merged/` | Verkehr + Wetter + Feiertage nach Datum zusammengeführt |
| 4 | `data/v4_cleaned/` | NaN-Zeilen entfernt |
| 5 | `data/v5_engineered/` | ML-fertige Features (zyklische + binäre Kodierung) |

### Ordner

| Ordner | Inhalt |
|---|---|
| `scripts/` | 16 Python-Skripte für die Datenverarbeitung (v1 → v5) und Analysen |
| `models/` | Trainings-Skripte für die ML-Modelle |
| `results/model_results/` | Metriken (CSV), Plots und Vorhersage-Vergleiche pro Modell |
| `Tutorials/` | Begleitende Lernmaterialien (Python, NumPy, Pandas, PyTorch) |

### Wichtige Skripte

| Skript | Beschreibung |
|---|---|
| `run_pipeline.py` | Master-Skript: führt die gesamte Pipeline durch (Datenverarbeitung + Modelltraining) |
| `models/linear_regression.py` | Lineares Regressionsmodell (PyTorch), 10 Modelle (5 Stationen × 2 Richtungen) |
| `models/mlp.py` | MLP-Modell (PyTorch) mit Early Stopping und LR-Scheduler, 10 Modelle |
| `models/mlp_tuning.py` | Optuna-Hyperparameter-Tuning für das MLP (optionaler Schritt, nicht in Pipeline) |

### Ergebnisse pro Modell

Unter `results/model_results/<modell>/` entstehen folgende Unterordner:

| Unterordner | Inhalt |
|---|---|
| `metrics/` | `all_metrics.csv` mit MAE, RMSE, R² pro Station |
| `model_weights/` | Gespeicherte Modell-Gewichte (`.pt`) |
| `predictions/` | `predictions_<station>.csv` mit Ist- und Prognosewerten pro Stunde |
| `plots/loss_curves/` | Trainings- und Validierungsverlust über Epochen |
| `plots/scatter/` | Predicted vs. Actual Streudiagramme |
| `plots/timeseries/` | Zeitreihe (erste 2 Wochen des Testsets) |
| `plots/residuals/` | Durchschnittliche Residuals nach Tagesstunde |
| `plots/tagesverlauf/` | Durchschnittlicher Tagesverlauf (Ist vs. Prognose) |
| `summary/` | R²-Übersichtsplot über alle Stationen |

## Installation

> Benötigt Python >= 3.11. Für das MLP wird eine NVIDIA-GPU mit CUDA 12.1 empfohlen (Fallback: CPU).

### Windows (mit CUDA)

Doppelklick auf `install_windows.bat` — erstellt `.venv` und installiert alle Pakete inklusive PyTorch CUDA 12.1.

Anschliessend Umgebung aktivieren:
```
.venv\Scripts\activate
```

### Mac / Linux (CPU)

```bash
bash install_mac_linux.sh
source .venv/bin/activate
```

PyTorch wird automatisch als CPU-Version installiert (CUDA nicht verfügbar auf Mac/Linux ohne NVIDIA-GPU).

## Reproduktion

Die gesamte Pipeline — von der Rohdatenverarbeitung bis zum fertigen Modelltraining — mit einem einzigen Befehl ausführen:

```bash
python run_pipeline.py
```

**Phase 1 – Datenverarbeitung:** 16 Skripte transformieren die Rohdaten schrittweise zu `data/v5_engineered/`.

**Phase 2 – Modelltraining:** `linear_regression.py` und `mlp.py` trainieren je 10 Modelle und speichern Resultate unter `results/model_results/`.

### Optionales Hyperparameter-Tuning (MLP)

```bash
python models/mlp_tuning.py
```

Führt 50 Optuna-Trials auf der Station `050_Brunnen_Mositunnel_R1` durch und gibt die besten Hyperparameter zur manuellen Übernahme in `mlp.py` aus. Dieser Schritt ist **nicht** Teil von `run_pipeline.py`.

## Reproduzierbarkeit

- **Zufalls-Seeds:** Beide Modelle setzen `torch.manual_seed(42)` und `numpy.random.seed(42)`.
- **Kein Datenleck:** `StandardScaler` wird ausschliesslich auf den Trainingsdaten gefittet und dann auf Validation/Test angewendet.
- **Chronologischer Split:** Kein `shuffle` — Daten werden zeitlich geordnet aufgeteilt (Linear: 80/20, MLP: 70/10/20).
- **Kein `shuffle` im DataLoader:** Zeitliche Reihenfolge bleibt auch während des Trainings erhalten.
