# Maturaarbeit: Verkehrsvolumen und Wetterdaten (MATA 2026)

Dieses Repository enthält den Code und die Datenverarbeitungs-Pipeline für meine Maturaarbeit zur Analyse und Vorhersage von Verkehrsvolumen in Abhängigkeit von Wetterdaten und Feiertagen im Kanton Schwyz.

## Datenquellen & Danksagung (Attribution)

Die in diesem Projekt verwendeten Rohdaten stammen aus den folgenden offiziellen Quellen:

*   **Verkehrsdaten:** Bundesamt für Strassen (ASTRA). Die Veröffentlichung der hier aufbereiteten Verkehrsdaten erfolgt mit freundlicher Genehmigung des ASTRA.
*   **Wetterdaten:** Bundesamt für Meteorologie und Klimatologie (MeteoSchweiz) über das [Daten-Portal](https://www.meteoschweiz.admin.ch/service-und-publikationen/applikationen/ext/daten-ohne-programmierkenntnisse-herunterladen.html)
*   **Feiertage:** Automatisch generiert über das Python-Modul `holidays` (basierend auf den offiziellen Feiertagen des Kantons Schwyz).

*Hinweis: Jegliche Weiterverwendung der rohen Verkehrsdaten erfordert eine eigene Abklärung mit den jeweiligen Ämtern.*

## Projektstruktur

Das Projekt ist vollständig reproduzierbar aufgebaut. Die Daten durchlaufen 5 Versionen:

1.  `data/v1_raw/`: Unveränderte Originaldaten
2.  `data/v2_intermediate/`: Erste Vorverarbeitung (stündliche Aggregation, Filterung)
3.  `data/v3_merged/`: Zusammengeführte Datensätze (Verkehr + Wetter + Feiertage)
4.  `data/v4_cleaned/`: Bereinigte Datensätze ohne fehlende Werte
5.  `data/v5_engineered/`: Finale Datensätze mit Machine-Learning-Features

### Weitere Ordner

*   `scripts/`: 16 Python-Skripte für die Datenverarbeitung (v1 → v5) und Analysen
*   `models/`: Trainings-Skripte für die Vorhersagemodelle (z.B. `linear_regression.py`)
*   `results/model_results/`: Trainierte Modelle, Metriken (CSV) und Visualisierungen pro Modell
*   `Tutorials/`: Begleitende Lernmaterialien (PyTorch-Kurs)

### Wichtige Skripte

| Skript | Beschreibung |
|---|---|
| `run_pipeline.py` | Master-Skript: führt die gesamte Pipeline aus (Datenverarbeitung + Modelltraining) |
| `scripts/*.py` | 16 Daten-Aufbereitungs- und Analyse-Skripte (werden von `run_pipeline.py` aufgerufen) |
| `models/linear_regression.py` | Lineares Regressionsmodell (PyTorch), trainiert 10 Modelle (5 Stationen × 2 Richtungen) |

## Installation (Setup)

Damit das Projekt einwandfrei ausgeführt werden kann, müssen die nötigen Python-Bibliotheken (siehe `requirements.txt`) installiert sein. 
Dafür gibt es im Hauptordner zwei fertige Installations-Skripte, welche automatisch eine virtuelle Umgebung (`.venv`) erstellen und alles Nötige installieren:

*   **Für Windows:** Einfach einen Doppelklick auf `install_windows.bat` machen.
*   **Für Mac/Linux:** Das Skript `install_mac_linux.sh` im Terminal ausführen (`bash install_mac_linux.sh`).

## Reproduktion

Um die gesamte Pipeline auszuführen – von der Rohdatenverarbeitung bis zum fertigen Modelltraining – einfach folgendes Skript ausführen:

```bash
python run_pipeline.py
```

Die Pipeline besteht aus zwei Phasen:
1.  **Phase 1 – Datenverarbeitung:** Die 16 Skripte in `scripts/` verarbeiten die Rohdaten schrittweise zu den fertigen Feature-Datensätzen (`data/v5_engineered/`).
2.  **Phase 2 – Modelltraining:** Die Trainings-Skripte in `models/` trainieren die Vorhersagemodelle und speichern Resultate, Metriken und Plots unter `results/model_results/`.
