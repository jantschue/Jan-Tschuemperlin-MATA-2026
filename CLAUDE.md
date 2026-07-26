# CLAUDE.md – Maturaarbeit Verkehrsvorhersage

Framework: **PyTorch** | Sprache: **Python 3**

---

## Projekt

Stündliche Verkehrsvorhersage auf Schweizer Strassen (ASTRA-Daten, Kanton Schwyz, 5 Messstationen).  
**Zielgrösse:** `volume` (Fahrzeuganzahl pro Stunde)

### Features

| Spalte | Beschreibung |
|--------|-------------|
| `datetime` | Zeitstempel |
| `volume` | Zielgrösse |
| `Year` | Jahr |
| `Hour_sin`, `Hour_cos` | Zyklische Kodierung Stunde |
| `DayOfWeek_sin`, `DayOfWeek_cos` | Zyklische Kodierung Wochentag |
| `Month_sin`, `Month_cos` | Zyklische Kodierung Monat |
| `DayOfYear_sin`, `DayOfYear_cos` | Zyklische Kodierung Jahrestag |
| `is_weekend` | Binär (0/1) |
| `is_holiday` | Binär (0/1) |
| `temp` | Temperatur |
| `rain_1h` | Niederschlag letzte Stunde |
| `sun_1h` | Sonnenstunden letzte Stunde |
| `snow_1h` | Schneefall letzte Stunde |
| `weather_cat` | Kategoriale Wetterkategorie |

### Modelle

| Modell | Rolle |
|--------|-------|
| MLP (Deep Neural Network) | Hauptmodell |
| Lineare Regression | Baseline |

### Wichtige Constraints

- **Kein random Train/Test-Split** – immer chronologisch
- **Keine Lag-Features** – strukturelle Langzeitprognose, kein 1h-Ahead-Forecasting
- **Kein LSTM/RNN** – Zeitfeatures sind explizit kodiert, nicht implizit zu lernen

---

## Referenz-Notebook

Wenn im Projektverzeichnis ein `pytorch_workflow.ipynb` vorhanden ist, hat dessen Code-Stil und Struktur Vorrang. Vor dem Schreiben von neuem Code dieses Notebook lesen und Konventionen (Klassennamen, Schichtaufbau, Trainingsschleife, etc.) übernehmen.

---

## Coding Guidelines

### Skript-Header

Jedes Skript beginnt mit einem deutschen Docstring:

```python
"""
Dieses Skript lädt und bereinigt die Rohdaten der ASTRA-Zählstellen und speichert
den verarbeiteten Datensatz als CSV-Datei für die weitere Modellierung.
"""
```

### Reproduzierbarkeit

- Einstiegsskript `run_pipeline.py` führt alle Schritte in Reihenfolge aus
- Fester Seed überall: `torch.manual_seed`, `random.seed`, `numpy.random.seed`
- Nur relative Pfade (kein `/home/user/...`)

> **WICHTIG – Änderungen immer auf Folgewirkungen prüfen:**
> Wird **irgendetwas** geändert (Skript, Modell, Daten-Schema, Pfade, Feature-Liste,
> Hyperparameter, Ordnerstruktur …), muss **immer** geprüft werden, ob das
> Auswirkungen auf **andere Skripte** hat. Wenn ja, sind diese **sofort mit
> anzupassen**, sodass das Projekt **jederzeit vollständig und fehlerfrei mit
> `python run_pipeline.py` reproduzierbar** bleibt. Eine Änderung gilt erst als
> fertig, wenn die gesamte Pipeline wieder durchläuft und alle abhängigen
> Skripte (inkl. Analyse-Skripte) auf den aktuellen Stand bezogen sind.

### Abhängigkeiten

`requirements.txt` mit gepinnten Versionen, bei jeder Paketänderung aktualisieren:

```
torch==2.3.0
pandas==2.2.1
scikit-learn==1.4.2
numpy==1.26.4
matplotlib==3.8.4
```

### Dokumentation

`README.md` enthält: Projektbeschreibung, Verzeichnisbaum, Setup-Anleitung, `python run_pipeline.py`-Kurzanleitung.  
Installationsskripte parallel zu `requirements.txt` aktuell halten: `install_mac_linux.sh` / `install_windows.bat`

> **Das `README.md` muss immer aktuell sein.** Bei jeder Änderung, die Inhalt,
> Verzeichnisbaum, Setup, Abhängigkeiten oder den Ablauf betrifft, wird das README
> im selben Zug mitgepflegt – es darf nie veraltet zum Code stehen.

### Code-Stil

- Variablen und Funktionen: **Englisch**
- Kommentare und Docstrings: **Deutsch**
- Keine Magic Numbers – Hyperparameter oben im Skript oder in Konfigdatei sammeln
- Jede Funktion erhält einen kurzen deutschen Docstring