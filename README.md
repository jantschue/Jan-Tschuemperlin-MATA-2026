# Maturaarbeit: Verkehrsvolumen und Wetterdaten (MATA 2026)

Dieses Repository enthält den Code und die Datenverarbeitungs-Pipeline für meine Maturaarbeit zur Analyse und Vorhersage von Verkehrsvolumen in Abhängigkeit von Wetterdaten und Feiertagen im Kanton Schwyz.

## Datenquellen & Danksagung (Attribution)

Die in diesem Projekt verwendeten Rohdaten stammen aus den folgenden offiziellen Quellen:

*   **Verkehrsdaten:** Bundesamt für Strassen (ASTRA). Die Veröffentlichung der hier aufbereiteten Verkehrsdaten erfolgt mit freundlicher Genehmigung des ASTRA.
*   **Wetterdaten:** Bundesamt für Meteorologie und Klimatologie (MeteoSchweiz) über das [Daten-Portal](https://www.meteoschweiz.admin.ch/service-und-publikationen/applikationen/ext/daten-ohne-programmierkenntnisse-herunterladen.html)
*   **Feiertage:** Offizielle Publikationen des Kantons Schwyz (Beispiel: [Feiertage im Kanton Schwyz](https://www.sz.ch/public/upload/assets/86157/Feiertage_im_Kanton_Schwyz_fuer_das_Jahr_2026.pdf?fp=1))

*Hinweis: Jegliche Weiterverwendung der rohen Verkehrsdaten erfordert eine eigene Abklärung mit den jeweiligen Ämtern.*

## Projektstruktur

Das Projekt ist vollständig reproduzierbar aufgebaut. Die Daten durchlaufen 5 Versionen:

1.  `data/v1_raw/`: Unveränderte Originaldaten
2.  `data/v2_intermediate/`: Erste Vorverarbeitung (stündliche Aggregation, Filterung)
3.  `data/v3_merged/`: Zusammengeführte Datensätze (Verkehr + Wetter + Feiertage)
4.  `data/v4_cleaned/`: Bereinigte Datensätze ohne fehlende Werte
5.  `data/v5_engineered/`: Finale Datensätze mit Machine-Learning-Features

## Reproduktion

Um die gesamte Daten-Pipeline auszuführen und alle Ergebnisse und Plots von Grund auf neu zu generieren, führen Sie einfach folgendes Skript aus:

```bash
python run_pipeline.py
```
