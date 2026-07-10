# Prompt für Claude Code: Titel zu allen eigenen Grafiken hinzufügen

Füge jeder der unten aufgeführten, selbst erstellten Grafiken (eigene Darstellung) einen
kurzen, aussagekräftigen Titel direkt im Plot hinzu. Der Titel wird im
erzeugenden Skript gesetzt (`ax.set_title(...)` bzw. `fig.suptitle(...)` bei
Mehrfach-Plots) und muss beim erneuten Ausführen der Pipeline automatisch mit
ausgegeben werden.

## Vorgaben

- Titel exakt wie in der Spalte **Titel** übernehmen (Schweizer Schreibweise, kein Gedankenstrich).
- Titel klar lesbar setzen: `fontsize` passend zur bestehenden Schrift, `fontweight="bold"`, genügend Abstand (`pad`), Layout mit `tight_layout()` / `bbox_inches="tight"` prüfen, damit nichts abgeschnitten wird.
- Bei mehreren Teilplots (Raster) den Gesamttitel als `fig.suptitle(...)` setzen, die bestehenden Teil-Titel der Subplots (z. B. Stationsnamen) unverändert lassen.
- Keine Doppelspurigkeit erzeugen: Falls ein Skript bereits einen Titel setzt, den bestehenden durch den vorgegebenen Titel ersetzen statt zusätzlich einen zweiten hinzuzufügen.
- **CLAUDE.md beachten:** deutscher Docstring/Kommentar bei Änderungen, keine Magic Numbers (Titeltext ggf. als Konstante oben im Skript), nur relative Pfade.
- **Folgewirkungen prüfen:** Nach den Änderungen die Grafiken neu erzeugen und sicherstellen, dass `python run_pipeline.py` weiterhin vollständig und fehlerfrei durchläuft. Betroffene Skripte, README und (falls vorhanden) das Abbildungsverzeichnis auf den aktuellen Stand bringen.
- Wo das erzeugende Skript in der Tabelle mit „prüfen“ markiert ist: Skript zuerst anhand des angegebenen PNG-Dateinamens im Repo lokalisieren, dann Titel einfügen.

## Grafiken und Titel

| Abb. | Inhalt | Titel | Erzeugendes Skript / PNG |
|------|--------|-------|--------------------------|
| 1 | Flussdiagramm klassische Programmierung vs. ML | `Klassische Programmierung vs. maschinelles Lernen` | `Abbildungen_MATA/Code/create_ml_schema_plot.py` → `klassische_vs_ml_programmierung.png` |
| 2 | Suche der bestpassenden Geraden (Temperatur/Glace) | `Suche nach der bestpassenden Regressionsgeraden` | `Abbildungen_MATA/Code/abbildung_2_regression_diagramm.py` |
| 3 | Regressionsgerade f(x)=8x+20 mit Vorhersage | `Lineare Regression am Beispiel Temperatur und Glaceverkauf` | `Abbildungen_MATA/Code/create_icecream_plot.py` → `temperatur_glace_regression.png` |
| 7 | Lineare Aktivierung und ReLU | `Lineare Aktivierungsfunktion und ReLU im Vergleich` | `Abbildungen_MATA/Code/abbildung_5_linear_vs_relu.py` |
| 8 | Schema des verwendeten MLP | `Aufbau des verwendeten MLP` | `Abbildungen_MATA/Code/create_mlp_schema.py` → `mlp_schema.png` |
| 9 | Tagesverlauf Wetter und Verkehrsvolumen | `Tagesverlauf von Wetter und Verkehrsvolumen` | `scripts/plot_hourly_averages_clean.py` |
| 10 | Karte der fünf Zählstellen | `Lage der fünf Zählstellen im Kanton Schwyz` | `Abbildungen_MATA/Code/plot_zaehlstellen_karte.py` |
| 11 | Korrelationsmatrix der Merkmale | `Korrelationsmatrix der Eingabemerkmale` | `scripts/create_correlation_matrix_v8.py` |
| 12 | Netzgrösse gegen Test-R² (Brunnen R1) | `Einfluss der Netzgrösse auf das Test-R² (Brunnen R1)` | `Abbildungen_MATA/Code/abbildung_parameter_vs_performance.py` → `abbildung_parameter_vs_performance_v8.png` |
| 13 | Balken R² lineare Regression je Datenreihe | `Bestimmtheitsmass R² der linearen Regression je Datenreihe` | `scripts/plot_r2_summary_clean.py` |
| 14 | Prognose vs. Messwert lineare Regression (Schwyz R1) | `Lineare Regression: Prognose und Messwerte (Schwyz R1, zwei Wochen)` | `scripts/plot_timeseries_clean.py` → `timeseries_clean_Schwyz_R1.png` |
| 15 | Balken R² DNN je Datenreihe | `Bestimmtheitsmass R² des DNN je Datenreihe` | `scripts/plot_r2_summary_clean_mlp.py` |
| 16 | Prognose vs. Messwert DNN (Schwyz R1) | `DNN: Prognose und Messwerte (Schwyz R1, zwei Wochen)` | `scripts/plot_timeseries_clean_mlp.py` → `timeseries_clean_Schwyz_R1.png` |
| 17 | Trainingsverlauf DNN (R² und Verlust) | `Trainingsverlauf des DNN (Schwyz R1)` | `scripts/plot_trainingsverlauf_r2_schwyz_r1.py` → `trainingsverlauf_r2_720_Schwyz_R1_v8.png` |
| 18 | Streudiagramm-Raster Prognose vs. Messwert je Zählstelle (R1) | `Prognose gegen Messwert je Zählstelle (Fahrtrichtung R1)` | Skript anhand Raster-Streudiagramm (5 Stationen, R1) **prüfen** und lokalisieren |
| 19 / 20 | Permutations-Wichtigkeit DNN vs. lineare Regression | `Wichtigkeit der Merkmalsgruppen: DNN vs. lineare Regression` | `Abbildungen_MATA/Code/abbildung_permutation_importance.py` → `abbildung_permutation_importance_v8.png` (bzw. `scripts/permutation_importance.py`) **prüfen**, welche Ausgabe im Dokument verwendet wird |
| 21 | Veränderung des R² durch Corona-Ausschluss | `Veränderung des R² durch den Corona-Ausschluss` | `Abbildungen_MATA/Code/abbildung_r2_vergleich_lr_dnn.py` → `abbildung_r2_vergleich_lr_dnn_v8.png` **prüfen** |
| 22 | Trainingszeit lineare Regression vs. DNN | `Trainingszeit von linearer Regression und DNN je Datenreihe` | `Abbildungen_MATA/Code/abbildung_trainingszeit_lr_dnn.py` → `abbildung_trainingszeit_lr_dnn_v8.png` |
| 23 | Genauigkeit gegen Rechenaufwand | `Genauigkeit gegen Rechenaufwand beider Modelle` | `Abbildungen_MATA/Code/abbildung_genauigkeit_vs_aufwand.py` → `abbildung_genauigkeit_vs_aufwand_v8.png` |

## Nicht enthalten (bewusst weggelassen)

Die Abbildungen 4 (Underfitting/Overfitting), 5 (Aufbau neuronales Netz) und 6
(einzelnes Neuron) stammen nicht aus eigenen Skripten (externe bzw. schematische
Darstellungen) und werden hier nicht per Code angepasst. Falls diese doch als
eigene Darstellung gelten und einen Titel bekommen sollen, bitte separat angeben.

## Abschluss

Nach dem Einfügen der Titel alle betroffenen Grafiken neu erzeugen, das Ergebnis
kurz sichten (Titel vorhanden, nichts abgeschnitten) und bestätigen, dass
`python run_pipeline.py` vollständig durchläuft.
