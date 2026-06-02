# Feiertags-Feature-Vektoren

Hier liegen die rohen (unskalierten) 16-Feature-Vektoren aller Test-Set-Stunden,
die auf einen Schwyzer Feiertag fallen. Die Seite "Feiertags-Analyse"
(`src/components/FeiertagsAnalyse.jsx`) braucht sie für das Counterfactual:
denselben Stunden-Vektor einmal mit `is_holiday=1` und einmal mit `is_holiday=0`
durch das MLP schicken, um den isolierten Effekt des Feiertags-Flags zu messen.

Dateinamen-Schema: `{station_id}_holidays.json`

Format:

```json
[
  { "datetime": "2024-11-01T00:00:00", "f": [2024, 0.0, 1.0, ..., 1, -1.2, 0.0, 0.0, 0.0, 2] },
  ...
]
```

- `f` ist der Feature-Vektor in exakt der Reihenfolge aus `models/mlp.py` (FEATURES).
- `f[10]` ist `is_holiday` (für das Counterfactual auf 0 gesetzt).
- Der Test-Split ist identisch mit jenem des MLP, daher stimmen die Datetimes mit
  den `results/{station_id}_daily.json` überein. `mlpForward(f, mlp)` reproduziert
  exakt das `pred_mlp` aus der daily-Datei.

Erzeugt von `export_weights.py` (Funktion `export_holiday_features`). Die
Feiertags-Datumslogik ist identisch zu `src/utils/swissHolidays.js`. Berchtoldstag
(2.1.) ist enthalten, damit die Webapp dessen optionales Counterfactual zeigen kann.
