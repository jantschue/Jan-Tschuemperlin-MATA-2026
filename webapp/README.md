# Verkehrsvorhersage SZ – Webapp

Single-Page-Webapp zur Visualisierung der Verkehrsvorhersage-Modelle aus der Maturaarbeit 2026.
Reines Frontend: alle Modell-Gewichte und Tagesergebnisse werden aus statischen JSON-Dateien
geladen, kein Backend nötig.

## Lokal starten

```bash
cd webapp
npm install
npm run dev
```

Die App ist dann unter http://localhost:5173 erreichbar.

## Vier Ansichten

| Ansicht | Beschreibung |
|---------|--------------|
| Stationskarte | Leaflet-Karte aller 10 Messstellen, gefärbt nach MLP-R²; sortierbare Übersichtstabelle |
| Live-Vorhersage | Eingabeformular (Uhrzeit, Wetter, Wochentag …) und Vorhersage in Echtzeit (MLP + LR) |
| Datums-Analyse | Stündlicher Vergleich Tatsächlich vs. Vorhersage für ein gewähltes Datum |
| Feature-Sensitivität | Wie reagiert das MLP auf Änderungen einzelner Features? (2×2 Grid + Feature-Wichtigkeit) |

## Daten-Layout

```
webapp/public/data/
  stations.json                      # Stations-Liste mit Metriken
  weights/{station_id}_mlp.json      # MLP-Gewichte
  weights/{station_id}_linear.json   # Lineare-Regression-Gewichte
  results/{station_id}_daily.json    # Stündliche Vorhersagen pro Datum
```

Die Platzhalter-Dateien werden mit `node scripts/generate_placeholders.mjs` neu erzeugt
(nicht-deterministisch nicht, da fester Seed).

## Exportieren der echten PyTorch-Modelle

### MLP-Weights (`{station_id}_mlp.json`)

```python
"""
Exportiert die Gewichte eines trainierten PyTorch-MLP nach JSON für die Webapp.
"""
import json
import torch

# Modell und Scaler laden (Pfade entsprechend anpassen)
model = torch.load("models/050_R1_mlp.pt", map_location="cpu")
model.eval()

input_scaler = torch.load("models/050_R1_input_scaler.pt")
output_scaler = torch.load("models/050_R1_output_scaler.pt")

# Alle Linear-Schichten in Reihenfolge extrahieren
layers = []
for module in model.modules():
    if isinstance(module, torch.nn.Linear):
        layers.append({
            "weight": module.weight.detach().cpu().tolist(),
            "bias":   module.bias.detach().cpu().tolist()
        })

payload = {
    "layers": layers,
    "activation": "relu",
    "input_scaler": {
        "mean": input_scaler.mean_.tolist(),
        "std":  input_scaler.scale_.tolist()
    },
    "output_scaler": {
        "mean": output_scaler.mean_.tolist(),
        "std":  output_scaler.scale_.tolist()
    }
}

with open("webapp/public/data/weights/050_R1_mlp.json", "w") as f:
    json.dump(payload, f)
```

### Lineare Regression (`{station_id}_linear.json`)

```python
"""
Exportiert ein scikit-learn LinearRegression Modell nach JSON.
"""
import json
import joblib

model = joblib.load("models/050_R1_linear.pkl")
input_scaler = joblib.load("models/050_R1_input_scaler.pkl")
output_scaler = joblib.load("models/050_R1_output_scaler.pkl")

payload = {
    "weight": model.coef_.tolist(),
    "bias":   float(model.intercept_),
    "input_scaler": {
        "mean": input_scaler.mean_.tolist(),
        "std":  input_scaler.scale_.tolist()
    },
    "output_scaler": {
        "mean": output_scaler.mean_.tolist(),
        "std":  output_scaler.scale_.tolist()
    }
}

with open("webapp/public/data/weights/050_R1_linear.json", "w") as f:
    json.dump(payload, f)
```

### Tagesergebnisse (`{station_id}_daily.json`)

```python
"""
Exportiert die stündlichen Vorhersagen eines Modells über den Test-Zeitraum nach JSON.
"""
import json
import pandas as pd

# Test-DataFrame mit echten und vorhergesagten Werten
df = pd.read_csv("results/050_R1_predictions.csv")
# erwartete Spalten: datetime, actual, pred_mlp, pred_linear

records = df.assign(
    datetime=pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
).to_dict(orient="records")

with open("webapp/public/data/results/050_R1_daily.json", "w") as f:
    json.dump(records, f)
```

### `stations.json`

```python
"""
Erzeugt die zentrale stations.json aus einem Dict mit Metadaten und Test-Metriken.
"""
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

STATIONS = [
    {"id": "050_R1", "name": "Brunnen Mositunnel", "direction": "Nord",
     "lat": 46.999, "lng": 8.608},
    # … weitere Stationen
]

def metrics_for(csv_path):
    df = pd.read_csv(csv_path)
    out = {}
    for name, col in [("linear", "pred_linear"), ("mlp", "pred_mlp")]:
        out[name] = {
            "mae":  float(mean_absolute_error(df["actual"], df[col])),
            "rmse": float(np.sqrt(mean_squared_error(df["actual"], df[col]))),
            "r2":   float(r2_score(df["actual"], df[col]))
        }
    return out

result = []
for s in STATIONS:
    s["metrics"] = metrics_for(f"results/{s['id']}_predictions.csv")
    result.append(s)

with open("webapp/public/data/stations.json", "w") as f:
    json.dump(result, f, indent=2)
```

## Deployment auf Vercel

1. Repo bei Vercel importieren (oder `vercel` CLI im Ordner `webapp/` ausführen).
2. Build-Settings:
   - Framework Preset: **Vite**
   - Build Command: `npm run build`
   - Output Directory: `dist`
3. Push auf `main` → Vercel deployt automatisch.

## Tech-Stack

- React 18 (Hooks, funktionale Komponenten)
- Vite als Build-System
- Tailwind CSS (Dark Mode standardmässig, kein Toggle)
- Recharts für alle Diagramme
- Leaflet + react-leaflet für die Karte (CartoDB Dark Matter Tiles)
- Reiner Client – kein Backend, keine API-Aufrufe
