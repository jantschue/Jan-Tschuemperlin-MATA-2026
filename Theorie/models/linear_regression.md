# Erklärung: `models/linear_regression.py`

**Originaldatei:** `models/linear_regression.py`

**Zusammenfassung:** Dieses Skript definiert ein lineares Regressionsmodell als PyTorch-`nn.Module`, trainiert es für jede der 10 Messstationen (5 Stationen × 2 Richtungen) und speichert Gewichte, Vorhersagen, Metriken und Evaluationsplots. Das Modell dient als Baseline zum Vergleich mit dem komplexeren MLP. Die Architektur besteht aus einer einzigen linearen Schicht, die die Transformation $\hat{y} = Wx + b$ durchführt.

---

## 1. Imports

```python
import os
import json
from pathlib import Path
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

Jeder Import wird hier einzeln erklärt:

**`import os`**
Das `os`-Modul ist Teil der Python-Standardbibliothek und stellt Betriebssystemfunktionen bereit (Dateipfade, Prozesse, etc.). In diesem Skript wird es zwar importiert, aber eigentlich nie direkt verwendet, da `pathlib.Path` die Pfadverwaltung übernimmt. Es handelt sich um einen historischen Import.

**`import json`**
Das `json`-Modul ermöglicht das Lesen und Schreiben von JSON-Dateien (JavaScript Object Notation). Es wird verwendet, um Split-Informationen (Start- und Enddaten von Train/Test) in `split_info_{name}.json` zu speichern.

**`from pathlib import Path`**
`pathlib.Path` ist eine objektorientierte Klasse für Dateipfade. Statt Zeichenketten-Konkatenation (`"ordner" + "/" + "datei"`) kann man den `/`-Operator verwenden: `DATA_DIR / filename`. Das funktioniert plattformübergreifend (Windows, Mac, Linux).

**`import time`**
Das `time`-Modul misst Zeit. `time.time()` gibt die aktuelle Zeit in Sekunden seit dem 1. Januar 1970 zurück. Vor und nach dem Training aufgerufen ergibt die Differenz die Trainingszeit in Sekunden.

**`import torch`**
Das zentrale Paket von PyTorch. Es stellt Tensoren (mehrdimensionale Arrays mit GPU-Unterstützung), die automatische Differenzierung (Autograd) und alle Berechnungsoperationen bereit. Vergleiche `01_pytorch_workflow.ipynb`, Zelle 2: `import torch`.

**`import torch.nn as nn`**
Das Unterpaket `nn` (neural networks) enthält alle Bausteine für neuronale Netze: `nn.Module` (Basisklasse), `nn.Linear` (lineare Schicht), `nn.MSELoss` (Verlustfunktion), `nn.ReLU` (Aktivierungsfunktion) und viele weitere. Vergleiche `01_pytorch_workflow.ipynb`, Zelle 2: `from torch import nn # nn contains all of pytorch's building blocks for neural networks`.

**`import numpy as np`**
NumPy ist die wichtigste Python-Bibliothek für numerische Berechnungen auf Arrays. Sie wird hier für Arrayoperationen (`np.sqrt`), den Seed (`np.random.seed`) und als Zwischenformat beim Konvertieren zwischen PyTorch-Tensoren und sklearn-Funktionen verwendet.

**`import pandas as pd`**
Pandas ist eine Bibliothek für tabellarische Daten in Form von DataFrames (Tabellen mit benannten Spalten und Zeilenindizes). Sie wird zum Laden der CSV-Dateien, zur Datumsverwaltung (`pd.to_datetime`) und zur Aggregation von Metriken verwendet.

**`import matplotlib.pyplot as plt`**
Die Standard-Plotting-Bibliothek in Python. Alle Diagramme (Loss-Kurven, Scatter-Plots, Zeitreihen, Residuen, Tagesverlauf) werden damit erstellt und als PNG gespeichert.

**`import matplotlib.dates as mdates`**
Eine Erweiterung von Matplotlib speziell für die Formatierung von Datumsachsen. Wird in Plot C (Zeitreihe) verwendet, um die x-Achse korrekt als Datum zu beschriften.

**`from sklearn.preprocessing import StandardScaler, LabelEncoder`**
Aus der scikit-learn Bibliothek (Machine-Learning-Werkzeugkasten in Python):
- `StandardScaler` normalisiert numerische Features auf Mittelwert 0 und Standardabweichung 1.
- `LabelEncoder` konvertiert kategoriale Werte (z.B. Wetterklassen als Strings) in Ganzzahlen, die als Modell-Input verwendet werden können.

**`from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score`**
Evaluationsmetriken aus scikit-learn:
- `mean_absolute_error`: mittlerer absoluter Fehler (MAE)
- `mean_squared_error`: mittlerer quadratischer Fehler (MSE), wird hier als Basis für RMSE verwendet
- `r2_score`: Bestimmtheitsmass $R^2$

---

## 2. Seeds und Device

```python
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**`torch.manual_seed(42)`**
Setzt den internen Zufallsgenerator von PyTorch auf einen festen Startwert (42 ist eine übliche Konvention). Dadurch sind die zufällig initialisierten Startgewichte des Modells bei jedem Ausführen identisch. Vergleiche `00_pytorch_fundamentals.ipynb`, Abschnitt "Reproducibility (trying to take random out of random)":

```python
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
```

**Warum Reproduzierbarkeit wichtig ist:** Neuronale Netze werden mit zufälligen Anfangsgewichten gestartet. Ohne festen Seed würde dasselbe Skript bei zwei Läufen unterschiedliche Ergebnisse liefern, was wissenschaftliche Vergleiche erschwert. Der Seed garantiert, dass alle Experimente unter denselben Anfangsbedingungen starten.

**`np.random.seed(42)`**
Setzt den Zufallsgenerator von NumPy ebenfalls auf 42. NumPy wird im Hintergrund von scikit-learn verwendet (z.B. `LabelEncoder`), weshalb auch dieser Seed gesetzt wird.

**`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`**
Wählt automatisch die GPU (CUDA), falls eine verfügbar ist, sonst die CPU. Dieser Ausdruck stammt direkt aus `01_pytorch_workflow.ipynb`, Abschnitt 6 "Putting it all together":

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Der einzige Unterschied: Im Skript wird `torch.device(...)` explizit aufgerufen, was ein `torch.device`-Objekt erzeugt statt eines einfachen Strings. Beide Varianten funktionieren mit `.to(device)`, aber das Objekt ist die sauberere Form.

---

## 3. Pfade und Konfiguration

```python
DATA_DIR = Path("data/v5_engineered")
RESULTS_DIR = Path("results/model_results/linear_regression")

DATASETS = [
    "050_Brunnen_Mositunnel_R1_engineered.csv",
    "050_Brunnen_Mositunnel_R2_engineered.csv",
    ...
]
```

**`DATA_DIR = Path("data/v5_engineered")`**
Relativer Pfad zu den aufbereiteten CSV-Datensätzen. `v5_engineered` bedeutet Version 5 nach Feature-Engineering (zyklische Zeitfeatures, Wetterdaten, etc. sind bereits enthalten).

**`RESULTS_DIR = Path("results/model_results/linear_regression")`**
Zielordner für alle Ausgaben dieses Modells: Metriken, Gewichte, Plots, Vorhersagen.

**`DATASETS`**
Liste aller 10 Dateinamen (5 Stationen × 2 Fahrtrichtungen R1/R2). Die Schleife in `main()` iteriert über diese Liste.

```python
(RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "model_weights").mkdir(parents=True, exist_ok=True)
...
```

**`mkdir(parents=True, exist_ok=True)`**
Erstellt den Ordner, falls er noch nicht existiert.
- `parents=True`: erstellt auch übergeordnete Ordner, falls nötig (z.B. `results/` und `model_results/`).
- `exist_ok=True`: kein Fehler, wenn der Ordner bereits vorhanden ist.

---

## 4. Modellklasse `LinearRegressionModel`

```python
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
```

Diese Klasse ist das eigentliche Modell. Sie ist direkt von `LinearRegressionModelV2` aus `01_pytorch_workflow.ipynb`, Zelle 55, abgeleitet. Der einzige Unterschied: `in_features` ist variabel statt fest auf 1 gesetzt, da wir 16 Features haben.

**`class LinearRegressionModel(nn.Module):`**
Die Klasse erbt von `nn.Module`, der Basisklasse für alle neuronalen Netze in PyTorch. Vergleiche `01_pytorch_workflow.ipynb`, Abschnitt "PyTorch model building essentials": "torch.nn.Module: The base class for all neural network modules, if you subclass it, you should overwrite forward()".

**`def __init__(self, input_dim):`**
Der Konstruktor der Klasse. Der Parameter `input_dim` legt die Anzahl der Eingabe-Features fest. Bei 16 Features wäre `input_dim=16`.

**`super().__init__()`**
Ruft den Konstruktor der Elternklasse `nn.Module` auf. Dieser Aufruf ist zwingend erforderlich: ohne ihn würde das interne Registrierungssystem von PyTorch nicht initialisiert, und Methoden wie `.parameters()`, `.to(device)`, `.train()`, `.eval()` würden nicht funktionieren.

**`self.linear_layer = nn.Linear(in_features=input_dim, out_features=1)`**
Erstellt eine einzelne lineare Schicht. `nn.Linear` führt die Berechnung

$$\hat{y} = x W^T + b$$

durch, wobei:
- $x$ der Eingabe-Tensor der Form $(\text{batch\_size}, \text{input\_dim})$ ist,
- $W$ die Gewichtsmatrix der Form $(1, \text{input\_dim})$ ist (zufällig initialisiert),
- $b$ der Bias (Skalar, zufällig initialisiert) ist,
- $\hat{y}$ der Ausgabe-Tensor der Form $(\text{batch\_size}, 1)$ ist.

Vergleiche `01_pytorch_workflow.ipynb`, Zelle 55: `self.linear_layer = nn.Linear(in_features=1, out_features=1)`. Durch das Registrieren unter `self.linear_layer` weiss PyTorch, dass die Gewichte und Biases dieser Schicht lernbare Parameter sind, die bei `.parameters()` zurückgegeben und durch den Optimizer aktualisiert werden.

**`def forward(self, x: torch.Tensor) -> torch.Tensor:`**
Definiert, was beim Aufrufen des Modells (`model(x)`) berechnet wird. Diese Methode muss in jeder `nn.Module`-Unterklasse überschrieben werden. Vergleiche `01_pytorch_workflow.ipynb`, Abschnitt "PyTorch model building essentials": "def forward(): All nn.Module subclasses require a forward() method, this defines the computation that will take place on the data passed to the particular nn.Module". Der Typ-Hinweis `: torch.Tensor -> torch.Tensor` ist optional, verbessert aber die Lesbarkeit.

**`return self.linear_layer(x)`**
Ruft die `forward()`-Methode von `nn.Linear` auf. Da das Modell nur aus dieser einen Schicht besteht und keine Aktivierungsfunktion verwendet, ist das Modell äquivalent zu einer klassischen multiplen linearen Regression. Für Regressionsprobleme (Vorhersage einer kontinuierlichen Zahl) ist keine Aktivierungsfunktion am Ausgang gewünscht.

---

## 5. `main()` - Datenvorbereitung pro Station

```python
def main():
    all_metrics = []

    for i, filename in enumerate(DATASETS):
        name = filename.replace('.csv', '')
        file_path = DATA_DIR / filename

        if not file_path.exists():
            print(f"Warnung: Datei {file_path} nicht gefunden, wird übersprungen.")
            continue

        df = pd.read_csv(file_path).dropna()
```

**`all_metrics = []`**
Leere Python-Liste, die nach jedem Modell ein Wörterbuch mit den Metriken (MAE, RMSE, R²) aufnimmt. Am Ende wird diese Liste in eine CSV-Datei umgewandelt.

**`for i, filename in enumerate(DATASETS):`**
Iteriert über alle 10 Dateinamen. `enumerate()` gibt den Index `i` (0, 1, 2, ..., 9) zusammen mit dem Wert `filename` zurück. Der Index `i` wird später benötigt, um die erste Zeile der Metriken-CSV zu überschreiben statt anzuhängen.

**`name = filename.replace('.csv', '')`**
Erzeugt einen Bezeichner ohne Dateiendung, z.B. `"050_Brunnen_Mositunnel_R1_engineered"`. Dieser Name wird als Grundlage für alle Ausgabedateinamen verwendet.

**`file_path = DATA_DIR / filename`**
Setzt den vollständigen Pfad zusammen: `data/v5_engineered/050_Brunnen_Mositunnel_R1_engineered.csv`.

**`if not file_path.exists(): ... continue`**
Prüft, ob die Datei existiert. Falls nicht, wird eine Warnung ausgegeben und die aktuelle Iteration übersprungen. Das macht das Skript robust gegenüber fehlenden Datensätzen.

**`df = pd.read_csv(file_path).dropna()`**
Lädt die CSV-Datei als Pandas-DataFrame und entfernt sofort alle Zeilen mit fehlenden Werten (NaN). PyTorch-Tensoren können keine NaN-Werte enthalten; ein einziger NaN-Wert würde den gesamten Loss zu `nan` machen.

```python
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
```

**`df['datetime'] = pd.to_datetime(df['datetime'])`**
Konvertiert die `datetime`-Spalte von einem String (z.B. `"2019-01-01 00:00:00"`) in ein echtes Pandas-`Timestamp`-Objekt. Nur so können Datumsoperationen (Sortierung, Differenz von Zeitstempeln) korrekt funktionieren.

**`df.set_index('datetime', inplace=True)`**
Setzt den Zeitstempel als DataFrame-Index. Dadurch steht `datetime` nicht mehr als Feature-Spalte zur Verfügung, was korrekt ist: der rohe Zeitstempel soll nicht als Zahl ins Modell eingehen.

**`df.sort_index(inplace=True)`**
Sortiert den DataFrame nach dem Zeitstempel in aufsteigender Reihenfolge. Dies ist Voraussetzung für den chronologischen Split.

```python
        feats = [
            'Year', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
            'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos',
            'is_weekend', 'is_holiday', 'temp', 'rain_1h', 'sun_1h', 'snow_1h', 'weather_cat'
        ]

        if 'weather_cat' in df.columns and df['weather_cat'].dtype == 'object':
            df['weather_cat'] = LabelEncoder().fit_transform(df['weather_cat'])

        feature_cols = [c for c in feats if c in df.columns]

        X = df[feature_cols].values
        y = df['volume'].values.reshape(-1, 1)
```

**`feats = [...]`**
Eine Python-Liste mit 16 Feature-Namen. Diese Liste definiert, welche Spalten als Eingabe-Features verwendet werden sollen. Die Features sind:
- Zeitfeatures: `Year`, und zyklische Kodierungen für Stunde, Wochentag, Monat, Jahrestag (Sinus/Kosinus-Paare)
- Binäre Flags: `is_weekend`, `is_holiday`
- Wetterdaten: `temp`, `rain_1h`, `sun_1h`, `snow_1h`, `weather_cat`

**`if 'weather_cat' in df.columns and df['weather_cat'].dtype == 'object':`**
Prüft, ob die Wetterklassen-Spalte vorhanden und noch als Text (Strings) gespeichert ist.

**`LabelEncoder().fit_transform(df['weather_cat'])`**
Konvertiert die Wetterklassen-Strings (z.B. `"rainy"`, `"sunny"`, `"cloudy"`) in aufeinanderfolgende Ganzzahlen (z.B. 0, 1, 2). `nn.Linear` kann nur numerische Eingaben verarbeiten.

**`feature_cols = [c for c in feats if c in df.columns]`**
List-Comprehension: Nimmt nur die Features aus `feats` auf, die tatsächlich in der CSV vorhanden sind. Falls eine Spalte fehlt, wird sie stillschweigend übersprungen. Das macht den Code robuster gegenüber unterschiedlichen CSV-Varianten.

**`X = df[feature_cols].values`**
Extrahiert die Feature-Matrix als NumPy-Array der Form `(n_samples, n_features)`. `.values` gibt das zugrundeliegende NumPy-Array zurück (ohne Pandas-Index).

**`y = df['volume'].values.reshape(-1, 1)`**
Extrahiert die Zielvariable `volume` als Spaltenvektor der Form `(n_samples, 1)`. `.reshape(-1, 1)` macht aus einem 1D-Array `[a, b, c, ...]` ein 2D-Array `[[a], [b], [c], ...]`. Das `-1` bedeutet: "berechne diese Dimension automatisch" (hier: Anzahl Stichproben). Die 2D-Form ist erforderlich, weil `nn.Linear` mit `out_features=1` ebenfalls einen 2D-Tensor zurückgibt.

---

## 6. `main()` - Chronologischer Train/Test-Split

```python
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        df_test = df.iloc[split_idx:].copy()
```

**`split_idx = int(len(df) * 0.8)`**
Berechnet den Trennindex für den 80/20-Split. `int()` rundet die Dezimalzahl auf eine ganze Zahl ab. Beispiel: Bei 50'000 Datenpunkten ist `split_idx = 40'000`.

**`X_train, X_test = X[:split_idx], X[split_idx:]`**
Schneidet die Feature-Matrix chronologisch in zwei Teile:
- `X_train`: die ersten 80% (ältere Daten)
- `X_test`: die letzten 20% (neuere Daten)

Da die Daten zuvor nach Datum sortiert wurden, lernt das Modell aus der Vergangenheit und wird auf der Zukunft getestet. Vergleiche `01_pytorch_workflow.ipynb`, Abschnitt "Splitting data into training and test sets":

```python
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
```

**Warum kein zufälliger Split?**
Bei Zeitreihendaten wäre ein zufälliger Split methodisch falsch (Data Leakage): Wenn man Messungen aus der Zukunft im Training verwendet, kann das Modell diese Information nutzen, die es in der Praxis nicht haben würde. Das Modell würde dann auf dem Testset zu gut abschneiden, ohne wirklich Vorhersagekraft zu besitzen.

**`df_test = df.iloc[split_idx:].copy()`**
Speichert den Testabschnitt des DataFrames (mit datetime-Index) separat für die Plots. `.copy()` erzeugt eine unabhängige Kopie, sodass spätere Änderungen am `df_test` den ursprünglichen DataFrame `df` nicht beeinflussen.

---

## 7. `main()` - StandardScaler und Tensor-Konvertierung

```python
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)
        y_test_scaled = y_scaler.transform(y_test)
```

**Konzeptuelle Erklärung: Was macht `StandardScaler`?**

`StandardScaler` normalisiert jede Feature-Spalte so, dass sie nach der Transformation Mittelwert 0 und Standardabweichung 1 hat:

$$x' = \frac{x - \mu}{\sigma}$$

wobei $\mu$ der Mittelwert und $\sigma$ die Standardabweichung der jeweiligen Spalte aus den Trainingsdaten sind.

**Warum ist Normalisierung für neuronale Netze wichtig?**
Ohne Normalisierung haben Features sehr unterschiedliche Grössenordnungen: `Year` liegt um 2020, `Hour_sin` zwischen -1 und 1, `temp` zwischen -10 und 35, `rain_1h` zwischen 0 und 50. Ein Gradient-Abstieg auf unnormierten Daten konvergiert viel langsamer, weil die Gradienten für unterschiedliche Features stark in ihrer Grössenordnung variieren. Normalisierte Eingaben beschleunigen und stabilisieren das Training erheblich.

**`scaler.fit_transform(X_train)`**
Zwei Operationen in einem Schritt:
1. `fit`: berechnet Mittelwert und Standardabweichung jeder Spalte aus `X_train`
2. `transform`: normalisiert `X_train` mit diesen Werten

Wichtig: `fit` darf nur auf den Trainingsdaten aufgerufen werden.

**`scaler.transform(X_test)`**
Auf den Testdaten wird nur `transform` aufgerufen, nicht `fit`. Die Testdaten werden mit den Statistiken (Mittelwert, Standardabweichung) der Trainingsdaten normalisiert. Das ist methodisch korrekt: in der Realität kennt man zum Zeitpunkt des Trainings die statistischen Eigenschaften der Zukunft (Testset) nicht.

**`y_scaler.fit_transform(y_train)`**
Auch die Zielvariable `volume` wird skaliert. Dadurch liegt der Loss nicht im Bereich von Tausenden von Fahrzeugen, sondern etwa bei Werten um 0 bis 3. Das verbessert die numerische Stabilität des Trainings.

```python
        X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
        X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32).to(device)
        y_test_t  = torch.tensor(y_test_scaled,  dtype=torch.float32).to(device)
```

**`torch.tensor(..., dtype=torch.float32)`**
Konvertiert das NumPy-Array in einen PyTorch-Tensor. Der Datentyp `float32` (32-Bit Gleitkommazahl) ist der Standard in PyTorch und in `nn.Linear`. Vergleiche `00_pytorch_fundamentals.ipynb`, Abschnitt "Tensor datatypes": PyTorch arbeitet standardmässig mit `float32`. Der Grund: GPU-Berechnungen sind mit `float32` deutlich schneller als mit `float64`.

**`.to(device)`**
Verschiebt den Tensor auf das ausgewählte Gerät (CPU oder GPU). Alle Operationen zwischen Tensoren müssen auf demselben Gerät stattfinden. Vergleiche `00_pytorch_fundamentals.ipynb`, Abschnitt "Putting tensors on the GPU":

```python
tensor_on_gpu = tensor.to(device)
```

---

## 8. `main()` - Modell-Setup

```python
        model = LinearRegressionModel(input_dim=len(feature_cols)).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        epochs = 200
```

**`model = LinearRegressionModel(input_dim=len(feature_cols)).to(device)`**
Erstellt für jede Station eine neue, unabhängige Modellinstanz. `input_dim=len(feature_cols)` übergibt die Anzahl der verfügbaren Features. `.to(device)` verschiebt alle Parameter des Modells auf das Gerät. Vergleiche `01_pytorch_workflow.ipynb`, Abschnitt 6.2: `model_1 = LinearRegressionModelV2(); model_1.to(device)`.

**`loss_fn = nn.MSELoss()`**
Die **Mean Squared Error Loss** (mittlerer quadratischer Fehler) als Verlustfunktion:

$$L_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

Sie misst, wie weit die Vorhersagen $\hat{y}_i$ von den echten Werten $y_i$ entfernt sind. Grosse Abweichungen werden durch das Quadrieren stärker bestraft als kleine.

**Unterschied zum Kurs-Notebook:** In `01_pytorch_workflow.ipynb` wird `nn.L1Loss()` (Mean Absolute Error) verwendet:
$$L_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |\hat{y}_i - y_i|$$

Der Vorteil von MSE ist, dass er überall differenzierbar ist (L1 hat bei 0 eine nicht-differenzierbare Ecke), was die Gradientenberechnung vereinfacht. MSE ist für Regressionsaufgaben üblicher und wird hier eingesetzt, obwohl er im Kurs nicht explizit behandelt wurde.

**`optimizer = torch.optim.Adam(model.parameters(), lr=0.01)`**

- **`model.parameters()`:** Gibt einen Iterator über alle lernbaren Parameter des Modells zurück (hier: Gewichtsmatrix und Bias von `self.linear_layer`). Vergleiche `01_pytorch_workflow.ipynb`, Zelle 15: `list(model_0.parameters())`. Im Notebook wird erklärt: "torch.nn.Parameter: What parameters should our model try and learn". Der Optimizer erhält diesen Iterator, um zu wissen, welche Werte er anpassen soll.

- **`lr=0.01`:** Die Lernrate (Learning Rate) steuert die Schrittgrösse bei der Parameteranpassung. Eine Lernrate von 0.01 bedeutet: der Parameter wird um maximal 1% seines skalierten Gradienten verändert. Zu hohe Lernraten führen zu instabilem Training (der Loss springt unkontrolliert), zu tiefe Lernraten verlangsamen die Konvergenz stark. Typische Grössenordnungen: $10^{-4}$ bis $10^{-1}$. Im Kurs-Notebook wird ebenfalls `lr=0.01` bei SGD verwendet.

**Konzeptuelle Erklärung: Adam vs. SGD**

Im Kurs-Notebook (`01_pytorch_workflow.ipynb`, Zelle 25 und 60) wird `torch.optim.SGD` (Stochastic Gradient Descent) verwendet:
```python
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
```

**Adam** (Adaptive Moment Estimation, 2014 von Kingma & Ba) ist eine Weiterentwicklung:
- **SGD** aktualisiert alle Parameter mit derselben festen Lernrate: $\theta \leftarrow \theta - \alpha \cdot \nabla L$
- **Adam** berechnet für jeden Parameter eine individuelle, adaptive Lernrate basierend auf zwei exponentiell gleitenden Durchschnitten:
  - $m_t$: erster Moment (gleitender Durchschnitt des Gradienten selbst)
  - $v_t$: zweiter Moment (gleitender Durchschnitt des quadrierten Gradienten)

Die Update-Regel von Adam lautet:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

Der praktische Vorteil: Adam konvergiert in den meisten Fällen schneller und robuster als SGD, weil Parameter mit grossen Gradienten kleinere Schritte machen und Parameter mit kleinen Gradienten grössere Schritte. Adam ist heute der meistverwendete Optimizer für neuronale Netze. **Dieser Optimizer wird im Kurs-Notebook nicht behandelt.**

**`epochs = 200`**
Die Anzahl der vollständigen Trainingsdurchläufe über alle Trainingsdaten. Nach 200 Epochen wird das Training beendet.

---

## 9. `main()` - Trainings-Loop

```python
        epoch_count = []
        loss_values = []
        test_loss_values = []

        for epoch in range(epochs):
            model.train()

            y_pred = model(X_train_t)
            loss = loss_fn(y_pred, y_train_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.inference_mode():
                test_pred = model(X_test_t)
                test_loss = loss_fn(test_pred, y_test_t)

            if epoch % 10 == 0:
                epoch_count.append(epoch)
                loss_values.append(loss.item())
                test_loss_values.append(test_loss.item())
```

Dies ist das Herzstück des Skripts. Der gesamte Loop entspricht direkt dem Trainings-Loop aus `01_pytorch_workflow.ipynb`, Zellen 27 und 61. Die Kommentare im Code (`# 0. Loop through the data`, `# 1. Forward pass`, etc.) wurden direkt aus dem Notebook übernommen.

**`epoch_count = []`, `loss_values = []`, `test_loss_values = []`**
Leere Listen zum Aufzeichnen der Loss-Werte über die Epochen. Vergleiche `01_pytorch_workflow.ipynb`: "Track different values".

**`for epoch in range(epochs):`**
Äussere Schleife über alle Epochen. Jede Epoche = ein vollständiger Durchlauf durch alle Trainingsdaten.

---

**Schritt 0: `model.train()`**

Versetzt das Modell in den Trainings-Modus. Für das lineare Regressionsmodell ohne Dropout oder BatchNorm macht das keinen funktionalen Unterschied. Es ist aber gute Praxis, diesen Aufruf immer zu machen, damit beim späteren Hinzufügen von Dropout oder BatchNorm das Verhalten korrekt ist. Vergleiche `01_pytorch_workflow.ipynb`: `model_0.train() # train mode in PyTorch sets all parameters that require gradients to require gradients`.

---

**Schritt 1: `y_pred = model(X_train_t)` - Forward Pass**

Ruft implizit `model.forward(X_train_t)` auf. Die Eingabe `X_train_t` wird durch `self.linear_layer` geleitet und das Ergebnis $\hat{y}$ zurückgegeben. Dabei baut PyTorch automatisch einen Berechnungsgraphen (Computational Graph) auf: PyTorch merkt sich, welche Operationen auf welchen Tensoren ausgeführt wurden. Dieser Graph wird in Schritt 4 für die Backpropagation traversiert. Vergleiche `01_pytorch_workflow.ipynb`, Abschnitt "Building a training loop": "Forward pass (this involves data moving through our model's forward() functions) to make predictions on data".

---

**Schritt 2: `loss = loss_fn(y_pred, y_train_t)` - Verlustberechnung**

Berechnet den MSE-Loss zwischen den Vorhersagen `y_pred` und den echten Werten `y_train_t`. Das Ergebnis ist ein Skalar-Tensor (eine einzige Zahl). Vergleiche `01_pytorch_workflow.ipynb`, Schritt 2: "Calculate the loss".

---

**Schritt 3: `optimizer.zero_grad()` - Gradienten zurücksetzen**

Setzt alle Gradienten aller Parameter auf 0. Dies ist notwendig, weil PyTorch Gradienten standardmässig akkumuliert: ohne diesen Schritt würde in jeder Epoche der neue Gradient zum alten addiert werden, was zu falschen Updates führt. Vergleiche `01_pytorch_workflow.ipynb`, Kommentar: "by default how the optimizer changes will accumulate through the loop so... we have to zero them above in step 3 for the next iteration of the loop".

---

**Schritt 4: `loss.backward()` - Backpropagation**

Berechnet die Gradienten aller Parameter bezüglich des Losses. PyTorch traversiert den in Schritt 1 aufgebauten Berechnungsgraphen rückwärts (daher "backward") und wendet die Kettenregel der Differentialrechnung an:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W}$$

Für MSE und eine lineare Schicht ergibt sich konkret:
$$\frac{\partial L_{\text{MSE}}}{\partial W} = \frac{2}{n} X^T (\hat{y} - y)$$

Die Gradienten werden in den `.grad`-Attributen der Parameter gespeichert (z.B. `model.linear_layer.weight.grad`). Vergleiche `01_pytorch_workflow.ipynb`, Schritt 4: "Perform backpropagation on the loss with respect to the parameters of the model".

---

**Schritt 5: `optimizer.step()` - Parameterupdate**

Adam liest die in `.grad` gespeicherten Gradienten und aktualisiert die Parameter gemäss seiner Update-Regel. Die Gewichte und der Bias der linearen Schicht werden in Richtung eines niedrigeren Losses verändert. Vergleiche `01_pytorch_workflow.ipynb`, Schritt 5: "Step the optimizer (perform gradient descent)".

---

**Testing-Block (innerhalb der Epoch-Schleife):**

```python
            model.eval()
            with torch.inference_mode():
                test_pred = model(X_test_t)
                test_loss = loss_fn(test_pred, y_test_t)
```

**`model.eval()`**
Versetzt das Modell in den Evaluationsmodus. Vergleiche `01_pytorch_workflow.ipynb`: `model_0.eval() # turns off gradient tracking and also (dropout/batch norm layers)`.

**`with torch.inference_mode():`**
Kontext-Manager, der die automatische Gradientenberechnung (Autograd) deaktiviert. Da bei der Evaluation keine Backpropagation stattfindet, spart das Speicher und Rechenzeit. Vergleiche `01_pytorch_workflow.ipynb`, Zelle 20:
```python
with torch.inference_mode():
    y_preds = model_0(X_test)
# Note: in older PyTorch code you might also see torch.no_grad()
```

**`if epoch % 10 == 0:`**
Speichert nur jeden 10. Epochen-Wert für die Loss-Kurve (Epoche 0, 10, 20, ..., 190). Das reduziert den Speicherbedarf und hält die Darstellung übersichtlich.

**`loss.item()`**
Konvertiert den Skalar-Tensor in eine Python-`float`-Zahl. Python-Listen können keine Tensoren direkt aufnehmen.

---

## 10. `main()` - Evaluation auf dem Testset

```python
        model.eval()
        with torch.inference_mode():
            preds_t = model(X_test_t)
            preds_scaled = preds_t.cpu().numpy()

        preds = y_scaler.inverse_transform(preds_scaled)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
```

**`preds_t.cpu().numpy()`**
Zwei Schritte:
1. `.cpu()`: verschiebt den Tensor von der GPU auf die CPU (falls nötig). Vergleiche `00_pytorch_fundamentals.ipynb`, Abschnitt "Moving tensors back to the CPU": `tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()`.
2. `.numpy()`: konvertiert den PyTorch-Tensor in ein NumPy-Array. Das ist nötig, weil sklearn-Funktionen NumPy-Arrays erwarten.

**`y_scaler.inverse_transform(preds_scaled)`**
Macht die Skalierung rückgängig und transformiert die normierten Vorhersagen zurück in Fahrzeuge pro Stunde:
$$x = x' \cdot \sigma + \mu$$
Ohne diesen Schritt würden MAE und RMSE auf skalierten Werten (um 0) berechnet und wären nicht als physikalische Grösse interpretierbar.

**Metriken:**

- $\text{MAE} = \frac{1}{n} \sum_{i=1}^n |\hat{y}_i - y_i|$ in Fahrzeugen/h: durchschnittlicher absoluter Fehler.
- $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2}$ in Fahrzeugen/h: wie MAE, aber grosse Fehler werden stärker gewichtet.
- $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$: Anteil der durch das Modell erklärten Varianz. $R^2 = 1$ bedeutet perfekte Vorhersage, $R^2 = 0$ ist gleichwertig mit dem Mittelwert als Vorhersage, $R^2 < 0$ ist schlechter als der Mittelwert.

Diese Metriken werden im Kurs-Notebook nicht behandelt.

---

## 11. `main()` - Speichern des Modells

```python
        torch.save(model.state_dict(), RESULTS_DIR / "model_weights" / f"lr_{name}.pt")
```

**`model.state_dict()`**
Gibt ein geordnetes Wörterbuch (OrderedDict) aller lernbaren Parameter zurück, z.B.:
```
{'linear_layer.weight': tensor([[0.23, -0.11, ...]]),
 'linear_layer.bias':   tensor([0.05])}
```
Vergleiche `01_pytorch_workflow.ipynb`, Abschnitt 4 "Saving a model": `torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)`. Das Notebook erklärt auch, warum man den `state_dict` statt des gesamten Modells speichert: "Since we saved our model's state_dict() rather the entire model, we'll create a new instance of our model class and load the saved state_dict() into that."

**`torch.save(...)`**
Serialisiert das Objekt mit Python's Pickle-Format. Die `.pt`-Datei enthält ausschliesslich die Zahlenwerte der gelernten Parameter, nicht die Modellarchitektur. Zum Laden muss man zuerst eine neue Modellinstanz erstellen und dann `model.load_state_dict(torch.load(...))` aufrufen.

---

## 12. Bezug zu den Notebooks

| Konzept im Skript | Quelle |
|---|---|
| `class X(nn.Module)` Subclassing | `01_pytorch_workflow.ipynb`, Zelle 12 und 55 |
| `nn.Linear(in_features, out_features)` | `01_pytorch_workflow.ipynb`, Zelle 55 |
| `super().__init__()` | `01_pytorch_workflow.ipynb`, Zelle 12 |
| `model.parameters()` | `01_pytorch_workflow.ipynb`, Zelle 15 und 25 |
| `def forward(self, x)` | `01_pytorch_workflow.ipynb`, Abschnitt "PyTorch model building essentials" |
| 5-Schritte Trainings-Loop | `01_pytorch_workflow.ipynb`, Zellen 27 und 61 |
| `model.train()` / `model.eval()` | `01_pytorch_workflow.ipynb`, Zellen 27 und 61 |
| `torch.inference_mode()` | `01_pytorch_workflow.ipynb`, Zelle 20 und 65 |
| `torch.save()` / `state_dict()` | `01_pytorch_workflow.ipynb`, Abschnitte 4 und 5 |
| `torch.manual_seed(42)` | `00_pytorch_fundamentals.ipynb`, Abschnitt "Reproducibility" |
| `device = "cuda" if ... else "cpu"` | `00_pytorch_fundamentals.ipynb`, Abschnitt "Running tensors on the GPU"; `01_pytorch_workflow.ipynb`, Abschnitt 6 |
| `.to(device)` für Tensor und Modell | `00_pytorch_fundamentals.ipynb`, Abschnitt "Putting tensors on the GPU" |
| `dtype=torch.float32` | `00_pytorch_fundamentals.ipynb`, Abschnitt "Tensor datatypes" |
| `.cpu().numpy()` | `00_pytorch_fundamentals.ipynb`, Abschnitt "Moving tensors back to the CPU" |
| Chronologischer 80/20-Split | `01_pytorch_workflow.ipynb`, Abschnitt "Splitting data into training and test sets" |

**Konzepte, die über die Notebooks hinausgehen:**
- `nn.MSELoss()`: Im Notebook wird `nn.L1Loss()` verwendet. MSE bestraft grosse Ausreisser stärker.
- `torch.optim.Adam`: Im Notebook wird `torch.optim.SGD` verwendet. Adam ist adaptiv und konvergiert in der Praxis schneller.
- `StandardScaler` von scikit-learn: Nicht im Notebook behandelt, aber wichtig für stabiles Training.
- `LabelEncoder` von scikit-learn: Nicht im Notebook behandelt.
- Metriken MAE, RMSE, R²: Nicht im Notebook behandelt.
- `model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})`: Im Notebook wird das Laden ohne device-Transfer gezeigt.

---

## 13. Häufige Stolpersteine

**1. `fit_transform` vs `transform` auf dem Testset**
`scaler.fit_transform(X_train)` darf nur auf Trainingsdaten angewendet werden. Auf Testdaten immer nur `.transform()`. Andernfalls fliesst statistische Information über das Testset in die Normalisierung ein (Data Leakage), was die Metriken verfälscht.

**2. `optimizer.zero_grad()` vergessen**
Ohne diesen Aufruf akkumulieren sich die Gradienten über Epochen hinweg. Das Training würde mit unkorrekten Gradienten arbeiten und instabil werden.

**3. `inverse_transform` nach der Vorhersage**
Die Modellvorhersagen sind skaliert. Für interpretierbare Metriken (MAE in Fahrzeugen/h) muss `y_scaler.inverse_transform()` aufgerufen werden, bevor MAE/RMSE/R² berechnet werden.

**4. `.to(device)` für Modell und Daten**
Sowohl das Modell als auch alle Eingabe-Tensoren müssen auf demselben Gerät liegen. Wenn das Modell auf der GPU ist, aber die Eingabedaten auf der CPU, entsteht ein Laufzeitfehler.

**5. Chronologischer Split ist zwingend**
Ein zufälliger Split wäre bei Zeitreihendaten methodisch falsch. Das Modell würde dann Informationen aus der Zukunft im Training verwenden.

**6. `y.reshape(-1, 1)` notwendig**
`nn.Linear(out_features=1)` gibt einen Tensor der Form `(n, 1)` zurück. Die Zielgrösse muss dieselbe Form haben, damit der Loss korrekt berechnet wird. Ein 1D-Array würde einen Shape-Fehler verursachen.

**7. `loss.item()` statt `loss` in der Liste**
Tensoren mit `requires_grad=True` in einer Python-Liste zu speichern, hält den gesamten Berechnungsgraphen im Speicher. `.item()` extrahiert nur den numerischen Wert.

---

## Anmerkung

- `import os` (Zeile 5) wird im gesamten Skript nicht verwendet. Es kann entfernt werden, ohne die Funktionalität zu beeinflussen.
- Das Skript trainiert kein Validierungsset: der Test-Loss wird bereits während des Trainings berechnet und für die Loss-Kurve verwendet. Da kein Early Stopping angewendet wird, besteht das Risiko des Overfittings in den letzten Epochen. Beim MLP wird dieses Problem durch ein separates Validierungsset und Early Stopping behoben.
- `y_test_scaled` wird berechnet (`y_test_scaled = y_scaler.transform(y_test)`), aber nie verwendet. Der Test-Loss in der Trainingsschleife wird auf den skalierten Testdaten (`y_test_t`) berechnet, und die finalen Metriken werden auf den unsk alierten Originaldaten (`y_test`) berechnet. Die Variable `y_test_scaled` ist überflüssig.
