# Erklärung: `models/mlp.py`

**Originaldatei:** `models/mlp.py`

**Zusammenfassung:** Dieses Skript definiert ein tiefes Multi-Layer Perceptron (MLP) mit Batch-Normalisierung und Dropout, trainiert es für jede der 10 Messstationen und speichert Gewichte, Vorhersagen und Evaluationsplots. Gegenüber dem linearen Baseline-Modell enthält das MLP mehrere versteckte Schichten, die nicht-lineare Zusammenhänge zwischen den Features und dem Verkehrsvolumen lernen können. Zusätzliche Techniken wie DataLoader, Learning-Rate-Scheduler und Early Stopping machen das Training stabiler und effizienter.

---

## 1. Imports

```python
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

Imports, die bereits in `linear_regression.py` erklärt wurden (`json`, `time`, `pathlib.Path`, `numpy`, `pandas`, `torch`, `torch.nn`, `matplotlib`, `sklearn.preprocessing`, `sklearn.metrics`), werden hier nur kurz erwähnt. Neu hinzugekommen sind:

**`from torch.utils.data import Dataset, DataLoader`**
Aus dem Unterpaket `torch.utils.data`:
- `Dataset`: abstrakte Basisklasse für benutzerdefinierte Datensätze. Man erbt von ihr und implementiert `__len__` und `__getitem__`.
- `DataLoader`: Wrapper um ein `Dataset`, der das Laden von Daten in Batches, das optionale Mischen (Shuffle) und das parallele Laden (num_workers) übernimmt.

Diese beiden Klassen sind im Kurs-Notebook `01_pytorch_workflow.ipynb` nicht behandelt. Sie sind jedoch ein zentrales Konzept in PyTorch für das Training mit grossen Datensätzen.

---

## 2. Seeds und Device

```python
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Identisch mit `linear_regression.py`. Vergleiche `00_pytorch_fundamentals.ipynb`, Abschnitt "Reproducibility" und `01_pytorch_workflow.ipynb`, Abschnitt 6. Für die ausführliche Erklärung siehe `Theorie/models/linear_regression.md`, Abschnitt 2.

---

## 3. Hyperparameter-Konfiguration

```python
HIDDEN_DIMS   = [256, 128, 128, 64]
DROPOUT       = 0.2582
BATCH_SIZE    = 128
MAX_EPOCHS    = 500
LEARNING_RATE = 0.0003352
WEIGHT_DECAY  = 0.0016735
LR_PATIENCE   = 15
LR_FACTOR     = 0.5
ES_PATIENCE   = 10
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.10
```

Alle Hyperparameter sind am Anfang des Skripts als Konstanten definiert, damit sie leicht übersichtlich geändert werden können (keine "Magic Numbers" im Code). Die Werte stammen aus dem Optuna-Tuning (`mlp_tuning.py`).

**`HIDDEN_DIMS = [256, 128, 128, 64]`**
Liste der Grössen der vier versteckten Schichten. Das Netz hat also die Struktur:

$$\text{Input}(16) \to 256 \to 128 \to 128 \to 64 \to \text{Output}(1)$$

Die abnehmende Grösse ist eine gängige Praxis: breite erste Schicht erfasst viele Features, spätere Schichten verdichten die Repräsentation.

**`DROPOUT = 0.2582`**
Die Dropout-Rate: 25.82% der Neuronen werden in jeder Trainingsepoche zufällig deaktiviert (auf 0 gesetzt). Dieser Wert wurde durch Optuna-Tuning ermittelt.

**`BATCH_SIZE = 128`**
Anzahl der Trainingsbeispiele, die in einem Schritt durch das Netz verarbeitet werden. Kleinere Batches bedeuten mehr Updates pro Epoche, aber mehr Rauschen in den Gradienten.

**`MAX_EPOCHS = 500`**
Maximale Anzahl von Trainingsdurchläufen. In der Praxis endet das Training durch Early Stopping deutlich früher.

**`LEARNING_RATE = 0.0003352`**
Die initiale Lernrate für Adam. Dieser sehr genaue Wert stammt aus dem Optuna-Tuning.

**`WEIGHT_DECAY = 0.0016735`**
Regularisierungsparameter. Addiert eine Strafe auf die L2-Norm der Gewichte zur Loss-Funktion (L2-Regularisierung), um Overfitting zu reduzieren. Wird direkt im Adam-Optimizer übergeben.

**`LR_PATIENCE = 15`, `LR_FACTOR = 0.5`**
Parameter für den Learning-Rate-Scheduler `ReduceLROnPlateau`: Wenn sich der Validation Loss 15 Epochen lang nicht verbessert, wird die Lernrate mit Faktor 0.5 halbiert.

**`ES_PATIENCE = 10`**
Early-Stopping-Geduld: Wenn sich der Validation Loss 10 Epochen lang nicht verbessert, wird das Training abgebrochen.

**`TRAIN_RATIO = 0.70`, `VAL_RATIO = 0.10`**
Das MLP verwendet einen 70/10/20-Split (Training/Validation/Test). Das Validierungsset wird für Early Stopping und den LR-Scheduler verwendet. Der lineare Baseline-Modell verwendet keinen Validierungsset (80/20).

---

## 4. Dataset-Klasse `VerkehrsDataset`

```python
class VerkehrsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

Diese Klasse ist im Kurs-Notebook nicht behandelt, ist aber ein fundamentales Konzept in PyTorch.

**`class VerkehrsDataset(Dataset):`**
Erbt von `torch.utils.data.Dataset`. Um den `DataLoader` verwenden zu können, muss man entweder eine bestehende Dataset-Klasse verwenden oder eine eigene implementieren, die von `Dataset` erbt.

**`def __init__(self, X, y):`**
Konstruktor: Nimmt die Feature-Matrix `X` und die Zielvariable `y` als NumPy-Arrays entgegen und konvertiert sie sofort in `float32`-Tensoren. Die Konvertierung findet einmalig hier statt (nicht bei jedem `__getitem__`-Aufruf), was effizienter ist.

**`def __len__(self) -> int:`**
Pflichtmethode. Der `DataLoader` ruft diese Methode auf, um die Gesamtanzahl der Datenpunkte zu kennen. Gibt die Anzahl der Zeilen zurück.

**`def __getitem__(self, idx):`**
Pflichtmethode. Gibt ein einzelnes Datenpaar (Feature-Vektor, Zielwert) für den Index `idx` zurück. Der `DataLoader` ruft diese Methode in einer Schleife auf und fasst die Einzelergebnisse zu einem Batch zusammen.

**Konzeptuelle Erklärung: Warum `Dataset` und `DataLoader`?**

Beim Training mit dem linearen Regressionsmodell werden alle Trainingsdaten auf einmal durch das Netz gegeben (Full-Batch Gradient Descent). Das ist bei kleinen Datensätzen möglich, skaliert aber nicht auf grosse Datenmengen, die nicht in den Speicher passen. Der `DataLoader` unterteilt den Datensatz in kleinere Batches und gibt sie nacheinander an das Modell. Mini-Batch-Gradient-Descent hat ausserdem einen regularisierenden Effekt: das Rauschen in den Batch-Gradienten hilft, lokale Minima zu verlassen.

---

## 5. Modellklasse `MLP`

```python
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

Die MLP-Klasse ist das Kernstück des Projekts und geht inhaltlich deutlich über das Kurs-Notebook hinaus.

**`class MLP(nn.Module):`**
Erbt von `nn.Module`, genau wie `LinearRegressionModel`. Vergleiche `01_pytorch_workflow.ipynb`, Zelle 12 und 55.

**`def __init__(self, input_dim, hidden_dims, dropout):`**
Der Konstruktor nimmt drei Parameter:
- `input_dim`: Anzahl der Eingabe-Features (z.B. 16)
- `hidden_dims`: Liste der Schichtgrössen, z.B. `[256, 128, 128, 64]`
- `dropout`: Dropout-Rate, z.B. `0.2582`

**`super().__init__()`**
Pflicht bei jeder `nn.Module`-Unterklasse. Vergleiche `01_pytorch_workflow.ipynb`, Zelle 12 und 55.

**`layers = []` und `in_dim = input_dim`**
`layers` ist eine Python-Liste, in die Schritt für Schritt alle Schichten des Netzes eingefügt werden. `in_dim` verfolgt die aktuelle Eingabedimension: sie startet bei `input_dim` und wird nach jeder Schicht auf `out_dim` gesetzt.

**`for i, out_dim in enumerate(hidden_dims):`**
Schleife über jede versteckte Schicht. Bei `hidden_dims = [256, 128, 128, 64]` läuft die Schleife 4 Mal.

**`layers.append(nn.Linear(in_dim, out_dim))`**
Fügt eine lineare Schicht hinzu. Im ersten Durchlauf: `nn.Linear(16, 256)`. Vergleiche `01_pytorch_workflow.ipynb`, Zelle 55.

**`layers.append(nn.BatchNorm1d(out_dim))`**
**Batch-Normalisierung** (nicht im Kurs-Notebook behandelt). Diese Schicht normalisiert die Ausgabe der linearen Schicht über den aktuellen Batch hinweg:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

Dann skaliert und verschiebt sie die normalisierten Werte mit lernbaren Parametern $\gamma$ und $\beta$:

$$y_i = \gamma \hat{x}_i + \beta$$

Wobei $\mu_B$ und $\sigma_B^2$ der Mittelwert und die Varianz über den Batch sind.

**Warum Batch-Normalisierung?**
Ohne Normalisierung können die Aktivierungen in tiefen Netzen im Laufe des Trainings sehr grosse oder sehr kleine Werte annehmen (Internal Covariate Shift). Batch-Normalisierung hält die Aktivierungen in einem stabilen Bereich, was:
1. das Training deutlich beschleunigt (höhere Lernraten möglich)
2. weniger anfällig für die Initialisierung der Gewichte macht
3. einen leichten Regularisierungseffekt hat

`nn.BatchNorm1d` ist die Version für 1D-Daten (im Gegensatz zu `BatchNorm2d` für Bilder).

**`layers.append(nn.ReLU())`**
**Rectified Linear Unit** Aktivierungsfunktion:
$$\text{ReLU}(x) = \max(0, x)$$

Negative Aktivierungen werden auf 0 gesetzt, positive bleiben unverändert. ReLU ist die meistverwendete Aktivierungsfunktion in tiefen neuronalen Netzen.

**Warum eine Aktivierungsfunktion?**
Ohne Aktivierungsfunktion wäre die gesamte Abfolge linearer Schichten äquivalent zu einer einzigen linearen Schicht ($W_2 (W_1 x + b_1) + b_2 = W_{\text{eff}} x + b_{\text{eff}}$). Nicht-lineare Aktivierungsfunktionen ermöglichen dem Netz, beliebige nicht-lineare Zusammenhänge zu lernen.

**Hinweis:** Im Kurs-Notebook `01_pytorch_workflow.ipynb` werden keine Aktivierungsfunktionen behandelt, da das Kurs-Beispiel ein lineares Regressionsproblem ist.

**`if i < len(hidden_dims) - 1: layers.append(nn.Dropout(dropout))`**
**Dropout** (nicht im Kurs-Notebook behandelt): Setzt in jeder Trainings-Iteration einen zufälligen Anteil (hier 25.82%) der Neuronen auf 0. Beim letzten Hidden Layer (Index 3 von 4) wird kein Dropout angewendet, da die Ausgabe dieser Schicht direkt in die Output-Schicht geht.

**Warum Dropout?**
Dropout ist eine Regularisierungstechnik. Indem zufällig Neuronen deaktiviert werden, kann sich das Netz nicht auf einzelne Neuronen verlassen (Co-Adaptation). Das zwingt das Netz, robustere, redundante Repräsentationen zu lernen und reduziert Overfitting. Im Evaluationsmodus (`model.eval()`) ist Dropout automatisch deaktiviert: alle Neuronen sind aktiv.

**`in_dim = out_dim`**
Aktualisiert die Eingabedimension für die nächste Schicht. Nach der ersten Schicht (`in_dim=16`, `out_dim=256`) wird `in_dim=256` für die nächste Iteration.

**`layers.append(nn.Linear(in_dim, 1))`**
Fügt die Ausgabeschicht hinzu: eine lineare Schicht, die von der letzten Hidden-Layer-Grösse (64) auf einen einzelnen Ausgabewert abbildet. Keine Aktivierungsfunktion, da Regression.

**`self.net = nn.Sequential(*layers)`**
`nn.Sequential` ist ein Container, der Schichten in einer definierten Reihenfolge stapelt. Der `*`-Operator entpackt die Liste in einzelne Argumente. Wenn `model.net(x)` aufgerufen wird, wird `x` der Reihe nach durch alle Schichten in `layers` geleitet.

Die resultierende Netzarchitektur (für `hidden_dims=[256, 128, 128, 64]`):

```
Input(16)
  → Linear(16→256) → BatchNorm1d(256) → ReLU → Dropout(0.26)
  → Linear(256→128) → BatchNorm1d(128) → ReLU → Dropout(0.26)
  → Linear(128→128) → BatchNorm1d(128) → ReLU → Dropout(0.26)
  → Linear(128→64) → BatchNorm1d(64) → ReLU
  → Linear(64→1)
Output(1)
```

**`def forward(self, x: torch.Tensor) -> torch.Tensor:`**
Ruft einfach `self.net(x)` auf, was den Tensor durch alle Schichten in `nn.Sequential` schickt. Vergleiche `01_pytorch_workflow.ipynb`, Abschnitt "PyTorch model building essentials".

---

## 6. `main()` - Datenvorbereitung und 70/10/20-Split

```python
        n         = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
        df_test = df.iloc[val_end:].copy()
```

Das MLP verwendet einen dreifachen chronologischen Split: 70% Training, 10% Validation, 20% Test.

**`train_end = int(n * TRAIN_RATIO)`**
Trennpunkt zwischen Training und Validierung: die ersten 70% der Daten.

**`val_end = int(n * (TRAIN_RATIO + VAL_RATIO))`**
Trennpunkt zwischen Validierung und Test: nach 80% der Daten.

**`X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]`**
Chronologischer Dreifach-Split. Drei aufeinanderfolgende Zeitabschnitte.

**Warum ein Validierungsset?**
Das Validierungsset hat zwei Aufgaben in diesem Skript:
1. **Early Stopping:** Der Validation Loss zeigt, ob das Modell auf ungesehenen Daten besser oder schlechter wird. Verschlechtert er sich, wird das Training gestoppt.
2. **LR-Scheduler:** Der Scheduler beobachtet den Validation Loss und reduziert die Lernrate, wenn keine Verbesserung eintritt.

Das Testset wird hingegen erst nach dem Training einmalig für die finale Evaluation verwendet.

---

## 7. `main()` - Skalierung und DataLoader

```python
        x_scaler  = StandardScaler()
        X_train_s = x_scaler.fit_transform(X_train)
        X_val_s   = x_scaler.transform(X_val)
        X_test_s  = x_scaler.transform(X_test)

        y_scaler  = StandardScaler()
        y_train_s = y_scaler.fit_transform(y_train)
        y_val_s   = y_scaler.transform(y_val)
```

Wie beim linearen Modell, jedoch mit drei statt zwei Teilmengen. Der Scaler wird nur auf `X_train` gefittet. Für die Erklärung von `StandardScaler` siehe `Theorie/models/linear_regression.md`, Abschnitt 7.

```python
        train_loader = DataLoader(
            VerkehrsDataset(X_train_s, y_train_s), batch_size=BATCH_SIZE, shuffle=False
        )
        val_loader = DataLoader(
            VerkehrsDataset(X_val_s, y_val_s), batch_size=BATCH_SIZE, shuffle=False
        )
```

**`DataLoader(VerkehrsDataset(...), batch_size=BATCH_SIZE, shuffle=False)`**
Erstellt einen DataLoader. Der DataLoader iteriert über das Dataset und gibt bei jeder Iteration einen Batch von `BATCH_SIZE=128` Datenpunkten zurück.

- **`batch_size=128`:** Pro Trainingsschritt werden 128 Datenpunkte durch das Netz geschickt. Bei z.B. 35'000 Trainingspunkten sind das etwa 274 Batches pro Epoche.
- **`shuffle=False`:** Die Daten werden nicht gemischt. Da es sich um Zeitreihendaten handelt, würde ein Shuffle die zeitliche Reihenfolge zerstören. Für das Training eines MLPs ohne Lag-Features spielt die Reihenfolge innerhalb einer Epoche zwar keine direkte Rolle, aber es ist methodisch konsistent.

---

## 8. `main()` - Modell-Setup mit Scheduler und Early Stopping

```python
        model     = MLP(len(feature_cols), HIDDEN_DIMS, DROPOUT).to(device)
        loss_fn   = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=LR_PATIENCE, factor=LR_FACTOR
        )

        best_val_loss = float("inf")
        best_weights  = None
        no_improve    = 0
```

**`model = MLP(len(feature_cols), HIDDEN_DIMS, DROPOUT).to(device)`**
Erstellt eine neue MLP-Instanz für jede Station. Für `.to(device)` und `model.parameters()` vergleiche `01_pytorch_workflow.ipynb`, Abschnitt 6.2.

**`optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)`**
Adam-Optimizer mit Weight Decay. Für eine Erklärung von Adam vs. SGD vergleiche `Theorie/models/linear_regression.md`, Abschnitt 8.

**`weight_decay=WEIGHT_DECAY`** (nicht im Kurs-Notebook behandelt)
Weight Decay implementiert L2-Regularisierung. Bei jedem Update wird eine Strafe auf die Gewichtsgrösse addiert:
$$L_{\text{total}} = L_{\text{MSE}} + \lambda \sum_j W_j^2$$
wobei $\lambda = \text{weight\_decay}$ der Regularisierungsparameter ist. Grosse Gewichte werden bestraft, was das Modell dazu zwingt, viele kleine statt wenige grosse Gewichte zu lernen. Das reduziert Overfitting.

**`scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(...)`** (nicht im Kurs-Notebook behandelt)

`ReduceLROnPlateau` reduziert die Lernrate automatisch, wenn ein überwachter Wert (hier der Validation Loss) aufhört, sich zu verbessern:
- `mode="min"`: die Lernrate wird reduziert, wenn der überwachte Wert nicht mehr sinkt
- `patience=LR_PATIENCE=15`: warte 15 Epochen ohne Verbesserung, bevor die Lernrate angepasst wird
- `factor=LR_FACTOR=0.5`: die Lernrate wird um den Faktor 0.5 reduziert (halbiert)

**Warum ein LR-Scheduler?**
Zu Beginn des Trainings sind grosse Lernraten sinnvoll (schnelles Konvergieren). Später, wenn sich das Modell bereits nahe am Optimum befindet, sind kleinere Lernraten nötig, um das Optimum genau zu treffen, ohne darüber hinaus zu schiessen.

**`best_val_loss = float("inf")`**
Initialisiert den besten bekannten Validation Loss mit Unendlich. Jede erste Messung wird kleiner sein.

**`best_weights = None`**
Speicherplatz für die Gewichte des besten Modells (jenes mit dem niedrigsten Validation Loss).

**`no_improve = 0`**
Zähler für die Anzahl der Epochen ohne Verbesserung des Validation Loss.

---

## 9. `main()` - Trainings-Loop mit Batches, Scheduler und Early Stopping

```python
        for epoch in range(MAX_EPOCHS):

            model.train()
            epoch_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                y_pred = model(X_batch)
                loss   = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item() * len(X_batch)

            epoch_train_loss /= len(train_loader.dataset)
```

**Äussere Schleife: `for epoch in range(MAX_EPOCHS):`**
Iteriert über maximal 500 Epochen. Das Training kann durch Early Stopping vorher beendet werden.

**`model.train()`**
Wichtig beim MLP (im Gegensatz zum linearen Modell): BatchNorm verhält sich im Trainings- und Evaluationsmodus unterschiedlich:
- **Trainings-Modus:** BatchNorm verwendet die Statistiken des aktuellen Batches (Batch-Mittelwert/-Varianz).
- **Eval-Modus:** BatchNorm verwendet die während des Trainings akkumulierten, laufenden Mittelwerte (Running Mean/Var).

Auch Dropout ist nur im Trainings-Modus aktiv. Vergleiche `01_pytorch_workflow.ipynb`, Kommentar: `model_0.train() # train mode in PyTorch sets all parameters that require gradients to require gradients`.

**`epoch_train_loss = 0.0`**
Akkumulator für den Loss über alle Batches der Epoche.

**Innere Schleife: `for X_batch, y_batch in train_loader:`**
Iteriert über alle Batches des Training-DataLoaders. Bei 35'000 Trainingspunkten und `BATCH_SIZE=128` sind das ca. 274 Batches pro Epoche. Pro Batch werden die 5 Trainingsschritte (Forward Pass, Loss, Zero Grad, Backward, Step) ausgeführt. Die Struktur der 5 Schritte ist identisch mit `01_pytorch_workflow.ipynb`, Zellen 27 und 61.

**`X_batch, y_batch = X_batch.to(device), y_batch.to(device)`**
Der DataLoader gibt CPU-Tensoren zurück (da das `VerkehrsDataset` keine Device-Angabe macht). Diese werden hier auf das Berechnungsgerät verschoben. Vergleiche `00_pytorch_fundamentals.ipynb`, Abschnitt "Putting tensors on the GPU".

**`epoch_train_loss += loss.item() * len(X_batch)`**
Gewichtete Summation des Batch-Losses. Da der letzte Batch kleiner sein kann, wird mit `len(X_batch)` gewichtet.

**`epoch_train_loss /= len(train_loader.dataset)`**
Normiert den akkumulierten Loss durch die Gesamtanzahl der Trainingsbeispiele: ergibt den durchschnittlichen Loss pro Datenpunkt über die gesamte Epoche.

```python
            model.eval()
            with torch.inference_mode():
                epoch_val_loss = 0.0
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    epoch_val_loss += loss_fn(model(X_batch), y_batch).item() * len(X_batch)
                epoch_val_loss /= len(val_loader.dataset)
```

**Validierungs-Block:**
Wie beim Training, aber:
- `model.eval()`: Dropout deaktiviert, BatchNorm verwendet Running Mean/Var. Vergleiche `01_pytorch_workflow.ipynb`, Kommentar: `model_1.eval() # turns off dropout and batchnorm training behaviour`.
- `with torch.inference_mode()`: kein Berechnungsgraph, kein Speicher für Gradienten. Vergleiche `01_pytorch_workflow.ipynb`, Zellen 20 und 65.

```python
            scheduler.step(epoch_val_loss)
```

**`scheduler.step(epoch_val_loss)`**
Übergibt den aktuellen Validation Loss an den Scheduler. Dieser prüft, ob eine Verbesserung eingetreten ist. Falls nicht (seit 15 Epochen), halbiert er die Lernrate.

```python
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= ES_PATIENCE:
                    print(f"  Early Stop bei Epoche {epoch} ...")
                    break
```

**Early Stopping** (nicht im Kurs-Notebook behandelt):

**`if epoch_val_loss < best_val_loss:`**
Prüft, ob der aktuelle Validation Loss besser als der bisher beste ist.

**`best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}`**
Speichert eine Kopie der aktuellen Modellgewichte. `.cpu()` verschiebt auf die CPU (um GPU-Speicher zu sparen). `.clone()` erstellt eine tiefe Kopie (ohne Clone würden Änderungen an den Modellgewichten auch `best_weights` verändern). `model.state_dict()` gibt ein Wörterbuch aller Parameter zurück. Vergleiche `01_pytorch_workflow.ipynb`, Abschnitt 4.

**`no_improve += 1`**
Zählt die Anzahl der Epochen ohne Verbesserung.

**`if no_improve >= ES_PATIENCE: break`**
Bricht das Training ab, wenn 10 Epochen ohne Verbesserung vergangen sind.

**Warum Early Stopping?**
Ohne Early Stopping würde das Modell möglicherweise über das Optimum hinaus trainieren. Nach einer Weile beginnt der Validation Loss zu steigen (Overfitting: das Modell lernt die Trainingsdaten auswendig, kann aber nicht mehr generalisieren), während der Training Loss weiter sinkt. Early Stopping stoppt das Training genau dann, wenn der Validation Loss am niedrigsten ist.

```python
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
```

**Bestes Modell laden:**
Nach dem Training wird nicht das letzte Modell für die Evaluation verwendet, sondern das beste, das während des Trainings gefunden wurde. `best_weights` enthält die gespeicherten Gewichte als CPU-Tensoren, die hier wieder auf das Berechnungsgerät verschoben werden. Vergleiche `01_pytorch_workflow.ipynb`, Abschnitt 5 "Loading a PyTorch model": `loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))`.

---

## 10. `main()` - Evaluation auf dem Testset

```python
        model.eval()
        with torch.inference_mode():
            preds_scaled = model(
                torch.tensor(X_test_s, dtype=torch.float32).to(device)
            ).cpu().numpy()

        preds = y_scaler.inverse_transform(preds_scaled)
```

Das Testset wird direkt als Tensor übergeben (ohne DataLoader), da alle Testdaten auf einmal verarbeitet werden können. Das Ergebnis wird mit `y_scaler.inverse_transform` zurück in Fahrzeuge/h konvertiert. Für die Erklärung von `model.eval()`, `torch.inference_mode()`, `.cpu().numpy()` und `inverse_transform` vergleiche `Theorie/models/linear_regression.md`, Abschnitte 9 und 10.

---

## 11. `main()` - Speichern des Modells

```python
        torch.save(model.state_dict(), RESULTS_DIR / "model_weights" / f"mlp_{name}.pt")
```

Identisch mit dem linearen Modell. Es wird der `state_dict` des besten Modells (das zuvor mit `model.load_state_dict(best_weights)` geladen wurde) gespeichert. Vergleiche `01_pytorch_workflow.ipynb`, Abschnitt 4.

---

## 12. Bezug zu den Notebooks

| Konzept im Skript | Quelle |
|---|---|
| `class X(nn.Module)` Subclassing | `01_pytorch_workflow.ipynb`, Zellen 12 und 55 |
| `nn.Linear` | `01_pytorch_workflow.ipynb`, Zelle 55 |
| `super().__init__()` | `01_pytorch_workflow.ipynb`, Zellen 12 und 55 |
| `model.parameters()` | `01_pytorch_workflow.ipynb`, Zellen 15 und 25 |
| `def forward(self, x)` | `01_pytorch_workflow.ipynb`, Abschnitt "PyTorch model building essentials" |
| 5-Schritte Trainings-Loop | `01_pytorch_workflow.ipynb`, Zellen 27 und 61 |
| `model.train()` / `model.eval()` | `01_pytorch_workflow.ipynb`, Kommentar Zelle 61: "turns off dropout and batchnorm training behaviour" |
| `torch.inference_mode()` | `01_pytorch_workflow.ipynb`, Zellen 20 und 65 |
| `torch.save()` / `state_dict()` | `01_pytorch_workflow.ipynb`, Abschnitte 4 und 5 |
| `load_state_dict()` | `01_pytorch_workflow.ipynb`, Abschnitt 5 |
| `torch.manual_seed(42)` | `00_pytorch_fundamentals.ipynb`, Abschnitt "Reproducibility" |
| `device = "cuda" if ... else "cpu"` | `01_pytorch_workflow.ipynb`, Abschnitt 6 |
| `.to(device)` | `00_pytorch_fundamentals.ipynb`, Abschnitt "Putting tensors on the GPU" |
| `.cpu().numpy()` | `00_pytorch_fundamentals.ipynb`, Abschnitt "Moving tensors back to the CPU" |
| Chronologischer Split | `01_pytorch_workflow.ipynb`, Abschnitt "Splitting data" |

**Konzepte, die über die Notebooks hinausgehen:**
- `torch.utils.data.Dataset` und `DataLoader`: Batch-Training mit Mini-Batches
- `nn.BatchNorm1d`: Batch-Normalisierung
- `nn.ReLU()`: Aktivierungsfunktion (Kurs nur lineare Regression)
- `nn.Dropout()`: Regularisierung
- `nn.Sequential`: Container für Schichten-Stapel
- `torch.optim.Adam`: Kurs verwendet SGD
- `weight_decay` im Optimizer: L2-Regularisierung
- `torch.optim.lr_scheduler.ReduceLROnPlateau`: adaptiver LR-Scheduler
- Early Stopping mit Speicherung der besten Gewichte
- Dreifacher Train/Val/Test-Split
- `nn.MSELoss`: Kurs verwendet L1Loss

---

## 13. Häufige Stolpersteine

**1. `model.train()` vs `model.eval()` beim MLP ist entscheidend**
Beim linearen Modell macht das keinen Unterschied. Beim MLP ist der Unterschied wichtig: Dropout ist im Eval-Modus deaktiviert, BatchNorm verwendet im Eval-Modus die Running Statistics statt der Batch-Statistiken. Ein vergessenes `model.eval()` vor der Evaluation führt zu zufällig schwankenden Ergebnissen (Dropout aktiv) und falschen Normalisierungen.

**2. Warum wird das Modell nach dem Training neu geladen?**
`model.load_state_dict(best_weights)` lädt nicht das letzte Modell (nach `MAX_EPOCHS` oder Early Stop), sondern das beste (niedrigster Validation Loss während des Trainings). Das letzte Modell könnte bereits überfittet sein.

**3. `fit_transform` nur auf Trainingsdaten**
Gilt hier genauso wie beim linearen Modell. Das Validierungsset und das Testset werden nur mit `.transform()` normalisiert.

**4. `no_improve = 0` muss nach Verbesserung zurückgesetzt werden**
Wenn eine Verbesserung auftritt, wird `no_improve = 0` gesetzt. Andernfalls würde ein einmaliger schlechter Wert nicht mehr "aufgeholt" werden können.

**5. BatchNorm-Verhalten im Trainings-Loop**
Im Training wird die Batch-Statistik verwendet. Wenn der Batch sehr klein ist (z.B. letzter Batch einer Epoche), kann die BatchNorm-Schätzung ungenau sein. Deshalb gilt: kein BatchNorm bei `batch_size=1`.

**6. Kein `y_test_s` benötigt**
Die finalen Metriken werden auf dem unsk alierten `y_test` (original in Fahrzeugen/h) berechnet, nachdem `inverse_transform` angewendet wurde. Das skalierte `y_test_s` wird nicht benötigt.

**7. `best_weights` auf CPU zwischenspeichern**
Die Gewichte werden auf der CPU gespeichert (`v.cpu().clone()`), um GPU-Speicher zu sparen. Beim Laden werden sie wieder auf das `device` verschoben.

---

## Anmerkung

- Das Kommentar im Code `# TEST_RATIO = 0.20 (implizit: ...)` ist leicht irreführend: `split_info` nennt den Split `"chronological 70/15/15"`, obwohl tatsächlich 70/10/20 gemeint ist (Zelle 148: `"split_type": "chronological 70/15/15"`). Der tatsächliche Split ist 70/10/20.
- Der Zeitreihen-Plot C enthält eine elaborierte Logik zur Lückenerkennung (Block mit mindestens 336 zusammenhängenden Stunden suchen). Das ist korrekt und sinnvoll für Stationen mit Datenlücken, aber macht den Code deutlich komplizierter als nötig für einen ersten Leser.
- Die `weight_decay`-Regularisierung und Batch-Normalisierung erfüllen beide eine ähnliche Funktion (Overfitting reduzieren). Es ist nicht ungewöhnlich, beide gleichzeitig zu verwenden, aber es ist konzeptionell redundant.
