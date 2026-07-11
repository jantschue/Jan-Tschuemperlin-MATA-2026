# Theorie-Ordner: Zeilengenaue Erklärungen der Modellskripte

Dieser Ordner enthält didaktische Erklärungsdateien zu den Modellskripten des Projekts. Er ist für die Prüfungsvorbereitung gedacht und nicht Teil des produktiven Codes. Der Ordner ist in `.gitignore` eingetragen und wird nicht ins Repository commitet.

---

## Übersicht der erklärten Dateien

| Markdown-Datei | Originaldatei | Modell |
|---|---|---|
| `models/linear_regression.md` | `models/linear_regression.py` | Lineare Regression (Baseline) |
| `models/mlp.md` | `models/mlp.py` | MLP (Hauptmodell) |
| `models/mlp_tuning.md` | `models/mlp_tuning.py` | MLP Hyperparameter-Tuning mit Optuna |

---

## Empfohlene Lesereihenfolge

**Schritt 1: Lineare Regression (Einstieg)**
Beginne mit `models/linear_regression.md`. Das lineare Modell ist einfacher aufgebaut und führt alle grundlegenden Konzepte ein, die auch im MLP wieder auftauchen: `nn.Module`, `nn.Linear`, der 5-Schritte-Trainings-Loop, `StandardScaler`, chronologischer Split, `model.train()` / `model.eval()`, `torch.inference_mode()`, `torch.save()`.

**Schritt 2: MLP (Hauptmodell)**
Lies danach `models/mlp.md`. Das MLP baut direkt auf den Konzepten der linearen Regression auf und fügt hinzu: `VerkehrsDataset`, `DataLoader`, `nn.BatchNorm1d`, `nn.ReLU`, `nn.Dropout`, `nn.Sequential`, Weight Decay, `ReduceLROnPlateau`-Scheduler, Early Stopping und den 70/10/20-Split.

**Schritt 3: Optuna-Tuning (Hyperparameter-Suche)**
Zuletzt `models/mlp_tuning.md`. Dieses Skript setzt das Wissen aus `mlp.md` voraus und erklärt, wie die Hyperparameter (Schichtgrössen, Dropout, Lernrate, etc.) systematisch gesucht wurden. Es behandelt Optuna, den TPE-Sampler und Pruning.

---

## Welche Datei gehört zu welchem Modell?

**Lineares Regressionsmodell (Baseline):**
- `models/linear_regression.md` erklärt `models/linear_regression.py`
- Eine einzige lineare Schicht: $\hat{y} = Wx + b$
- Kein Validierungsset, kein Early Stopping
- Optimizer: Adam, Loss: MSE, Split: 80/20

**MLP (Multi-Layer Perceptron, Hauptmodell):**
- `models/mlp.md` erklärt `models/mlp.py`
- 4 versteckte Schichten mit BatchNorm, ReLU, Dropout
- Validierungsset, Early Stopping, LR-Scheduler
- Optimizer: Adam mit Weight Decay, Loss: MSE, Split: 70/10/20
- `models/mlp_tuning.md` erklärt `models/mlp_tuning.py`
- Automatische Hyperparameter-Suche mit Optuna (TPE-Sampler, Median-Pruner)

---

## Herkunft der Konzepte

### Konzepte, die direkt aus den Kurs-Notebooks stammen

Die folgenden Konzepte wurden in `00_pytorch_fundamentals.ipynb` und `01_pytorch_workflow.ipynb` behandelt und tauchen in den Modellskripten wieder auf:

| Konzept | Kurs-Notebook | Abschnitt |
|---|---|---|
| Tensoren, `dtype=torch.float32` | `00_pytorch_fundamentals.ipynb` | "Tensor datatypes" |
| `torch.manual_seed(42)` | `00_pytorch_fundamentals.ipynb` | "Reproducibility" |
| `device = "cuda" if ... else "cpu"` | `00_pytorch_fundamentals.ipynb` | "Running tensors on the GPU" |
| `.to(device)` | `00_pytorch_fundamentals.ipynb` | "Putting tensors on the GPU" |
| `.cpu().numpy()` | `00_pytorch_fundamentals.ipynb` | "Moving tensors back to the CPU" |
| `class X(nn.Module)` Subclassing | `01_pytorch_workflow.ipynb` | Zelle 12, 55 |
| `nn.Linear(in_features, out_features)` | `01_pytorch_workflow.ipynb` | Zelle 55 |
| `super().__init__()` | `01_pytorch_workflow.ipynb` | Zelle 12, 55 |
| `def forward(self, x)` | `01_pytorch_workflow.ipynb` | Abschnitt "PyTorch model building essentials" |
| `model.parameters()` | `01_pytorch_workflow.ipynb` | Zelle 15, 25 |
| 5-Schritte Trainings-Loop | `01_pytorch_workflow.ipynb` | Zellen 27, 61 |
| `model.train()` / `model.eval()` | `01_pytorch_workflow.ipynb` | Zellen 27, 61 |
| `torch.inference_mode()` | `01_pytorch_workflow.ipynb` | Zellen 20, 65 |
| `torch.save()` / `state_dict()` | `01_pytorch_workflow.ipynb` | Abschnitte 4, 5 |
| `load_state_dict()` | `01_pytorch_workflow.ipynb` | Abschnitt 5 |
| Chronologischer Train/Test-Split | `01_pytorch_workflow.ipynb` | Abschnitt "Splitting data" |

### Konzepte, die über die Kurs-Notebooks hinausgehen

Die folgenden Konzepte sind im Projekt verwendet, aber nicht in den Kurs-Notebooks behandelt. Sie werden in den Erklärungsdateien eigenständig erklärt:

| Konzept | Erklärt in |
|---|---|
| `nn.MSELoss()` (statt L1Loss) | `linear_regression.md`, Abschnitt 8 |
| `torch.optim.Adam` (statt SGD) | `linear_regression.md`, Abschnitt 8 |
| `StandardScaler` von scikit-learn | `linear_regression.md`, Abschnitt 7 |
| `LabelEncoder` von scikit-learn | `linear_regression.md`, Abschnitt 5 |
| Metriken: MAE, RMSE, R² | `linear_regression.md`, Abschnitt 10 |
| `torch.utils.data.Dataset` | `mlp.md`, Abschnitt 4 |
| `torch.utils.data.DataLoader` | `mlp.md`, Abschnitt 7 |
| `nn.BatchNorm1d` | `mlp.md`, Abschnitt 5 |
| `nn.ReLU()` | `mlp.md`, Abschnitt 5 |
| `nn.Dropout()` | `mlp.md`, Abschnitt 5 |
| `nn.Sequential` | `mlp.md`, Abschnitt 5 |
| `weight_decay` im Adam-Optimizer | `mlp.md`, Abschnitt 8 |
| `ReduceLROnPlateau` LR-Scheduler | `mlp.md`, Abschnitt 8 |
| Early Stopping | `mlp.md`, Abschnitt 9 |
| Dreifacher Train/Val/Test-Split | `mlp.md`, Abschnitt 6 |
| Optuna: `create_study`, `optimize` | `mlp_tuning.md`, Abschnitt 5 |
| TPE-Sampler | `mlp_tuning.md`, Abschnitt 5 |
| Median-Pruner | `mlp_tuning.md`, Abschnitt 5 |
| `trial.suggest_*()` Methoden | `mlp_tuning.md`, Abschnitt 4 |
| `trial.report()` / `should_prune()` | `mlp_tuning.md`, Abschnitt 4 |
