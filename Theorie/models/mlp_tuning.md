# Erklärung: `models/mlp_tuning.py`

**Originaldatei:** `models/mlp_tuning.py`

**Zusammenfassung:** Dieses Skript sucht mit Hilfe der Bibliothek Optuna automatisch nach den besten Hyperparametern für das MLP-Modell. Es läuft 150 Versuche (Trials) durch, von denen jeder eine andere Kombination von Hyperparametern (Anzahl Schichten, Schichtgrössen, Dropout, Lernrate, Weight Decay, Batch-Grösse) testet. Am Ende werden die besten Hyperparameter als JSON-Datei gespeichert und auf der Konsole ausgegeben, damit sie manuell in `mlp.py` eingetragen werden können. Optuna und alle Konzepte in diesem Skript gehen vollständig über die Kurs-Notebooks hinaus.

---

## 1. Imports

```python
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
```

Bekannte Imports (`json`, `time`, `pathlib.Path`, `numpy`, `pandas`, `torch`, `torch.nn`, `sklearn`, `torch.utils.data`) wurden bereits in `Theorie/models/linear_regression.md` und `Theorie/models/mlp.md` erklärt.

**`import optuna`**
Optuna ist eine Python-Bibliothek für automatische Hyperparameter-Optimierung (Automated Machine Learning, AutoML). Sie implementiert verschiedene Such-Algorithmen (z.B. TPE, CMA-ES) und Pruning-Strategien (frühzeitiges Abbrechen von schlechten Versuchen). Optuna ist nicht im Kurs-Notebook behandelt.

---

## 2. Konfiguration

```python
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR    = Path("data/v5_engineered")
RESULTS_DIR = Path("results/model_results/mlp")

TUNING_DATASET = "050_Brunnen_Mositunnel_R1_engineered.csv"
N_TRIALS       = 150
TUNING_EPOCHS  = 150
ES_PATIENCE    = 25

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
```

**`TUNING_DATASET = "050_Brunnen_Mositunnel_R1_engineered.csv"`**
Das Tuning wird nur auf einem einzigen Datensatz durchgeführt (Station 050, Richtung R1). Das spart Rechenzeit: für 150 Versuche wäre es viel zu aufwändig, alle 10 Stationen zu optimieren. Die Annahme ist, dass die optimalen Hyperparameter über alle Stationen hinweg ähnlich sind.

**`N_TRIALS = 150`**
Anzahl der Optuna-Versuche (Trials). Jeder Versuch trainiert ein Modell mit einer anderen Hyperparameter-Kombination und gibt den Validation Loss zurück. Optuna nutzt die Ergebnisse der vorherigen Versuche, um intelligentere Kombinationen für die nächsten zu wählen.

**`TUNING_EPOCHS = 150`**
Gegenüber dem finalen Training (`MAX_EPOCHS=500`) wird jeder Trial nur für 150 Epochen trainiert. Das macht jeden Trial schneller. Wird durch Early Stopping oft noch früher beendet.

**`ES_PATIENCE = 25`**
Early-Stopping-Geduld beim Tuning. Grosszügiger als beim finalen Training (25 statt 10), damit auch langsam konvergierende Konfigurationen eine faire Chance bekommen.

---

## 3. `VerkehrsDataset` und `MLP` Klassen

```python
class VerkehrsDataset(Dataset):
    ...

class MLP(nn.Module):
    ...
```

Diese beiden Klassen sind identisch mit denen in `mlp.py`. Sie wurden im Tuning-Skript dupliziert, damit das Skript eigenständig ohne Import aus `mlp.py` funktioniert. Für die vollständige Erklärung beider Klassen vergleiche `Theorie/models/mlp.md`, Abschnitte 4 und 5.

---

## 4. `objective()` - Die Zielfunktion

```python
def objective(trial, X_train_s, y_train_s, X_val_s, y_val_s, input_dim):
    n_layers     = trial.suggest_int("n_layers", 2, 4)
    hidden_dims  = [trial.suggest_categorical(f"n_units_l{i}", [32, 64, 128, 256, 512])
                    for i in range(n_layers)]
    dropout      = trial.suggest_float("dropout", 0.0, 0.3)
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [128, 256, 512])
```

Die `objective()`-Funktion ist das Herzstück des Optuna-Skripts. Sie wird für jeden Trial einmal aufgerufen und gibt am Ende einen einzigen Wert zurück: den Validation Loss des trainierten Modells. Optuna versucht, diesen Wert zu minimieren.

**`trial`**
Ein Optuna-`Trial`-Objekt. Es stellt Methoden bereit, mit denen die Funktion Hyperparameter-Werte für diesen spezifischen Versuch vorschlägt (`suggest_int`, `suggest_float`, `suggest_categorical`). Optuna protokolliert, welche Werte in welchem Trial verwendet wurden.

**`trial.suggest_int("n_layers", 2, 4)`**
Schlägt einen ganzzahligen Wert zwischen 2 und 4 (inklusive) vor. Der Name `"n_layers"` wird als Schlüssel protokolliert.

**`trial.suggest_categorical(f"n_units_l{i}", [32, 64, 128, 256, 512])`**
Wählt einen Wert aus einer vorgegebenen Liste aus. Für jede der `n_layers` Schichten wird eine Grösse aus `[32, 64, 128, 256, 512]` gewählt. Der Name `"n_units_l0"`, `"n_units_l1"`, etc. wird protokolliert.

**`trial.suggest_float("dropout", 0.0, 0.3)`**
Schlägt einen kontinuierlichen Wert zwischen 0.0 und 0.3 vor.

**`trial.suggest_float("lr", 1e-4, 1e-2, log=True)`**
Schlägt eine Lernrate vor. `log=True` bedeutet, dass die Werte logarithmisch gleichmässig verteilt werden: $10^{-4}$, $10^{-3.5}$, $10^{-3}$, $10^{-2.5}$, $10^{-2}$. Das ist sinnvoll, weil Lernraten typischerweise in Grössenordnungen gedacht werden, nicht linear.

**`trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)`**
Schlägt einen Weight-Decay-Wert vor, ebenfalls logarithmisch.

**`trial.suggest_categorical("batch_size", [128, 256, 512])`**
Wählt eine Batch-Grösse aus drei Optionen.

**Konzeptuelle Erklärung: Was ist der Suchraum?**

Der gesamte Hyperparameter-Raum, den Optuna durchsucht, ist:
- Anzahl Schichten: 2, 3 oder 4
- Grösse jeder Schicht: 32, 64, 128, 256 oder 512
- Dropout: jeder Wert zwischen 0 und 30%
- Lernrate: jeder Wert zwischen $10^{-4}$ und $10^{-2}$
- Weight Decay: jeder Wert zwischen $10^{-5}$ und $10^{-2}$
- Batch-Grösse: 128, 256 oder 512

Die Anzahl möglicher diskreter Kombinationen ist sehr gross (bei 3 Schichten z.B. $5^3 \times 3 = 375$ allein für Schichtgrössen und Batch-Grösse). Optuna sucht diesen Raum intelligent, nicht brute-force.

```python
    train_loader = DataLoader(
        VerkehrsDataset(X_train_s, y_train_s), batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        VerkehrsDataset(X_val_s, y_val_s), batch_size=batch_size, shuffle=False
    )

    model     = MLP(input_dim, hidden_dims, dropout).to(device)
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```

DataLoader, Modell, Loss und Optimizer werden für jeden Trial neu erstellt. Das ist wichtig: jeder Trial startet mit einem frisch initialisierten Modell.

```python
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(TUNING_EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss   = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            epoch_val_loss = 0.0
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                epoch_val_loss += loss_fn(model(X_batch), y_batch).item() * len(X_batch)
            epoch_val_loss /= len(val_loader.dataset)
```

Dieser Trainings-Loop ist vereinfacht: ohne LR-Scheduler, aber mit Early Stopping. Für die Erklärung der einzelnen Schritte (Forward Pass, Loss, Zero Grad, Backward, Step, `model.train()`, `model.eval()`, `torch.inference_mode()`) vergleiche `Theorie/models/mlp.md`, Abschnitt 9, und `01_pytorch_workflow.ipynb`, Zellen 27 und 61.

```python
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
```

**`trial.report(epoch_val_loss, epoch)`**
Meldet den aktuellen Validation Loss an Optuna zurück. Optuna kann damit entscheiden, ob dieser Trial vorzeitig abgebrochen werden soll (Pruning).

**`if trial.should_prune(): raise optuna.exceptions.TrialPruned()`**
**Pruning:** Wenn Optuna erkennt, dass dieser Trial auf einem schlechten Pfad ist (der Loss ist schlechter als der Median der abgeschlossenen Trials bei derselben Epoche), wird der Trial durch eine Exception abgebrochen. Das spart Rechenzeit, weil aussichtslose Konfigurationen früh beendet werden.

```python
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                break

    return best_val_loss
```

**Early Stopping im Tuning:** Analoges Muster wie in `mlp.py`, aber ohne Speicherung der besten Gewichte (beim Tuning werden keine Gewichte gespeichert). Am Ende wird der beste Validation Loss zurückgegeben. Optuna protokolliert diesen Wert als Ergebnis des Trials.

---

## 5. `main()` - Studie erstellen und optimieren

```python
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name="mlp_tuning",
        storage="sqlite:///optuna_mlp.db",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=None),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20),
    )
```

**`optuna.logging.set_verbosity(optuna.logging.WARNING)`**
Reduziert die Ausgaben von Optuna auf Warnungen. Ohne diese Zeile würde Optuna für jeden Trial eine Zeile ausgeben, was bei 150 Trials sehr viel wäre.

**`optuna.create_study(...)`**
Erstellt eine Optuna-Studie. Eine Studie ist eine Sammlung von zusammengehörenden Trials.

**`study_name="mlp_tuning"`**
Name der Studie (für die Protokollierung).

**`storage="sqlite:///optuna_mlp.db"`**
Speichert die Studie in einer SQLite-Datenbank. Das ermöglicht das Fortsetzen des Tunings nach einem Absturz oder einem Unterbruch. Wenn das Skript erneut ausgeführt wird, werden bereits abgeschlossene Trials nicht wiederholt.

**`load_if_exists=True`**
Falls die Datenbank bereits eine Studie mit demselben Namen enthält, wird diese geladen und fortgesetzt.

**`direction="minimize"`**
Das Ziel ist, den Validation Loss zu minimieren (kleinere Werte sind besser).

**`sampler=optuna.samplers.TPESampler(seed=None)`**
Der **TPE-Sampler** (Tree-structured Parzen Estimator) ist der Standard-Algorithmus von Optuna. Er modelliert die Verteilung der Hyperparameter basierend auf den bisherigen Ergebnissen und schlägt für den nächsten Trial Werte vor, die in Bereichen mit niedrigen Loss-Werten liegen.

**Wie funktioniert TPE konzeptuell?**
TPE trennt die bisherigen Trials in zwei Gruppen: die mit niedrigen Loss-Werten (gute Trials) und die mit hohen (schlechte Trials). Für den nächsten Trial werden Hyperparameter-Werte bevorzugt, die bei den guten Trials häufig vorkamen. Das ist intelligenter als zufälliges Suchen (Random Search), aber weniger komplex als Gradient-basierte Methoden.

**`pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)`**
Der **Median-Pruner** bricht einen Trial ab, wenn sein Loss in einer bestimmten Epoche schlechter ist als der Median aller abgeschlossenen Trials bei derselben Epoche.
- `n_startup_trials=10`: Die ersten 10 Trials werden nicht gepruned (zu wenig Referenzpunkte für den Median).
- `n_warmup_steps=20`: In den ersten 20 Epochen eines Trials wird noch nicht gepruned (das Modell braucht Zeit zum Einlaufen).

```python
    def wrapped_objective(trial):
        return objective(trial, X_train_s, y_train_s, X_val_s, y_val_s, input_dim)

    study.optimize(wrapped_objective, n_trials=N_TRIALS, show_progress_bar=True)
```

**`wrapped_objective`**
Eine innere Funktion, die `objective` aufruft und die vorbereiteten Daten einschliesst. Optuna erwartet eine Funktion, die nur `trial` als Argument nimmt. Das Wrapper-Muster ermöglicht es, zusätzliche Argumente über den Closure-Mechanismus von Python weiterzugeben.

**`study.optimize(wrapped_objective, n_trials=N_TRIALS, show_progress_bar=True)`**
Startet die Optimierung: `wrapped_objective` wird 150 Mal aufgerufen. `show_progress_bar=True` zeigt einen Fortschrittsbalken in der Konsole.

---

## 6. `main()` - Ergebnisse auswerten

```python
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"Abgeschlossene Trials: {len(completed)} | Geprunte Trials: {len(pruned)}")

    best = study.best_trial
    print(f"Bester Trial: #{best.number}")
    print(f"Bester Validation Loss: {best.value:.6f}")
```

**`study.trials`**
Liste aller Trials der Studie (abgeschlossen, gepruned, fehlgeschlagen).

**`t.state == optuna.trial.TrialState.COMPLETE`**
Filtert nach abgeschlossenen Trials (d.h. Trials, die bis zum Ende liefen und einen Wert zurückgaben).

**`study.best_trial`**
Das `Trial`-Objekt mit dem niedrigsten `value` (Validation Loss) unter allen abgeschlossenen Trials.

```python
    hidden_dims   = [best.params[f"n_units_l{i}"] for i in range(best.params["n_layers"])]
    dropout       = best.params["dropout"]
    learning_rate = best.params["lr"]
    weight_decay  = best.params["weight_decay"]
    batch_size    = best.params["batch_size"]
```

**`best.params`**
Wörterbuch der Hyperparameter des besten Trials. Die Keys entsprechen den Namen, die bei `trial.suggest_*()` vergeben wurden.

**`hidden_dims = [best.params[f"n_units_l{i}"] for i in range(best.params["n_layers"])]`**
Rekonstruiert die Liste der Schichtgrössen aus den einzeln gespeicherten Werten. Wenn der beste Trial `n_layers=4` und `n_units_l0=256`, `n_units_l1=128`, `n_units_l2=128`, `n_units_l3=64` hatte, ergibt das `hidden_dims = [256, 128, 128, 64]`.

```python
    best_params = {
        "HIDDEN_DIMS":   hidden_dims,
        "DROPOUT":       dropout,
        "LEARNING_RATE": learning_rate,
        "WEIGHT_DECAY":  weight_decay,
        "BATCH_SIZE":    batch_size,
        "best_val_loss": best.value,
        "trial_number":  best.number,
        ...
    }
    best_params_path = RESULTS_DIR / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)
```

Die besten Hyperparameter werden als JSON gespeichert. Dann müssen sie manuell in `mlp.py` eingetragen werden (das Skript tut das nicht automatisch). Dieser manuelle Schritt ist auf der Konsole klar kommuniziert:

```
  HIDDEN_DIMS   = [256, 128, 128, 64]
  DROPOUT       = 0.26
  LEARNING_RATE = 0.0003
  WEIGHT_DECAY  = 0.0017
  BATCH_SIZE    = 128
```

---

## 7. Bezug zu den Notebooks

Das gesamte Skript geht vollständig über den Inhalt der Kurs-Notebooks hinaus. Folgende Konzepte, die aus den Notebooks stammen, werden im Tuning-Skript verwendet:

| Konzept im Skript | Quelle |
|---|---|
| `class X(nn.Module)`, `nn.Linear`, `forward()` | `01_pytorch_workflow.ipynb`, Zellen 12, 55 |
| 5-Schritte Trainings-Loop | `01_pytorch_workflow.ipynb`, Zellen 27, 61 |
| `model.train()` / `model.eval()` | `01_pytorch_workflow.ipynb`, Zellen 27, 61 |
| `torch.inference_mode()` | `01_pytorch_workflow.ipynb`, Zellen 20, 65 |
| `torch.manual_seed(42)` | `00_pytorch_fundamentals.ipynb`, Abschnitt "Reproducibility" |
| `.to(device)` | `00_pytorch_fundamentals.ipynb`, Abschnitt "Putting tensors on the GPU" |

**Konzepte, die vollständig über die Notebooks hinausgehen:**
- Optuna: Hyperparameter-Optimierung, TPE-Sampler, Pruning, `study.optimize()`
- `trial.suggest_int()`, `trial.suggest_float()`, `trial.suggest_categorical()`
- `trial.report()` und `trial.should_prune()`
- `optuna.pruners.MedianPruner`: frühzeitiges Abbrechen von schlechten Trials
- `optuna.samplers.TPESampler`: intelligente Suchstrategie
- SQLite-Persistenz der Tuning-Datenbank
- `study.best_trial.params`: Zugriff auf beste Hyperparameter
- Alle Konzepte aus `mlp.md` (BatchNorm, Dropout, DataLoader, etc.)

---

## 8. Häufige Stolpersteine

**1. Kein fester Seed für den TPE-Sampler**
`TPESampler(seed=None)` bedeutet, dass das Tuning nicht reproduzierbar ist. Jedes Mal, wenn das Skript von Null startet, wählt der Sampler andere Startpunkte. Mit `seed=42` wäre die Reihenfolge der Trials reproduzierbar. Da `load_if_exists=True` gesetzt ist, werden abgeschlossene Trials aber wiederverwendet.

**2. Das Tuning läuft nur auf einer Station**
Die gefundenen Hyperparameter sind optimal für Station 050 R1. Für andere Stationen könnten andere Werte besser sein. Die Annahme, dass die Hyperparameter übertragbar sind, ist eine pragmatische Vereinfachung.

**3. Die Hyperparameter müssen manuell in `mlp.py` eingetragen werden**
Das Tuning-Skript schreibt die Ergebnisse nicht automatisch in `mlp.py`. Der Kommentar im Code (`"Beste Hyperparameter – manuell in mlp.py eintragen"`) weist explizit darauf hin.

**4. Pruning kann gute Konfigurationen frühzeitig abbrechen**
Ein Trial kann gepruned werden, auch wenn er langfristig gut wäre (z.B. weil er langsam konvergiert). Das `n_warmup_steps=20` gibt Trials etwas Zeit, aber es bleibt eine Heuristik.

**5. Validation Loss ≠ Testset-Performance**
Das Tuning optimiert auf dem Validierungsset. Wenn man zu viele Trials durchführt, kann es zur Überanpassung an das Validierungsset kommen (das Modell "kennt" implizit das Validierungsset durch die vielen Trials). Deshalb wird am Ende auf dem Testset evaluiert, das während des Tunings nicht berührt wurde.

**6. `best_val_loss` wird zurückgegeben, nicht gespeichert**
Im Tuning-Skript werden keine Modellgewichte gespeichert. Die `objective()`-Funktion trainiert ein Modell, misst den Validation Loss und wirft das Modell danach weg. Nur die Hyperparameter werden gespeichert. Das finale Modell wird mit diesen Hyperparametern in `mlp.py` neu trainiert.

---

## Anmerkung

- Der gesamte Datenlade- und Vorbereitungscode in `main()` ist identisch mit dem aus `mlp.py`. Eine Refaktorierung (gemeinsame Hilfsfunktionen auslagern) würde den Code wartbarer machen, wurde aber bewusst nicht durchgeführt, um die Skripte eigenständig zu halten.
- `optuna.logging.set_verbosity(optuna.logging.WARNING)` am Anfang von `main()` sollte besser am Modulniveau stehen (ausserhalb von `main()`), damit es auch bei Import des Moduls wirksam ist. Das ist ein kosmetischer Punkt ohne funktionale Auswirkung.
- Die Klassen `VerkehrsDataset` und `MLP` sind Duplikate aus `mlp.py`. In einem produktiven Projekt würde man diese Klassen in ein gemeinsames Modul auslagern (z.B. `models/architecture.py`) und importieren.
