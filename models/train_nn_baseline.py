"""
"Trainiert ein erstes Baseline-Neuronales-Netz mit PyTorch für jeden Standort.
Führt ein One-Hot-Encoding durch, skaliert alle Features und das Target und trennt die Daten
chronologisch (80% Train, 20% Test), um Data-Leakage zu vermeiden.
Gibt am Ende die Evaluierungsmetriken aus, speichert Loss-/Genauigkeitsgraphen 
für den Trainingsverlauf und exportiert die Vorhersagen für das Dashboard."
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Pfade relativ zum Skript definieren
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_dir = os.path.join(base_dir, "data", "engineered_features")
output_dir = os.path.dirname(os.path.abspath(__file__)) # Selber Ordner (Test_model)

# Einfaches Multi-Layer Perceptron in PyTorch
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1) # Output = volume
        )
        
    def forward(self, x):
        return self.layers(x)

def train_model(X_train, y_train, X_test, y_test_unscaled, y_scaler, input_dim, epochs=30, batch_size=256, lr=0.001):
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_maes = []
    val_r2s = []
    
    for e in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            
        avg_train_loss = epoch_loss / len(dataset)
        train_losses.append(avg_train_loss)
        
        model.eval()
        with torch.no_grad():
            X_test_tensor_dev = X_test_tensor.to(device)
            y_pred_scaled = model(X_test_tensor_dev).cpu().numpy()
            
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
        val_mae = mean_absolute_error(y_test_unscaled, y_pred)
        val_r2 = r2_score(y_test_unscaled, y_pred)
        
        val_maes.append(val_mae)
        val_r2s.append(val_r2)
            
    return model, device, train_losses, val_maes, val_r2s

def main():
    print("Starte Baseline Neural Network Training (PyTorch) mit Dashboard Export...\n")
    
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"Keine Datensätze in {input_dir} gefunden.")
        return
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0].replace('_engineered', '')
        
        print(f"--- Verarbeite Datensatz: {base_name} ---")
        df = pd.read_csv(file_path)
        
        df = df.dropna()
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(by='datetime').reset_index(drop=True)
        
        df_encoded = pd.get_dummies(df, columns=['weather_cat'], drop_first=True)
        
        X = df_encoded.drop(columns=['datetime', 'volume'])
        y = df_encoded['volume'].values
        
        split_index = int(len(X) * 0.8)
        
        X_train, X_test = X.iloc[:split_index].values, X.iloc[split_index:].values
        y_train, y_test = y[:split_index], y[split_index:]
        datetime_test = df['datetime'].iloc[split_index:].dt.strftime('%Y-%m-%d %H:%M').values
        
        X_scaler = StandardScaler()
        X_train_scaled = X_scaler.fit_transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)
        
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        print("Trainiere MLP (dies kann einen Moment dauern)...")
        input_dim = X_train_scaled.shape[1]
        model, device, train_losses, val_maes, val_r2s = train_model(
            X_train_scaled, y_train_scaled, X_test_scaled, y_test, y_scaler,
            input_dim=input_dim, epochs=30, batch_size=256
        )
        
        # Finale Predictions (für Export)
        model.eval()
        with torch.no_grad():
            X_test_tensor_dev = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
            y_pred_scaled = model(X_test_tensor_dev).cpu().numpy()
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
        
        print(f"Resultate für {base_name}:")
        print(f"  - Finaler MAE:  {val_maes[-1]:.2f} Fahrzeuge")
        print(f"  - Finaler R²:   {val_r2s[-1]:.4f}\n")
        
        # --- JSON Export für das Dashboard (letzte 2 Wochen = 336 Stunden) ---
        pred_dir = os.path.join(output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        
        last_n = min(336, len(y_test))
        export_df = pd.DataFrame({
            "datetime": datetime_test[-last_n:],
            "real": y_test[-last_n:],
            "predicted": np.round(y_pred[-last_n:], 1)
        })
        export_df.to_json(os.path.join(pred_dir, f"{base_name}_predictions.json"), orient="records")
        
        # Plotting (wie bisher)
        plt.figure(figsize=(14, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, 31), train_losses, label='Train MSE Loss (Scaled)', color='blue')
        plt.title(f'Trainings-Loss pro Epoche\n({base_name})')
        plt.xlabel('Epoche')
        plt.ylabel('Loss (MSE)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, 31), val_r2s, label='Test R² Score', color='green')
        plt.title(f'Bestimmtheitsmass (R²) auf Testdaten\n({base_name})')
        plt.xlabel('Epoche')
        plt.ylabel('R² Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{base_name}_training_curves.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
