import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from efficient_kan import KAN
import copy

import warnings
warnings.filterwarnings('ignore')

class MLP(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        layers = []
        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i+1]))
            if i < len(architecture) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, architecture, epochs=20, lr=1e-3, patience=5):
    model = MLP(architecture)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_t)
        preds = torch.argmax(test_out, dim=1).numpy()
        
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    return acc, prec, rec, f1

def train_kan(X_train, y_train, X_val, y_val, X_test, y_test, architecture, grid=5, k=3, epochs=20, lr=1e-3, patience=5):
    model = KAN(layers_hidden=architecture, grid_size=grid, spline_order=k)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # train step
        model.train()
        for batch_X, batch_y in train_loader:
            def closure():
                optimizer.zero_grad()
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                return loss
            optimizer.step(closure)
        
        # eval step
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # PyKAN state dict contains splines coefs
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    with torch.no_grad():
        test_pred = model(torch.tensor(X_test, dtype=torch.float32))
        preds = torch.argmax(test_pred, dim=1).numpy()
        
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro')
    rec = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    return acc, prec, rec, f1

def evaluate_kan_mlp(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['label', 'load']).values
    y = df['label'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    kan_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    mlp_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    
    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]
        
        # Split train_full into train and val for early stopping (e.g. 15% of train = ~12% overall)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.15, stratify=y_train_full, random_state=42)
            
        scaler = MinMaxScaler(feature_range=(0,1))
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        arch = [X.shape[1], 20, 2]
        
        # Train MLP
        acc_m, prec_m, rec_m, f1_m = train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, arch)
        mlp_metrics['acc'].append(acc_m)
        mlp_metrics['prec'].append(prec_m)
        mlp_metrics['rec'].append(rec_m)
        mlp_metrics['f1'].append(f1_m)
        
        # Train KAN
        acc_k, prec_k, rec_k, f1_k = train_kan(X_train, y_train, X_val, y_val, X_test, y_test, arch)
        kan_metrics['acc'].append(acc_k)
        kan_metrics['prec'].append(prec_k)
        kan_metrics['rec'].append(rec_k)
        kan_metrics['f1'].append(f1_k)
        
        print(f"  Fold {fold}: KAN Acc={acc_k:.4f}, MLP Acc={acc_m:.4f}", flush=True)
        fold += 1
        
    def avg_metrics(m):
        return {k: np.mean(v) for k, v in m.items()}
        
    return avg_metrics(kan_metrics), avg_metrics(mlp_metrics)

def main():
    data_dir = r"c:/Users/tahah/OneDrive/Desktop/Moving-window-based-feature-extraction-method-for-vibration-based-condition-monitoring-main/data/processed"
    windows = [300, 400, 500, 600, 700, 800]
    out_dir = r"c:/Users/tahah/OneDrive/Desktop/Moving-window-based-feature-extraction-method-for-vibration-based-condition-monitoring-main/results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Tables for Acc, Prec, Rec
    acc_table = pd.DataFrame(index=['KAN', 'MLP'], columns=windows)
    prec_table = pd.DataFrame(index=['KAN', 'MLP'], columns=windows)
    rec_table = pd.DataFrame(index=['KAN', 'MLP'], columns=windows)
    f1_table = pd.DataFrame(index=['KAN', 'MLP'], columns=windows)
    
    for W in windows:
        filepath = os.path.join(data_dir, f"features_W{W}.csv")
        if not os.path.exists(filepath):
            continue
        print(f"\nEvaluating KAN and MLP for window size W={W}...", flush=True)
        kan_res, mlp_res = evaluate_kan_mlp(filepath)
        
        for name, res in [('KAN', kan_res), ('MLP', mlp_res)]:
            acc_table.loc[name, W] = res['acc'] * 100
            prec_table.loc[name, W] = res['prec'] * 100
            rec_table.loc[name, W] = res['rec'] * 100
            f1_table.loc[name, W] = res['f1'] * 100
            
    acc_table.to_csv(os.path.join(out_dir, "dl_accuracy.csv"))
    prec_table.to_csv(os.path.join(out_dir, "dl_precision.csv"))
    rec_table.to_csv(os.path.join(out_dir, "dl_recall.csv"))
            
    print("\n--- Accuracy (%) ---")
    print(acc_table.to_markdown())

if __name__ == "__main__":
    main()
