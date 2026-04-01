"""
unsupervised_03_deep_learning.py
--------------------------------
Implements lightweight Deep Learning approaches for anomaly detection:
  1. Dense Autoencoder (AE)
  2. PatchCore (Coreset Memory Bank on handcrafted features)

Evaluates Models on Validation and Test sets using:
  - AE: Reconstruction Error (MSE)
  - PatchCore: Nearest-Neighbor Distance to nominal memory bank
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, pairwise_distances

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "unsupervised")
RESULTS_DIR = os.path.join(BASE_DIR, "results_unsupervised")
WINDOWS = [300, 400, 500, 600, 700, 800, 900, 1000]

class Autoencoder(nn.Module):
    def __init__(self, input_dim=44):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

def k_center_greedy(features, fraction=0.1):
    """
    K-Center Greedy algorithm to build a coreset memory bank containing
    'fraction' amount of the most representative nominal features.
    """
    n_samples = features.shape[0]
    n_select = max(1, int(n_samples * fraction))
    
    # Initialize with a random point
    coreset_idx = [np.random.randint(0, n_samples)]
    min_distances = pairwise_distances(features, features[coreset_idx], metric='euclidean').flatten()
    
    for _ in range(1, n_select):
        # Find the point that is farthest from the current coreset
        idx = np.argmax(min_distances)
        coreset_idx.append(idx)
        # Update distances
        dist_new = pairwise_distances(features, features[[idx]], metric='euclidean').flatten()
        min_distances = np.minimum(min_distances, dist_new)
        
    return features[coreset_idx]

def eval_model(y_true, scores, threshold):
    y_pred = (scores > threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return auc, p, r, f1

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("======================================================")
    print(" Deep Learning & PatchCore - Gearbox Fault Detection")
    print(f" Using Backend: {device}")
    print("======================================================")
    
    all_results = []
    
    for W in WINDOWS:
        train_path = os.path.join(DATA_DIR, f"train_W{W}.csv")
        val_path = os.path.join(DATA_DIR, f"val_W{W}.csv")
        test_path = os.path.join(DATA_DIR, f"test_W{W}.csv")
        if not os.path.exists(train_path):
            continue
            
        print(f"\n--- Training Deep Learning & PatchCore for W={W} ---")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        feat_cols = [c for c in train_df.columns if c not in ['load', 'label']]
        X_train, y_train = train_df[feat_cols].values, train_df['label'].values
        X_val, y_val = val_df[feat_cols].values, val_df['label'].values
        X_test, y_test = test_df[feat_cols].values, test_df['label'].values
        
        # Scale features using training set stats
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        
        # -------------------------------------------------------------
        # 1. PatchCore (Feature-based Coreset)
        # -------------------------------------------------------------
        print("  [PatchCore] Building Coreset Memory Bank...")
        # Store 10% representative healthy behavior
        memory_bank = k_center_greedy(X_train_s, fraction=0.1) 
        
        val_dists = pairwise_distances(X_val_s, memory_bank, metric='euclidean')
        test_dists = pairwise_distances(X_test_s, memory_bank, metric='euclidean')
        
        # Anomaly score is distance to the nearest neighbor in the memory bank
        patchcore_val_scores = val_dists.min(axis=1)
        patchcore_test_scores = test_dists.min(axis=1)
        
        # Threshold validation
        best_f1_pc, best_th_pc = 0, 0
        for th in np.linspace(patchcore_val_scores.min(), patchcore_val_scores.max(), 100):
            yp = (patchcore_val_scores > th).astype(int)
            _, _, f, _ = precision_recall_fscore_support(y_val, yp, average='binary', zero_division=0)
            if f > best_f1_pc:
                best_f1_pc, best_th_pc = f, th
                
        auc_pc, p_pc, r_pc, f1_pc = eval_model(y_test, patchcore_test_scores, best_th_pc)
        print(f"  PatchCore       | Test AUC: {auc_pc:.4f} | F1: {f1_pc:.4f} | Prec: {p_pc:.4f} | Rec: {r_pc:.4f}")
        all_results.append({"W": W, "Model": "PatchCore", "AUC": auc_pc, "F1": f1_pc, "Precision": p_pc, "Recall": r_pc, "Threshold": best_th_pc})
        
        # -------------------------------------------------------------
        # 2. Dense Autoencoder
        # -------------------------------------------------------------
        print("  [Autoencoder] Training Neural Network...")
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
        
        dataset_train = TensorDataset(X_train_t, X_train_t)
        loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)
        
        model = Autoencoder(input_dim=len(feat_cols)).to(device)
        criterion = nn.MSELoss(reduction='none') # For per-sample reconstruction error
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        epochs = 100
        model.train()
        for epoch in range(epochs):
            for batch_x, _ in loader_train:
                optimizer.zero_grad()
                x_rec, _ = model(batch_x)
                loss = torch.mean(criterion(x_rec, batch_x))
                loss.backward()
                optimizer.step()
                
        # Inference
        model.eval()
        with torch.no_grad():
            val_rec, _ = model(X_val_t)
            # Anomaly score is mean squared error per sample over the 44 features
            ae_val_scores = torch.mean(criterion(val_rec, X_val_t), dim=1).cpu().numpy()
            
            test_rec, _ = model(X_test_t)
            ae_test_scores = torch.mean(criterion(test_rec, X_test_t), dim=1).cpu().numpy()
            
        best_f1_ae, best_th_ae = 0, 0
        for th in np.linspace(ae_val_scores.min(), ae_val_scores.max(), 100):
            yp = (ae_val_scores > th).astype(int)
            _, _, f, _ = precision_recall_fscore_support(y_val, yp, average='binary', zero_division=0)
            if f > best_f1_ae:
                best_f1_ae, best_th_ae = f, th
                
        auc_ae, p_ae, r_ae, f1_ae = eval_model(y_test, ae_test_scores, best_th_ae)
        print(f"  Autoencoder     | Test AUC: {auc_ae:.4f} | F1: {f1_ae:.4f} | Prec: {p_ae:.4f} | Rec: {r_ae:.4f}")
        all_results.append({"W": W, "Model": "Autoencoder", "AUC": auc_ae, "F1": f1_ae, "Precision": p_ae, "Recall": r_ae, "Threshold": best_th_ae})
        
        # -------------------------------------------------------------
        # 3. Save combined predictions
        # -------------------------------------------------------------
        test_df["PatchCore_score"] = patchcore_test_scores
        test_df["Autoencoder_score"] = ae_test_scores
        
        # Merge baseline preds if they exist
        baseline_preds_path = os.path.join(RESULTS_DIR, f"baseline_test_preds_W{W}.csv")
        if os.path.exists(baseline_preds_path):
            base_df = pd.read_csv(baseline_preds_path)
            for c in base_df.columns:
                if c.endswith("_score") and c not in test_df.columns:
                    test_df[c] = base_df[c]
                    
        test_df.to_csv(os.path.join(RESULTS_DIR, f"all_test_preds_W{W}.csv"), index=False)
        
    res_df = pd.DataFrame(all_results)
    metrics_path = os.path.join(RESULTS_DIR, "dl_metrics.csv")
    res_df.to_csv(metrics_path, index=False)
    print("\n[OK] Deep Learning and PatchCore training complete.")
    print(f"Results saved in {metrics_path}")

if __name__ == "__main__":
    main()
