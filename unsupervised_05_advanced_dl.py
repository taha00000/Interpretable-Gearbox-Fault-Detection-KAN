"""
unsupervised_05_advanced_dl.py
------------------------------
Implements additional state-of-the-art Semi-Supervised Anomaly Detection (SSAD)
methods trained exclusively on 'good' (healthy) data:
  1. Variational Autoencoder (VAE)
  2. Deep SVDD (Support Vector Data Description)
  3. Teacher-Student Knowledge Distillation
  4. Self-Supervised Learning (SSL) Denoising Autoencoder

Evaluates metrics and appends to dl_metrics.csv.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "unsupervised")
RESULTS_DIR = os.path.join(BASE_DIR, "results_unsupervised_pruned")
WINDOWS = [300, 400, 500, 600, 700, 800, 900, 1000]

def eval_model(y_true, scores, threshold):
    y_pred = (scores > threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = 0.5
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return auc, p, r, f1

# -------------------------------------------------------------
# 1. Variational Autoencoder (VAE)
# -------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, indim=44, zdim=8):
        super(VAE, self).__init__()
        self.enc = nn.Sequential(nn.Linear(indim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.fc_mu = nn.Linear(16, zdim)
        self.fc_var = nn.Linear(16, zdim)
        self.dec = nn.Sequential(nn.Linear(zdim, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, indim))
        
    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.dec(z), mu, logvar

# -------------------------------------------------------------
# 2. Deep SVDD
# -------------------------------------------------------------
class DeepSVDD(nn.Module):
    def __init__(self, indim=44, outdim=16):
        super(DeepSVDD, self).__init__()
        self.net = nn.Sequential(nn.Linear(indim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, outdim))
    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------
# 3. Teacher-Student
# -------------------------------------------------------------
class TSNetwork(nn.Module):
    def __init__(self, indim=44):
        super(TSNetwork, self).__init__()
        self.net = nn.Sequential(nn.Linear(indim, 32), nn.ReLU(), nn.Linear(32, 16))
    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------
# 4. SSL (Denoising Autoencoder)
# -------------------------------------------------------------
class DAE(nn.Module):
    def __init__(self, indim=44):
        super(DAE, self).__init__()
        self.enc = nn.Sequential(nn.Linear(indim, 32), nn.ReLU(), nn.Linear(32, 8))
        self.dec = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, indim))
    def forward(self, x):
        return self.dec(self.enc(x))

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("======================================================")
    print(f" Advanced SSAD Deep Learning Models | Backend: {device}")
    print("======================================================")
    
    all_results = []
    
    for W in WINDOWS:
        train_path = os.path.join(DATA_DIR, f"train_W{W}.csv")
        if not os.path.exists(train_path): continue
        print(f"\n--- Training Horizon Architectures for W={W} ---")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(os.path.join(DATA_DIR, f"val_W{W}.csv"))
        test_df = pd.read_csv(os.path.join(DATA_DIR, f"test_W{W}.csv"))
        
        fcols = ["S1_rms", "S1_std", "S1_var", "S4_mean", "S2_shape", "S2_skew", "S1_shape", "S1_p2p", "S4_shape", "S3_mean", "S2_std", "S2_rms", "S3_shape"]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_df[fcols].values)
        X_va, X_te = scaler.transform(val_df[fcols].values), scaler.transform(test_df[fcols].values)
        y_va, y_te = val_df['label'].values, test_df['label'].values
        
        Xt_tr = torch.FloatTensor(X_tr).to(device)
        Xt_va, Xt_te = torch.FloatTensor(X_va).to(device), torch.FloatTensor(X_te).to(device)
        loader = DataLoader(TensorDataset(Xt_tr, Xt_tr), batch_size=64, shuffle=True)
        
        # 1. VAE
        print("  Training VAE...")
        vae = VAE(indim=len(fcols)).to(device)
        opt_v = optim.Adam(vae.parameters(), lr=1e-3)
        for _ in range(30):
            for bx, _ in loader:
                opt_v.zero_grad()
                x_rec, mu, logvar = vae(bx)
                recon_loss = nn.MSELoss()(x_rec, bx)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                (recon_loss + 0.001*kld_loss).backward()
                opt_v.step()
        vae.eval()
        with torch.no_grad():
            rec_va, _, _ = vae(Xt_va)
            rec_te, _, _ = vae(Xt_te)
            s_va_vae = torch.mean((rec_va - Xt_va)**2, dim=1).cpu().numpy()
            s_te_vae = torch.mean((rec_te - Xt_te)**2, dim=1).cpu().numpy()
            
        # 2. Deep SVDD
        print("  Training Deep SVDD...")
        svdd = DeepSVDD(indim=len(fcols)).to(device)
        opt_s = optim.Adam(svdd.parameters(), lr=1e-3, weight_decay=1e-5)
        with torch.no_grad(): c = torch.mean(svdd(Xt_tr), dim=0) # Initialize center
        for _ in range(30):
            for bx, _ in loader:
                opt_s.zero_grad()
                loss = torch.mean(torch.sum((svdd(bx) - c)**2, dim=1))
                loss.backward()
                opt_s.step()
        svdd.eval()
        with torch.no_grad():
            s_va_svdd = torch.sum((svdd(Xt_va) - c)**2, dim=1).cpu().numpy()
            s_te_svdd = torch.sum((svdd(Xt_te) - c)**2, dim=1).cpu().numpy()
            
        # 3. Teacher-Student
        print("  Training Teacher-Student Network...")
        teacher = TSNetwork(indim=len(fcols)).to(device)
        for p in teacher.parameters(): p.requires_grad = False # Freeze teacher
        student = TSNetwork(indim=len(fcols)).to(device)
        opt_ts = optim.Adam(student.parameters(), lr=1e-3)
        for _ in range(30):
            for bx, _ in loader:
                opt_ts.zero_grad()
                nn.MSELoss()(student(bx), teacher(bx)).backward()
                opt_ts.step()
        student.eval()
        with torch.no_grad():
            s_va_ts = torch.mean((student(Xt_va) - teacher(Xt_va))**2, dim=1).cpu().numpy()
            s_te_ts = torch.mean((student(Xt_te) - teacher(Xt_te))**2, dim=1).cpu().numpy()
            
        # 4. SSL Denoising AE
        print("  Training SSL Denoising Autoencoder...")
        dae = DAE(indim=len(fcols)).to(device)
        opt_d = optim.Adam(dae.parameters(), lr=1e-3)
        for _ in range(30):
            for bx, _ in loader:
                opt_d.zero_grad()
                noisy = bx + 0.5 * torch.randn_like(bx)
                nn.MSELoss()(dae(noisy), bx).backward()
                opt_d.step()
        dae.eval()
        with torch.no_grad():
            s_va_dae = torch.mean((dae(Xt_va) - Xt_va)**2, dim=1).cpu().numpy()
            s_te_dae = torch.mean((dae(Xt_te) - Xt_te)**2, dim=1).cpu().numpy()
            
        # Collect & Evaluate
        scores_dict = {"VAE": s_va_vae, "DeepSVDD": s_va_svdd, "TeacherStudent": s_va_ts, "SSL_DAE": s_va_dae}
        test_scores_dict = {"VAE": s_te_vae, "DeepSVDD": s_te_svdd, "TeacherStudent": s_te_ts, "SSL_DAE": s_te_dae}
        
        preds_df = pd.read_csv(os.path.join(RESULTS_DIR, f"all_test_preds_W{W}.csv"))
        
        for name in scores_dict:
            val_sc = scores_dict[name]
            te_sc = test_scores_dict[name]
            best_f1, best_th = 0, 0
            for th in np.linspace(val_sc.min(), val_sc.max(), 100):
                yp = (val_sc > th).astype(int)
                f = precision_recall_fscore_support(y_va, yp, average='binary', zero_division=0)[2]
                if f > best_f1: 
                    best_f1, best_th = f, th
            
            auc, p, r, f1 = eval_model(y_te, te_sc, best_th)
            print(f"  {name:15s} | Test AUC: {auc:.4f} | F1: {f1:.4f}")
            all_results.append({"W": W, "Model": name, "AUC": auc, "F1": f1, "Precision": p, "Recall": r, "Threshold": best_th})
            
            preds_df[f"{name}_score"] = te_sc
            
        preds_df.to_csv(os.path.join(RESULTS_DIR, f"all_test_preds_W{W}.csv"), index=False)
        
    res_df = pd.DataFrame(all_results)
    
    old_dl_path = os.path.join(RESULTS_DIR, "dl_metrics.csv")
    if os.path.exists(old_dl_path):
        old_dl = pd.read_csv(old_dl_path)
        # Drop old VAE/DeepSVDD etc if rerunning
        old_dl = old_dl[~old_dl['Model'].isin(res_df['Model'].unique())]
        res_df = pd.concat([old_dl, res_df], ignore_index=True)
        
    res_df.to_csv(old_dl_path, index=False)
    print("\n[OK] Advanced DL Models complete. Metrics appended to dl_metrics.csv.")

if __name__ == "__main__":
    main()
