import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1
setup_code = """import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, pairwise_distances
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(".")
DATASET_DIR = os.path.join(BASE_DIR, "Gearbox Dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

WINDOWS = [300, 400, 500, 600, 700, 800, 900, 1000]
SEED = 42

sys.path.insert(0, os.path.join(BASE_DIR, "efficient_kan"))
from efficient_kan import KAN
"""
cells.append(nbf.v4.new_markdown_cell("## Setup & Imports"))
cells.append(nbf.v4.new_code_cell(setup_code))

# Cell 2
extract_code = """# 1) Import all the data from the 4 sensors and calculate the 11 statistical features for all of them.
import glob
from scipy.stats import skew, kurtosis

FEATURE_NAMES = ["mean", "rms", "std", "var", "skew", "kurtosis", "p2p", "crest", "shape", "margin", "impulse"]
CHANNELS = ["S1", "S2", "S3", "S4"]

def compute_11_features(signal):
    rms_val = np.sqrt(np.mean(signal**2))
    margin_denom = np.mean(np.sqrt(np.abs(signal)))**2
    return [
        np.mean(signal), rms_val, np.std(signal), np.var(signal), skew(signal), kurtosis(signal),
        np.ptp(signal), np.max(np.abs(signal)) / rms_val if rms_val > 0 else 0,
        rms_val / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0,
        np.max(np.abs(signal)) / margin_denom if margin_denom > 0 else 0,
        np.max(np.abs(signal)) / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0
    ]

def extract_features(W):
    all_data = []
    filepaths = glob.glob(os.path.join(DATASET_DIR, "**", "*.txt"), recursive=True)
    for filepath in filepaths:
        fname = os.path.basename(filepath).lower()
        if "healthy" in fname: label = 0
        elif "broken" in fname: label = 1
        else: continue
            
        load_val = 0
        for l in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            if f"{l}hz" in fname or f"{l}load" in fname: load_val = l
                
        try:
            df = pd.read_csv(filepath, sep='\\t', header=None)
            if df.shape[1] < 4: df = pd.read_csv(filepath, sep=',', header=None)
            series = df.iloc[:, :4].values
        except: continue
            
        for start in range(0, series.shape[0] - W + 1, W):
            row = []
            for c in range(4): row.extend(compute_11_features(series[start:start+W, c]))
            row.extend([load_val, label])
            all_data.append(row)
            
    cols = [f"{c}_{f}" for c in CHANNELS for f in FEATURE_NAMES] + ["load", "label"]
    pd.DataFrame(all_data, columns=cols).to_csv(os.path.join(PROCESSED_DIR, f"features_W{W}.csv"), index=False)

for W in WINDOWS:
    if not os.path.exists(os.path.join(PROCESSED_DIR, f"features_W{W}.csv")):
        extract_features(W)
"""
cells.append(nbf.v4.new_markdown_cell("## 1) Feature Extraction (44 Features)"))
cells.append(nbf.v4.new_code_cell(extract_code))

# Cell 3
sup44_code = """# 2) Sweep supervised models (including KAN) from W = 300 to 1000 using the 44 feature vectors
from IPython.display import display

def get_supervised_models(): return {"DT": DecisionTreeClassifier(random_state=SEED), "RF": RandomForestClassifier(n_estimators=100, random_state=SEED), "SVM": SVC(kernel="rbf", random_state=SEED), "NB": GaussianNB(), "KNN": KNeighborsClassifier(n_neighbors=5), "GBC": GradientBoostingClassifier(n_estimators=100, random_state=SEED), "LR": LogisticRegression(max_iter=1000, random_state=SEED)}

def train_kan_supervised(X_tr, y_tr, X_te, y_te):
    model = KAN(layers_hidden=[X_tr.shape[1], max(4, X_tr.shape[1]//2), 2], grid_size=5, spline_order=3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)), batch_size=512, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(30):
        for bx, by in loader:
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()
    model.eval()
    with torch.no_grad(): return accuracy_score(y_te, torch.argmax(model(torch.tensor(X_te, dtype=torch.float32)), dim=1).numpy()) * 100

results_44F = pd.DataFrame(index=list(get_supervised_models().keys()) + ["KAN"], columns=WINDOWS, dtype=float)

for W in WINDOWS:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"features_W{W}.csv"))
    X, y = df.drop(columns=["label", "load"]).values, df["label"].values
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fold_accs = {m: [] for m in results_44F.index}
    
    for tr, te in skf.split(X, y):
        sc = MinMaxScaler(); X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
        models = get_supervised_models()
        for mname, clf in models.items():
            clf.fit(X_tr, y[tr])
            fold_accs[mname].append(accuracy_score(y[te], clf.predict(X_te)) * 100)
        fold_accs["KAN"].append(train_kan_supervised(X_tr, y[tr], X_te, y[te]))
        
    for m in results_44F.index: results_44F.loc[m, W] = np.mean(fold_accs[m])

print("--- Supervised Models Accuracy (%) [44 Features] ---")
display(results_44F.round(2))
"""
cells.append(nbf.v4.new_markdown_cell("## 2) Supervised Models Sweep (W=300 to 1000) on 44 Features"))
cells.append(nbf.v4.new_code_cell(sup44_code))

# Cell 4
twoNN_code = """# 3) Calculate 2-NN at W = 1000
df_1000 = pd.read_csv(os.path.join(PROCESSED_DIR, f"features_W1000.csv"))
X_1000_h = StandardScaler().fit_transform(df_1000[df_1000['label'] == 0].drop(columns=['load', 'label']).values)

nn_twonn = NearestNeighbors(n_neighbors=3, metric='euclidean', n_jobs=-1).fit(X_1000_h)
dists, _ = nn_twonn.kneighbors(X_1000_h)

r1, r2 = dists[:, 1], dists[:, 2]
valid = r1 > 1e-10
mu = r2[valid] / r1[valid]
id_est = len(mu) / np.sum(np.log(mu))
print(f"TWO-NN Intrinsic Dimension for W=1000: {id_est:.2f}")
"""
cells.append(nbf.v4.new_markdown_cell("## 3) Two-NN Intrinsic Dimension Calculation at W=1000"))
cells.append(nbf.v4.new_code_cell(twoNN_code))

# Cell 5
kan13_code = """# 4) Run 100-epoch KAN on W = 1000 to find true feature importance
df = pd.read_csv(os.path.join(PROCESSED_DIR, f"features_W1000.csv"))
feat_cols = list(df.drop(columns=["label", "load"]).columns)
X_s = MinMaxScaler().fit_transform(df[feat_cols].values)
y = df["label"].values

n_f = X_s.shape[1]
model = KAN(layers_hidden=[n_f, n_f//2, 2], grid_size=5, spline_order=3)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(X_s, dtype=torch.float32), torch.tensor(y, dtype=torch.long)), batch_size=512, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

model.train()
for _ in range(100):
    for bx, by in loader:
        opt.zero_grad()
        crit(model(bx), by).backward()
        opt.step()

score = model.layers[0].spline_weight.detach().cpu().abs().mean(dim=(0, 2)).numpy()
imp_df = pd.DataFrame({"Feature": feat_cols, "Importance": score}).sort_values("Importance", ascending=False).reset_index(drop=True)

TOP_13_VECTORS = imp_df.head(13)["Feature"].tolist()
print("--- Top 13 Most Important Feature Vectors ---")
print(TOP_13_VECTORS)
"""
cells.append(nbf.v4.new_markdown_cell("## 4) 100-Epoch KAN for Feature Importance (Top 13 Vectors)"))
cells.append(nbf.v4.new_code_cell(kan13_code))

# Cell 6
unsup_code = """# 5) Run The 9 unsupervised anomaly detection algorithms on 44 vectors vs 13 vectors across W=300-1000
class AE(nn.Module):
    def __init__(self, d): super().__init__(); self.e = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,8), nn.ReLU()); self.d = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self, x): return self.d(self.e(x))
class VAE(nn.Module):
    def __init__(self, d): super().__init__(); self.e = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU()); self.mu = nn.Linear(16,8); self.var = nn.Linear(16,8); self.d = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self, x): h=self.e(x); m, lv=self.mu(h), self.var(h); return self.d(m + torch.randn_like(torch.exp(0.5*lv))*torch.exp(0.5*lv)), m, lv
class DeepSVDD(nn.Module):
    def __init__(self, d): super().__init__(); self.n = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,16))
    def forward(self, x): return self.n(x)
class TSNetwork(nn.Module):
    def __init__(self, d): super().__init__(); self.n = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16))
    def forward(self, x): return self.n(x)
class DAE(nn.Module):
    def __init__(self, d): super().__init__(); self.e=nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,8)); self.d=nn.Sequential(nn.Linear(8,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self, x): return self.d(self.e(x))

def k_center(f, frac=0.1):
    c = [np.random.randint(0, f.shape[0])]; dists = pairwise_distances(f, f[c], metric='euclidean').flatten()
    for _ in range(max(1, int(f.shape[0]*frac)) - 1): idx=np.argmax(dists); c.append(idx); dists=np.minimum(dists, pairwise_distances(f, f[[idx]]).flatten())
    return f[c]

def run_unsup(fts):
    res = pd.DataFrame(index=["IsolationForest", "OC-SVM", "LOF", "PatchCore", "Autoencoder", "VAE", "DeepSVDD", "TeacherStudent", "SSL_DAE"], columns=WINDOWS, dtype=float)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for W in WINDOWS:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, f"features_W{W}.csv"))
        tr, te = train_test_split(df[df["label"]==0], test_size=0.3, random_state=SEED)
        te = pd.concat([te, df[df["label"]==1]])
        X_tr, X_te = StandardScaler().fit(tr[fts].values).transform(tr[fts].values), StandardScaler().fit(tr[fts].values).transform(te[fts].values)
        y_te = te["label"].values
        
        res.loc["IsolationForest", W] = roc_auc_score(y_te, -IsolationForest(n_estimators=100, random_state=SEED).fit(X_tr).decision_function(X_te))
        res.loc["OC-SVM", W] = roc_auc_score(y_te, -OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_tr).decision_function(X_te))
        res.loc["LOF", W] = roc_auc_score(y_te, -LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X_tr).decision_function(X_te))
        res.loc["PatchCore", W] = roc_auc_score(y_te, pairwise_distances(X_te, k_center(X_tr), metric='euclidean').min(axis=1))
        
        Xt_tr, Xt_te = torch.tensor(X_tr, dtype=torch.float32).to(dev), torch.tensor(X_te, dtype=torch.float32).to(dev)
        ldr = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt_tr, Xt_tr), batch_size=256, shuffle=True)
        
        # AE
        ae = AE(len(fts)).to(dev); o = torch.optim.Adam(ae.parameters(), lr=1e-3)
        for _ in range(15):
            for b,_ in ldr: o.zero_grad(); nn.MSELoss()(ae(b),b).backward(); o.step()
        with torch.no_grad(): res.loc["Autoencoder",W] = roc_auc_score(y_te, torch.mean((ae(Xt_te)-Xt_te)**2, dim=1).cpu())
        
        # VAE
        vae = VAE(len(fts)).to(dev); o = torch.optim.Adam(vae.parameters(), lr=1e-3)
        for _ in range(15):
            for b,_ in ldr: o.zero_grad(); x,m,lv=vae(b); (nn.MSELoss()(x,b) - 0.0005*torch.sum(1+lv-m.pow(2)-lv.exp())).backward(); o.step()
        with torch.no_grad(): res.loc["VAE",W] = roc_auc_score(y_te, torch.mean((vae(Xt_te)[0]-Xt_te)**2, dim=1).cpu())
        
        # DeepSVDD
        svdd = DeepSVDD(len(fts)).to(dev); o = torch.optim.Adam(svdd.parameters(), lr=1e-3)
        with torch.no_grad(): c = torch.mean(svdd(Xt_tr), dim=0)
        for _ in range(15):
            for b,_ in ldr: o.zero_grad(); torch.mean(torch.sum((svdd(b)-c)**2,dim=1)).backward(); o.step()
        with torch.no_grad(): res.loc["DeepSVDD",W] = roc_auc_score(y_te, torch.sum((svdd(Xt_te)-c)**2, dim=1).cpu())
        
        # TS
        t, s = TSNetwork(len(fts)).to(dev), TSNetwork(len(fts)).to(dev); o = torch.optim.Adam(s.parameters(), lr=1e-3)
        for p in t.parameters(): p.requires_grad=False
        for _ in range(15):
            for b,_ in ldr: o.zero_grad(); nn.MSELoss()(s(b),t(b)).backward(); o.step()
        with torch.no_grad(): res.loc["TeacherStudent",W] = roc_auc_score(y_te, torch.mean((s(Xt_te)-t(Xt_te))**2, dim=1).cpu())
        
        # SSL DAE
        dae = DAE(len(fts)).to(dev); o = torch.optim.Adam(dae.parameters(), lr=1e-3)
        for _ in range(15):
            for b,_ in ldr: o.zero_grad(); nn.MSELoss()(dae(b + 0.5*torch.randn_like(b)),b).backward(); o.step()
        with torch.no_grad(): res.loc["SSL_DAE",W] = roc_auc_score(y_te, torch.mean((dae(Xt_te)-Xt_te)**2, dim=1).cpu())
        
    return res

print("--- Unsupervised AUC (%) [44 Features] ---")
res_44 = run_unsup(feat_cols)
display((res_44 * 100).round(2))

print("\\n--- Unsupervised AUC (%) [Top 13 KAN Features] ---")
res_13 = run_unsup(TOP_13_VECTORS)
display((res_13 * 100).round(2))
"""
cells.append(nbf.v4.new_markdown_cell("## 5) Unsupervised Anomaly Detection Sweep (44 vs 13 Features)"))
cells.append(nbf.v4.new_code_cell(unsup_code))

# Cell 7
sup13_code = """# 6) Sweep supervised models (including KAN) from W = 300 to 1000 using the 13 feature vectors
results_13F = pd.DataFrame(index=list(get_supervised_models().keys()) + ["KAN"], columns=WINDOWS, dtype=float)

for W in WINDOWS:
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"features_W{W}.csv"))
    X, y = df[TOP_13_VECTORS].values, df["label"].values
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    fold_accs = {m: [] for m in results_13F.index}
    
    for tr, te in skf.split(X, y):
        sc = MinMaxScaler(); X_tr, X_te = sc.fit_transform(X[tr]), sc.transform(X[te])
        models = get_supervised_models()
        for mname, clf in models.items():
            clf.fit(X_tr, y[tr])
            fold_accs[mname].append(accuracy_score(y[te], clf.predict(X_te)) * 100)
        fold_accs["KAN"].append(train_kan_supervised(X_tr, y[tr], X_te, y[te]))
        
    for m in results_13F.index: results_13F.loc[m, W] = np.mean(fold_accs[m])

print("--- Supervised Models Accuracy (%) [13 Features] ---")
display(results_13F.round(2))
"""
cells.append(nbf.v4.new_markdown_cell("## 6) Supervised Models Sweep (W=300 to 1000) on 13 Features"))
cells.append(nbf.v4.new_code_cell(sup13_code))

nb.cells = cells
with open("Gearbox_Fault_Detection_Consolidated.ipynb", "w") as f:
    nbf.write(nb, f)
