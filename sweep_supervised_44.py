import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(".")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
WINDOWS = [300, 400, 500, 600, 700, 800, 900, 1000]
SEED = 42

sys.path.insert(0, os.path.join(BASE_DIR, "efficient_kan"))
from efficient_kan import KAN

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
    if not os.path.exists(os.path.join(PROCESSED_DIR, f"features_W{W}.csv")): continue
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
print(results_44F.round(2).to_markdown())
