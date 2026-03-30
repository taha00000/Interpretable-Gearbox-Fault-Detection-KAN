"""
04_kan_training.py
------------------
Trains and benchmarks two neural architectures against each other:
  - KAN  [N -> N//2 -> 2]  -- cubic B-spline edges (efficient_kan implementation)
  - MLP  [N -> N//2 -> 2]  -- ReLU activations, equivalent parameter budget

N is the number of input features, determined dynamically from the dataset
(currently 44: 11 features x 4 sensors, matching Hassan et al. 2026).

Both are evaluated under identical 10-fold stratified cross-validation with
MinMax-scaled features, using the same random seed and fold splits as the
baseline classifiers in 03_baseline_ml.py.

The trained KAN model from the BEST fold of W=600 is saved to model/ for
later use in 05_interpretability_and_pruning.py.

Outputs (saved to results/):
  dl_accuracy.csv              -- accuracy  (%) for KAN and MLP across all window sizes
  dl_precision.csv             -- precision (%)
  dl_recall.csv                -- recall    (%)
  dl_f1.csv                    -- F1        (%)
  dl_results_detailed.csv      -- per-fold metrics for KAN and MLP
  dl_summary.csv               -- mean +/- std summary for all metrics

model/ (saved artefacts):
  kan_best_W600.pt          -- state_dict of best-fold KAN on W=600
  kan_best_W600_scaler.npy  -- MinMaxScaler parameters for that fold
"""

import os
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficient_kan"))
from efficient_kan import KAN

warnings.filterwarnings("ignore")

# -- Paths ---------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR   = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "model")
WINDOWS   = [300, 400, 500, 600, 700, 800]
SAVE_MODEL_FOR_W = 600   # window size whose best-fold model is saved for XAI

# -- Hyper-parameters ----------------------------------------------------------
# Architecture is set dynamically based on input feature count (see evaluate_window)
GRID_SIZE    = 5
SPLINE_ORDER = 3
EPOCHS       = 50
LR           = 1e-3
BATCH_SIZE   = 512
PATIENCE     = 10
N_FOLDS      = 10    # matches Hassan et al. 2026 (10-fold CV)
SEED         = 42


# -- MLP definition ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, arch: list):
        super().__init__()
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i + 1]))
            if i < len(arch) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -- Generic training loop -----------------------------------------------------
def train_model(model, X_tr, y_tr, X_val, y_val, X_te, y_te,
                use_closure=False):
    """Train *any* nn.Module with early stopping; return test metrics."""
    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_te_t  = torch.tensor(X_te,  dtype=torch.float32)

    loader    = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr_t, y_tr_t),
        batch_size=BATCH_SIZE, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val   = float("inf")
    best_state = None
    patience   = 0

    for _ in range(EPOCHS):
        model.train()
        for bx, by in loader:
            if use_closure:
                def closure():
                    optimizer.zero_grad()
                    loss = criterion(model(bx), by)
                    loss.backward()
                    return loss
                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience   = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_te_t), dim=1).numpy()

    return (
        accuracy_score (y_te, preds),
        precision_score(y_te, preds, average="macro", zero_division=0),
        recall_score   (y_te, preds, average="macro", zero_division=0),
        f1_score       (y_te, preds, average="macro", zero_division=0),
        model
    )


# -- Per-window evaluation -----------------------------------------------------
def evaluate_window(filepath: str, W: int, save_kan_model: bool = False):
    df = pd.read_csv(filepath)
    X  = df.drop(columns=["label", "load"]).values
    y  = df["label"].values

    # Dynamic architecture: [N_features -> N_features//2 -> 2]
    n_features = X.shape[1]
    architecture = [n_features, n_features // 2, 2]
    print(f"  Architecture: {architecture}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    kan_buf = {k: [] for k in ("acc", "prec", "rec", "f1")}
    mlp_buf = {k: [] for k in ("acc", "prec", "rec", "f1")}

    best_kan_acc   = -1
    best_kan_model = None
    best_scaler    = None

    per_fold_rows = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr_full, X_te = X[tr_idx], X[te_idx]
        y_tr_full, y_te = y[tr_idx], y[te_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_full, y_tr_full,
            test_size=0.15, stratify=y_tr_full, random_state=SEED
        )
        scaler = MinMaxScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_val  = scaler.transform(X_val)
        X_te_s = scaler.transform(X_te)

        # -- KAN ---------------------------------------------------------------
        kan = KAN(layers_hidden=architecture,
                  grid_size=GRID_SIZE, spline_order=SPLINE_ORDER)
        acc_k, prec_k, rec_k, f1_k, kan = train_model(
            kan, X_tr, y_tr, X_val, y_val, X_te_s, y_te, use_closure=True)

        kan_buf["acc"].append(acc_k);  kan_buf["prec"].append(prec_k)
        kan_buf["rec"].append(rec_k);  kan_buf["f1"].append(f1_k)
        per_fold_rows.append({
            "Model": "KAN", "W": W, "Fold": fold,
            "Accuracy": acc_k * 100, "Precision": prec_k * 100,
            "Recall": rec_k * 100, "F1": f1_k * 100,
        })

        if save_kan_model and acc_k > best_kan_acc:
            best_kan_acc   = acc_k
            best_kan_model = copy.deepcopy(kan.state_dict())
            best_scaler    = (scaler.data_min_.copy(),
                              scaler.data_max_.copy(),
                              scaler.scale_.copy())

        # -- MLP ---------------------------------------------------------------
        mlp = MLP(architecture)
        acc_m, prec_m, rec_m, f1_m, _ = train_model(
            mlp, X_tr, y_tr, X_val, y_val, X_te_s, y_te, use_closure=False)

        mlp_buf["acc"].append(acc_m);  mlp_buf["prec"].append(prec_m)
        mlp_buf["rec"].append(rec_m);  mlp_buf["f1"].append(f1_m)
        per_fold_rows.append({
            "Model": "MLP", "W": W, "Fold": fold,
            "Accuracy": acc_m * 100, "Precision": prec_m * 100,
            "Recall": rec_m * 100, "F1": f1_m * 100,
        })

        print(f"  Fold {fold:2d}: KAN {acc_k*100:.2f}%  |  MLP {acc_m*100:.2f}%",
              flush=True)

    if save_kan_model and best_kan_model is not None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(best_kan_model,
                   os.path.join(MODEL_DIR, f"kan_best_W{SAVE_MODEL_FOR_W}.pt"))
        np.save(os.path.join(MODEL_DIR,
                             f"kan_best_W{SAVE_MODEL_FOR_W}_scaler.npy"),
                np.array(best_scaler, dtype=object))
        print(f"  [Saved] Best-fold KAN -> model/kan_best_W{SAVE_MODEL_FOR_W}.pt")

    kan_avg = {k: np.mean(v) * 100       for k, v in kan_buf.items()}
    kan_std = {k + "_std": np.std(v) * 100 for k, v in kan_buf.items()}
    mlp_avg = {k: np.mean(v) * 100       for k, v in mlp_buf.items()}
    mlp_std = {k + "_std": np.std(v) * 100 for k, v in mlp_buf.items()}
    return {**kan_avg, **kan_std}, {**mlp_avg, **mlp_std}, per_fold_rows


# -- Main ----------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    acc_tbl  = pd.DataFrame(index=["KAN", "MLP"], columns=WINDOWS, dtype=float)
    prec_tbl = pd.DataFrame(index=["KAN", "MLP"], columns=WINDOWS, dtype=float)
    rec_tbl  = pd.DataFrame(index=["KAN", "MLP"], columns=WINDOWS, dtype=float)
    f1_tbl   = pd.DataFrame(index=["KAN", "MLP"], columns=WINDOWS, dtype=float)

    all_fold_rows = []
    summary_rows  = []

    for W in WINDOWS:
        fp = os.path.join(DATA_DIR, f"features_W{W}.csv")
        if not os.path.exists(fp):
            print(f"[SKIP] {fp} -- run 02_feature_extraction.py first.")
            continue
        save_flag = (W == SAVE_MODEL_FOR_W)
        print(f"\nEvaluating KAN & MLP for W={W}...", flush=True)
        kan_res, mlp_res, fold_rows = evaluate_window(fp, W, save_kan_model=save_flag)
        all_fold_rows.extend(fold_rows)

        for tag, res in (("KAN", kan_res), ("MLP", mlp_res)):
            acc_tbl.loc[tag, W]  = res["acc"]
            prec_tbl.loc[tag, W] = res["prec"]
            rec_tbl.loc[tag, W]  = res["rec"]
            f1_tbl.loc[tag, W]   = res["f1"]
            summary_rows.append({
                "Model": tag, "W": W,
                "Accuracy":  f"{res['acc']:.2f} +/- {res['acc_std']:.2f}",
                "Precision": f"{res['prec']:.2f} +/- {res['prec_std']:.2f}",
                "Recall":    f"{res['rec']:.2f} +/- {res['rec_std']:.2f}",
                "F1":        f"{res['f1']:.2f} +/- {res['f1_std']:.2f}",
            })

    # Save mean-only tables (backward compatible)
    acc_tbl.to_csv(os.path.join(OUT_DIR,  "dl_accuracy.csv"))
    prec_tbl.to_csv(os.path.join(OUT_DIR, "dl_precision.csv"))
    rec_tbl.to_csv(os.path.join(OUT_DIR,  "dl_recall.csv"))
    f1_tbl.to_csv(os.path.join(OUT_DIR,   "dl_f1.csv"))

    # Save per-fold detailed results
    pd.DataFrame(all_fold_rows).to_csv(
        os.path.join(OUT_DIR, "dl_results_detailed.csv"), index=False)

    # Save summary with +/- std
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUT_DIR, "dl_summary.csv"), index=False)

    pd.set_option("display.float_format", "{:.2f}".format)
    print("\n-- Accuracy (%) --")
    print(acc_tbl.to_string())
    print(f"\nResults saved to {OUT_DIR}/")
    print(f"  Per-fold details -> dl_results_detailed.csv")
    print(f"  Summary (+/-std) -> dl_summary.csv")


if __name__ == "__main__":
    main()
