"""
05_interpretability_and_pruning.py
-----------------------------------
Implements the two core XAI novelty contributions of the paper:

  NC2 — Spline Activation Visualisation
        For each of the 40 input features, sweeps the feature value
        from min to max while holding all others at their mean, and
        records the model's predicted fault probability.  The resulting
        curves reveal threshold-like, sigmoid-like, or flat (irrelevant)
        activation profiles — directly connecting B-spline geometry to
        physical fault mechanics.

  NC3 — Pruning + Validation
        Ranks all 40 features by their L1 importance score (mean absolute
        spline weight across all output nodes).  Prunes features below a
        threshold, then re-trains all 7 baseline ML classifiers on ONLY
        the surviving features and compares accuracy against the full
        40-feature baseline.

NOTE ON MODEL CONSISTENCY
---------------------------------------------------------------------------
This script uses the SAME efficient_kan library and architecture as
04_kan_training.py.  For XAI analysis, one KAN is trained on the FULL
dataset (no train/test split) at a representative window size — this is
standard practice in interpretable ML: the benchmark accuracy numbers come
from 5-fold CV (04_), while the final deployed model used for explanation
is trained on all available data.  For windows other than the primary one,
the analysis is still run on the respective full datasets.
---------------------------------------------------------------------------

Outputs saved to results/:
  feature_importance_W<W>.csv          — L1 norm ranking, all 40 features
  feature_importance_bar_W<W>.png      — horizontal bar chart (top 20)
  marginal_activations_W<W>/           — per-feature fault-probability curves
      <feature_name>.png               — one PNG per feature (40 total)
  top_activations_grid_W<W>.png        — 4×5 grid of the 20 most important
  pruning_validation_W<W>.csv          — ML accuracy: full vs pruned features
  pruning_comparison_W<W>.png          — bar chart comparison
"""

import os
import sys
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficient_kan"))
from efficient_kan import KAN

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR   = os.path.join(BASE_DIR, "results")
WINDOWS   = [300, 400, 500, 600, 700, 800]

# Primary window for in-depth analysis (all 44 per-feature spline plots).
# Other windows get importance ranking + pruning validation only.
PRIMARY_W = 600

# ── KAN hyper-parameters (must match 04_kan_training.py) ─────────────────────
# ARCHITECTURE is set dynamically from feature count at runtime
GRID_SIZE    = 5
SPLINE_ORDER = 3
EPOCHS_XAI   = 80          # slightly more epochs for full-dataset XAI model
LR           = 1e-3
BATCH_SIZE   = 512
PATIENCE     = 15
SEED         = 42

# Pruning: fraction of features to keep (top-K by L1 norm)
PRUNE_THRESHOLD = 0.05     # features with importance < 5 % of max are pruned
N_FOLDS_PRUNE   = 10       # matches Hassan et al. 2026 (10-fold CV)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_dataset(W: int):
    fp = os.path.join(DATA_DIR, f"features_W{W}.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"{fp} — run 02_feature_extraction.py first.")
    df      = pd.read_csv(fp)
    X       = df.drop(columns=["label", "load"]).values
    y       = df["label"].values
    feats   = list(df.drop(columns=["label", "load"]).columns)
    return X, y, feats


def train_kan_full(X_scaled: np.ndarray, y: np.ndarray) -> KAN:
    """Train a KAN on the FULL (scaled) dataset for XAI analysis."""
    n_features   = X_scaled.shape[1]
    architecture = [n_features, n_features // 2, 2]
    model = KAN(layers_hidden=architecture,
                grid_size=GRID_SIZE, spline_order=SPLINE_ORDER)
    Xt = torch.tensor(X_scaled, dtype=torch.float32)
    yt = torch.tensor(y,        dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt),
        batch_size=BATCH_SIZE, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_loss  = float("inf")
    best_state = None
    patience   = 0

    for _ in range(EPOCHS_XAI):
        model.train()
        epoch_loss = 0.0
        for bx, by in loader:
            def closure():
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                return loss
            l = optimizer.step(closure)
            epoch_loss += float(l)

        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            patience   = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ── NC2: L1 Feature Importance ────────────────────────────────────────────────
def compute_l1_importance(model: KAN, n_features: int = 40) -> np.ndarray:
    """
    Compute an importance score per input feature using the L1 norm of the
    spline weights in the first KAN layer.

    For efficient_kan, model.layers[0].spline_weight has shape
    (out_features, in_features, grid_size + spline_order).
    We average |weight| across output nodes and grid points,
    giving one score per input feature.
    """
    layer = model.layers[0]
    # spline_weight: (out, in, G+k)
    w     = layer.spline_weight.detach().cpu()          # (20, 40, G+k)
    score = w.abs().mean(dim=(0, 2)).numpy()            # (40,)
    return score


# ── NC2: Marginal Activation Curves ──────────────────────────────────────────
def marginal_activation(model: KAN,
                        X_scaled: np.ndarray,
                        feature_idx: int,
                        n_points: int = 200) -> tuple:
    """
    Sweep feature `feature_idx` from 0 to 1 (MinMax range) while holding
    all other features at their mean value, and record the predicted fault
    probability P(faulty | x).

    Returns (x_grid, fault_probability_curve).
    """
    x_mean  = torch.tensor(X_scaled.mean(axis=0), dtype=torch.float32)
    x_grid  = torch.linspace(0.0, 1.0, n_points)
    probs   = []

    model.eval()
    with torch.no_grad():
        for val in x_grid:
            x_in = x_mean.clone()
            x_in[feature_idx] = val
            logits = model(x_in.unsqueeze(0))        # (1, 2)
            p_fault = torch.softmax(logits, dim=1)[0, 1].item()
            probs.append(p_fault)

    return x_grid.numpy(), np.array(probs)


def plot_single_activation(x_vals, y_vals, feature_name, W, save_path):
    """Plot and save one marginal-activation curve."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x_vals, y_vals, color="#d62728", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Decision threshold")
    ax.set_xlabel("Normalised feature value", fontsize=10)
    ax.set_ylabel("P(faulty)", fontsize=10)
    ax.set_title(f"{feature_name}  (W={W})", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_activation_grid(model, X_scaled, feature_names, importance, W, out_dir,
                         n_top=20, n_cols=4):
    """4×5 grid of the n_top most-important feature activation curves."""
    top_idx = np.argsort(importance)[::-1][:n_top]
    n_rows  = (n_top + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()

    for plot_i, feat_i in enumerate(top_idx):
        x_v, y_v = marginal_activation(model, X_scaled, int(feat_i))
        ax = axes[plot_i]
        ax.plot(x_v, y_v, color="#1f77b4", linewidth=1.8)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
        ax.set_title(feature_names[feat_i], fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)

    for j in range(plot_i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"KAN Marginal Activation Curves — Top {n_top} Features  (W={W})",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, f"top_activations_grid_W{W}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Activation grid → {path}")


def plot_importance_bar(imp_df, W, out_dir, top_n=20):
    """Horizontal bar chart of top-N feature importances."""
    top = imp_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(top["Feature"][::-1], top["Importance"][::-1],
                   color="#2ca02c", edgecolor="white")
    ax.set_xlabel("L1 Importance Score (mean |spline weight|)", fontsize=11)
    ax.set_title(f"KAN Feature Importance — Top {top_n}  (W={W})", fontsize=12)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"feature_importance_bar_W{W}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] Importance bar chart → {path}")


# ── NC3: Pruning Validation ───────────────────────────────────────────────────
def get_baseline_models():
    return {
        "DT" : DecisionTreeClassifier(random_state=SEED),
        "RF" : RandomForestClassifier(n_estimators=100, random_state=SEED),
        "SVM": SVC(kernel="rbf", random_state=SEED),
        "NB" : GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GBC": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "LR" : LogisticRegression(max_iter=1000, random_state=SEED),
    }


def cv_accuracy(X: np.ndarray, y: np.ndarray, model_name: str) -> float:
    """5-fold CV accuracy for one sklearn model on given feature matrix."""
    from sklearn.preprocessing import StandardScaler
    models = get_baseline_models()
    clf    = models[model_name]
    skf    = StratifiedKFold(n_splits=N_FOLDS_PRUNE, shuffle=True, random_state=SEED)
    accs   = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr]);  X_te = sc.transform(X[te])
        clf_clone = copy.deepcopy(clf)
        clf_clone.fit(X_tr, y[tr])
        accs.append(accuracy_score(y[te], clf_clone.predict(X_te)))
    return float(np.mean(accs)) * 100


def kan_cv_accuracy(X: np.ndarray, y: np.ndarray) -> float:
    """10-fold CV accuracy for KAN on given feature matrix."""
    in_dim = X.shape[1]
    arch   = [in_dim, max(4, in_dim // 2), 2]
    skf    = StratifiedKFold(n_splits=N_FOLDS_PRUNE, shuffle=True, random_state=SEED)
    accs   = []
    for tr, te in skf.split(X, y):
        X_tr_f, X_te = X[tr], X[te]
        y_tr_f, y_te = y[tr], y[te]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr_f, y_tr_f, test_size=0.15,
            stratify=y_tr_f, random_state=SEED)
        sc  = MinMaxScaler()
        X_tr  = sc.fit_transform(X_tr)
        X_val = sc.transform(X_val)
        X_te  = sc.transform(X_te)

        kan = KAN(layers_hidden=arch,
                  grid_size=GRID_SIZE, spline_order=SPLINE_ORDER)
        Xt = torch.tensor(X_tr,  dtype=torch.float32)
        yt = torch.tensor(y_tr,  dtype=torch.long)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        yv = torch.tensor(y_val, dtype=torch.long)
        Xte= torch.tensor(X_te,  dtype=torch.float32)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=BATCH_SIZE, shuffle=True)
        opt   = torch.optim.Adam(kan.parameters(), lr=LR)
        crit  = nn.CrossEntropyLoss()
        best_l= float("inf"); best_s=None; pat=0

        for _ in range(EPOCHS_XAI):
            kan.train()
            for bx, by in loader:
                def closure():
                    opt.zero_grad()
                    l = crit(kan(bx), by)
                    l.backward()
                    return l
                opt.step(closure)
            kan.eval()
            with torch.no_grad():
                vl = crit(kan(Xv), yv).item()
            if vl < best_l:
                best_l = vl; best_s = copy.deepcopy(kan.state_dict()); pat=0
            else:
                pat += 1
                if pat >= PATIENCE: break
        if best_s: kan.load_state_dict(best_s)
        kan.eval()
        with torch.no_grad():
            preds = torch.argmax(kan(Xte), dim=1).numpy()
        accs.append(accuracy_score(y_te, preds))
    return float(np.mean(accs)) * 100


def run_pruning_validation(X: np.ndarray, y: np.ndarray,
                           feature_names: list, importance: np.ndarray,
                           W: int, out_dir: str):
    """
    NC3: Select surviving features by L1 importance threshold,
    then benchmark all classifiers on full vs pruned feature sets.
    """
    max_imp   = importance.max()
    survivors = np.where(importance >= PRUNE_THRESHOLD * max_imp)[0]
    pruned    = np.where(importance <  PRUNE_THRESHOLD * max_imp)[0]

    print(f"\n  Pruning: {len(survivors)}/{len(feature_names)} features survive "
          f"(threshold = {PRUNE_THRESHOLD*100:.0f}% of max importance)")
    if len(survivors) == 0:
        print("  [WARNING] No features survived pruning — lowering threshold.")
        survivors = np.argsort(importance)[::-1][:10]

    survivor_names = [feature_names[i] for i in survivors]
    X_pruned       = X[:, survivors]

    # Save survivor list
    surv_df = pd.DataFrame({
        "Feature"   : survivor_names,
        "Index"     : survivors,
        "Importance": importance[survivors]
    }).sort_values("Importance", ascending=False)
    surv_df.to_csv(os.path.join(out_dir, f"pruned_survivors_W{W}.csv"), index=False)

    # Benchmark all classifiers
    print("  Running 5-fold CV on full (40) vs pruned feature sets…")
    rows = []
    model_names = list(get_baseline_models().keys()) + ["KAN"]

    for mname in model_names:
        if mname == "KAN":
            acc_full   = kan_cv_accuracy(X,        y)
            acc_pruned = kan_cv_accuracy(X_pruned,  y)
        else:
            acc_full   = cv_accuracy(X,        y, mname)
            acc_pruned = cv_accuracy(X_pruned,  y, mname)

        delta = acc_pruned - acc_full
        rows.append({
            "Model"        : mname,
            "Full_40_feats": round(acc_full,   2),
            "Pruned_feats" : round(acc_pruned, 2),
            "n_pruned"     : len(survivors),
            "Delta_%"      : round(delta,      2),
        })
        print(f"    {mname:5s}  Full={acc_full:.2f}%  "
              f"Pruned={acc_pruned:.2f}%  Δ={delta:+.2f}%", flush=True)

    val_df = pd.DataFrame(rows)
    val_df.to_csv(os.path.join(out_dir, f"pruning_validation_W{W}.csv"), index=False)

    # Comparison plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(rows))
    w     = 0.35
    ax.bar(x_pos - w/2, val_df["Full_40_feats"], w, label="Full (40 features)",
           color="#1f77b4", edgecolor="white")
    ax.bar(x_pos + w/2, val_df["Pruned_feats"],  w, label=f"Pruned ({len(survivors)} features)",
           color="#ff7f0e", edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(val_df["Model"], fontsize=10)
    ax.set_ylabel("5-Fold CV Accuracy (%)", fontsize=11)
    ax.set_title(f"Pruning Validation — Full vs Pruned Feature Set  (W={W})", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(min(val_df[["Full_40_feats","Pruned_feats"]].min()) - 5, 101)
    plt.tight_layout()
    path = os.path.join(out_dir, f"pruning_comparison_W{W}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] Pruning comparison plot → {path}")

    return val_df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for W in WINDOWS:
        print(f"\n{'='*60}")
        print(f"  Interpretability & Pruning Analysis  —  W={W}")
        print(f"{'='*60}")

        try:
            X, y, feature_names = load_dataset(W)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        # Scale the full dataset for training the XAI model
        scaler   = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Train KAN on full dataset for XAI ────────────────────────────────
        print(f"  Training KAN on full dataset (N={len(X)})…", flush=True)
        model = train_kan_full(X_scaled, y)

        # ── NC2a: L1 feature importance ───────────────────────────────────────
        importance = compute_l1_importance(model, n_features=len(feature_names))
        imp_df = pd.DataFrame({
            "Feature"   : feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        imp_df.to_csv(os.path.join(OUT_DIR, f"feature_importance_W{W}.csv"), index=False)
        print(f"  Top 5 features: {', '.join(imp_df['Feature'].head(5).tolist())}")

        plot_importance_bar(imp_df, W, OUT_DIR)

        # ── NC2b: Per-feature marginal activation curves ──────────────────────
        print(f"  Generating marginal activation curves for all {len(feature_names)} features…",
              flush=True)
        act_dir = os.path.join(OUT_DIR, f"marginal_activations_W{W}")
        os.makedirs(act_dir, exist_ok=True)

        for idx, fname in enumerate(feature_names):
            x_v, y_v = marginal_activation(model, X_scaled, idx)
            plot_single_activation(x_v, y_v, fname, W,
                                   os.path.join(act_dir, f"{fname}.png"))

        print(f"  [Saved] {len(feature_names)} activation curves → {act_dir}/")

        # 4×5 grid of top-20 for the paper
        plot_activation_grid(model, X_scaled, feature_names, importance,
                             W, OUT_DIR, n_top=20, n_cols=4)

        # ── NC3: Pruning + Validation ─────────────────────────────────────────
        run_pruning_validation(X, y, feature_names, importance, W, OUT_DIR)

    print(f"\n{'='*60}")
    print("  All interpretability analyses complete.")
    print(f"  Outputs → {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
