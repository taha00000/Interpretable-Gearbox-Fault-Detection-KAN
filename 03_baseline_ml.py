"""
03_baseline_ml.py
-----------------
Evaluates seven traditional ML classifiers on each of the six
moving-window datasets (W = 300 → 800), using identical 5-fold
stratified cross-validation to the KAN experiments in 04_kan_training.py.

Classifiers: Decision Tree (DT), Random Forest (RF), SVM, Naïve Bayes (NB),
             K-Nearest Neighbour (KNN), Gradient Boosting (GBC),
             Logistic Regression (LR).

Outputs (saved to results/):
  baseline_accuracy.csv   — accuracy  (%) per model × window size
  baseline_precision.csv  — precision (%) per model × window size
  baseline_recall.csv     — recall    (%) per model × window size
  baseline_f1.csv         — F1        (%) per model × window size
  baseline_results_detailed.csv — per-fold metrics for all models
  baseline_summary.csv    — mean ± std summary for all metrics
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "results")
WINDOWS  = [300, 400, 500, 600, 700, 800]
N_FOLDS  = 10
SEED     = 42


# ── Model definitions ─────────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        "DT" : DecisionTreeClassifier(random_state=SEED),
        "RF" : RandomForestClassifier(n_estimators=100, random_state=SEED),
        "SVM": SVC(kernel="rbf", random_state=SEED),
        "NB" : GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GBC": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "LR" : LogisticRegression(max_iter=1000, random_state=SEED),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_window(filepath: str, W: int) -> tuple:
    """Run 5-fold CV for all 7 classifiers on one window-size dataset.
    Returns (summary_dict, per_fold_rows)."""
    df = pd.read_csv(filepath)
    X  = df.drop(columns=["label", "load"]).values
    y  = df["label"].values

    models = get_models()
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    acc_buf  = {m: [] for m in models}
    prec_buf = {m: [] for m in models}
    rec_buf  = {m: [] for m in models}
    f1_buf   = {m: [] for m in models}

    per_fold_rows = []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        for name, clf in models.items():
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            a = accuracy_score(y_te, y_pred)
            p = precision_score(y_te, y_pred, average="macro", zero_division=0)
            r = recall_score(y_te, y_pred,  average="macro", zero_division=0)
            f = f1_score(y_te, y_pred, average="macro", zero_division=0)
            acc_buf[name].append(a)
            prec_buf[name].append(p)
            rec_buf[name].append(r)
            f1_buf[name].append(f)
            per_fold_rows.append({
                "Model": name, "W": W, "Fold": fold,
                "Accuracy": a * 100, "Precision": p * 100,
                "Recall": r * 100, "F1": f * 100,
            })

    summary = {
        name: {
            "acc":      np.mean(acc_buf[name])  * 100,
            "acc_std":  np.std(acc_buf[name])    * 100,
            "prec":     np.mean(prec_buf[name]) * 100,
            "prec_std": np.std(prec_buf[name])  * 100,
            "rec":      np.mean(rec_buf[name])  * 100,
            "rec_std":  np.std(rec_buf[name])   * 100,
            "f1":       np.mean(f1_buf[name])   * 100,
            "f1_std":   np.std(f1_buf[name])    * 100,
        }
        for name in models
    }
    return summary, per_fold_rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model_names = list(get_models().keys())

    acc_tbl  = pd.DataFrame(index=model_names, columns=WINDOWS, dtype=float)
    prec_tbl = pd.DataFrame(index=model_names, columns=WINDOWS, dtype=float)
    rec_tbl  = pd.DataFrame(index=model_names, columns=WINDOWS, dtype=float)
    f1_tbl   = pd.DataFrame(index=model_names, columns=WINDOWS, dtype=float)

    all_fold_rows = []
    summary_rows  = []

    for W in WINDOWS:
        fp = os.path.join(DATA_DIR, f"features_W{W}.csv")
        if not os.path.exists(fp):
            print(f"[SKIP] {fp} not found — run 02_feature_extraction.py first.")
            continue
        print(f"Evaluating baseline ML classifiers for W={W}…", flush=True)
        res, fold_rows = evaluate_window(fp, W)
        all_fold_rows.extend(fold_rows)
        for name in model_names:
            acc_tbl.loc[name, W]  = res[name]["acc"]
            prec_tbl.loc[name, W] = res[name]["prec"]
            rec_tbl.loc[name, W]  = res[name]["rec"]
            f1_tbl.loc[name, W]   = res[name]["f1"]
            summary_rows.append({
                "Model": name, "W": W,
                "Accuracy":  f"{res[name]['acc']:.2f} ± {res[name]['acc_std']:.2f}",
                "Precision": f"{res[name]['prec']:.2f} ± {res[name]['prec_std']:.2f}",
                "Recall":    f"{res[name]['rec']:.2f} ± {res[name]['rec_std']:.2f}",
                "F1":        f"{res[name]['f1']:.2f} ± {res[name]['f1_std']:.2f}",
            })

    # Save mean-only tables (backward compatible)
    acc_tbl.to_csv(os.path.join(OUT_DIR,  "baseline_accuracy.csv"))
    prec_tbl.to_csv(os.path.join(OUT_DIR, "baseline_precision.csv"))
    rec_tbl.to_csv(os.path.join(OUT_DIR,  "baseline_recall.csv"))
    f1_tbl.to_csv(os.path.join(OUT_DIR,   "baseline_f1.csv"))

    # Save per-fold detailed results
    pd.DataFrame(all_fold_rows).to_csv(
        os.path.join(OUT_DIR, "baseline_results_detailed.csv"), index=False)

    # Save summary with ± std
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUT_DIR, "baseline_summary.csv"), index=False)

    pd.set_option("display.float_format", "{:.2f}".format)
    print("\n── Accuracy (%) ──")
    print(acc_tbl.to_string())
    print("\n── Precision (%) ──")
    print(prec_tbl.to_string())
    print("\n── Recall (%) ──")
    print(rec_tbl.to_string())
    print("\n── F1 (%) ──")
    print(f1_tbl.to_string())
    print(f"\nResults saved to {OUT_DIR}/")
    print(f"  Per-fold details → baseline_results_detailed.csv")
    print(f"  Summary (±std)   → baseline_summary.csv")


if __name__ == "__main__":
    main()
