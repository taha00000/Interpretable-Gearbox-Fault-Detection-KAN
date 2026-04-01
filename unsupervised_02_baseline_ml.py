"""
unsupervised_02_baseline_ml.py
------------------------------
Trains baseline unsupervised/one-class machine learning models:
  1. Isolation Forest
  2. One-Class SVM (OC-SVM)
  3. Local Outlier Factor (LOF)

Trains the models exclusively on the healthy Training Set.
Computes anomaly scores and evaluates thresholding on Val and Test sets.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "unsupervised")
RESULTS_DIR = os.path.join(BASE_DIR, "results_unsupervised")
WINDOWS = [300, 400, 500, 600, 700, 800, 900, 1000]

def eval_model(y_true, scores, threshold):
    # Higher score = more anomalous
    y_pred = (scores > threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError: # handle edge cases where only 1 class is present (shouldn't happen here)
        auc = 0.5
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return auc, p, r, f1

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []
    
    print("======================================================")
    print(" Baseline Unsupervised ML - Gearbox Fault Detection")
    print("======================================================")
    
    for W in WINDOWS:
        train_path = os.path.join(DATA_DIR, f"train_W{W}.csv")
        val_path = os.path.join(DATA_DIR, f"val_W{W}.csv")
        test_path = os.path.join(DATA_DIR, f"test_W{W}.csv")
        
        if not os.path.exists(train_path):
            continue
            
        print(f"\n--- Training baselines for W={W} ---")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        feat_cols = [c for c in train_df.columns if c not in ['load', 'label']]
        
        X_train = train_df[feat_cols].values
        X_val = val_df[feat_cols].values
        X_test = test_df[feat_cols].values
        
        y_val = val_df['label'].values
        y_test = test_df['label'].values
        
        # Scale features using training set stats
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)
        
        models = {
            "IsolationForest": IsolationForest(n_estimators=100, random_state=42),
            "OC-SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
            "LOF": LocalOutlierFactor(n_neighbors=20, novelty=True)
        }
        
        for name, clf in models.items():
            clf.fit(X_train_s)
            
            # Scikit-Learn convention: decision_function computes a score where
            # < 0 is an outlier/anomaly, and > 0 is an inlier/healthy.
            # We want an Anomaly Score (higher = more anomalous).
            # So, we negate the decision_function.
            val_scores = -clf.decision_function(X_val_s)
            test_scores = -clf.decision_function(X_test_s)
            
            # Find the best threshold using the validation set to maximize F1-score
            best_thresh = 0
            best_f1 = 0
            for th in np.linspace(val_scores.min(), val_scores.max(), 100):
                yp = (val_scores > th).astype(int)
                _, _, f, _ = precision_recall_fscore_support(y_val, yp, average='binary', zero_division=0)
                if f > best_f1:
                    best_f1 = f
                    best_thresh = th
                    
            # Evaluate on Test Set using the discovered threshold
            auc, p, r, f1 = eval_model(y_test, test_scores, best_thresh)
            print(f"  {name:15s} | Test AUC: {auc:.4f} | F1: {f1:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")
            
            all_results.append({
                "W": W,
                "Model": name,
                "AUC": auc,
                "F1": f1,
                "Precision": p,
                "Recall": r,
                "Threshold": best_thresh
            })
            
            # Save predictions for plotting later
            test_df[f"{name}_score"] = test_scores
            
        test_df.to_csv(os.path.join(RESULTS_DIR, f"baseline_test_preds_W{W}.csv"), index=False)
        
    res_df = pd.DataFrame(all_results)
    metrics_path = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
    res_df.to_csv(metrics_path, index=False)
    print("\n[OK] Baseline unsupervised ML training complete.")
    print(f"Results saved in {metrics_path}")

if __name__ == "__main__":
    main()
