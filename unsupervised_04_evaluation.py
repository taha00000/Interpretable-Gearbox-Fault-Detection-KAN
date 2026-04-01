"""
unsupervised_04_evaluation.py
-----------------------------
Phase 4: Evaluation & Analysis
Aggregates metrics from baseline models and deep learning models.
Generates comprehensive ROC-AUC curves and probability density
plots (distributions) of the anomaly scores to contrast Healthy vs Faulty.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_unsupervised")
WINDOWS = [300, 400, 500, 600, 700, 800, 900, 1000]

def main():
    print("======================================================")
    print(" Evaluation & Plotting - Gearbox Fault Detection")
    print("======================================================")
    
    # 1. Combine Metrics
    base_path = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
    dl_path = os.path.join(RESULTS_DIR, "dl_metrics.csv")
    
    dfs = []
    if os.path.exists(base_path): dfs.append(pd.read_csv(base_path))
    if os.path.exists(dl_path): dfs.append(pd.read_csv(dl_path))
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, "combined_metrics.csv"), index=False)
        print("\n[OK] Combined metrics saved to combined_metrics.csv")
        
        # Print summary table using AUC as the main comparative metric
        try:
            pivot = combined.pivot(index='Model', columns='W', values='AUC').round(4)
            print("\n--- Summary of ROC-AUC across Window Sizes ---")
            print(pivot.to_string())
        except Exception as e:
            print("Could not create pivot summary:", e)
    
    # 2. Plotting for W=400 (or the first available if 400 doesn't exist)
    target_W = 1000
    preds_path = os.path.join(RESULTS_DIR, f"all_test_preds_W{target_W}.csv")
    if not os.path.exists(preds_path):
        for W in WINDOWS:
            if os.path.exists(os.path.join(RESULTS_DIR, f"all_test_preds_W{W}.csv")):
                target_W = W
                preds_path = os.path.join(RESULTS_DIR, f"all_test_preds_W{target_W}.csv")
                break
                
    if not os.path.exists(preds_path):
        print("No prediction files found. Exiting.")
        return
        
    print(f"\nGenerating plots for W={target_W} based on file: {preds_path}")
    df = pd.read_csv(preds_path)
    score_cols = [c for c in df.columns if c.endswith("_score")]
    
    y_true = df["label"].values
    
    # --- Plot ROC Curves ---
    plt.figure(figsize=(10, 8))
    for col in score_cols:
        model_name = col.replace("_score", "")
        scores = df[col].values
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'ROC Curves for Unsupervised Anomaly Detection (W={target_W})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    roc_out = os.path.join(RESULTS_DIR, f"roc_curves_W{target_W}.png")
    plt.savefig(roc_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] ROC Curve plotted to       -> {roc_out}")
    
    # --- Plot Anomaly Score Distributions (for Baseline best: LOF, and DL best: PatchCore) ---
    models_to_plot = ["PatchCore", "LOF"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, model in enumerate(models_to_plot):
        col = f"{model}_score"
        if col not in df.columns:
            continue
            
        ax = axes[i]
        healthy_scores = df[df["label"] == 0][col]
        faulty_scores = df[df["label"] == 1][col]
        
        sns.kdeplot(healthy_scores, ax=ax, fill=True, color='green', label='Healthy (Normal)', alpha=0.5)
        sns.kdeplot(faulty_scores, ax=ax, fill=True, color='red', label='Faulty (Anomaly)', alpha=0.5)
        
        ax.set_title(f"Score Distribution: {model} (W={target_W})")
        ax.set_xlabel("Anomaly Score (Higher = More Anomalous)")
        ax.set_ylabel("Kernel Density Estimate")
        ax.legend()
        
    plt.tight_layout()
    dist_out = os.path.join(RESULTS_DIR, f"score_distributions_W{target_W}.png")
    plt.savefig(dist_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] Density Distributions to   -> {dist_out}")
    
    print("\n[OK] Phase 4 Evaluation complete.")

if __name__ == "__main__":
    main()
