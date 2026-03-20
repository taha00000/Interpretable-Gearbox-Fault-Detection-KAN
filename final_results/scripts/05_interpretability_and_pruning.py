import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from kan import KAN

import warnings
warnings.filterwarnings('ignore')

def get_data(W):
    filepath = r"c:/Users/tahah/OneDrive/Desktop/Moving-window-based-feature-extraction-method-for-vibration-based-condition-monitoring-main/data/processed/features_W" + str(W) + ".csv"
    df = pd.read_csv(filepath)
    X = df.drop(columns=['label', 'load']).values
    y = df['label'].values
    features = list(df.drop(columns=['label', 'load']).columns)
    return X, y, features

def main():
    windows = [300, 400, 500, 600, 700, 800]
    out_dir = r"c:/Users/tahah/OneDrive/Desktop/Moving-window-based-feature-extraction-method-for-vibration-based-condition-monitoring-main/results"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Train on W = 600 for Interpretability and Pruning (assuming it's a good representative size)
    W = 600
    print(f"\n--- Running Interpretability & Pruning Analysis on W={W} ---")
    X, y, feature_names = get_data(W)
    X_train, X_test, y_train, y_test = X, X, y, y # Using full dataset for pure analysis/visualization as standard in XAI 
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    dataset = {
        'train_input': torch.tensor(X_train_scaled, dtype=torch.float32),
        'train_label': torch.tensor(y_train, dtype=torch.long),
        'test_input': torch.tensor(X_train_scaled, dtype=torch.float32),
        'test_label': torch.tensor(y_train, dtype=torch.long)
    }
    
    arch = [40, 20, 2]
    model = KAN(width=arch, grid=5, k=3, seed=42)
    
    # Minimal training loop for the analysis model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(50): # few epochs enough to shape early splines
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(dataset['train_input']), dataset['train_label'])
            loss.backward()
            return loss
        optimizer.step(closure)
        
    # Plot complete network
    fig_folder = os.path.join(out_dir, f'kan_plots_W{W}')
    os.makedirs(fig_folder, exist_ok=True)
    # PyKAN plot functionality:
    # We catch errors gracefully in case PyKAN version differs
    try:
        model.plot(folder=fig_folder, beta=100)
        print(f"Saved complete network plot to {fig_folder}")
    except Exception as e:
        print(f"Could not generate plot: {e}")
        
    # Feature Importance (L1 Norm)
    # PyKAN provides attribute_scores or we can approximate by getting weights of first layer
    try:
        scores = model.feature_score
        importance = scores.detach().cpu().numpy()
        
        # Save rankings
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        imp_df = imp_df.sort_values(by='Importance', ascending=False)
        imp_df.to_csv(os.path.join(out_dir, f'feature_importance_W{W}.csv'), index=False)
        print("Saved Feature Importances.")
        
        # Plot top 10
        plt.figure(figsize=(10, 6))
        plt.barh(imp_df['Feature'].head(10)[::-1], imp_df['Importance'].head(10)[::-1])
        plt.xlabel("L1 Norm (Importance)")
        plt.title(f"KAN Feature Importance (W={W})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'feature_importance_bar_W{W}.png'))
        plt.close()
    except Exception as e:
        print(f"Could not extract feature scores: {e}")
        
    # Pruning
    print("\n--- Pruning Network ---")
    try:
        # PyKAN prune() signature can vary by version; this repo uses positional threshold.
        # threshold removes edges with small activation/scale.
        model = model.prune(1e-2)

        # Save surviving feature set for interpretability.
        # `acts_scale[0]` corresponds to the first layer edge scales; we reduce across the next-layer dim.
        acts0 = model.acts_scale[0].detach().cpu().numpy()
        if acts0.ndim >= 2:
            survival_score = np.sum(np.abs(acts0), axis=1)
        else:
            survival_score = np.abs(acts0)

        survivors = np.where(survival_score > 0)[0]
        pruned_df = pd.DataFrame(
            {
                "FeatureIndex": survivors,
                "Feature": [feature_names[i] for i in survivors],
                "SurvivalScore": survival_score[survivors],
            }
        ).sort_values("SurvivalScore", ascending=False)

        pruned_df.to_csv(os.path.join(out_dir, f'pruned_survivors_W{W}.csv'), index=False)
        print(f"Network pruned! Surviving input features: {len(survivors)}")

        # Re-plot pruned model for paper-ready visuals.
        pruned_fig_folder = os.path.join(out_dir, f'kan_plots_W{W}_pruned')
        os.makedirs(pruned_fig_folder, exist_ok=True)
        try:
            model.plot(folder=pruned_fig_folder, beta=100)
            print(f"Saved pruned network plot to {pruned_fig_folder}")
        except Exception as e:
            print(f"Could not generate pruned network plot: {e}")

        # Optional: save updated feature importance after pruning.
        try:
            pruned_scores = model.feature_score.detach().cpu().numpy()
            pruned_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': pruned_scores})
            pruned_imp_df = pruned_imp_df.sort_values(by='Importance', ascending=False)
            pruned_imp_df.to_csv(os.path.join(out_dir, f'feature_importance_pruned_W{W}.csv'), index=False)
        except Exception as e:
            print(f"Could not extract pruned feature scores: {e}")
    except Exception as e:
        print(f"Pruning failed or unavailable: {e}")

if __name__ == "__main__":
    main()
