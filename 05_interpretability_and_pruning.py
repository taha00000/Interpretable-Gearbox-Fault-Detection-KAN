import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    # Optional resume control:
    #   python 05_interpretability_and_pruning.py --windows 400 500
    # Defaults to all window sizes.
    if "--windows" in sys.argv:
        idx = sys.argv.index("--windows")
        parsed = []
        for a in sys.argv[idx + 1:]:
            if a.startswith("--"):
                break
            try:
                parsed.append(int(a))
            except ValueError:
                pass
        if parsed:
            windows = parsed
    out_dir = r"c:/Users/tahah/OneDrive/Desktop/Moving-window-based-feature-extraction-method-for-vibration-based-condition-monitoring-main/results"
    os.makedirs(out_dir, exist_ok=True)

    arch = [40, 20, 2]
    grid, k = 5, 3
    epochs_for_xai = 50  # minimal shaping of early splines
    # Full KAN graph rendering can generate thousands of PNGs and is the most crash-prone part.
    # We still compute CSV novelty outputs for all windows, but only render plots for representative W.
    SAVE_NETWORK_PLOTS_WINDOWS = {600}

    def window_is_complete(W: int) -> bool:
        # Treat pruning novelty as "complete" only when both ranking + surviving feature list exist.
        # Plot regeneration is optional and can be rerun if desired.
        pruned_survivors = os.path.join(out_dir, f'pruned_survivors_W{W}.csv')
        pruned_importance = os.path.join(out_dir, f'feature_importance_pruned_W{W}.csv')
        return os.path.exists(pruned_survivors) and os.path.exists(pruned_importance)

    for W in windows:
        if window_is_complete(W):
            print(f"\n--- Skipping W={W} (pruning outputs already exist) ---", flush=True)
            continue

        print(f"\n--- Running Interpretability & Pruning Analysis on W={W} ---", flush=True)
        X, y, feature_names = get_data(W)

        # Using full dataset for interpretability/visualization (no train/test split needed for edge ranking).
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        train_input = torch.tensor(X_scaled, dtype=torch.float32)
        train_label = torch.tensor(y, dtype=torch.long)

        model = KAN(width=arch, grid=grid, k=k, seed=42)

        # Minimal training loop for the analysis model.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(epochs_for_xai):
            def closure():
                optimizer.zero_grad()
                loss = criterion(model(train_input), train_label)
                loss.backward()
                return loss
            optimizer.step(closure)

        # Plot complete network
        if W in SAVE_NETWORK_PLOTS_WINDOWS:
            fig_folder = os.path.join(out_dir, f'kan_plots_W{W}')
            os.makedirs(fig_folder, exist_ok=True)
            try:
                model.plot(folder=fig_folder, beta=100)
                print(f"Saved complete network plot to {fig_folder}")
            except Exception as e:
                print(f"Could not generate plot: {e}")

        # Feature Importance (L1-like score)
        try:
            scores = model.feature_score
            importance = scores.detach().cpu().numpy()

            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            imp_df = imp_df.sort_values(by='Importance', ascending=False)
            imp_df.to_csv(os.path.join(out_dir, f'feature_importance_W{W}.csv'), index=False)

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
            model = model.prune(1e-2)

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

            if W in SAVE_NETWORK_PLOTS_WINDOWS:
                pruned_fig_folder = os.path.join(out_dir, f'kan_plots_W{W}_pruned')
                os.makedirs(pruned_fig_folder, exist_ok=True)
                try:
                    model.plot(folder=pruned_fig_folder, beta=100)
                    print(f"Saved pruned network plot to {pruned_fig_folder}")
                except Exception as e:
                    print(f"Could not generate pruned network plot: {e}")

            try:
                pruned_scores = model.feature_score.detach().cpu().numpy()
                pruned_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': pruned_scores})
                pruned_imp_df = pruned_imp_df.sort_values(by='Importance', ascending=False)
                pruned_imp_df.to_csv(
                    os.path.join(out_dir, f'feature_importance_pruned_W{W}.csv'), index=False
                )
            except Exception as e:
                print(f"Could not extract pruned feature scores: {e}")

        except Exception as e:
            print(f"Pruning failed or unavailable: {e}")

if __name__ == "__main__":
    main()
