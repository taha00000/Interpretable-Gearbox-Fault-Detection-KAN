"""
06_paper_figures.py
-------------------
Generates publication-ready figures, tables, confusion matrices, and
statistical significance tests for the IBCAST 2026 conference paper.

Prerequisites:
  - Run 03_baseline_ml.py  (produces baseline_results_detailed.csv)
  - Run 04_kan_training.py (produces dl_results_detailed.csv)
  - Run 05_interpretability_and_pruning.py (produces feature importance + pruning CSVs)

Outputs (saved to results/paper_figures/):
  confusion_matrices_W600.png       -- confusion matrices for key models
  accuracy_comparison_bar.png       -- grouped bar chart across window sizes
  accuracy_vs_window_line.png       -- line chart: accuracy vs window size
  statistical_tests.csv             -- paired t-test / Wilcoxon results
  latex_tables.txt                  -- LaTeX-formatted tables for paper
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

warnings.filterwarnings("ignore")

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
RESULTS    = os.path.join(BASE_DIR, "results")
FIG_DIR    = os.path.join(RESULTS, "paper_figures")
WINDOWS    = [300, 400, 500, 600, 700, 800]
PRIMARY_W  = 600
SEED       = 42


# -- Style setup ---------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# IEEE-friendly color palette
COLORS = {
    "KAN": "#1f77b4",
    "MLP": "#ff7f0e",
    "DT":  "#2ca02c",
    "RF":  "#d62728",
    "SVM": "#9467bd",
    "NB":  "#8c564b",
    "KNN": "#e377c2",
    "GBC": "#7f7f7f",
    "LR":  "#bcbd22",
}


# -- Load data -----------------------------------------------------------------
def load_detailed_results():
    """Load per-fold results from both baseline and DL scripts."""
    bl_path = os.path.join(RESULTS, "baseline_results_detailed.csv")
    dl_path = os.path.join(RESULTS, "dl_results_detailed.csv")

    frames = []
    if os.path.exists(bl_path):
        frames.append(pd.read_csv(bl_path))
        print(f"  Loaded {bl_path}")
    else:
        print(f"  [WARNING] {bl_path} not found. Run 03_baseline_ml.py first.")

    if os.path.exists(dl_path):
        frames.append(pd.read_csv(dl_path))
        print(f"  Loaded {dl_path}")
    else:
        print(f"  [WARNING] {dl_path} not found. Run 04_kan_training.py first.")

    if not frames:
        raise FileNotFoundError("No detailed result files found. Run 03 and 04 first.")

    return pd.concat(frames, ignore_index=True)


def load_mean_tables():
    """Load the backward-compatible mean accuracy tables."""
    bl = pd.read_csv(os.path.join(RESULTS, "baseline_accuracy.csv"), index_col=0)
    dl = pd.read_csv(os.path.join(RESULTS, "dl_accuracy.csv"), index_col=0)
    combined = pd.concat([dl, bl])
    combined.columns = [int(c) for c in combined.columns]
    return combined


# -- Figure 1: Confusion matrices for key models at W=600 ---------------------
def generate_confusion_matrices(df, W=PRIMARY_W):
    """Generate confusion matrices for key models using per-fold predictions.
    Since we only have aggregate metrics, we reconstruct approximate CMs
    from the per-fold accuracy on the dataset."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression

    # Load dataset
    fp = os.path.join(DATA_DIR, f"features_W{W}.csv")
    if not os.path.exists(fp):
        print(f"  [SKIP] Cannot generate confusion matrices: {fp} not found.")
        return
    data = pd.read_csv(fp)
    X = data.drop(columns=["label", "load"]).values
    y = data["label"].values

    models = {
        "KAN": None,  # handled separately
        "MLP": None,  # handled separately
        "SVM": SVC(kernel="rbf", random_state=SEED),
        "RF":  RandomForestClassifier(n_estimators=100, random_state=SEED),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cm_dict = {}

    # ML models
    for name in ["SVM", "RF"]:
        clf = models[name]
        all_y_true, all_y_pred = [], []
        for tr, te in skf.split(X, y):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr])
            X_te = sc.transform(X[te])
            clf_copy = type(clf)(**clf.get_params())
            clf_copy.fit(X_tr, y[tr])
            preds = clf_copy.predict(X_te)
            all_y_true.extend(y[te])
            all_y_pred.extend(preds)
        cm_dict[name] = confusion_matrix(all_y_true, all_y_pred)

    # KAN and MLP
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, "efficient_kan"))

    try:
        from efficient_kan import KAN as KANModel
        import torch
        import torch.nn as nn
        import copy
        from sklearn.model_selection import train_test_split

        arch = [40, 20, 2]
        for tag, ModelClass, use_closure in [("KAN", KANModel, True), ("MLP", None, False)]:
            all_y_true, all_y_pred = [], []
            for tr, te in skf.split(X, y):
                X_tr_f, X_te_raw = X[tr], X[te]
                y_tr_f, y_te_raw = y[tr], y[te]
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_tr_f, y_tr_f, test_size=0.15, stratify=y_tr_f, random_state=SEED)
                sc = MinMaxScaler()
                X_tr_s = sc.fit_transform(X_tr)
                X_val_s = sc.transform(X_val)
                X_te_s = sc.transform(X_te_raw)

                if tag == "KAN":
                    model = KANModel(layers_hidden=arch, grid_size=5, spline_order=3)
                else:
                    layers = []
                    for i in range(len(arch) - 1):
                        layers.append(nn.Linear(arch[i], arch[i+1]))
                        if i < len(arch) - 2:
                            layers.append(nn.ReLU())
                    model = nn.Sequential(*layers)

                Xt = torch.tensor(X_tr_s, dtype=torch.float32)
                yt = torch.tensor(y_tr, dtype=torch.long)
                Xv = torch.tensor(X_val_s, dtype=torch.float32)
                yv = torch.tensor(y_val, dtype=torch.long)
                loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(Xt, yt), batch_size=512, shuffle=True)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                crit = nn.CrossEntropyLoss()
                best_l = float("inf"); best_s = None; pat = 0
                for _ in range(50):
                    model.train()
                    for bx, by in loader:
                        if use_closure:
                            def closure():
                                opt.zero_grad()
                                l = crit(model(bx), by)
                                l.backward()
                                return l
                            opt.step(closure)
                        else:
                            opt.zero_grad()
                            l = crit(model(bx), by)
                            l.backward()
                            opt.step()
                    model.eval()
                    with torch.no_grad():
                        vl = crit(model(Xv), yv).item()
                    if vl < best_l:
                        best_l = vl; best_s = copy.deepcopy(model.state_dict()); pat = 0
                    else:
                        pat += 1
                        if pat >= 10: break
                if best_s: model.load_state_dict(best_s)
                model.eval()
                with torch.no_grad():
                    preds = torch.argmax(model(torch.tensor(X_te_s, dtype=torch.float32)), dim=1).numpy()
                all_y_true.extend(y_te_raw)
                all_y_pred.extend(preds)
            cm_dict[tag] = confusion_matrix(all_y_true, all_y_pred)
    except Exception as e:
        print(f"  [WARNING] Could not generate KAN/MLP confusion matrices: {e}")

    # Plot
    plot_order = [m for m in ["KAN", "MLP", "SVM", "RF"] if m in cm_dict]
    n = len(plot_order)
    if n == 0:
        print("  [SKIP] No confusion matrices to plot.")
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, name in zip(axes, plot_order):
        cm = cm_dict[name]
        disp = ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Faulty"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontsize=11, fontweight="bold")

    fig.suptitle(f"Confusion Matrices (W={W}, 5-Fold CV Aggregated)", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"confusion_matrices_W{W}.png")
    plt.savefig(path)
    plt.close()
    print(f"  [Saved] {path}")


# -- Figure 2: Accuracy comparison bar chart -----------------------------------
def generate_accuracy_bar_chart(df):
    """Grouped bar chart: accuracy per model for selected window sizes."""
    selected_W = [300, 500, 600, 800]
    models_order = ["KAN", "MLP", "DT", "RF", "SVM", "NB", "KNN", "GBC", "LR"]

    fig, ax = plt.subplots(figsize=(10, 5))
    n_models = len(models_order)
    n_groups = len(selected_W)
    bar_width = 0.8 / n_models
    x = np.arange(n_groups)

    for i, model in enumerate(models_order):
        means = []
        stds  = []
        for W in selected_W:
            subset = df[(df["Model"] == model) & (df["W"] == W)]
            if len(subset) > 0:
                means.append(subset["Accuracy"].mean())
                stds.append(subset["Accuracy"].std())
            else:
                means.append(0)
                stds.append(0)
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width * 0.9, yerr=stds,
               label=model, color=COLORS.get(model, "#333"),
               edgecolor="white", linewidth=0.5, capsize=2)

    ax.set_xlabel("Window Size (samples)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"W={w}" for w in selected_W])
    ax.set_ylim(85, 101)
    ax.legend(ncol=3, loc="lower right", framealpha=0.9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "accuracy_comparison_bar.png")
    plt.savefig(path)
    plt.close()
    print(f"  [Saved] {path}")


# -- Figure 3: Accuracy vs window size line chart ------------------------------
def generate_accuracy_line_chart(df):
    """Line chart showing accuracy vs window size for each model."""
    models_order = ["KAN", "MLP", "SVM", "RF", "LR"]
    markers = {"KAN": "o", "MLP": "s", "SVM": "^", "RF": "D", "LR": "v"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in models_order:
        means = []
        stds  = []
        ws    = []
        for W in WINDOWS:
            subset = df[(df["Model"] == model) & (df["W"] == W)]
            if len(subset) > 0:
                means.append(subset["Accuracy"].mean())
                stds.append(subset["Accuracy"].std())
                ws.append(W)
        if ws:
            ax.errorbar(ws, means, yerr=stds,
                        label=model, marker=markers.get(model, "o"),
                        color=COLORS.get(model, "#333"),
                        linewidth=1.8, markersize=6, capsize=3)

    ax.set_xlabel("Window Size (samples)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. Window Size")
    ax.set_xticks(WINDOWS)
    ax.set_ylim(92, 100.5)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "accuracy_vs_window_line.png")
    plt.savefig(path)
    plt.close()
    print(f"  [Saved] {path}")


# -- Statistical significance tests -------------------------------------------
def run_statistical_tests(df, W=PRIMARY_W):
    """Run paired t-test and Wilcoxon signed-rank test: KAN vs each baseline."""
    kan_accs = df[(df["Model"] == "KAN") & (df["W"] == W)].sort_values("Fold")["Accuracy"].values
    if len(kan_accs) == 0:
        print(f"  [SKIP] No KAN results for W={W}.")
        return

    other_models = [m for m in df["Model"].unique() if m != "KAN"]
    rows = []

    for other in sorted(other_models):
        other_accs = df[(df["Model"] == other) & (df["W"] == W)].sort_values("Fold")["Accuracy"].values
        if len(other_accs) != len(kan_accs):
            continue
        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(kan_accs, other_accs)
        # Wilcoxon signed-rank (may fail with n < 6)
        try:
            w_stat, w_pval = stats.wilcoxon(kan_accs, other_accs)
        except ValueError:
            w_stat, w_pval = np.nan, np.nan

        diff_mean = np.mean(kan_accs - other_accs)
        rows.append({
            "Comparison": f"KAN vs {other}",
            "KAN_Mean": f"{np.mean(kan_accs):.2f}",
            "Other_Mean": f"{np.mean(other_accs):.2f}",
            "Mean_Diff": f"{diff_mean:+.2f}",
            "t_statistic": f"{t_stat:.4f}",
            "t_p_value": f"{t_pval:.4f}",
            "t_significant": "Yes" if t_pval < 0.05 else "No",
            "Wilcoxon_p_value": f"{w_pval:.4f}" if not np.isnan(w_pval) else "N/A",
        })

    test_df = pd.DataFrame(rows)
    path = os.path.join(FIG_DIR, "statistical_tests.csv")
    test_df.to_csv(path, index=False)
    print(f"  [Saved] {path}")
    print(f"\n  Statistical Tests (KAN vs others, W={W}):")
    print(test_df.to_string(index=False))
    return test_df


# -- LaTeX tables --------------------------------------------------------------
def generate_latex_tables(df):
    """Generate LaTeX-formatted tables for the paper."""
    lines = []
    lines.append("%" + "="*70)
    lines.append("% TABLE I: Classification Accuracy (Mean +/- Std)")
    lines.append("%" + "="*70)
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{Classification Accuracy (\%) for All Models Across Window Sizes}")
    lines.append(r"\label{tab:accuracy}")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l" + "c" * len(WINDOWS) + "}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Model} & " + " & ".join([f"\\textbf{{W={w}}}" for w in WINDOWS]) + r" \\")
    lines.append(r"\hline")

    models_order = ["KAN", "MLP", "DT", "RF", "SVM", "NB", "KNN", "GBC", "LR"]
    for model in models_order:
        cells = []
        for W in WINDOWS:
            subset = df[(df["Model"] == model) & (df["W"] == W)]
            if len(subset) > 0:
                m = subset["Accuracy"].mean()
                s = subset["Accuracy"].std()
                # Bold the best result per column
                cell = f"{m:.2f}$\\pm${s:.2f}"
            else:
                cell = "--"
            cells.append(cell)
        row_label = f"\\textbf{{{model}}}" if model == "KAN" else model
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")
        if model == "MLP":
            lines.append(r"\hline")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    # TABLE II: F1-Score
    lines.append("%" + "="*70)
    lines.append("% TABLE II: F1-Score (Mean +/- Std)")
    lines.append("%" + "="*70)
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{F1-Score (\%) for All Models Across Window Sizes}")
    lines.append(r"\label{tab:f1}")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l" + "c" * len(WINDOWS) + "}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Model} & " + " & ".join([f"\\textbf{{W={w}}}" for w in WINDOWS]) + r" \\")
    lines.append(r"\hline")

    for model in models_order:
        cells = []
        for W in WINDOWS:
            subset = df[(df["Model"] == model) & (df["W"] == W)]
            if len(subset) > 0:
                m = subset["F1"].mean()
                s = subset["F1"].std()
                cell = f"{m:.2f}$\\pm${s:.2f}"
            else:
                cell = "--"
            cells.append(cell)
        row_label = f"\\textbf{{{model}}}" if model == "KAN" else model
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")
        if model == "MLP":
            lines.append(r"\hline")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    text = "\n".join(lines)
    path = os.path.join(FIG_DIR, "latex_tables.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"  [Saved] {path}")
    return text


# -- Main ----------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 60)
    print("  06_paper_figures.py — Generating publication-ready outputs")
    print("=" * 60)

    # Load data
    print("\nLoading per-fold detailed results...")
    df = load_detailed_results()
    print(f"  Total rows: {len(df)}, Models: {sorted(df['Model'].unique())}")
    print(f"  Windows: {sorted(df['W'].unique())}")

    # Confusion matrices
    print(f"\nGenerating confusion matrices (W={PRIMARY_W})...")
    generate_confusion_matrices(df, W=PRIMARY_W)

    # Accuracy comparison bar chart
    print("\nGenerating accuracy comparison bar chart...")
    generate_accuracy_bar_chart(df)

    # Accuracy vs window size line chart
    print("\nGenerating accuracy vs. window size line chart...")
    generate_accuracy_line_chart(df)

    # Statistical significance tests
    print(f"\nRunning statistical significance tests (W={PRIMARY_W})...")
    run_statistical_tests(df, W=PRIMARY_W)

    # LaTeX tables
    print("\nGenerating LaTeX tables...")
    generate_latex_tables(df)

    print(f"\n{'='*60}")
    print(f"  All paper figures saved to {FIG_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
