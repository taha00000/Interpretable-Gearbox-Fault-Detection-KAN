import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── Cell 0: Title ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "# KAN Symbolic Health Equation Discovery for Gearbox Fault Detection\n"
    "\n"
    "**Core idea:** A KAN autoencoder is trained on *healthy data only* (no fault labels). "
    "After training, the learned B-spline activations on each edge are extracted and fitted "
    "to explicit symbolic functions (linear, sigmoid, log, etc.) via BIC-optimal curve fitting. "
    "These compose into human-readable **health equations** — e.g. "
    "`z₃ ≈ +0.82·σ(S2_kurt) + 0.41·log|S1_rms| − 0.17`. "
    "Anomaly detection is then dual-scored:\n"
    "- **Score A**: standard per-feature reconstruction MSE\n"
    "- **Score B** *(novel)*: how badly the test point violates the symbolic health equations\n"
    "- **Score C**: equal-weight combination of A and B\n\n"
    "Unlike weight-norm feature importance, Score B produces an interpretable, "
    "per-equation fault attribution that can be validated against gearbox physics.\n\n"
    "This pipeline deliberately avoids the TwoNN → bottleneck → feature-count argument "
    "from the companion notebook; the bottleneck size `B` is a tunable hyperparameter."
))

# ── Cell 1: Setup & Imports ───────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Setup & Imports"))
cells.append(nbf.v4.new_code_cell("""\
%pip install pandas numpy torch scikit-learn scipy matplotlib nbformat -q
import os, sys, glob, warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from IPython.display import display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR      = os.path.abspath(".")
DATASET_DIR   = os.path.join(BASE_DIR, "Gearbox Dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR   = os.path.join(BASE_DIR, "results_kan_symbolic")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOWS      = [300, 400, 500, 600, 700, 800, 900, 1000]
SEED         = 42
B            = 8        # bottleneck size — fixed hyperparameter, NOT from TwoNN
LAMBDA_REG   = 1e-4     # spline L1 regularisation weight
PRUNE_ALPHA  = 0.05     # edges below PRUNE_ALPHA * max_norm are pruned
EPOCHS       = 150
ABL_EPOCHS   = 100      # reduced epochs for ablation variants
np.random.seed(SEED)
torch.manual_seed(SEED)

sys.path.insert(0, os.path.join(BASE_DIR, "efficient_kan"))
from efficient_kan import KAN

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {dev}")
print(f"B={B}, lambda={LAMBDA_REG}, alpha={PRUNE_ALPHA}, epochs={EPOCHS}")
"""))

# ── Cell 2: Feature Extraction markdown ───────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "## 1) Feature Extraction (44 Features × 8 Window Sizes)\n\n"
    "11 statistical features × 4 sensor channels (S1–S4). "
    "Files are cached in `data/processed/features_W{W}.csv`."
))

# ── Cell 3: Feature Extraction code ───────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
FEATURE_NAMES = ["mean", "rms", "std", "var", "skew", "kurtosis", "p2p",
                 "crest", "shape", "margin", "impulse"]
CHANNELS = ["S1", "S2", "S3", "S4"]

def compute_11_features(signal):
    rms_val = np.sqrt(np.mean(signal**2))
    margin_denom = np.mean(np.sqrt(np.abs(signal)))**2
    return [
        np.mean(signal), rms_val, np.std(signal), np.var(signal),
        skew(signal), kurtosis(signal), np.ptp(signal),
        np.max(np.abs(signal)) / rms_val if rms_val > 0 else 0,
        rms_val / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0,
        np.max(np.abs(signal)) / margin_denom if margin_denom > 0 else 0,
        np.max(np.abs(signal)) / np.mean(np.abs(signal)) if np.mean(np.abs(signal)) > 0 else 0,
    ]

def extract_features(W):
    all_data = []
    filepaths = glob.glob(os.path.join(DATASET_DIR, "**", "*.txt"), recursive=True)
    for filepath in filepaths:
        fname = os.path.basename(filepath).lower()
        if "healthy" in fname:   label = 0
        elif "broken" in fname:  label = 1
        else:                    continue
        load_val = 0
        for l in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            if f"{l}hz" in fname or f"{l}load" in fname:
                load_val = l
        try:
            df = pd.read_csv(filepath, sep='\\t', header=None)
            if df.shape[1] < 4:
                df = pd.read_csv(filepath, sep=',', header=None)
            series = df.iloc[:, :4].values
        except:
            continue
        for start in range(0, series.shape[0] - W + 1, W):
            row = []
            for c in range(4):
                row.extend(compute_11_features(series[start:start+W, c]))
            row.extend([load_val, label])
            all_data.append(row)
    cols = [f"{ch}_{f}" for ch in CHANNELS for f in FEATURE_NAMES] + ["load", "label"]
    pd.DataFrame(all_data, columns=cols).to_csv(
        os.path.join(PROCESSED_DIR, f"features_W{W}.csv"), index=False)

for W in WINDOWS:
    if not os.path.exists(os.path.join(PROCESSED_DIR, f"features_W{W}.csv")):
        print(f"Extracting W={W}...")
        extract_features(W)
print("Feature extraction complete (or cached).")

# Load reference dataset and determine feature columns
df_1000 = pd.read_csv(os.path.join(PROCESSED_DIR, "features_W1000.csv"))
feat_cols  = [c for c in df_1000.columns if c not in ['load', 'label']]
n_features = len(feat_cols)
print(f"Feature columns ({n_features}): {feat_cols[:5]} ... {feat_cols[-5:]}")
"""))

# ── Cell 4: Stage 1 markdown ───────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "## Stage 1 — KAN Autoencoder Training (Healthy Data Only)\n\n"
    "Architecture: **[44 → 22 → B → 22 → 44]** with B=8 (fixed hyperparameter).  \n"
    "Loss: MSE(recon, x) + λ · regularization_loss()  \n"
    "The L1 + entropy spline regularisation from `efficient_kan` encourages smooth, "
    "symbolisable B-splines. Only reconstruction MSE is logged in the training curve — "
    "the regularisation term is kept separate so the curve shows pure reconstruction quality."
))

# ── Cell 5: KAN-AE Training ────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
df_healthy = df_1000[df_1000['label'] == 0].copy()
scaler_ae  = MinMaxScaler()
X_healthy  = scaler_ae.fit_transform(df_healthy[feat_cols].values).astype(np.float32)
print(f"Healthy samples: {X_healthy.shape[0]},  Features: {n_features},  Bottleneck B={B}")

# Build KAN-AE [44 -> 22 -> B -> 22 -> 44]
kan_ae = KAN(
    layers_hidden=[n_features, n_features // 2, B, n_features // 2, n_features],
    grid_size=5, spline_order=3
)
assert len(kan_ae.layers) == 4, "Expected 4 KANLinear layers"

Xt     = torch.tensor(X_healthy, dtype=torch.float32)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(Xt, Xt), batch_size=256, shuffle=True
)
opt       = torch.optim.Adam(kan_ae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

losses = []
kan_ae.train()
for epoch in range(EPOCHS):
    epoch_mse = 0.0
    for bx, bt in loader:
        opt.zero_grad()
        recon = kan_ae(bx)
        mse   = criterion(recon, bt)
        reg   = kan_ae.regularization_loss()
        loss  = mse + LAMBDA_REG * reg
        loss.backward()
        opt.step()
        epoch_mse += mse.item() * bx.size(0)  # log pure MSE only
    avg = epoch_mse / len(Xt)
    losses.append(avg)
    if (epoch + 1) % 30 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  MSE = {avg:.6f}")

# Save
torch.save(kan_ae.state_dict(), os.path.join(RESULTS_DIR, "kan_ae_symbolic.pt"))

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(losses, linewidth=1.5, color='#e74c3c')
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (reconstruction only)")
ax.set_title(f"KAN-AE Training Loss  [B={B}, λ={LAMBDA_REG}]")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "kan_ae_symbolic_loss.png"), dpi=150)
plt.show()
print(f"Final training MSE: {losses[-1]:.6f}")
"""))

# ── Cell 6: Stage 2 markdown ───────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "## Stage 2 — Edge Pruning & Dependency Graph\n\n"
    "For each edge (out_neuron j, in_feature i) in the first encoder layer (44→22), "
    "compute a combined importance norm:\n\n"
    "    combined_norm[j,i] = mean|scaled_spline_weight[j,i,:]| + |base_weight[j,i]|\n\n"
    "Edges below `PRUNE_ALPHA × max_norm` are pruned. The surviving sparse graph "
    "represents which feature→encoder connections the KAN found informative for "
    "reconstructing healthy data.  \n\n"
    "**Note:** We always use `layer.scaled_spline_weight` (= `spline_weight × spline_scaler`), "
    "not raw `spline_weight`, since `efficient_kan` has a standalone per-edge scale factor."
))

# ── Cell 7: Edge Pruning ───────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
kan_ae.eval()
kan_ae.cpu()   # keep on CPU for all subsequent analysis

layer0 = kan_ae.layers[0]  # KANLinear(44 -> 22)

# Use scaled_spline_weight (accounts for spline_scaler)
sw = layer0.scaled_spline_weight.detach().cpu()  # (22, 44, 8)
bw = layer0.base_weight.detach().cpu()            # (22, 44)

edge_norm    = sw.abs().mean(dim=2)                # (22, 44): mean over coeffs
combined_norm = edge_norm + bw.abs()               # (22, 44): combined importance

threshold     = PRUNE_ALPHA * combined_norm.max().item()
mask          = (combined_norm > threshold).numpy()  # (22, 44) bool

n_total      = 22 * 44
n_surviving  = int(mask.sum())
print(f"Pruning threshold: {threshold:.5f}")
print(f"Surviving edges:   {n_surviving}/{n_total}  ({100*n_surviving/n_total:.1f}%)")

# Which features have the most surviving connections
feat_edge_counts = mask.sum(axis=0)   # (44,)
feature_imp_df = pd.DataFrame({
    'feature':        feat_cols,
    'surviving_edges': feat_edge_counts,
    'mean_norm':       combined_norm.numpy().mean(axis=0),
}).sort_values('surviving_edges', ascending=False).reset_index(drop=True)
print("\\nTop 15 features by surviving encoder edges:")
display(feature_imp_df.head(15))
feature_imp_df.to_csv(os.path.join(RESULTS_DIR, "edge_feature_importance.csv"), index=False)

# Build the list of surviving (j, i) pairs
surviving_edges = [(j, i) for j in range(22) for i in range(44) if mask[j, i]]
print(f"\\nTotal edges for symbolic regression: {len(surviving_edges)}")

# 2-panel heatmap
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

im0 = axes[0].imshow(combined_norm.numpy(), aspect='auto', cmap='hot')
axes[0].set_title("Combined Edge Norm  (22 neurons × 44 features)", fontsize=11)
axes[0].set_xlabel("Input Feature Index"); axes[0].set_ylabel("Encoder Neuron")
plt.colorbar(im0, ax=axes[0], label='norm')

im1 = axes[1].imshow(mask.astype(float), aspect='auto', cmap='Blues')
axes[1].set_title(f"Surviving Edges  (α={PRUNE_ALPHA}): {n_surviving}/{n_total}", fontsize=11)
step = 4
axes[1].set_xticks(range(0, 44, step))
axes[1].set_xticklabels(feat_cols[::step], rotation=90, fontsize=7)
axes[1].set_xlabel("Input Feature"); axes[1].set_ylabel("Encoder Neuron")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "edge_pruning_heatmap.png"), dpi=150)
plt.show()
"""))

# ── Cell 8: Stage 3 markdown ───────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "## Stage 3 — Per-Edge Symbolic Regression\n\n"
    "For each surviving edge (j, i):\n"
    "1. **Sample the spline** — construct a `(200, 44)` input tensor with feature `i` swept "
    "over `[0, 1]` (MinMax-scaled range) and all other features held at 0.5. "
    "Evaluate the KANLinear forward for that edge only (spline + base contributions).\n"
    "2. **Fit 6 symbolic candidates** (linear, quadratic, sigmoid, sqrt, log, constant) "
    "via `scipy.optimize.curve_fit` with `p0 = 0.1·1`.\n"
    "3. **Select by BIC** — `BIC = n·log(MSE) + k·log(n)` — to penalise unnecessary complexity.\n\n"
    "Symbolisation quality is reported as the fraction of edges with R² > 0.95."
))

# ── Cell 9: Spline Sampling Helpers ───────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ── Spline edge sampler ───────────────────────────────────────────────────
def sample_spline_edge(layer, out_j, in_i, n_pts=200):
    layer.eval()
    x_vals = np.linspace(0.0, 1.0, n_pts)   # data is MinMax-scaled to [0,1]
    # Build (n_pts, in_features) tensor; all other features at 0.5 (midpoint)
    x_tensor = torch.full((n_pts, layer.in_features), 0.5, dtype=torch.float32)
    x_tensor[:, in_i] = torch.tensor(x_vals, dtype=torch.float32)
    with torch.no_grad():
        bases = layer.b_splines(x_tensor)            # (n_pts, in_features, n_coeff)
        coeff = layer.scaled_spline_weight[out_j, in_i, :]  # (n_coeff,)
        y_spline = (bases[:, in_i, :] * coeff.unsqueeze(0)).sum(dim=1).numpy()
        # Base contribution: base_weight[j,i] * SiLU(x_i)
        x_col   = x_tensor[:, in_i]
        silu_out = nn.SiLU()(x_col).numpy()
        y_base  = layer.base_weight[out_j, in_i].item() * silu_out
    return x_vals, y_spline + y_base

# ── Symbolic candidate functions ──────────────────────────────────────────
def sym_linear(x, a, b):        return a * x + b
def sym_quadratic(x, a, b, c):  return a * x**2 + b * x + c
def sym_sigmoid(x, a, b, c):
    return a / (1.0 + np.exp(-np.clip(b * (x - c), -100, 100)))
def sym_sqrt(x, a, b):          return a * np.sqrt(np.abs(x) + 1e-8) + b
def sym_log(x, a, b):           return a * np.log(np.abs(x) + 1e-8) + b
def sym_constant(x, a):         return np.full_like(x, a, dtype=float)

CANDIDATES = [
    ("linear",    sym_linear,    2),
    ("quadratic", sym_quadratic, 3),
    ("sigmoid",   sym_sigmoid,   3),
    ("sqrt",      sym_sqrt,      2),
    ("log",       sym_log,       2),
    ("constant",  sym_constant,  1),
]
CANDIDATES_DICT = {name: fn for name, fn, _ in CANDIDATES}

def bic_score(n, k, mse):
    mse = max(mse, 1e-12)
    return n * np.log(mse) + k * np.log(n)

def fit_symbolic(x_vals, y_vals):
    n      = len(x_vals)
    y_mean = y_vals.mean()
    ss_tot = ((y_vals - y_mean)**2).sum() + 1e-12
    best   = {"best_symbol": "constant", "params": [y_mean],
              "mse": ((y_vals - y_mean)**2).mean(), "r2": 0.0,
              "bic": bic_score(n, 1, ((y_vals - y_mean)**2).mean())}
    for sym_name, sym_fn, k in CANDIDATES:
        try:
            popt, _ = curve_fit(sym_fn, x_vals, y_vals,
                                p0=np.ones(k) * 0.1, maxfev=5000)
            y_pred = sym_fn(x_vals, *popt)
            mse    = float(np.mean((y_vals - y_pred)**2))
            bic    = bic_score(n, k, mse)
            r2     = float(1.0 - ((y_vals - y_pred)**2).sum() / ss_tot)
            if bic < best["bic"]:
                best = {"best_symbol": sym_name, "params": popt.tolist(),
                        "mse": mse, "r2": r2, "bic": bic}
        except Exception:
            continue
    return best

print(f"Helpers defined. Candidates: {[c[0] for c in CANDIDATES]}")
"""))

# ── Cell 10: Run Symbolic Regression ──────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
symbolic_results = []
n_edges = len(surviving_edges)
print(f"Running symbolic regression on {n_edges} surviving edges...")

for idx, (j, i) in enumerate(surviving_edges):
    x_vals, y_vals = sample_spline_edge(layer0, j, i, n_pts=200)
    result = fit_symbolic(x_vals, y_vals)
    symbolic_results.append({
        "out_neuron":   j,
        "in_feat_idx":  i,
        "in_feature":   feat_cols[i],
        "best_symbol":  result["best_symbol"],
        "params":       result["params"],
        "r2":           result["r2"],
        "fit_mse":      result["mse"],
        "bic":          result["bic"],
    })
    if (idx + 1) % 100 == 0:
        print(f"  {idx+1}/{n_edges} edges processed...")

sym_df = pd.DataFrame(symbolic_results)
sym_df.to_csv(os.path.join(RESULTS_DIR, "symbolic_regression_results.csv"), index=False)

print("\\nSymbol type distribution:")
display(sym_df["best_symbol"].value_counts().rename("count").to_frame())

print("\\nR² statistics:")
display(sym_df["r2"].describe().round(4).to_frame())

qual = float((sym_df["r2"] > 0.95).mean() * 100)
print(f"\\nSymbolisation quality (R²>0.95): {qual:.1f}%")
print(f"Mean R²: {sym_df['r2'].mean():.4f}")
"""))

# ── Cell 11: Visualise Top-10 Fits ─────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
top10 = sym_df.nlargest(10, "r2").reset_index(drop=True)

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
axes = axes.flatten()

for idx, row in top10.iterrows():
    j, i = int(row["out_neuron"]), int(row["in_feat_idx"])
    x_vals, y_vals = sample_spline_edge(layer0, j, i, n_pts=200)
    ax = axes[idx]
    ax.scatter(x_vals, y_vals, s=8, alpha=0.4, color='#3498db', label='B-spline')
    sym_fn = CANDIDATES_DICT[row["best_symbol"]]
    y_fit  = sym_fn(x_vals, *row["params"])
    ax.plot(x_vals, y_fit, 'r-', linewidth=2,
            label=f'{row["best_symbol"]}  R²={row["r2"]:.3f}')
    ax.set_title(f'n{j} ← {row["in_feature"]}', fontsize=8)
    ax.legend(fontsize=6); ax.set_xlabel("x (MinMax scaled)", fontsize=7)
    ax.set_ylabel("f(x)", fontsize=7)

plt.suptitle("Top-10 B-spline Edges vs Best Symbolic Fit", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "top10_symbolic_fits.png"), dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── Cell 12: Stage 4 markdown ──────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "## Stage 4 — Health Equation Extraction\n\n"
    "The encoder first layer maps 44 features → 22 intermediate neurons. "
    "For each neuron j, we collect all surviving symbolic edge functions and compose them into:\n\n"
    "    z_j  ≈  Σ_i  g_{ji}(feature_i)\n\n"
    "where each `g_{ji}` is now a named symbolic form (e.g. sigmoid, log, linear). "
    "These are the **health equations** — mathematical relationships the KAN learned "
    "hold in the healthy gearbox state.  \n"
    "Violations of these equations on test data form Score B."
))

# ── Cell 13: Health Equation Display ──────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
assert len(kan_ae.layers) == 4, "Expected 4 layers: [44->22, 22->B, B->22, 22->44]"

def _fmt_term(sym_name, params, feat_name):
    p = params
    if sym_name == "linear":
        return f"{p[0]:+.3f}*{feat_name} {p[1]:+.3f}"
    elif sym_name == "quadratic":
        return f"{p[0]:+.3f}*{feat_name}^2 {p[1]:+.3f}*{feat_name} {p[2]:+.3f}"
    elif sym_name == "sigmoid":
        return f"{p[0]:+.3f}*sigma({p[1]:.3f}*({feat_name} {p[2]:+.3f}))"
    elif sym_name == "sqrt":
        return f"{p[0]:+.3f}*sqrt|{feat_name}| {p[1]:+.3f}"
    elif sym_name == "log":
        return f"{p[0]:+.3f}*log|{feat_name}| {p[1]:+.3f}"
    else:  # constant
        return f"{p[0]:+.3f}"

health_equations = {}
print("=" * 72)
print("HEALTH EQUATIONS  (KAN encoder: 44 features → 22 intermediate neurons)")
print("=" * 72)

for j in range(22):
    edges_j = sym_df[sym_df["out_neuron"] == j]
    if len(edges_j) == 0:
        eq = f"z_{j:02d}  =  0  (no surviving edges)"
    else:
        terms = [_fmt_term(r["best_symbol"], r["params"], r["in_feature"])
                 for _, r in edges_j.iterrows()]
        eq = f"z_{j:02d}  ≈  " + "  +  ".join(terms)
    health_equations[j] = eq
    print(eq)

print("=" * 72)
print(f"\\nSymbolisation quality (R²>0.95): {qual:.1f}%")
print(f"Neurons with ≥1 surviving edge:  {len(sym_df['out_neuron'].unique())}/22")
"""))

# ── Cell 14: Stage 5 markdown ──────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "## Stage 5 — Dual Anomaly Scoring\n\n"
    "Three complementary scores — all unsupervised, no fault labels:\n\n"
    "| Score | Formula | Interpretable? |\n"
    "|---|---|---|\n"
    "| **A** — Reconstruction error | mean((x − KAN_AE(x))²) per feature | No |\n"
    "| **B** — Symbolic violation | Σ_j (h1_j − Σ_i g_{ji}(x_i))² | **Yes — per equation** |\n"
    "| **C** — Combined | 0.5·minmax(A) + 0.5·minmax(B) | Partial |\n\n"
    "Scaling discipline:\n"
    "- KAN-AE training: `MinMaxScaler` fit on **all healthy data** in `df_1000`\n"
    "- Evaluation split: `MinMaxScaler` refit on **training healthy split only** "
    "(70%) to prevent test-set leakage"
))

# ── Cell 15: Data Split ────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
df_all     = pd.read_csv(os.path.join(PROCESSED_DIR, "features_W1000.csv"))
df_h_all   = df_all[df_all['label'] == 0]
df_broken  = df_all[df_all['label'] == 1]

train_h, test_h = train_test_split(df_h_all, test_size=0.3, random_state=SEED)
test_df  = pd.concat([test_h, df_broken]).reset_index(drop=True)
y_test   = test_df['label'].values

# Scaler refit on training healthy split only — no leakage
scaler_test = MinMaxScaler().fit(train_h[feat_cols].values)
X_train_np  = scaler_test.transform(train_h[feat_cols].values).astype(np.float32)
X_test_np   = scaler_test.transform(test_df[feat_cols].values).astype(np.float32)

print(f"Train healthy:  {len(train_h)}")
print(f"Test  healthy:  {len(test_h)}")
print(f"Test  broken:   {len(df_broken)}")
print(f"Test  total:    {len(test_df)}  (label distribution: {np.bincount(y_test)})")
"""))

# ── Cell 16: Score A — Reconstruction Error ────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
kan_ae.eval()
X_test_t = torch.tensor(X_test_np, dtype=torch.float32)

with torch.no_grad():
    recon_test = kan_ae(X_test_t).numpy()

recon_error = (X_test_np - recon_test) ** 2   # (n_test, 44)
score_A     = recon_error.mean(axis=1)          # (n_test,)

auc_A = roc_auc_score(y_test, score_A)
print(f"Score A (KAN-AE Reconstruction)  AUC = {auc_A:.4f}")
"""))

# ── Cell 17: Score B — Symbolic Equation Violation ────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# Forward through encoder layer 0 only to get h1 (22-dim intermediate)
with torch.no_grad():
    h1_t  = kan_ae.layers[0](X_test_t)   # (n_test, 22)
    h1_np = h1_t.numpy()

# Symbolic prediction of h1 from scaled test features
sym_h1_pred = np.zeros((len(test_df), 22), dtype=np.float32)

for _, row in sym_df.iterrows():
    j   = int(row["out_neuron"])
    i   = int(row["in_feat_idx"])
    fn  = CANDIDATES_DICT[row["best_symbol"]]
    try:
        sym_h1_pred[:, j] += fn(X_test_np[:, i], *row["params"]).astype(np.float32)
    except Exception:
        pass   # skip numerically unstable evaluations

# Symbolic violation: squared residual per equation, summed
sym_residuals = (h1_np - sym_h1_pred) ** 2   # (n_test, 22)
score_B       = sym_residuals.sum(axis=1)      # (n_test,)

auc_B = roc_auc_score(y_test, score_B)
print(f"Score B (Symbolic Equation Violation)  AUC = {auc_B:.4f}")

if auc_B < 0.6:
    print("  [NOTE] Score B AUC is low — symbolic approximation quality may be "
          "insufficient. Check the ablation study (Stage 8) for diagnostics.")
"""))

# ── Cell 18: Score C + Score Distributions ────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
def minmax_norm(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)

score_C = 0.5 * minmax_norm(score_A) + 0.5 * minmax_norm(score_B)
auc_C   = roc_auc_score(y_test, score_C)

print(f"Score A (Reconstruction):  AUC = {auc_A:.4f}")
print(f"Score B (Symbolic):        AUC = {auc_B:.4f}")
print(f"Score C (Combined):        AUC = {auc_C:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
score_info = [
    (score_A, f"Score A — Recon\\nAUC={auc_A:.4f}"),
    (score_B, f"Score B — Symbolic\\nAUC={auc_B:.4f}"),
    (score_C, f"Score C — Combined\\nAUC={auc_C:.4f}"),
]
for ax, (sc, title) in zip(axes, score_info):
    ax.hist(sc[y_test==0], bins=40, alpha=0.6, label='Healthy', color='#2ecc71')
    ax.hist(sc[y_test==1], bins=40, alpha=0.6, label='Broken',  color='#e74c3c')
    ax.set_title(title, fontsize=10); ax.legend(fontsize=9)
    ax.set_xlabel("Score"); ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "score_distributions.png"), dpi=150)
plt.show()
"""))

# ── Cell 19: Stage 6 markdown ──────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "## Stage 6 — Full Benchmark: 9 Baselines + 3 KAN-AE Scores\n\n"
    "**12 methods × 8 window sizes** AUC table.\n\n"
    "- The 9 classic unsupervised detectors (IsolationForest, OC-SVM, LOF, PatchCore, "
    "Autoencoder, VAE, DeepSVDD, TeacherStudent, SSL_DAE) are re-trained from scratch "
    "for each window size using `StandardScaler`.\n"
    "- The 3 KAN-AE scores use the **single W=1000 model** evaluated across all windows "
    "(cross-window generalisation). The W=1000 `MinMaxScaler` is applied to other-window "
    "data — values may fall slightly outside [0,1] for small windows, which is expected "
    "and mirrors realistic deployment."
))

# ── Cell 20: Detector Definitions ─────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# ── Deep-learning baseline architectures ─────────────────────────────────
class AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,8), nn.ReLU())
        self.d = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self, x): return self.d(self.e(x))

class VAENet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.e   = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU())
        self.mu  = nn.Linear(16,8); self.var = nn.Linear(16,8)
        self.d   = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self, x):
        h = self.e(x); m, lv = self.mu(h), self.var(h)
        return self.d(m + torch.randn_like(torch.exp(0.5*lv)) * torch.exp(0.5*lv)), m, lv

class DeepSVDD(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,16))
    def forward(self, x): return self.n(x)

class TSNetwork(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16))
    def forward(self, x): return self.n(x)

class DAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,8))
        self.d = nn.Sequential(nn.Linear(8,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self, x): return self.d(self.e(x))

def k_center(f, frac=0.1):
    c = [np.random.randint(0, f.shape[0])]
    dists = pairwise_distances(f, f[c], metric='euclidean').flatten()
    for _ in range(max(1, int(f.shape[0]*frac)) - 1):
        idx = np.argmax(dists); c.append(idx)
        dists = np.minimum(dists, pairwise_distances(f, f[[idx]]).flatten())
    return f[c]

def run_baseline_detectors(windows=WINDOWS):
    names = ["IsolationForest","OC-SVM","LOF","PatchCore",
             "Autoencoder","VAE","DeepSVDD","TeacherStudent","SSL_DAE"]
    res = pd.DataFrame(index=names, columns=windows, dtype=float)
    for W in windows:
        df   = pd.read_csv(os.path.join(PROCESSED_DIR, f"features_W{W}.csv"))
        fc   = [c for c in df.columns if c not in ['load','label']]
        tr,te = train_test_split(df[df['label']==0], test_size=0.3, random_state=SEED)
        te    = pd.concat([te, df[df['label']==1]])
        sc    = StandardScaler().fit(tr[fc].values)
        X_tr, X_te = sc.transform(tr[fc].values), sc.transform(te[fc].values)
        y_te  = te['label'].values
        res.loc["IsolationForest",W] = roc_auc_score(y_te,
            -IsolationForest(n_estimators=100,random_state=SEED).fit(X_tr).decision_function(X_te))
        res.loc["OC-SVM",W] = roc_auc_score(y_te,
            -OneClassSVM(kernel="rbf",gamma="scale",nu=0.05).fit(X_tr).decision_function(X_te))
        res.loc["LOF",W] = roc_auc_score(y_te,
            -LocalOutlierFactor(n_neighbors=20,novelty=True).fit(X_tr).decision_function(X_te))
        res.loc["PatchCore",W] = roc_auc_score(y_te,
            pairwise_distances(X_te, k_center(X_tr), metric='euclidean').min(axis=1))
        Xtr_t = torch.tensor(X_tr,dtype=torch.float32)
        Xte_t = torch.tensor(X_te,dtype=torch.float32)
        ldr   = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtr_t,Xtr_t), batch_size=256, shuffle=True)
        d = len(fc)
        ae_b = AE(d); o=torch.optim.Adam(ae_b.parameters(),lr=1e-3)
        for _ in range(15):
            for b,_ in ldr: o.zero_grad(); nn.MSELoss()(ae_b(b),b).backward(); o.step()
        with torch.no_grad(): res.loc["Autoencoder",W]=roc_auc_score(y_te,
            torch.mean((ae_b(Xte_t)-Xte_t)**2,dim=1).cpu())
        vae_b = VAENet(d); o=torch.optim.Adam(vae_b.parameters(),lr=1e-3)
        for _ in range(15):
            for b,_ in ldr:
                o.zero_grad(); x,m,lv=vae_b(b)
                (nn.MSELoss()(x,b)-0.0005*torch.sum(1+lv-m.pow(2)-lv.exp())).backward(); o.step()
        with torch.no_grad(): res.loc["VAE",W]=roc_auc_score(y_te,
            torch.mean((vae_b(Xte_t)[0]-Xte_t)**2,dim=1).cpu())
        svdd = DeepSVDD(d); o=torch.optim.Adam(svdd.parameters(),lr=1e-3)
        with torch.no_grad(): c_svdd=torch.mean(svdd(Xtr_t),dim=0)
        for _ in range(15):
            for b,_ in ldr:
                o.zero_grad(); torch.mean(torch.sum((svdd(b)-c_svdd)**2,dim=1)).backward(); o.step()
        with torch.no_grad(): res.loc["DeepSVDD",W]=roc_auc_score(y_te,
            torch.sum((svdd(Xte_t)-c_svdd)**2,dim=1).cpu())
        t_net=TSNetwork(d); s_net=TSNetwork(d); o=torch.optim.Adam(s_net.parameters(),lr=1e-3)
        for p in t_net.parameters(): p.requires_grad=False
        for _ in range(15):
            for b,_ in ldr: o.zero_grad(); nn.MSELoss()(s_net(b),t_net(b)).backward(); o.step()
        with torch.no_grad(): res.loc["TeacherStudent",W]=roc_auc_score(y_te,
            torch.mean((s_net(Xte_t)-t_net(Xte_t))**2,dim=1).cpu())
        dae_b = DAE(d); o=torch.optim.Adam(dae_b.parameters(),lr=1e-3)
        for _ in range(15):
            for b,_ in ldr:
                o.zero_grad(); nn.MSELoss()(dae_b(b+0.5*torch.randn_like(b)),b).backward(); o.step()
        with torch.no_grad(): res.loc["SSL_DAE",W]=roc_auc_score(y_te,
            torch.mean((dae_b(Xte_t)-Xte_t)**2,dim=1).cpu())
        print(f"  W={W} baselines done.")
    return res

def run_kan_ae_scores(windows=WINDOWS):
    rows = ["KAN-AE-Recon","KAN-AE-Symbolic","KAN-AE-Combined"]
    res  = pd.DataFrame(index=rows, columns=windows, dtype=float)
    kan_ae.eval(); kan_ae.cpu()
    for W in windows:
        df_w = pd.read_csv(os.path.join(PROCESSED_DIR, f"features_W{W}.csv"))
        fc   = [c for c in df_w.columns if c not in ['load','label']]
        _,te_h = train_test_split(df_w[df_w['label']==0], test_size=0.3, random_state=SEED)
        te_w  = pd.concat([te_h, df_w[df_w['label']==1]]).reset_index(drop=True)
        y_w   = te_w['label'].values
        X_te_w = scaler_test.transform(te_w[fc].values).astype(np.float32)
        Xte_t  = torch.tensor(X_te_w, dtype=torch.float32)
        with torch.no_grad():
            recon_w = kan_ae(Xte_t).numpy()
            h1_w    = kan_ae.layers[0](Xte_t).numpy()
        sA = ((X_te_w - recon_w)**2).mean(axis=1)
        sp = np.zeros((len(te_w), 22), dtype=np.float32)
        for _, row in sym_df.iterrows():
            j2 = int(row["out_neuron"]); i2 = int(row["in_feat_idx"])
            if i2 >= X_te_w.shape[1]: continue
            try: sp[:, j2] += CANDIDATES_DICT[row["best_symbol"]](
                    X_te_w[:, i2], *row["params"]).astype(np.float32)
            except Exception: pass
        sB = ((h1_w - sp)**2).sum(axis=1)
        def _n(a):
            lo,hi=a.min(),a.max(); return (a-lo)/(hi-lo+1e-12)
        sC = 0.5*_n(sA) + 0.5*_n(sB)
        res.loc["KAN-AE-Recon",    W] = roc_auc_score(y_w, sA)
        res.loc["KAN-AE-Symbolic", W] = roc_auc_score(y_w, sB)
        res.loc["KAN-AE-Combined", W] = roc_auc_score(y_w, sC)
        print(f"  W={W} KAN-AE done: A={res.loc['KAN-AE-Recon',W]:.3f} "
              f"B={res.loc['KAN-AE-Symbolic',W]:.3f} C={res.loc['KAN-AE-Combined',W]:.3f}")
    return res

print("Benchmark functions defined (9 baselines + 3 KAN-AE scores).")
"""))

# ── Cell 21: Run Full Benchmark ────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
print("Running 9 baseline detectors across 8 window sizes (~5-15 min on CPU)...")
baseline_res = run_baseline_detectors()

print("\\nRunning KAN-AE scores across 8 window sizes...")
kan_res = run_kan_ae_scores()

full_bench     = pd.concat([baseline_res, kan_res])
full_bench_pct = (full_bench * 100).round(2)

print("\\nAUC (%) — All 12 Methods × 8 Window Sizes:")
display(full_bench_pct)
full_bench.to_csv(os.path.join(RESULTS_DIR, "benchmark_auc.csv"))

# Best method per window
best_per_w = full_bench.idxmax(axis=0)
print("\\nBest method per window:")
for W in WINDOWS:
    print(f"  W={W:4d}: {best_per_w[W]:20s}  AUC={full_bench.loc[best_per_w[W],W]*100:.2f}%")
"""))

# ── Cell 22: Benchmark Heatmap ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
fig, ax = plt.subplots(figsize=(13, 8))
vals = full_bench_pct.values.astype(float)
im   = ax.imshow(vals, aspect='auto', cmap='RdYlGn', vmin=50, vmax=100)

ax.set_xticks(range(len(WINDOWS))); ax.set_xticklabels(WINDOWS, fontsize=10)
ax.set_yticks(range(len(full_bench))); ax.set_yticklabels(full_bench.index, fontsize=10)
ax.set_xlabel("Window Size", fontsize=11)
ax.set_title("AUC (%) — Full Benchmark  [12 Methods × 8 Window Sizes]", fontsize=12)

# Annotate cells
for i in range(len(full_bench)):
    for j in range(len(WINDOWS)):
        v = vals[i, j]
        ax.text(j, i, f"{v:.1f}", ha='center', va='center', fontsize=7,
                color='black' if 60 < v < 90 else 'white')

# Separator between 9 baselines and 3 KAN-AE rows
ax.axhline(8.5, color='black', linewidth=2.5, linestyle='--')
ax.text(len(WINDOWS) - 0.5, 8.5, ' KAN-AE →', va='bottom', ha='right',
        fontsize=9, color='black')

plt.colorbar(im, ax=ax, label='AUC (%)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "benchmark_heatmap.png"), dpi=150, bbox_inches='tight')
plt.show()
"""))

# ── Cell 23: Stage 7 markdown ──────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "## Stage 7 — Fault Localization via Equation Violation\n\n"
    "`sym_residuals` has shape `(n_test, 22)` — one violation score per encoder equation. "
    "By comparing healthy vs broken residuals per equation, we identify which equations "
    "(and therefore which input features) are most disrupted by the fault.  \n\n"
    "Since each equation involves specific sensor features, a violation pattern "
    "directly maps to physical sensor channels — **unsupervised fault attribution**, "
    "no fault labels used."
))

# ── Cell 24: Fault Localization ────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
healthy_mask = (y_test == 0)
broken_mask  = (y_test == 1)

mean_viol_healthy = sym_residuals[healthy_mask].mean(axis=0)   # (22,)
mean_viol_broken  = sym_residuals[broken_mask].mean(axis=0)    # (22,)
fault_ratio       = mean_viol_broken / (mean_viol_healthy + 1e-12)  # (22,)

top_k = 10
top_violated = np.argsort(fault_ratio)[::-1][:top_k]

print(f"Top {top_k} most violated equations in broken samples:")
print(f"{'Rank':>4}  {'Eq':>6}  {'Fault Ratio':>12}  {'Features involved'}")
print("-" * 65)
for rank, eq_idx in enumerate(top_violated, 1):
    feeding = sym_df[sym_df["out_neuron"] == eq_idx]["in_feature"].tolist()
    sensors = sorted(set(f.split("_")[0] for f in feeding))
    print(f"  {rank:2d}    z_{eq_idx:02d}   {fault_ratio[eq_idx]:>10.2f}x   "
          f"{', '.join(feeding[:4])}{'...' if len(feeding)>4 else ''}")
    print(f"                            → sensors: {sensors}")

# 2-panel bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

x_pos = np.arange(22); w = 0.35
axes[0].bar(x_pos-w/2, mean_viol_healthy, w, color='#2ecc71', alpha=0.8, label='Healthy')
axes[0].bar(x_pos+w/2, mean_viol_broken,  w, color='#e74c3c', alpha=0.8, label='Broken')
axes[0].set_xlabel("Encoder Neuron (equation index)", fontsize=10)
axes[0].set_ylabel("Mean Squared Violation", fontsize=10)
axes[0].set_title("Per-Equation Violation: Healthy vs Broken", fontsize=11)
axes[0].legend(fontsize=10); axes[0].set_xticks(x_pos); axes[0].set_xticklabels(x_pos, fontsize=7)

top_rev = top_violated[::-1]
axes[1].barh(range(top_k), fault_ratio[top_rev], color='#e74c3c', alpha=0.8)
axes[1].set_yticks(range(top_k))
axes[1].set_yticklabels([f"z_{i:02d}" for i in top_rev], fontsize=9)
axes[1].set_xlabel("Fault Ratio (broken / healthy)", fontsize=10)
axes[1].set_title(f"Top-{top_k} Most Violated Equations", fontsize=11)
axes[1].axvline(1.0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fault_localization.png"), dpi=150)
plt.show()

fault_loc_df = pd.DataFrame({
    "equation":    [f"z_{i:02d}" for i in range(22)],
    "mean_healthy": mean_viol_healthy,
    "mean_broken":  mean_viol_broken,
    "fault_ratio":  fault_ratio,
}).sort_values("fault_ratio", ascending=False)
fault_loc_df.to_csv(os.path.join(RESULTS_DIR, "fault_localization.csv"), index=False)
"""))

# ── Cell 25: Stage 8 markdown ──────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "## Stage 8 — Ablation Study\n\n"
    "Three ablations:\n\n"
    "1. **Regularisation** (λ=0 vs λ=1e-4): Does spline regularisation improve "
    "symbolisation quality (fraction of edges with R² > 0.95)?\n"
    "2. **Bottleneck size** B ∈ {4, 6, 8, 10, 12}: How does B affect reconstruction "
    "AUC (Score A) and symbolisation quality?\n"
    "3. **Scoring method**: Score A vs Score B vs Score C at W=1000."
))

# ── Cell 26: Ablation 1 — Regularisation Effect ───────────────────────────
cells.append(nbf.v4.new_code_cell("""\
print("Ablation 1: λ=0 (no regularisation) vs λ=1e-4...")
kan_ae_noreg = KAN(
    layers_hidden=[n_features, n_features//2, B, n_features//2, n_features],
    grid_size=5, spline_order=3
)
opt2 = torch.optim.Adam(kan_ae_noreg.parameters(), lr=1e-3)
kan_ae_noreg.train()
for epoch in range(ABL_EPOCHS):
    for bx, bt in loader:
        opt2.zero_grad()
        criterion(kan_ae_noreg(bx), bt).backward()   # no reg term
        opt2.step()

kan_ae_noreg.eval()
l0_noreg = kan_ae_noreg.layers[0]
sw_nr    = l0_noreg.scaled_spline_weight.detach().cpu()
bw_nr    = l0_noreg.base_weight.detach().cpu()
cn_nr    = (sw_nr.abs().mean(dim=2) + bw_nr.abs()).numpy()
thr_nr   = PRUNE_ALPHA * cn_nr.max()
mask_nr  = cn_nr > thr_nr
edges_nr = [(j2, i2) for j2 in range(22) for i2 in range(44) if mask_nr[j2, i2]]

r2s_noreg = []
for j2, i2 in edges_nr[:min(150, len(edges_nr))]:
    xv, yv = sample_spline_edge(l0_noreg, j2, i2)
    r2s_noreg.append(fit_symbolic(xv, yv)["r2"])

qual_noreg = float((np.array(r2s_noreg) > 0.95).mean() * 100) if r2s_noreg else 0.0

print(f"\\nAblation 1 Results (symbolisation quality, % R²>0.95):")
print(f"  λ=0    (no reg):   {qual_noreg:.1f}%   n_edges={len(edges_nr)}")
print(f"  λ=1e-4 (with reg): {qual:.1f}%   n_edges={len(surviving_edges)}")
"""))

# ── Cell 27: Ablation 2 — Bottleneck Size ─────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
B_values        = [4, 6, 8, 10, 12]
ablation_b_rows = []

for B_test in B_values:
    print(f"  B={B_test}...", end=" ", flush=True)
    if B_test == B:
        auc_bt = auc_A; q_bt = qual
    else:
        kan_tmp = KAN(
            layers_hidden=[n_features, n_features//2, B_test, n_features//2, n_features],
            grid_size=5, spline_order=3
        )
        opt_t = torch.optim.Adam(kan_tmp.parameters(), lr=1e-3)
        kan_tmp.train()
        for epoch in range(ABL_EPOCHS):
            for bx, bt in loader:
                opt_t.zero_grad()
                mse_t = criterion(kan_tmp(bx), bt)
                (mse_t + LAMBDA_REG * kan_tmp.regularization_loss()).backward()
                opt_t.step()
        kan_tmp.eval()
        Xte_cpu = torch.tensor(X_test_np, dtype=torch.float32)
        with torch.no_grad():
            rec_t = kan_tmp(Xte_cpu).numpy()
        auc_bt = roc_auc_score(y_test, ((X_test_np - rec_t)**2).mean(axis=1))
        l0_t  = kan_tmp.layers[0]
        sw_t  = l0_t.scaled_spline_weight.detach().abs()
        bw_t  = l0_t.base_weight.detach().abs()
        cn_t  = (sw_t.mean(dim=2) + bw_t).numpy()
        thr_t = PRUNE_ALPHA * cn_t.max()
        edges_t = [(j2,i2) for j2 in range(22) for i2 in range(44) if cn_t[j2,i2]>thr_t]
        r2s_t = []
        for j2, i2 in edges_t[:min(100, len(edges_t))]:
            xv, yv = sample_spline_edge(l0_t, j2, i2)
            r2s_t.append(fit_symbolic(xv, yv)["r2"])
        q_bt = float((np.array(r2s_t) > 0.95).mean() * 100) if r2s_t else 0.0
    ablation_b_rows.append({"B": B_test, "AUC_scoreA": auc_bt, "symb_quality_%": q_bt})
    print(f"AUC={auc_bt:.4f}, quality={q_bt:.1f}%")

ablation_b_df = pd.DataFrame(ablation_b_rows).set_index("B")
print("\\nAblation 2: Bottleneck Size")
display(ablation_b_df.round(4))
ablation_b_df.to_csv(os.path.join(RESULTS_DIR, "ablation_bottleneck.csv"))
"""))

# ── Cell 28: Ablation 3 + Summary Plot ────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
# Ablation 3 table
abl3_df = pd.DataFrame({
    "Method":        ["Score A (Recon)", "Score B (Symbolic)", "Score C (Combined)"],
    "AUC W=1000":    [round(auc_A,4), round(auc_B,4), round(auc_C,4)],
    "Interpretable": ["No", "Yes", "Partial"],
})
print("Ablation 3: Scoring Method Comparison")
display(abl3_df)

# 3-panel ablation figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel 1: regularisation
ax = axes[0]
ax.bar(["λ=0\\n(no reg)", "λ=1e-4\\n(with reg)"],
       [qual_noreg, qual], color=['#e74c3c','#2ecc71'], alpha=0.85, edgecolor='grey')
ax.set_ylabel("Symbolisation Quality (% R²>0.95)", fontsize=10)
ax.set_title("Abl. 1: Regularisation Effect", fontsize=11)
ax.set_ylim(0, 110)
for i, v in enumerate([qual_noreg, qual]):
    ax.text(i, v+2, f"{v:.1f}%", ha='center', fontsize=10)

# Panel 2: bottleneck size (dual y-axis)
ax  = axes[1]
ax2 = ax.twinx()
ax.plot(ablation_b_df.index, ablation_b_df["AUC_scoreA"]*100,
        'o-', color='#3498db', label='AUC Score A (%)', linewidth=2, markersize=7)
ax2.plot(ablation_b_df.index, ablation_b_df["symb_quality_%"],
         's--', color='#e74c3c', label='Symb. Quality (%)', linewidth=2, markersize=7)
ax.axvline(B, color='gray', linestyle=':', linewidth=1.5, label=f'B={B} (default)')
ax.set_xlabel("Bottleneck B", fontsize=10); ax.set_ylabel("AUC (%)", fontsize=10)
ax2.set_ylabel("Symbolisation Quality (%)", fontsize=10)
ax.set_title("Abl. 2: Bottleneck Size", fontsize=11)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=8)

# Panel 3: scoring method
ax = axes[2]
ax.bar(["Score A\\n(Recon)", "Score B\\n(Symbolic)", "Score C\\n(Combined)"],
       [auc_A*100, auc_B*100, auc_C*100],
       color=['#3498db','#e74c3c','#9b59b6'], alpha=0.85, edgecolor='grey')
ax.set_ylabel("AUC (%)", fontsize=10)
ax.set_title("Abl. 3: Scoring Method (W=1000)", fontsize=11)
ax.set_ylim(50, 102)
ax.axhline(90, color='gray', linestyle='--', linewidth=1)
for i, v in enumerate([auc_A*100, auc_B*100, auc_C*100]):
    ax.text(i, v+0.5, f"{v:.1f}%", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "ablation_summary.png"), dpi=150)
plt.show()

# Save combined ablation table
abl_rows = (
    [{"ablation":"Reg",       "variant":"λ=0",     "metric":"symb_quality_%", "value":qual_noreg},
     {"ablation":"Reg",       "variant":"λ=1e-4",  "metric":"symb_quality_%", "value":qual}] +
    [{"ablation":"Bottleneck","variant":f"B={b}",  "metric":"AUC_scoreA",     "value":ablation_b_df.loc[b,"AUC_scoreA"]}
     for b in B_values] +
    [{"ablation":"Scoring",   "variant":"ScoreA",  "metric":"AUC_W1000",      "value":auc_A},
     {"ablation":"Scoring",   "variant":"ScoreB",  "metric":"AUC_W1000",      "value":auc_B},
     {"ablation":"Scoring",   "variant":"ScoreC",  "metric":"AUC_W1000",      "value":auc_C}]
)
pd.DataFrame(abl_rows).to_csv(os.path.join(RESULTS_DIR, "ablation_table.csv"), index=False)
"""))

# ── Cell 29: Summary markdown ──────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Consolidated Summary & Conclusions"))

# ── Cell 30: Consolidated Summary ─────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""\
print("=" * 72)
print("  KAN SYMBOLIC HEALTH EQUATION DISCOVERY — CONSOLIDATED SUMMARY")
print("=" * 72)

print(f"\\n1. Model: KAN-AE [44→22→{B}→22→44]  λ={LAMBDA_REG}  epochs={EPOCHS}")
print(f"   Final training MSE: {losses[-1]:.6f}")

print(f"\\n2. Edge Pruning  (α={PRUNE_ALPHA}):")
print(f"   Surviving edges: {n_surviving}/{n_total}  ({100*n_surviving/n_total:.1f}%)")

print(f"\\n3. Symbolic Regression:")
print(f"   Symbolisation quality (R²>0.95): {qual:.1f}%")
print(f"   Mean R²: {sym_df['r2'].mean():.4f}")
print(f"   Symbol distribution:")
for sym, cnt in sym_df['best_symbol'].value_counts().items():
    print(f"     {sym:12s}: {cnt}")

print(f"\\n4. Anomaly Detection  (W=1000):")
print(f"   Score A (Reconstruction): AUC = {auc_A:.4f}")
print(f"   Score B (Symbolic):       AUC = {auc_B:.4f}")
print(f"   Score C (Combined):       AUC = {auc_C:.4f}")

print(f"\\n5. Sample Health Equations (first 5 neurons):")
for j in range(min(5, 22)):
    print(f"   {health_equations[j]}")

print(f"\\n6. Benchmark position (W=1000, Score C vs all 12 methods):")
all_w1000    = full_bench[1000].sort_values(ascending=False)
rank_c       = list(all_w1000.index).index("KAN-AE-Combined") + 1
print(f"   KAN-AE-Combined ranks #{rank_c} of {len(all_w1000)}")
print(f"   Best overall: {all_w1000.index[0]} = {all_w1000.iloc[0]*100:.2f}%")

print(f"\\n7. Fault localisation top-3 most violated equations:")
for eq_idx in top_violated[:3]:
    feats = sym_df[sym_df['out_neuron']==eq_idx]['in_feature'].tolist()
    sensors = sorted(set(f.split('_')[0] for f in feats))
    print(f"   z_{eq_idx:02d}: fault_ratio={fault_ratio[eq_idx]:.2f}x  sensors={sensors}")

print(f"\\n8. Ablation highlights:")
print(f"   Reg: λ=0 quality={qual_noreg:.1f}%  vs  λ=1e-4 quality={qual:.1f}%")
best_b = ablation_b_df['AUC_scoreA'].idxmax()
print(f"   Best bottleneck: B={best_b}  AUC={ablation_b_df.loc[best_b,'AUC_scoreA']:.4f}")

print(f"\\n9. All outputs saved to: {RESULTS_DIR}/")
import os as _os
for f in sorted(_os.listdir(RESULTS_DIR)):
    print(f"   {f}")
print("=" * 72)
"""))

# ── Assemble and write notebook ────────────────────────────────────────────
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": ".venv",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10.12",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
    },
}

out_path = "KAN_Symbolic_Health_Equations.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"Notebook written to: {out_path}")
print(f"Total cells: {len(nb.cells)}")
