"""
export_for_benchmark.py
=======================
Trains all models (same hyperparams as benchmark_inference.py) and exports
them to formats loadable by the C++ benchmark in cpp_benchmark/.

Exports
-------
  cpp_benchmark/exports/
    ae.pt              – AE TorchScript
    vae.pt             – VAE TorchScript (returns tuple: recon, mu, logvar)
    svdd.pt            – DeepSVDD TorchScript
    ts.pt              – Teacher+Student combined TorchScript (returns tuple)
    dae.pt             – SSL-DAE TorchScript
    kan_ae.pt          – Full KAN-AE TorchScript  (44→22→8→22→44)
    kan_layer0.pt      – KAN layer 0 TorchScript  (44→22)
    kan_layers1to3.pt  – KAN layers 1-3 TorchScript (22→8→22→44)
    test_data_mm.bin   – X_te_mm  float32 row-major (n_test × 44)
    test_data_std.bin  – X_te_std float32 row-major (n_test × 44)
    test_shape.bin     – [n_test, n_features] int32 × 2
    coreset.bin        – PatchCore coreset float32 row-major
    coreset_shape.bin  – [n_coreset, n_features] int32 × 2
    mahal_mu.bin       – Mahalanobis centre float32 (latent_dim,)
    mahal_cov_inv.bin  – Inverse covariance float32 row-major (latent_dim × latent_dim)
    mahal_dim.bin      – [latent_dim] int32 × 1
    sym_layer0.json    – Layer-0 symbolic data (normalized=false)
    sym_layer1.json    – Layer-1 symbolic data (normalized=true)
    sym_layer2.json    – Layer-2 symbolic data (normalized=true)
    sym_layer3.json    – Layer-3 symbolic data (normalized=true)

Usage
-----
    python export_for_benchmark.py
"""

import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path = [p for p in sys.path if not p.startswith("/home/suleiman/.local/lib")]

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import pairwise_distances

BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
EXPORT_DIR    = os.path.join(BASE_DIR, "cpp_benchmark", "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, "efficient_kan"))
from efficient_kan import KAN

# ── Hyperparameters (must match benchmark_inference.py exactly) ───────────────
W           = 1000
SEED        = 42
B           = 8
EPOCHS      = 150
PRUNE_ALPHA = 0.05

FEATURE_NAMES = ["mean", "rms", "std", "var", "skew", "kurt", "p2p",
                 "crest", "shape", "margin", "impulse"]
CHANNELS      = ["S1", "S2", "S3", "S4"]
feat_cols     = [f"{ch}_{f}" for ch in CHANNELS for f in FEATURE_NAMES]
n_features    = len(feat_cols)  # 44


# ── Symbolic candidates (identical to benchmark_inference.py) ─────────────────
def sym_linear(x, a, b):        return a * x + b
def sym_quadratic(x, a, b, c):  return a * x**2 + b * x + c
def sym_sigmoid(x, a, b, c):
    return a / (1.0 + np.exp(-np.clip(b * (x - c), -100, 100)))
def sym_tanh(x, a, b, c):       return a * np.tanh(np.clip(b * (x - c), -50, 50))
def sym_gaussian(x, a, b, c):
    return a * np.exp(-np.clip(((x - c) / (abs(b) + 1e-6))**2, 0, 100))
def sym_hinge(x, a, b, c):      return a * np.maximum(0.0, x - c) + b
def sym_sqrt(x, a, b):          return a * np.sqrt(np.abs(x) + 1e-8) + b
def sym_log(x, a, b):           return a * np.log(np.abs(x) + 1e-8) + b
def sym_constant(x, a):         return np.full_like(x, a, dtype=float)
def sym_exp(x, a, b):           return a * np.exp(np.clip(b * x, -50, 50))
def sym_power(x, a, b):         return a * (np.abs(x) + 1e-8) ** np.clip(b, -5, 5)
def sym_rational(x, a, b):      return a / (np.abs(b) + np.abs(x) + 1e-8)
def sym_sin(x, a, b, c):        return a * np.sin(np.clip(b * x + c, -100, 100))

CANDIDATES = [
    ("linear",    sym_linear,    2, [0.1, 0.1]),
    ("quadratic", sym_quadratic, 3, [0.1, 0.1, 0.1]),
    ("sigmoid",   sym_sigmoid,   3, [1.0, 5.0, 0.5]),
    ("tanh",      sym_tanh,      3, [1.0, 5.0, 0.5]),
    ("gaussian",  sym_gaussian,  3, [1.0, 0.3, 0.5]),
    ("hinge",     sym_hinge,     3, [1.0, 0.0, 0.5]),
    ("sqrt",      sym_sqrt,      2, [0.1, 0.1]),
    ("log",       sym_log,       2, [0.1, 0.1]),
    ("constant",  sym_constant,  1, [0.0]),
    ("exp",       sym_exp,       2, [0.1, 1.0]),
    ("power",     sym_power,     2, [0.1, 1.0]),
    ("rational",  sym_rational,  2, [0.1, 1.0]),
    ("sin",       sym_sin,       3, [1.0, 6.0, 0.0]),
]
_POLYNOMIAL = {"linear", "quadratic", "constant"}
R2_GAIN_MIN = 0.0005


def bic_score(n, k, mse):
    return n * np.log(max(mse, 1e-12)) + k * np.log(n)


def fit_symbolic(x_vals, y_vals):
    n      = len(x_vals)
    y_mean = float(y_vals.mean())
    ss_tot = float(((y_vals - y_mean)**2).sum() + 1e-12)
    fits   = [{"best_symbol": "constant", "params": [y_mean],
               "mse": float(((y_vals - y_mean)**2).mean()), "r2": 0.0,
               "bic": bic_score(n, 1, float(((y_vals - y_mean)**2).mean()))}]
    for sym_name, sym_fn, k, p0 in CANDIDATES:
        try:
            popt, _ = curve_fit(sym_fn, x_vals, y_vals,
                                p0=np.asarray(p0, dtype=float), maxfev=10000)
            y_pred  = sym_fn(x_vals, *popt)
            mse     = float(np.mean((y_vals - y_pred)**2))
            r2      = float(1.0 - ((y_vals - y_pred)**2).sum() / ss_tot)
            if np.isfinite(mse) and np.isfinite(r2):
                fits.append({"best_symbol": sym_name, "params": popt.tolist(),
                             "r2": r2, "bic": bic_score(n, k, mse)})
        except Exception:
            continue
    poly_fits  = [f for f in fits if f["best_symbol"] in _POLYNOMIAL]
    nonpoly    = [f for f in fits if f["best_symbol"] not in _POLYNOMIAL]
    best_poly  = max(poly_fits, key=lambda d: d["r2"]) if poly_fits else None
    r2_poly    = best_poly["r2"] if best_poly is not None else -np.inf
    qualifying = [f for f in nonpoly if f["r2"] > r2_poly + R2_GAIN_MIN]
    if qualifying:
        return max(qualifying, key=lambda d: d["r2"])
    return min(poly_fits, key=lambda d: d["bic"]) if best_poly else min(fits, key=lambda d: d["bic"])


def _symbolise_layer(model, layer_idx, X_healthy_mm, alpha=PRUNE_ALPHA, n_pts=200):
    layer = model.layers[layer_idx]
    sw    = layer.scaled_spline_weight.detach().cpu()
    bw    = layer.base_weight.detach().cpu()
    comb  = sw.abs().mean(dim=2) + bw.abs()
    mask  = (comb > alpha * comb.max().item()).numpy()
    out_dim, in_dim = mask.shape

    X0_t = torch.tensor(X_healthy_mm, dtype=torch.float32)
    with torch.no_grad():
        H_in = X0_t
        for k in range(layer_idx):
            H_in = model.layers[k](H_in)
        H_in = H_in.numpy()

    feat_means = H_in.mean(axis=0)
    H_min = H_in.min(axis=0); H_max = H_in.max(axis=0)
    span  = np.maximum(H_max - H_min, 1e-9)

    surviving = [(j, i) for j in range(out_dim) for i in range(in_dim) if mask[j, i]]
    rows = []
    for j, i in surviving:
        x_norm = np.quantile((H_in[:, i] - H_min[i]) / span[i],
                             np.linspace(0.02, 0.98, n_pts)).astype(np.float32)
        x_orig = x_norm * span[i] + H_min[i]
        x_tensor = torch.tensor(np.tile(feat_means.astype(np.float32), (n_pts, 1)))
        x_tensor[:, i] = torch.tensor(x_orig.astype(np.float32))
        with torch.no_grad():
            bases    = layer.b_splines(x_tensor)
            coeff    = layer.scaled_spline_weight[j, i, :]
            y_spline = (bases[:, i, :] * coeff.unsqueeze(0)).sum(dim=1).numpy()
            y_base   = layer.base_weight[j, i].item() * nn.SiLU()(x_tensor[:, i]).numpy()
        res = fit_symbolic(x_norm.astype(np.float64), (y_spline + y_base).astype(np.float64))
        rows.append({"out_neuron": j, "in_feat_idx": i,
                     "best_symbol": res["best_symbol"], "params": res["params"],
                     "r2": res["r2"], "x_min": float(H_min[i]), "x_span": float(span[i])})
    return pd.DataFrame(rows), H_in, out_dim


def _make_sym_evaluator_normalized(sym_df, out_dim):
    """Vectorised evaluator, normalized=True. Used for layers 1-3."""
    VECTORIZED_FNS = {
        "linear":    lambda X, A, B:    A * X + B,
        "quadratic": lambda X, A, B, C: A * X**2 + B * X + C,
        "sigmoid":   lambda X, A, B, C: A / (1.0 + np.exp(-np.clip(B * (X - C), -100, 100))),
        "tanh":      lambda X, A, B, C: A * np.tanh(np.clip(B * (X - C), -50, 50)),
        "gaussian":  lambda X, A, B, C: A * np.exp(-np.clip(((X - C) / (np.abs(B) + 1e-6))**2, 0, 100)),
        "hinge":     lambda X, A, B, C: A * np.maximum(0.0, X - C) + B,
        "sqrt":      lambda X, A, B:    A * np.sqrt(np.abs(X) + 1e-8) + B,
        "log":       lambda X, A, B:    A * np.log(np.abs(X) + 1e-8) + B,
        "constant":  lambda X, A:       A * np.ones_like(X),
        "exp":       lambda X, A, B:    A * np.exp(np.clip(B * X, -50, 50)),
        "power":     lambda X, A, B:    A * (np.abs(X) + 1e-8) ** np.clip(B, -5, 5),
        "rational":  lambda X, A, B:    A / (np.abs(B) + np.abs(X) + 1e-8),
        "sin":       lambda X, A, B, C: A * np.sin(np.clip(B * X + C, -100, 100)),
    }
    if sym_df is None or len(sym_df) == 0:
        return None
    sym_df   = sym_df.reset_index(drop=True)
    n_edges  = len(sym_df)
    out_cols = sym_df["out_neuron"].astype(int).values
    routing  = np.zeros((n_edges, out_dim), dtype=np.float64)
    for k, j in enumerate(out_cols):
        routing[k, j] = 1.0
    type_buckets = {}
    for k in range(n_edges):
        sym = sym_df.at[k, "best_symbol"]
        type_buckets.setdefault(sym, []).append(k)
    compiled = []
    for sym, k_list in type_buckets.items():
        col_idx    = sym_df["in_feat_idx"].astype(int).values[k_list]
        params_sub = sym_df["params"].iloc[k_list].tolist()
        n_p        = len(params_sub[0])
        param_arrs = [np.array([p[i] for p in params_sub], dtype=np.float64) for i in range(n_p)]
        grp_routing = routing[np.array(k_list, dtype=int), :]
        x_mins  = sym_df["x_min"].values.astype(np.float64)[k_list]
        x_spans = sym_df["x_span"].values.astype(np.float64)[k_list]
        compiled.append((VECTORIZED_FNS[sym], col_idx, param_arrs, grp_routing, x_mins, x_spans))

    def _eval(X_in):
        X64 = X_in.astype(np.float64)
        out = np.zeros((len(X64), out_dim), dtype=np.float64)
        for vec_fn, col_idx, param_arrs, grp_routing, x_mins, x_spans in compiled:
            X_cols = X64[:, col_idx]
            X_norm = (X_cols - x_mins) / (x_spans + 1e-12)
            out += vec_fn(X_norm, *param_arrs) @ grp_routing
        return out
    return _eval


def _calibrate_layer(sym_df_l, in_arr, model, layer_idx, out_dim):
    X_t = torch.tensor(in_arr.astype(np.float32))
    with torch.no_grad():
        true_out = model.layers[layer_idx](X_t).numpy()
    ev  = _make_sym_evaluator_normalized(sym_df_l, out_dim)
    raw = ev(in_arr) if ev is not None else np.zeros((len(in_arr), out_dim))
    scale = np.ones(out_dim); bias = np.zeros(out_dim)
    for j in range(out_dim):
        if np.std(raw[:, j]) < 1e-9:
            scale[j] = 0.0; bias[j] = float(true_out[:, j].mean())
        else:
            s, b = np.polyfit(raw[:, j], true_out[:, j], 1)
            scale[j] = s; bias[j] = b
    return scale, bias, true_out


# ── DL architectures ──────────────────────────────────────────────────────────
class _AE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,8), nn.ReLU())
        self.d = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self, x): return self.d(self.e(x))

class _VAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.e  = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU())
        self.mu = nn.Linear(16,8); self.lv = nn.Linear(16,8)
        self.d  = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self, x):
        h = self.e(x); m, lv = self.mu(h), self.lv(h)
        return self.d(m + torch.randn_like(torch.exp(0.5*lv)) * torch.exp(0.5*lv)), m, lv

class _SVDD(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,16))
    def forward(self, x): return self.n(x)

class _TS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,16))
    def forward(self, x): return self.n(x)

class _DAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.e = nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Linear(32,8))
        self.d = nn.Sequential(nn.Linear(8,32), nn.ReLU(), nn.Linear(32,d))
    def forward(self, x): return self.d(self.e(x))

class _TSWrapper(nn.Module):
    """Times both student and teacher in a single traced module."""
    def __init__(self, snet, tnet):
        super().__init__()
        self.snet = snet; self.tnet = tnet
    def forward(self, x):
        return self.snet(x), self.tnet(x)

class _KANLayer0(nn.Module):
    """Wraps just layer 0 of KAN for isolated tracing."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        return self.layer(x)

class _KANLayers1to3(nn.Module):
    """Wraps layers 1-3 of KAN for tracing (input: layer-0 output, 22-dim)."""
    def __init__(self, layers):
        super().__init__()
        self.l1 = layers[0]
        self.l2 = layers[1]
        self.l3 = layers[2]
    def forward(self, x):
        return self.l3(self.l2(self.l1(x)))


def _k_center(f, frac=0.1):
    n = f.shape[0]; ns = max(1, int(n * frac))
    c = [np.random.randint(0, n)]
    dists = pairwise_distances(f, f[c], metric='euclidean').flatten()
    for _ in range(ns - 1):
        idx = np.argmax(dists); c.append(idx)
        dists = np.minimum(dists, pairwise_distances(f, f[[idx]]).flatten())
    return f[c]


def _trace(mod, example, name):
    """Trace a module to TorchScript and save; returns path."""
    mod.eval()
    with torch.no_grad():
        scripted = torch.jit.trace(mod, example, strict=False)
    path = os.path.join(EXPORT_DIR, name)
    scripted.save(path)
    print(f"  Saved {name}")
    return path


def _save_sym_layer(sym_df, layer_idx, out_dim, normalized, scale, bias, filename):
    """Serialise symbolic layer data to JSON for C++ loading."""
    edges = []
    for _, row in sym_df.iterrows():
        edges.append({
            "out_j":  int(row["out_neuron"]),
            "in_i":   int(row["in_feat_idx"]),
            "sym":    str(row["best_symbol"]),
            "params": [float(v) for v in row["params"]],
            "x_min":  float(row["x_min"]),
            "x_span": float(row["x_span"]),
        })
    doc = {
        "layer_idx":  layer_idx,
        "out_dim":    out_dim,
        "normalized": normalized,
        "scale":      scale.tolist(),
        "bias":       bias.tolist(),
        "edges":      edges,
    }
    path = os.path.join(EXPORT_DIR, filename)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  Saved {filename}  ({len(edges)} edges)")


def _save_f32(arr, filename):
    path = os.path.join(EXPORT_DIR, filename)
    arr.astype(np.float32).tofile(path)
    print(f"  Saved {filename}  shape={arr.shape}")


def _save_i32(arr, filename):
    path = os.path.join(EXPORT_DIR, filename)
    np.array(arr, dtype=np.int32).tofile(path)
    print(f"  Saved {filename}  values={list(arr)}")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Export: W={W}, seed={SEED}")
    print("=" * 60)

    # ── Data (identical split to benchmark_inference.py) ──────────────────────
    df_w    = pd.read_csv(os.path.join(PROCESSED_DIR, f"features_W{W}.csv"))
    df_h    = df_w[df_w['label'] == 0]
    df_b    = df_w[df_w['label'] == 1]
    train_h, test_h = train_test_split(df_h, test_size=0.3, random_state=SEED)
    test_df = pd.concat([test_h, df_b]).reset_index(drop=True)
    n_test  = len(test_df)
    print(f"Train healthy: {len(train_h)}  Test: {n_test}")

    sc_mm  = MinMaxScaler().fit(train_h[feat_cols].values)
    sc_std = StandardScaler().fit(train_h[feat_cols].values)

    X_tr_mm      = sc_mm.transform(train_h[feat_cols].values).astype(np.float32)
    X_te_mm      = sc_mm.transform(test_df[feat_cols].values).astype(np.float32)
    X_healthy_mm = sc_mm.transform(df_h[feat_cols].values).astype(np.float32)
    X_tr_std     = sc_std.transform(train_h[feat_cols].values).astype(np.float32)
    X_te_std     = sc_std.transform(test_df[feat_cols].values).astype(np.float32)

    Xtr_s  = torch.tensor(X_tr_std, dtype=torch.float32)
    Xtr_m  = torch.tensor(X_tr_mm,  dtype=torch.float32)
    Xte_s  = torch.tensor(X_te_std, dtype=torch.float32)
    Xte_t  = torch.tensor(X_te_mm,  dtype=torch.float32)
    d      = n_features
    ldr_s  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_s, Xtr_s), batch_size=256, shuffle=True)

    # ── Save test data + shape ─────────────────────────────────────────────────
    print("\n[Data]")
    _save_i32([n_test, n_features], "test_shape.bin")
    _save_f32(X_te_mm,  "test_data_mm.bin")
    _save_f32(X_te_std, "test_data_std.bin")

    # ── PatchCore coreset ──────────────────────────────────────────────────────
    print("\n[PatchCore]")
    coreset = _k_center(X_tr_std, frac=0.1)
    _save_i32([coreset.shape[0], coreset.shape[1]], "coreset_shape.bin")
    _save_f32(coreset, "coreset.bin")

    # ── Classical sklearn models ───────────────────────────────────────────────
    print("\n[Classical sklearn]")

    # 1. IsolationForest — export tree split structures
    clf_if = IsolationForest(n_estimators=100, random_state=SEED).fit(X_tr_std)
    trees_export = []
    for est, feats in zip(clf_if.estimators_, clf_if.estimators_features_):
        t = est.tree_
        # Map local node feature indices → original feature indices
        node_feats_global = [
            int(feats[fi]) if fi >= 0 else -2
            for fi in t.feature
        ]
        trees_export.append({
            "node_feature":   node_feats_global,
            "threshold":      t.threshold.tolist(),
            "children_left":  t.children_left.tolist(),
            "children_right": t.children_right.tolist(),
            "n_node_samples": t.n_node_samples.tolist(),
        })
    n_samp_per_tree = int(clf_if.max_samples_)
    if_doc = {"n_trees": 100, "n_samples_per_tree": n_samp_per_tree,
               "trees": trees_export}
    if_path = os.path.join(EXPORT_DIR, "if_trees.json")
    with open(if_path, "w") as f:
        json.dump(if_doc, f)
    print(f"  Saved if_trees.json  (100 trees, {n_samp_per_tree} samples/tree,"
          f" {os.path.getsize(if_path)/1024:.0f} KB)")

    # 2. OC-SVM — export support vectors, dual coefficients, gamma, intercept
    clf_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_tr_std)
    sv  = clf_svm.support_vectors_.astype(np.float32)   # (n_sv, 44)
    dc  = clf_svm.dual_coef_.astype(np.float32).flatten()  # (n_sv,)
    _save_f32(sv,  "svm_sv.bin")
    _save_f32(dc,  "svm_dual_coef.bin")
    _save_i32([sv.shape[0], sv.shape[1]], "svm_sv_shape.bin")
    with open(os.path.join(EXPORT_DIR, "svm_params.json"), "w") as f:
        json.dump({"gamma": float(clf_svm._gamma),
                   "intercept": float(clf_svm.intercept_[0])}, f)
    print(f"  Saved svm_sv.bin {sv.shape}, svm_dual_coef.bin ({len(dc)} SVs)")

    # 3. LOF — compute and export training data, per-point lrd and k-distance
    # Use NearestNeighbors directly to avoid relying on sklearn LOF internals
    # (attribute names differ across sklearn versions)
    from sklearn.neighbors import NearestNeighbors
    LOF_K = 20
    clf_lof = LocalOutlierFactor(n_neighbors=LOF_K, novelty=True).fit(X_tr_std)
    lof_fit_X = X_tr_std.astype(np.float32)
    # Compute k-NN on training set (n_neighbors+1 to exclude self)
    nn_lof = NearestNeighbors(n_neighbors=LOF_K + 1, algorithm='auto').fit(X_tr_std)
    tr_dists, tr_idxs = nn_lof.kneighbors(X_tr_std)
    # Exclude self (always distance 0 at index 0)
    tr_dists = tr_dists[:, 1:]    # (n_train, LOF_K)
    tr_idxs  = tr_idxs[:,  1:]    # (n_train, LOF_K)
    lof_kdist = tr_dists[:, -1].astype(np.float32)   # k-distance per training pt
    # reach_dist(i,j) = max(k_dist[j], dist(i,j))
    reach_d  = np.maximum(tr_dists, lof_kdist[tr_idxs])  # (n_train, LOF_K)
    lof_lrdof = (LOF_K / (reach_d.sum(axis=1) + 1e-12)).astype(np.float32)
    _save_f32(lof_fit_X, "lof_fit_X.bin")
    _save_f32(lof_lrdof, "lof_lrdof.bin")
    _save_f32(lof_kdist, "lof_kdist.bin")
    _save_i32([lof_fit_X.shape[0], lof_fit_X.shape[1]], "lof_shape.bin")
    with open(os.path.join(EXPORT_DIR, "lof_params.json"), "w") as f:
        json.dump({"n_neighbors": LOF_K,
                   "n_train": int(lof_fit_X.shape[0])}, f)
    print(f"  Saved lof_fit_X.bin {lof_fit_X.shape}, lof_lrdof.bin, lof_kdist.bin")

    # 4. SHAP / LIME background and sample data
    from sklearn.cluster import KMeans
    bg_model = KMeans(n_clusters=20, random_state=SEED, n_init=10).fit(X_tr_mm)
    shap_bg  = bg_model.cluster_centers_.astype(np.float32)  # (20, 44)
    _save_f32(shap_bg, "shap_bg.bin")
    _save_i32([shap_bg.shape[0], shap_bg.shape[1]], "shap_bg_shape.bin")

    # Single fault sample to explain (first fault in test set)
    fault_idx_arr = np.where(test_df['label'].values == 1)[0]
    xai_idx  = int(fault_idx_arr[0])
    xai_samp = X_te_mm[xai_idx:xai_idx + 1].astype(np.float32)   # (1, 44)
    _save_f32(xai_samp, "xai_sample.bin")

    # LIME: training distribution stats for perturbation
    lime_stats = np.vstack([X_tr_mm.mean(axis=0),
                             X_tr_mm.std(axis=0)]).astype(np.float32)  # (2, 44)
    _save_f32(lime_stats, "lime_tr_stats.bin")
    print(f"  Saved shap_bg.bin {shap_bg.shape}, xai_sample.bin, lime_tr_stats.bin")

    # ── DL baselines ──────────────────────────────────────────────────────────
    print("\n[DL Baselines]")

    ae = _AE(d); o = torch.optim.Adam(ae.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad(); nn.MSELoss()(ae(bx), bx).backward(); o.step()
    _trace(ae, Xte_s, "ae.pt")

    vae = _VAE(d); o = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad(); x_r, m, lv = vae(bx)
            (nn.MSELoss()(x_r, bx) - 0.0005 * torch.sum(1 + lv - m.pow(2) - lv.exp())).backward()
            o.step()
    _trace(vae, Xte_s, "vae.pt")

    svdd = _SVDD(d); o = torch.optim.Adam(svdd.parameters(), lr=1e-3)
    with torch.no_grad():
        c_sv = torch.mean(svdd(Xtr_s), dim=0)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad()
            torch.mean(torch.sum((svdd(bx) - c_sv)**2, dim=1)).backward(); o.step()
    _trace(svdd, Xte_s, "svdd.pt")

    tnet = _TS(d); snet = _TS(d)
    for p in tnet.parameters(): p.requires_grad = False
    o = torch.optim.Adam(snet.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad(); nn.MSELoss()(snet(bx), tnet(bx)).backward(); o.step()
    snet.eval(); tnet.eval()
    ts_wrapper = _TSWrapper(snet, tnet)
    _trace(ts_wrapper, Xte_s, "ts.pt")

    dae = _DAE(d); o = torch.optim.Adam(dae.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad()
            nn.MSELoss()(dae(bx + 0.5 * torch.randn_like(bx)), bx).backward(); o.step()
    _trace(dae, Xte_s, "dae.pt")

    # ── KAN-AE training ────────────────────────────────────────────────────────
    print("\n[KAN-AE] Training (150 epochs)...")
    model = KAN(layers_hidden=[n_features, n_features // 2, B, n_features // 2, n_features],
                grid_size=5, spline_order=3)
    ldr_m = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_m, Xtr_m), batch_size=256, shuffle=True)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(EPOCHS):
        for bx, bt in ldr_m:
            opt.zero_grad(); nn.MSELoss()(model(bx), bt).backward(); opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  epoch {ep+1}/{EPOCHS}")
    model.eval(); model.cpu()

    # ── KAN-AE TorchScript exports ─────────────────────────────────────────────
    print("\n[KAN-AE TorchScript]")
    _trace(model, Xte_t, "kan_ae.pt")

    l0_wrapper = _KANLayer0(model.layers[0])
    _trace(l0_wrapper, Xte_t, "kan_layer0.pt")

    # Compute layer-0 output to use as example for layers 1-3
    with torch.no_grad():
        h1_example = model.layers[0](Xte_t)
    l1to3_wrapper = _KANLayers1to3(list(model.layers[1:]))
    _trace(l1to3_wrapper, h1_example, "kan_layers1to3.pt")

    # ── Mahalanobis parameters ─────────────────────────────────────────────────
    print("\n[Mahalanobis]")
    with torch.no_grad():
        H_tr_l1 = model.layers[0](Xtr_m).numpy()
    mu_l1    = H_tr_l1.mean(axis=0).astype(np.float32)
    cov_inv  = np.linalg.pinv(np.cov(H_tr_l1.T)).astype(np.float32)
    latent_dim = mu_l1.shape[0]  # 22

    _save_i32([latent_dim], "mahal_dim.bin")
    _save_f32(mu_l1.reshape(1, -1),  "mahal_mu.bin")     # (1, 22) → flatten → (22,) when reading
    _save_f32(cov_inv, "mahal_cov_inv.bin")                # (22, 22)

    # ── Layer 0 symbolic regression ────────────────────────────────────────────
    print("\n[Symbolic] Layer 0...")
    layer0  = model.layers[0]
    sw0     = layer0.scaled_spline_weight.detach().abs().mean(dim=2).numpy()
    bw0     = layer0.base_weight.detach().abs().numpy()
    comb0   = sw0 + bw0
    thr0    = PRUNE_ALPHA * comb0.max()
    edges0  = [(j, i) for j in range(22) for i in range(n_features) if comb0[j, i] > thr0]
    means0  = X_healthy_mm.mean(axis=0)

    sym_rows_l0 = []
    for (j, i) in edges0:
        xv, yv = [], []
        # sample_spline_edge logic inlined
        half   = 100; n_pts = 200
        qs     = np.linspace(0.02, 0.98, half)
        x_q    = np.quantile(X_healthy_mm[:, i], qs).astype(np.float32)
        x_u    = np.linspace(0.0, 1.0, n_pts - half).astype(np.float32)
        x_vals = np.sort(np.concatenate([x_q, x_u]))
        x_tensor = torch.tensor(np.tile(means0.astype(np.float32), (n_pts, 1)))
        x_tensor[:, i] = torch.tensor(x_vals)
        with torch.no_grad():
            bases    = layer0.b_splines(x_tensor)
            coeff    = layer0.scaled_spline_weight[j, i, :]
            y_spline = (bases[:, i, :] * coeff.unsqueeze(0)).sum(dim=1).numpy()
            y_base   = layer0.base_weight[j, i].item() * nn.SiLU()(x_tensor[:, i]).numpy()
        res = fit_symbolic(x_vals.astype(np.float64), (y_spline + y_base).astype(np.float64))
        sym_rows_l0.append({"out_neuron": j, "in_feat_idx": i,
                            "best_symbol": res["best_symbol"],
                            "params": res["params"], "r2": res["r2"],
                            "x_min": 0.0, "x_span": 1.0})
    sym_df_l0 = pd.DataFrame(sym_rows_l0)

    # Calibrate layer 0 (normalized=False: inputs already in [0,1])
    VECTORIZED_FNS_L0 = {
        "linear":    lambda X, A, B:    A * X + B,
        "quadratic": lambda X, A, B, C: A * X**2 + B * X + C,
        "sigmoid":   lambda X, A, B, C: A / (1.0 + np.exp(-np.clip(B*(X-C), -100, 100))),
        "tanh":      lambda X, A, B, C: A * np.tanh(np.clip(B*(X-C), -50, 50)),
        "gaussian":  lambda X, A, B, C: A * np.exp(-np.clip(((X-C)/(np.abs(B)+1e-6))**2, 0, 100)),
        "hinge":     lambda X, A, B, C: A * np.maximum(0.0, X - C) + B,
        "sqrt":      lambda X, A, B:    A * np.sqrt(np.abs(X) + 1e-8) + B,
        "log":       lambda X, A, B:    A * np.log(np.abs(X) + 1e-8) + B,
        "constant":  lambda X, A:       A * np.ones_like(X),
        "exp":       lambda X, A, B:    A * np.exp(np.clip(B * X, -50, 50)),
        "power":     lambda X, A, B:    A * (np.abs(X) + 1e-8) ** np.clip(B, -5, 5),
        "rational":  lambda X, A, B:    A / (np.abs(B) + np.abs(X) + 1e-8),
        "sin":       lambda X, A, B, C: A * np.sin(np.clip(B * X + C, -100, 100)),
    }
    n_edges_l0  = len(sym_df_l0)
    out_cols_l0 = sym_df_l0["out_neuron"].astype(int).values
    routing_l0  = np.zeros((n_edges_l0, 22), dtype=np.float64)
    for k, j in enumerate(out_cols_l0):
        routing_l0[k, j] = 1.0
    type_buckets_l0 = {}
    for k in range(n_edges_l0):
        sym = sym_df_l0.at[k, "best_symbol"]
        type_buckets_l0.setdefault(sym, []).append(k)
    compiled_l0 = []
    for sym, k_list in type_buckets_l0.items():
        col_idx    = sym_df_l0["in_feat_idx"].astype(int).values[k_list]
        params_sub = sym_df_l0["params"].iloc[k_list].tolist()
        n_p        = len(params_sub[0])
        param_arrs = [np.array([p[i] for p in params_sub], dtype=np.float64) for i in range(n_p)]
        grp_routing = routing_l0[np.array(k_list, dtype=int), :]
        compiled_l0.append((VECTORIZED_FNS_L0[sym], col_idx, param_arrs, grp_routing))

    def eval_l0_raw(X_in):
        X64 = X_in.astype(np.float64)
        out = np.zeros((len(X64), 22), dtype=np.float64)
        for vec_fn, col_idx, param_arrs, grp_routing in compiled_l0:
            X_cols = X64[:, col_idx]
            out += vec_fn(X_cols, *param_arrs) @ grp_routing
        return out

    with torch.no_grad():
        h1_true_tr = layer0(Xtr_m).numpy()
    raw_cal    = eval_l0_raw(X_tr_mm.astype(np.float64))
    scales_l0  = np.ones(22, dtype=np.float64)
    biases_l0  = np.zeros(22, dtype=np.float64)
    for j in range(22):
        rj = raw_cal[:, j]; tj = h1_true_tr[:, j]
        if rj.std() < 1e-9:
            scales_l0[j] = 0.0; biases_l0[j] = float(tj.mean())
        else:
            s, b = np.polyfit(rj, tj, 1)
            scales_l0[j] = s; biases_l0[j] = b

    _save_sym_layer(sym_df_l0, 0, 22, False, scales_l0, biases_l0, "sym_layer0.json")

    # ── Layers 1-3 symbolic regression ────────────────────────────────────────
    print("\n[Symbolic] Layers 1-3...")
    sym_df_l1, H_l1, od1 = _symbolise_layer(model, 1, X_healthy_mm)
    sym_df_l2, H_l2, od2 = _symbolise_layer(model, 2, X_healthy_mm)
    sym_df_l3, H_l3, od3 = _symbolise_layer(model, 3, X_healthy_mm)

    sc1, bs1, _ = _calibrate_layer(sym_df_l1, H_l1, model, 1, od1)
    sc2, bs2, _ = _calibrate_layer(sym_df_l2, H_l2, model, 2, od2)
    sc3, bs3, _ = _calibrate_layer(sym_df_l3, H_l3, model, 3, od3)

    _save_sym_layer(sym_df_l1, 1, od1, True, sc1, bs1, "sym_layer1.json")
    _save_sym_layer(sym_df_l2, 2, od2, True, sc2, bs2, "sym_layer2.json")
    _save_sym_layer(sym_df_l3, 3, od3, True, sc3, bs3, "sym_layer3.json")

    print("\n" + "=" * 60)
    print("All exports complete. Files in:", EXPORT_DIR)
    for f in sorted(os.listdir(EXPORT_DIR)):
        sz = os.path.getsize(os.path.join(EXPORT_DIR, f)) / 1024
        print(f"  {f:<35} {sz:8.1f} KB")


if __name__ == "__main__":
    main()
