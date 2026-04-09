"""
multi_seed_eval.py
==================
Multi-seed evaluation of ALL methods at W=1000 (10 independent seeds).

Existing KAN-AE Scores A–F are loaded from results_kan_symbolic/multi_seed_auc.csv
(already computed). This script adds:
  • KAN-AE-SymRecon  (SR)  — requires KAN-AE retraining per seed for layers 1-3
  • SHAP-Weighted-Recon    — requires KAN-AE retraining per seed
  • LIME-Weighted-Recon    — requires KAN-AE retraining per seed
  • IsolationForest, OC-SVM, LOF, PatchCore,
    Autoencoder, VAE, DeepSVDD, TeacherStudent, SSL-DAE  (baselines only need data)

Outputs saved to results_kan_symbolic/:
  multi_seed_auc_full.csv     — per-seed AUCs for all 18 methods
  multi_seed_summary_full.csv — mean / std / min / max per method
  multi_seed_eval.log         — full console transcript

Usage:
    /home/suleiman/miniconda3/envs/go2-convex-mpc/bin/python multi_seed_eval.py
"""

# ── Path fix & stdlib ─────────────────────────────────────────────────────────
import sys
sys.path = [p for p in sys.path if not p.startswith("/home/suleiman/.local/lib")]

import os, warnings, time
warnings.filterwarnings("ignore")

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
from sklearn.metrics import roc_auc_score, pairwise_distances

# ── Tee stdout → log file ─────────────────────────────────────────────────────
class _Tee:
    def __init__(self, path):
        self._term = sys.stdout
        self._file = open(path, "w", buffering=1)
    def write(self, msg):
        self._term.write(msg); self._file.write(msg)
    def flush(self):
        self._term.flush(); self._file.flush()
    def close(self):
        self._file.close()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR   = os.path.join(BASE_DIR, "results_kan_symbolic")
os.makedirs(RESULTS_DIR, exist_ok=True)

_tee = _Tee(os.path.join(RESULTS_DIR, "multi_seed_eval.log"))
sys.stdout = _tee

sys.path.insert(0, os.path.join(BASE_DIR, "efficient_kan"))
from efficient_kan import KAN

# ── Hyperparameters (must match original KAN-AE run) ─────────────────────────
B           = 8
LAMBDA_REG  = 1e-4
PRUNE_ALPHA = 0.05
EPOCHS      = 150
W           = 1000
SEEDS       = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]

FEATURE_NAMES = ["mean", "rms", "std", "var", "skew", "kurt", "p2p",
                 "crest", "shape", "margin", "impulse"]
CHANNELS      = ["S1", "S2", "S3", "S4"]
feat_cols     = [f"{ch}_{f}" for ch in CHANNELS for f in FEATURE_NAMES]
n_features    = len(feat_cols)   # 44

# ── Symbolic candidate functions ──────────────────────────────────────────────
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
CANDIDATES_DICT = {name: fn for name, fn, _, _ in CANDIDATES}
_POLYNOMIAL     = {"linear", "quadratic", "constant"}
R2_GAIN_MIN     = 0.0005


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
            y_pred = sym_fn(x_vals, *popt)
            mse    = float(np.mean((y_vals - y_pred)**2))
            r2     = float(1.0 - ((y_vals - y_pred)**2).sum() / ss_tot)
            if np.isfinite(mse) and np.isfinite(r2):
                fits.append({"best_symbol": sym_name, "params": popt.tolist(),
                             "mse": mse, "r2": r2, "bic": bic_score(n, k, mse)})
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


def sample_spline_edge(layer, out_j, in_i, n_pts=200, data_dist=None, feat_means=None):
    layer.eval()
    half = n_pts // 2
    if data_dist is not None and len(data_dist) > 10:
        qs     = np.linspace(0.02, 0.98, half)
        x_q    = np.quantile(data_dist, qs).astype(np.float32)
        x_u    = np.linspace(0.0, 1.0, n_pts - half).astype(np.float32)
        x_vals = np.sort(np.concatenate([x_q, x_u]))
    else:
        x_vals = np.linspace(0.0, 1.0, n_pts).astype(np.float32)
    in_features = layer.in_features
    base_row = (np.full(in_features, 0.5, dtype=np.float32)
                if feat_means is None else np.asarray(feat_means, dtype=np.float32))
    x_tensor = torch.tensor(np.tile(base_row, (len(x_vals), 1)))
    x_tensor[:, in_i] = torch.tensor(x_vals)
    with torch.no_grad():
        bases    = layer.b_splines(x_tensor)
        coeff    = layer.scaled_spline_weight[out_j, in_i, :]
        y_spline = (bases[:, in_i, :] * coeff.unsqueeze(0)).sum(dim=1).numpy()
        x_col    = x_tensor[:, in_i]
        y_base   = layer.base_weight[out_j, in_i].item() * nn.SiLU()(x_col).numpy()
    return x_vals, y_spline + y_base


# ── Deep-layer symbolization ──────────────────────────────────────────────────
def _eval_layer_symbolic(sym_df_l, in_arr, out_dim):
    out = np.zeros((in_arr.shape[0], out_dim), dtype=np.float64)
    for _, row in sym_df_l.iterrows():
        j = int(row["out_neuron"]); i = int(row["in_feat_idx"])
        x_norm = (in_arr[:, i] - row["x_min"]) / (row["x_span"] + 1e-12)
        try:
            out[:, j] += CANDIDATES_DICT[row["best_symbol"]](x_norm, *row["params"])
        except Exception:
            pass
    return out


def _symbolise_layer(model, layer_idx, X_healthy_mm, alpha=PRUNE_ALPHA, n_pts=200):
    """Symbolise one KAN layer using actual activations from healthy data."""
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


def _calibrate_layer(sym_df_l, in_arr, model, layer_idx, out_dim):
    X_t = torch.tensor(in_arr.astype(np.float32))
    with torch.no_grad():
        true_out = model.layers[layer_idx](X_t).numpy()
    raw = _eval_layer_symbolic(sym_df_l, in_arr, out_dim)
    scale = np.ones(out_dim); bias = np.zeros(out_dim)
    for j in range(out_dim):
        if np.std(raw[:, j]) < 1e-9:
            scale[j] = 0.0; bias[j] = float(true_out[:, j].mean())
        else:
            s, b = np.polyfit(raw[:, j], true_out[:, j], 1)
            scale[j] = s; bias[j] = b
    return scale, bias, true_out


# ── Baseline DL architectures ─────────────────────────────────────────────────
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


def _k_center(f, frac=0.1):
    n = f.shape[0]; ns = max(1, int(n * frac))
    c = [np.random.randint(0, n)]
    dists = pairwise_distances(f, f[c], metric='euclidean').flatten()
    for _ in range(ns - 1):
        idx = np.argmax(dists); c.append(idx)
        dists = np.minimum(dists, pairwise_distances(f, f[[idx]]).flatten())
    return f[c]


# ══════════════════════════════════════════════════════════════════════════════
def run_new_methods_one_seed(seed_val):
    """Run SR, SHAP/LIME, and baselines for one seed. Returns dict of AUCs."""
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    results = {"seed": seed_val}

    # ── Data split (must match original KAN-AE run) ───────────────────────────
    df_w = pd.read_csv(os.path.join(PROCESSED_DIR, f"features_W{W}.csv"))
    df_h = df_w[df_w['label'] == 0]
    df_b = df_w[df_w['label'] == 1]
    train_h, test_h = train_test_split(df_h, test_size=0.3, random_state=seed_val)
    test_df = pd.concat([test_h, df_b]).reset_index(drop=True)
    y_t     = test_df['label'].values

    # Two scalers: MinMaxScaler for KAN-AE, StandardScaler for baselines
    sc_mm  = MinMaxScaler().fit(train_h[feat_cols].values)
    sc_std = StandardScaler().fit(train_h[feat_cols].values)

    X_tr_mm      = sc_mm.transform(train_h[feat_cols].values).astype(np.float32)
    X_te_mm      = sc_mm.transform(test_df[feat_cols].values).astype(np.float32)
    X_healthy_mm = sc_mm.transform(df_h[feat_cols].values).astype(np.float32)
    X_tr_std     = sc_std.transform(train_h[feat_cols].values).astype(np.float32)
    X_te_std     = sc_std.transform(test_df[feat_cols].values).astype(np.float32)

    d = n_features

    # ── 1. Classical & DL baselines (StandardScaler) ─────────────────────────
    Xtr_s = torch.tensor(X_tr_std, dtype=torch.float32)
    Xte_s = torch.tensor(X_te_std, dtype=torch.float32)

    results['IsolationForest'] = roc_auc_score(y_t,
        -IsolationForest(n_estimators=100, random_state=seed_val)
         .fit(X_tr_std).decision_function(X_te_std))

    results['OC-SVM'] = roc_auc_score(y_t,
        -OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
         .fit(X_tr_std).decision_function(X_te_std))

    results['LOF'] = roc_auc_score(y_t,
        -LocalOutlierFactor(n_neighbors=20, novelty=True)
         .fit(X_tr_std).decision_function(X_te_std))

    results['PatchCore'] = roc_auc_score(y_t,
        pairwise_distances(X_te_std, _k_center(X_tr_std, frac=0.1),
                           metric='euclidean').min(axis=1))

    ldr_s = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_s, Xtr_s), batch_size=256, shuffle=True)

    ae = _AE(d); o = torch.optim.Adam(ae.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s: o.zero_grad(); nn.MSELoss()(ae(bx), bx).backward(); o.step()
    ae.eval()
    with torch.no_grad():
        results['Autoencoder'] = roc_auc_score(y_t,
            torch.mean((ae(Xte_s) - Xte_s)**2, dim=1).numpy())

    vae = _VAE(d); o = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad(); x, m, lv = vae(bx)
            (nn.MSELoss()(x, bx) - 0.0005*torch.sum(1+lv-m.pow(2)-lv.exp())).backward(); o.step()
    vae.eval()
    with torch.no_grad():
        results['VAE'] = roc_auc_score(y_t,
            torch.mean((vae(Xte_s)[0] - Xte_s)**2, dim=1).numpy())

    svdd = _SVDD(d); o = torch.optim.Adam(svdd.parameters(), lr=1e-3)
    with torch.no_grad(): c_sv = torch.mean(svdd(Xtr_s), dim=0)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad(); torch.mean(torch.sum((svdd(bx)-c_sv)**2, dim=1)).backward(); o.step()
    svdd.eval()
    with torch.no_grad():
        results['DeepSVDD'] = roc_auc_score(y_t,
            torch.sum((svdd(Xte_s)-c_sv)**2, dim=1).numpy())

    tnet = _TS(d); snet = _TS(d)
    for p in tnet.parameters(): p.requires_grad = False
    o = torch.optim.Adam(snet.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad(); nn.MSELoss()(snet(bx), tnet(bx)).backward(); o.step()
    snet.eval(); tnet.eval()
    with torch.no_grad():
        results['TeacherStudent'] = roc_auc_score(y_t,
            torch.mean((snet(Xte_s)-tnet(Xte_s))**2, dim=1).numpy())

    dae = _DAE(d); o = torch.optim.Adam(dae.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad()
            nn.MSELoss()(dae(bx + 0.5*torch.randn_like(bx)), bx).backward(); o.step()
    dae.eval()
    with torch.no_grad():
        results['SSL-DAE'] = roc_auc_score(y_t,
            torch.mean((dae(Xte_s)-Xte_s)**2, dim=1).numpy())

    print(f"    baselines done.", flush=True)

    # ── 2. Train KAN-AE (needed for SR and SHAP/LIME) ─────────────────────────
    model = KAN(layers_hidden=[n_features, n_features//2, B, n_features//2, n_features],
                grid_size=5, spline_order=3)
    Xt  = torch.tensor(X_tr_mm, dtype=torch.float32)
    ldr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, Xt), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(EPOCHS):
        for bx, bt in ldr:
            opt.zero_grad(); nn.MSELoss()(model(bx), bt).backward(); opt.step()
    model.eval(); model.cpu()
    print(f"    KAN-AE trained.", flush=True)

    # ── 3. KAN SymRecon (Score SR): symbolise all 4 layers ────────────────────
    try:
        # Layer 0: survive + symbolic regression (same as original, for h1_sym)
        layer0 = model.layers[0]
        with torch.no_grad():
            sw0  = layer0.scaled_spline_weight.detach().abs().mean(dim=2).cpu().numpy()
            bw0  = layer0.base_weight.detach().abs().cpu().numpy()
            comb0 = sw0 + bw0
        thr0 = PRUNE_ALPHA * comb0.max()
        edges0 = [(j, i) for j in range(22) for i in range(n_features)
                          if comb0[j, i] > thr0]
        feat_means_l0 = X_healthy_mm.mean(axis=0)
        sym_rows = []
        for (j, i) in edges0:
            xv, yv = sample_spline_edge(layer0, j, i, n_pts=200,
                                         data_dist=X_healthy_mm[:, i],
                                         feat_means=feat_means_l0)
            best = fit_symbolic(xv.astype(np.float64), yv.astype(np.float64))
            sym_rows.append({"out_neuron": j, "in_feat_idx": i,
                             "best_symbol": best["best_symbol"],
                             "params": best["params"], "r2": best["r2"]})
        sym_df_s = pd.DataFrame(sym_rows)

        # GAM calibration layer 0 on all healthy
        Xh_t = torch.tensor(X_healthy_mm, dtype=torch.float32)
        with torch.no_grad():
            h1_cal = model.layers[0](Xh_t).numpy()
        raw_cal = np.zeros((len(X_healthy_mm), 22), dtype=np.float64)
        for _, row in sym_df_s.iterrows():
            j2, i2 = int(row['out_neuron']), int(row['in_feat_idx'])
            fn = CANDIDATES_DICT[row['best_symbol']]
            try: raw_cal[:, j2] += fn(X_healthy_mm[:, i2].astype(np.float64), *row['params'])
            except Exception: pass
        scales = np.ones(22, dtype=np.float64); biases = np.zeros(22, dtype=np.float64)
        for j2 in range(22):
            rj = raw_cal[:, j2]; tj = h1_cal[:, j2]
            if rj.std() < 1e-9:
                scales[j2] = 0.0; biases[j2] = float(tj.mean()); continue
            s, b = np.polyfit(rj, tj, 1); scales[j2] = s; biases[j2] = b

        # Layers 1–3
        sym_df_l1, H_l1, od1 = _symbolise_layer(model, 1, X_healthy_mm)
        sym_df_l2, H_l2, od2 = _symbolise_layer(model, 2, X_healthy_mm)
        sym_df_l3, H_l3, od3 = _symbolise_layer(model, 3, X_healthy_mm)
        sc1, bs1, _ = _calibrate_layer(sym_df_l1, H_l1, model, 1, od1)
        sc2, bs2, _ = _calibrate_layer(sym_df_l2, H_l2, model, 2, od2)
        sc3, bs3, _ = _calibrate_layer(sym_df_l3, H_l3, model, 3, od3)

        # Full symbolic forward on test
        Xte_f = X_te_mm.astype(np.float64)
        raw0  = np.zeros((len(Xte_f), 22), dtype=np.float64)
        for _, row in sym_df_s.iterrows():
            j2, i2 = int(row['out_neuron']), int(row['in_feat_idx'])
            try: raw0[:, j2] += CANDIDATES_DICT[row['best_symbol']](
                    Xte_f[:, i2], *row['params'])
            except Exception: pass
        h1s = raw0 * scales + biases
        h2s = _eval_layer_symbolic(sym_df_l1, h1s, od1) * sc1 + bs1
        h3s = _eval_layer_symbolic(sym_df_l2, h2s, od2) * sc2 + bs2
        h4s = _eval_layer_symbolic(sym_df_l3, h3s, od3) * sc3 + bs3

        score_SR = ((X_te_mm - h4s.astype(np.float32))**2).mean(axis=1)
        results['SR'] = roc_auc_score(y_t, score_SR)
        print(f"    SR={results['SR']:.4f}", flush=True)
    except Exception as e:
        print(f"    [SR failed: {e}]", flush=True)
        results['SR'] = float('nan')

    # ── 4. KAN-AE reconstruction on test (needed for SHAP/LIME) ──────────────
    Xte_t = torch.tensor(X_te_mm, dtype=torch.float32)
    with torch.no_grad():
        recon_te = model(Xte_t).numpy()
    score_A  = ((X_te_mm - recon_te)**2).mean(axis=1)
    sq_err   = (X_te_mm - recon_te)**2   # (n_test, 44)

    # ── 5. SHAP-Weighted-Recon ─────────────────────────────────────────────────
    try:
        import shap as _shap

        def _predict_A(X_np):
            Xt = torch.tensor(X_np.astype(np.float32))
            with torch.no_grad():
                rec = model(Xt).numpy()
            return np.mean((X_np - rec)**2, axis=1)

        bg        = _shap.kmeans(X_tr_mm, 20)
        expl      = _shap.KernelExplainer(_predict_A, bg)
        fault_idx = np.where(y_t == 1)[0]
        top_f     = fault_idx[np.argsort(score_A[fault_idx])[::-1][:20]]
        shap_vals = expl.shap_values(X_te_mm[top_f], nsamples=200, silent=True)
        shap_w    = np.abs(shap_vals).mean(axis=0)
        shap_w   /= shap_w.sum() + 1e-12
        results['SHAP-Weighted'] = roc_auc_score(y_t, (shap_w * sq_err).sum(axis=1))
        print(f"    SHAP={results['SHAP-Weighted']:.4f}", flush=True)
    except Exception as e:
        print(f"    [SHAP failed: {e}]", flush=True)
        results['SHAP-Weighted'] = float('nan')

    # ── 6. LIME-Weighted-Recon ─────────────────────────────────────────────────
    try:
        from lime import lime_tabular as _lime

        def _predict_A_lime(X_np):
            Xt = torch.tensor(X_np.astype(np.float32))
            with torch.no_grad():
                rec = model(Xt).numpy()
            return np.mean((X_np - rec)**2, axis=1)

        expl_lime   = _lime.LimeTabularExplainer(
            X_tr_mm, feature_names=feat_cols, mode='regression', verbose=False)
        fault_idx   = np.where(y_t == 1)[0]
        top_f       = fault_idx[np.argsort(score_A[fault_idx])[::-1][:20]]
        lime_global = np.zeros(n_features)
        for fi in top_f:
            exp = expl_lime.explain_instance(
                X_te_mm[fi], _predict_A_lime,
                num_features=n_features, num_samples=300)
            for feat_idx, wt in exp.as_map()[1]:
                lime_global[feat_idx] += abs(wt)
        lime_w  = lime_global / (lime_global.sum() + 1e-12)
        results['LIME-Weighted'] = roc_auc_score(y_t, (lime_w * sq_err).sum(axis=1))
        print(f"    LIME={results['LIME-Weighted']:.4f}", flush=True)
    except Exception as e:
        print(f"    [LIME failed: {e}]", flush=True)
        results['LIME-Weighted'] = float('nan')

    return results


# ── Ordered columns for final output ─────────────────────────────────────────
BASELINE_COLS = ['IsolationForest', 'OC-SVM', 'LOF', 'PatchCore',
                 'Autoencoder', 'VAE', 'DeepSVDD', 'TeacherStudent', 'SSL-DAE']
XAI_COLS      = ['SHAP-Weighted', 'LIME-Weighted']
KAN_COLS      = ['A', 'B', 'C', 'D', 'E', 'M', 'F', 'SR']
ALL_METHODS   = BASELINE_COLS + XAI_COLS + KAN_COLS


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t_start = time.time()
    print(f"Multi-seed evaluation  W={W}  N_seeds={len(SEEDS)}")
    print(f"New methods: baselines ({len(BASELINE_COLS)}) + XAI (2) + SR")
    print("=" * 70, flush=True)

    # ── Load existing KAN-AE A–F results ──────────────────────────────────────
    existing_path = os.path.join(RESULTS_DIR, "multi_seed_auc.csv")
    existing_df   = pd.read_csv(existing_path)
    print(f"Loaded existing KAN-AE results from {existing_path}")
    print(f"  Seeds: {sorted(existing_df['seed'].tolist())}")
    print(f"  Scores: {[c for c in existing_df.columns if c != 'seed']}", flush=True)

    # ── Run new methods per seed ──────────────────────────────────────────────
    new_results = []
    for seed in SEEDS:
        t0 = time.time()
        print(f"\n{'─'*70}")
        print(f"Seed {seed:2d}", flush=True)
        res = run_new_methods_one_seed(seed)
        res['elapsed_s'] = time.time() - t0
        new_results.append(res)
        print(f"  → IF={res['IsolationForest']:.4f}  SVM={res['OC-SVM']:.4f}  "
              f"LOF={res['LOF']:.4f}  AE={res['Autoencoder']:.4f}  "
              f"SR={res.get('SR', float('nan')):.4f}  "
              f"SHAP={res.get('SHAP-Weighted', float('nan')):.4f}  "
              f"LIME={res.get('LIME-Weighted', float('nan')):.4f}  "
              f"({res['elapsed_s']:.0f}s)", flush=True)

    # ── Merge with existing A–F ───────────────────────────────────────────────
    new_df = pd.DataFrame(new_results)
    # existing_df has columns: seed, A, B, C, D, E, M, F, n_edges
    merged = new_df.merge(
        existing_df[['seed', 'A', 'B', 'C', 'D', 'E', 'M', 'F']],
        on='seed', how='left')
    merged.to_csv(os.path.join(RESULTS_DIR, "multi_seed_auc_full.csv"), index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY  N={len(merged)} seeds  W={W}  total_time={time.time()-t_start:.0f}s")
    print(f"{'='*70}")
    print(f"{'Method':<22}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"{'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    summary_rows = []
    for method in ALL_METHODS:
        if method not in merged.columns:
            continue
        v = merged[method].dropna()
        if len(v) == 0:
            continue
        mn, sd, lo, hi = v.mean(), v.std(), v.min(), v.max()
        print(f"  {method:<20}  {mn:8.4f}  {sd:8.4f}  {lo:8.4f}  {hi:8.4f}")
        summary_rows.append({"method": method, "mean": mn, "std": sd,
                              "min": lo, "max": hi, "n_seeds": len(v)})

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(RESULTS_DIR, "multi_seed_summary_full.csv"), index=False)

    print(f"\nSaved →")
    print(f"  {os.path.join(RESULTS_DIR, 'multi_seed_auc_full.csv')}")
    print(f"  {os.path.join(RESULTS_DIR, 'multi_seed_summary_full.csv')}")
    print(f"  {os.path.join(RESULTS_DIR, 'multi_seed_eval.log')}")

    _tee.close()
