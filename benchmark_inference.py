"""
benchmark_inference.py
======================
Measures wall-clock inference time and model size for all 19 methods.
Uses seed=42, W=1000.  CPU only.

Outputs:
  results_kan_symbolic/inference_benchmark.csv
    columns: method, params, inference_ms_per_sample, auc_mean, auc_std

Usage:
    /home/suleiman/miniconda3/envs/go2-convex-mpc/bin/python benchmark_inference.py
"""

import sys
sys.path = [p for p in sys.path if not p.startswith("/home/suleiman/.local/lib")]

import os, time, pickle, warnings
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

BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR   = os.path.join(BASE_DIR, "results_kan_symbolic")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, "efficient_kan"))
from efficient_kan import KAN

# ── Hyperparameters (match multi_seed_eval.py) ────────────────────────────────
W           = 1000
SEED        = 42
B           = 8
EPOCHS      = 150
PRUNE_ALPHA = 0.05
N_REPEATS   = 20    # timing repetitions (median taken)

FEATURE_NAMES = ["mean", "rms", "std", "var", "skew", "kurt", "p2p",
                 "crest", "shape", "margin", "impulse"]
CHANNELS      = ["S1", "S2", "S3", "S4"]
feat_cols     = [f"{ch}_{f}" for ch in CHANNELS for f in FEATURE_NAMES]
n_features    = len(feat_cols)   # 44

# ── Symbolic candidates (match multi_seed_eval.py) ────────────────────────────
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


def _k_center(f, frac=0.1):
    n = f.shape[0]; ns = max(1, int(n * frac))
    c = [np.random.randint(0, n)]
    dists = pairwise_distances(f, f[c], metric='euclidean').flatten()
    for _ in range(ns - 1):
        idx = np.argmax(dists); c.append(idx)
        dists = np.minimum(dists, pairwise_distances(f, f[[idx]]).flatten())
    return f[c]


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def _time_inference(fn, n_reps=N_REPEATS):
    """Warm up then time fn() n_reps times, return median seconds."""
    fn()  # warmup
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Benchmark: W={W}, seed={SEED}, N_REPEATS={N_REPEATS}")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    df_w = pd.read_csv(os.path.join(PROCESSED_DIR, f"features_W{W}.csv"))
    df_h = df_w[df_w['label'] == 0]
    df_b = df_w[df_w['label'] == 1]
    train_h, test_h = train_test_split(df_h, test_size=0.3, random_state=SEED)
    test_df = pd.concat([test_h, df_b]).reset_index(drop=True)
    y_t = test_df['label'].values
    n_test = len(y_t)
    print(f"Train healthy: {len(train_h)}  Test: {n_test} (pos={y_t.sum()})")

    sc_mm  = MinMaxScaler().fit(train_h[feat_cols].values)
    sc_std = StandardScaler().fit(train_h[feat_cols].values)

    X_tr_mm      = sc_mm.transform(train_h[feat_cols].values).astype(np.float32)
    X_te_mm      = sc_mm.transform(test_df[feat_cols].values).astype(np.float32)
    X_healthy_mm = sc_mm.transform(df_h[feat_cols].values).astype(np.float32)
    X_tr_std     = sc_std.transform(train_h[feat_cols].values).astype(np.float32)
    X_te_std     = sc_std.transform(test_df[feat_cols].values).astype(np.float32)

    Xtr_s = torch.tensor(X_tr_std, dtype=torch.float32)
    Xte_s = torch.tensor(X_te_std, dtype=torch.float32)
    Xte_t = torch.tensor(X_te_mm,  dtype=torch.float32)

    d = n_features
    ldr_s = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_s, Xtr_s), batch_size=256, shuffle=True)

    # ── Load AUC summary from multi_seed results ───────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, "multi_seed_summary_full.csv")
    auc_summary = {}
    if os.path.exists(summary_path):
        df_sum = pd.read_csv(summary_path)
        for _, row in df_sum.iterrows():
            auc_summary[row['method']] = (float(row['mean']), float(row['std']))
    # Map paper method names to summary keys
    name_map = {
        'IsolationForest': 'IsolationForest',
        'OC-SVM': 'OC-SVM',
        'LOF': 'LOF',
        'PatchCore': 'PatchCore',
        'Autoencoder': 'Autoencoder',
        'VAE': 'VAE',
        'DeepSVDD': 'DeepSVDD',
        'TeacherStudent': 'TeacherStudent',
        'SSL-DAE': 'SSL-DAE',
        'SHAP-Weighted-Recon': 'SHAP-Weighted',
        'LIME-Weighted-Recon': 'LIME-Weighted',
        'KAN-AE-Recon (A)': 'A',
        'KAN-AE-Symbolic (B)': 'B',
        'KAN-AE-Combined (C)': 'C',
        'KAN-AE-Symbolic-D': 'D',
        'KAN-AE-Combined-E': 'E',
        'KAN-AE-Mahal (M)': 'M',
        'KAN-AE-Combined-F': 'F',
        'KAN-AE-SymRecon': 'SR',
    }

    rows = []  # output rows

    def record(display_name, params_val, ms_per_sample, auc_m=None, auc_s=None):
        key = name_map.get(display_name, display_name)
        if auc_m is None and key in auc_summary:
            auc_m, auc_s = auc_summary[key]
        rows.append({
            "method": display_name,
            "params": params_val,
            "inference_ms_per_sample": round(ms_per_sample, 5),
            "auc_mean": round(auc_m * 100, 2) if auc_m else None,
            "auc_std":  round(auc_s * 100, 3) if auc_s else None,
        })
        print(f"  {display_name:<30} params={params_val:<10} "
              f"ms/sample={ms_per_sample:.5f}  "
              f"AUC={auc_m*100:.2f}±{auc_s*100:.2f}%"
              if auc_m else f"  {display_name}")

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Isolation Forest
    print("\n[1/19] Isolation Forest")
    clf_if = IsolationForest(n_estimators=100, random_state=SEED).fit(X_tr_std)
    params_if = len(pickle.dumps(clf_if)) / 1024  # KB
    t = _time_inference(lambda: clf_if.decision_function(X_te_std))
    record("IsolationForest", f"{params_if:.1f} KB", t * 1000 / n_test)

    # 2. OC-SVM
    print("[2/19] OC-SVM")
    clf_svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_tr_std)
    params_svm = len(pickle.dumps(clf_svm)) / 1024
    t = _time_inference(lambda: clf_svm.decision_function(X_te_std))
    record("OC-SVM", f"{params_svm:.1f} KB", t * 1000 / n_test)

    # 3. LOF
    print("[3/19] LOF")
    clf_lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(X_tr_std)
    params_lof = len(pickle.dumps(clf_lof)) / 1024
    t = _time_inference(lambda: clf_lof.decision_function(X_te_std))
    record("LOF", f"{params_lof:.1f} KB", t * 1000 / n_test)

    # 4. PatchCore
    print("[4/19] PatchCore")
    coreset = _k_center(X_tr_std, frac=0.1)
    params_pc = coreset.nbytes / 1024  # KB
    t = _time_inference(lambda: pairwise_distances(X_te_std, coreset, metric='euclidean').min(axis=1))
    record("PatchCore", f"{params_pc:.1f} KB", t * 1000 / n_test)

    # 5. Autoencoder
    print("[5/19] Autoencoder")
    ae = _AE(d); o = torch.optim.Adam(ae.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s: o.zero_grad(); nn.MSELoss()(ae(bx), bx).backward(); o.step()
    ae.eval()
    params_ae = _count_params(ae)
    with torch.no_grad():
        def _ae_inf(): return torch.mean((ae(Xte_s) - Xte_s)**2, dim=1).numpy()
    t = _time_inference(lambda: ae.eval() or torch.no_grad().__enter__() or torch.mean((ae(Xte_s) - Xte_s)**2, dim=1).numpy())
    # simpler timing
    ae.eval()
    _ae_inf_times = []
    _ = ae(Xte_s)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch.mean((ae(Xte_s) - Xte_s)**2, dim=1).numpy()
        _ae_inf_times.append(time.perf_counter() - t0)
    t_ae = float(np.median(_ae_inf_times))
    with torch.no_grad(): 
        mse_scores = torch.mean((ae(Xte_s) - Xte_s)**2, dim=1).numpy()
        auc_ae = roc_auc_score(y_t, mse_scores)
        
    record("Autoencoder", params_ae, t_ae * 1000 / n_test)

    # 6. VAE
    print("[6/19] VAE")
    vae = _VAE(d); o = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad(); x, m, lv = vae(bx)
            (nn.MSELoss()(x, bx) - 0.0005*torch.sum(1+lv-m.pow(2)-lv.exp())).backward(); o.step()
    vae.eval()
    params_vae = _count_params(vae)
    _vae_times = []
    with torch.no_grad(): _ = vae(Xte_s)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            xr, _, _ = vae(Xte_s)
            _ = torch.mean((xr - Xte_s)**2, dim=1).numpy()
        _vae_times.append(time.perf_counter() - t0)
    record("VAE", params_vae, float(np.median(_vae_times)) * 1000 / n_test)

    # 7. DeepSVDD
    print("[7/19] DeepSVDD")
    svdd = _SVDD(d); o = torch.optim.Adam(svdd.parameters(), lr=1e-3)
    with torch.no_grad(): c_sv = torch.mean(svdd(Xtr_s), dim=0)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad(); torch.mean(torch.sum((svdd(bx)-c_sv)**2, dim=1)).backward(); o.step()
    svdd.eval()
    params_svdd = _count_params(svdd)
    _svdd_times = []
    with torch.no_grad(): _ = svdd(Xte_s)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch.sum((svdd(Xte_s)-c_sv)**2, dim=1).numpy()
        _svdd_times.append(time.perf_counter() - t0)
    record("DeepSVDD", params_svdd, float(np.median(_svdd_times)) * 1000 / n_test)

    # 8. Teacher-Student
    print("[8/19] Teacher-Student")
    tnet = _TS(d); snet = _TS(d)
    for p in tnet.parameters(): p.requires_grad = False
    o = torch.optim.Adam(snet.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad(); nn.MSELoss()(snet(bx), tnet(bx)).backward(); o.step()
    snet.eval(); tnet.eval()
    params_ts = _count_params(snet) + _count_params(tnet)
    _ts_times = []
    with torch.no_grad(): _ = snet(Xte_s)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch.mean((snet(Xte_s)-tnet(Xte_s))**2, dim=1).numpy()
        _ts_times.append(time.perf_counter() - t0)
    record("TeacherStudent", params_ts, float(np.median(_ts_times)) * 1000 / n_test)

    # 9. SSL-DAE
    print("[9/19] SSL-DAE")
    dae = _DAE(d); o = torch.optim.Adam(dae.parameters(), lr=1e-3)
    for _ in range(15):
        for bx, _ in ldr_s:
            o.zero_grad()
            nn.MSELoss()(dae(bx + 0.5*torch.randn_like(bx)), bx).backward(); o.step()
    dae.eval()
    params_dae = _count_params(dae)
    _dae_times = []
    with torch.no_grad(): _ = dae(Xte_s)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch.mean((dae(Xte_s)-Xte_s)**2, dim=1).numpy()
        _dae_times.append(time.perf_counter() - t0)
    record("SSL-DAE", params_dae, float(np.median(_dae_times)) * 1000 / n_test)

    # ── Train KAN-AE (needed for all KAN variants) ────────────────────────────
    print("\n[KAN-AE] Training (150 epochs)...")
    model = KAN(layers_hidden=[n_features, n_features//2, B, n_features//2, n_features],
                grid_size=5, spline_order=3)
    Xt  = torch.tensor(X_tr_mm, dtype=torch.float32)
    ldr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, Xt), batch_size=256, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(EPOCHS):
        for bx, bt in ldr:
            opt.zero_grad(); nn.MSELoss()(model(bx), bt).backward(); opt.step()
        if (ep+1) % 50 == 0: print(f"  epoch {ep+1}/{EPOCHS}")
    model.eval(); model.cpu()
    params_kan = _count_params(model)
    print(f"  KAN-AE params: {params_kan}")

    # Precompute reconstruction for shared use
    with torch.no_grad():
        recon_te = model(Xte_t).numpy()
    score_A  = ((X_te_mm - recon_te)**2).mean(axis=1)
    sq_err   = (X_te_mm - recon_te)**2

    # Helper: time KAN forward + MSE
    _kan_times = []
    with torch.no_grad(): _ = model(Xte_t)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            r = model(Xte_t).numpy()
            _ = ((X_te_mm - r)**2).mean(axis=1)
        _kan_times.append(time.perf_counter() - t0)
    t_kan_fwd = float(np.median(_kan_times))

    # ── Layer 0 symbolic (for B, C, SHAP, LIME, SR) ──────────────────────────
    print("\n[Symbolic] Building layer 0 equations...")
    layer0 = model.layers[0]
    with torch.no_grad():
        sw0   = layer0.scaled_spline_weight.detach().abs().mean(dim=2).cpu().numpy()
        bw0   = layer0.base_weight.detach().abs().cpu().numpy()
        comb0 = sw0 + bw0
    thr0   = PRUNE_ALPHA * comb0.max()
    edges0 = [(j, i) for j in range(22) for i in range(n_features) if comb0[j,i] > thr0]
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
    print(f"  Layer 0: {len(edges0)} surviving edges, {len(sym_df_s)} symbolised")

    # Calibrate layer 0
    Xh_t = torch.tensor(X_healthy_mm, dtype=torch.float32)
    with torch.no_grad():
        h1_cal = model.layers[0](Xh_t).numpy()
    raw_cal = np.zeros((len(X_healthy_mm), 22), dtype=np.float64)
    for _, row in sym_df_s.iterrows():
        j2, i2 = int(row['out_neuron']), int(row['in_feat_idx'])
        fn = CANDIDATES_DICT[row['best_symbol']]
        try: raw_cal[:, j2] += fn(X_healthy_mm[:, i2].astype(np.float64), *row['params'])
        except Exception: pass
    scales_l0 = np.ones(22, dtype=np.float64); biases_l0 = np.zeros(22, dtype=np.float64)
    for j2 in range(22):
        rj = raw_cal[:, j2]; tj = h1_cal[:, j2]
        if rj.std() < 1e-9:
            scales_l0[j2] = 0.0; biases_l0[j2] = float(tj.mean()); continue
        s, b = np.polyfit(rj, tj, 1); scales_l0[j2] = s; biases_l0[j2] = b

    # Score B: symbolic violation on layer 0 output
    Xte_f  = X_te_mm.astype(np.float64)
    raw0_te = np.zeros((len(Xte_f), 22), dtype=np.float64)
    for _, row in sym_df_s.iterrows():
        j2, i2 = int(row['out_neuron']), int(row['in_feat_idx'])
        try:
            raw0_te[:, j2] += CANDIDATES_DICT[row['best_symbol']](Xte_f[:, i2], *row['params'])
        except Exception: pass

    with torch.no_grad():
        h1_true_te = model.layers[0](Xte_t).numpy()
    h1_sym_te = raw0_te * scales_l0 + biases_l0
    score_B = ((h1_true_te - h1_sym_te)**2).sum(axis=1)

    # Compute score B normalisation ranges on training healthy
    raw0_tr = np.zeros((len(X_tr_mm), 22), dtype=np.float64)
    for _, row in sym_df_s.iterrows():
        j2, i2 = int(row['out_neuron']), int(row['in_feat_idx'])
        try:
            raw0_tr[:, j2] += CANDIDATES_DICT[row['best_symbol']](
                X_tr_mm[:, i2].astype(np.float64), *row['params'])
        except Exception: pass
    with torch.no_grad():
        h1_true_tr = model.layers[0](torch.tensor(X_tr_mm)).numpy()
    h1_sym_tr = raw0_tr * scales_l0 + biases_l0
    scA_tr = ((X_tr_mm - recon_te[:len(X_tr_mm)])**2).mean(axis=1) if len(X_tr_mm) < len(recon_te) else score_A[:len(X_tr_mm)]
    # Use actual training reconstruction for normalisation
    with torch.no_grad():
        recon_tr = model(torch.tensor(X_tr_mm)).numpy()
    sA_tr = ((X_tr_mm - recon_tr)**2).mean(axis=1)
    sB_tr = ((h1_true_tr - h1_sym_tr)**2).sum(axis=1)
    mn_A, mx_A = sA_tr.min(), sA_tr.max()
    mn_B, mx_B = sB_tr.min(), sB_tr.max()
    norm_A = (score_A - mn_A) / (mx_A - mn_A + 1e-12)
    norm_B = (score_B - mn_B) / (mx_B - mn_B + 1e-12)
    score_C = 0.5 * norm_A + 0.5 * norm_B

    # ── 10. SHAP-Weighted ──────────────────────────────────────────────────────
    print("\n[10/19] SHAP-Weighted")
    shap_w = None
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
        print(f"  SHAP weights computed ({shap_w.shape})")
    except Exception as e:
        print(f"  [SHAP weight computation failed: {e}]")
        shap_w = np.ones(n_features) / n_features

    # Inference: KAN forward + weighted MSE (weights are precomputed)
    _shap_times = []
    with torch.no_grad(): _ = model(Xte_t)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            r = model(Xte_t).numpy()
            _ = (shap_w * (X_te_mm - r)**2).sum(axis=1)
        _shap_times.append(time.perf_counter() - t0)
    record("SHAP-Weighted-Recon", params_kan, float(np.median(_shap_times)) * 1000 / n_test)

    # ── 11. LIME-Weighted ──────────────────────────────────────────────────────
    print("[11/19] LIME-Weighted")
    lime_w = None
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
        lime_w = lime_global / (lime_global.sum() + 1e-12)
        print(f"  LIME weights computed")
    except Exception as e:
        print(f"  [LIME weight computation failed: {e}]")
        lime_w = np.ones(n_features) / n_features

    _lime_times = []
    with torch.no_grad(): _ = model(Xte_t)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            r = model(Xte_t).numpy()
            _ = (lime_w * (X_te_mm - r)**2).sum(axis=1)
        _lime_times.append(time.perf_counter() - t0)
    record("LIME-Weighted-Recon", params_kan, float(np.median(_lime_times)) * 1000 / n_test)

    # ── 12. KAN-AE-Recon (A) ──────────────────────────────────────────────────
    print("[12/19] KAN-AE-Recon (A)")
    record("KAN-AE-Recon (A)", params_kan, t_kan_fwd * 1000 / n_test)

    # ── 13. KAN-AE-Symbolic (B) ───────────────────────────────────────────────
    print("[13/19] KAN-AE-Symbolic (B)")
    # Inference: KAN forward pass (for h1_true) + symbolic evaluation (for h1_sym)
    # We need both; time the full Score-B computation
    _bscoredef_rows = list(sym_df_s.iterrows())  # cache iteration
    _B_times = []
    with torch.no_grad(): _ = model.layers[0](Xte_t)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            h1t = model.layers[0](Xte_t).numpy()
        raw = np.zeros((len(Xte_f), 22), dtype=np.float64)
        for _, row in _bscoredef_rows:
            j2, i2 = int(row['out_neuron']), int(row['in_feat_idx'])
            try:
                raw[:, j2] += CANDIDATES_DICT[row['best_symbol']](Xte_f[:, i2], *row['params'])
            except Exception: pass
        h1s = raw * scales_l0 + biases_l0
        _ = ((h1t - h1s)**2).sum(axis=1)
        _B_times.append(time.perf_counter() - t0)
    record("KAN-AE-Symbolic (B)", params_kan, float(np.median(_B_times)) * 1000 / n_test)

    # ── 14. KAN-AE-Combined (C) ───────────────────────────────────────────────
    print("[14/19] KAN-AE-Combined (C)")
    # Same as Score A + Score B; time combined (essentially max of A and B inference)
    _C_times = []
    with torch.no_grad(): _ = model(Xte_t)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            r = model(Xte_t).numpy()
            h1t = model.layers[0](Xte_t).numpy()
        raw = np.zeros((len(Xte_f), 22), dtype=np.float64)
        for _, row in _bscoredef_rows:
            j2, i2 = int(row['out_neuron']), int(row['in_feat_idx'])
            try:
                raw[:, j2] += CANDIDATES_DICT[row['best_symbol']](Xte_f[:, i2], *row['params'])
            except Exception: pass
        h1s = raw * scales_l0 + biases_l0
        sA = ((X_te_mm - r)**2).mean(axis=1)
        sB = ((h1t - h1s)**2).sum(axis=1)
        nA = (sA - sA.min()) / (sA.max() - sA.min() + 1e-12)
        nB = (sB - sB.min()) / (sB.max() - sB.min() + 1e-12)
        _ = 0.5 * nA + 0.5 * nB
        _C_times.append(time.perf_counter() - t0)
    record("KAN-AE-Combined (C)", params_kan, float(np.median(_C_times)) * 1000 / n_test)

    # ── 15. KAN-AE-Symbolic-D ─────────────────────────────────────────────────
    print("[15/19] KAN-AE-Symbolic-D (≈ Score B; different layer config)")
    # D uses R²-weighted symbolic residuals; inference cost same as B
    record("KAN-AE-Symbolic-D", params_kan, float(np.median(_B_times)) * 1000 / n_test)

    # ── 16. KAN-AE-Combined-E ─────────────────────────────────────────────────
    print("[16/19] KAN-AE-Combined-E (≈ Score C; alternating layers)")
    record("KAN-AE-Combined-E", params_kan, float(np.median(_C_times)) * 1000 / n_test)

    # ── 17. KAN-AE-Mahal (M) ──────────────────────────────────────────────────
    print("[17/19] KAN-AE-Mahal (M)")
    # M = Mahalanobis distance in latent space; inference: KAN forward + latent extraction + dist
    with torch.no_grad():
        H_in_l1 = model.layers[0](torch.tensor(X_tr_mm)).numpy()
    cov    = np.cov(H_in_l1.T)
    try:
        cov_inv = np.linalg.pinv(cov)
    except Exception:
        cov_inv = np.eye(22)
    _M_times = []
    with torch.no_grad(): _ = model.layers[0](Xte_t)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            z = model.layers[0](Xte_t).numpy()
        mu = H_in_l1.mean(axis=0)
        diff = z - mu
        _ = np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
        _M_times.append(time.perf_counter() - t0)
    record("KAN-AE-Mahal (M)", params_kan, float(np.median(_M_times)) * 1000 / n_test)

    # ── 18. KAN-AE-Combined-F ─────────────────────────────────────────────────
    print("[18/19] KAN-AE-Combined-F (Recon + Mahal blend)")
    _F_times = []
    with torch.no_grad(): _ = model(Xte_t)  # warmup
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        with torch.no_grad():
            r = model(Xte_t).numpy()
            z = model.layers[0](Xte_t).numpy()
        sA = ((X_te_mm - r)**2).mean(axis=1)
        mu = H_in_l1.mean(axis=0); diff = z - mu
        sM = np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
        nA = (sA - sA.min()) / (sA.max() - sA.min() + 1e-12)
        nM = (sM - sM.min()) / (sM.max() - sM.min() + 1e-12)
        _ = 0.5 * nA + 0.5 * nM
        _F_times.append(time.perf_counter() - t0)
    record("KAN-AE-Combined-F", params_kan, float(np.median(_F_times)) * 1000 / n_test)

    # ── 19. KAN-AE-SymRecon (SR) — NO NEURAL NETWORK AT INFERENCE ────────────
    print("\n[19/19] KAN-AE-SymRecon (SR) — building full symbolic forward pass...")
    try:
        sym_df_l1, H_l1, od1 = _symbolise_layer(model, 1, X_healthy_mm)
        sym_df_l2, H_l2, od2 = _symbolise_layer(model, 2, X_healthy_mm)
        sym_df_l3, H_l3, od3 = _symbolise_layer(model, 3, X_healthy_mm)
        sc1, bs1, _ = _calibrate_layer(sym_df_l1, H_l1, model, 1, od1)
        sc2, bs2, _ = _calibrate_layer(sym_df_l2, H_l2, model, 2, od2)
        sc3, bs3, _ = _calibrate_layer(sym_df_l3, H_l3, model, 3, od3)

        # Count symbolic parameters: sum of len(params) per row + calibration scalars
        def _sym_param_count(df):
            return int(df['params'].apply(len).sum()) + 2 * int(df[['out_neuron']].max().values[0] + 1)
        sym_params_l0 = len(sym_df_s) * 3 + 2 * 22  # ~3 params/edge + scale+bias per neuron
        sym_params_l1 = _sym_param_count(sym_df_l1) if len(sym_df_l1) else 0
        sym_params_l2 = _sym_param_count(sym_df_l2) if len(sym_df_l2) else 0
        sym_params_l3 = _sym_param_count(sym_df_l3) if len(sym_df_l3) else 0
        total_sym_params = sym_params_l0 + sym_params_l1 + sym_params_l2 + sym_params_l3
        print(f"  Symbolic params: L0={sym_params_l0}, L1={sym_params_l1}, "
              f"L2={sym_params_l2}, L3={sym_params_l3} → total={total_sym_params}")

        # Precompile layer-row lookups for fast inference timing
        l0_rows = [(int(r['out_neuron']), int(r['in_feat_idx']),
                    CANDIDATES_DICT[r['best_symbol']], r['params'])
                   for _, r in sym_df_s.iterrows()]
        l1_rows = [(int(r['out_neuron']), int(r['in_feat_idx']),
                    CANDIDATES_DICT[r['best_symbol']], r['params'], r['x_min'], r['x_span'])
                   for _, r in sym_df_l1.iterrows()] if len(sym_df_l1) else []
        l2_rows = [(int(r['out_neuron']), int(r['in_feat_idx']),
                    CANDIDATES_DICT[r['best_symbol']], r['params'], r['x_min'], r['x_span'])
                   for _, r in sym_df_l2.iterrows()] if len(sym_df_l2) else []
        l3_rows = [(int(r['out_neuron']), int(r['in_feat_idx']),
                    CANDIDATES_DICT[r['best_symbol']], r['params'], r['x_min'], r['x_span'])
                   for _, r in sym_df_l3.iterrows()] if len(sym_df_l3) else []

        def _sr_inference(X_in):
            """Full symbolic forward pass — no PyTorch."""
            Xf = X_in.astype(np.float64)
            # Layer 0
            h0 = np.zeros((len(Xf), 22), dtype=np.float64)
            for j2, i2, fn, p in l0_rows:
                try: h0[:, j2] += fn(Xf[:, i2], *p)
                except Exception: pass
            h1 = h0 * scales_l0 + biases_l0
            # Layer 1
            if l1_rows:
                h1b = np.zeros((len(Xf), od1), dtype=np.float64)
                for j2, i2, fn, p, xm, xs in l1_rows:
                    x_n = (h1[:, i2] - xm) / (xs + 1e-12)
                    try: h1b[:, j2] += fn(x_n, *p)
                    except Exception: pass
                h2 = h1b * sc1 + bs1
            else:
                h2 = h1[:, :od1] * sc1 + bs1
            # Layer 2
            if l2_rows:
                h2b = np.zeros((len(Xf), od2), dtype=np.float64)
                for j2, i2, fn, p, xm, xs in l2_rows:
                    x_n = (h2[:, i2] - xm) / (xs + 1e-12)
                    try: h2b[:, j2] += fn(x_n, *p)
                    except Exception: pass
                h3 = h2b * sc2 + bs2
            else:
                h3 = h2[:, :od2] * sc2 + bs2
            # Layer 3
            if l3_rows:
                h3b = np.zeros((len(Xf), od3), dtype=np.float64)
                for j2, i2, fn, p, xm, xs in l3_rows:
                    x_n = (h3[:, i2] - xm) / (xs + 1e-12)
                    try: h3b[:, j2] += fn(x_n, *p)
                    except Exception: pass
                h4 = h3b * sc3 + bs3
            else:
                h4 = h3[:, :od3] * sc3 + bs3
            return ((X_in - h4.astype(np.float32))**2).mean(axis=1)

        # Warmup + time SR inference (pure numpy, no torch)
        _ = _sr_inference(X_te_mm)
        _sr_times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            _ = _sr_inference(X_te_mm)
            _sr_times.append(time.perf_counter() - t0)
        t_sr = float(np.median(_sr_times))
        record("KAN-AE-SymRecon", total_sym_params, t_sr * 1000 / n_test)

    except Exception as e:
        print(f"  [SR failed: {e}]")
        record("KAN-AE-SymRecon", "N/A", float('nan'))

    # ── Save results ──────────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "inference_benchmark.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved → {out_path}")
    print(df_out.to_string(index=False))

    # Print summary table for paper
    print(f"\n{'='*60}")
    print("PAPER TABLE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'Params':<12} {'ms/sample':>10} {'AUC (%)':>12}")
    print(f"{'─'*30} {'─'*12} {'─'*10} {'─'*12}")
    for r in rows:
        auc_str = f"{r['auc_mean']:.2f}±{r['auc_std']:.2f}" if r['auc_mean'] else "N/A"
        print(f"{r['method']:<30} {str(r['params']):<12} {r['inference_ms_per_sample']:>10.5f} {auc_str:>12}")


if __name__ == "__main__":
    main()
