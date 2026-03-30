"""
02_feature_extraction.py
------------------------
Sliding non-overlapping window feature extraction pipeline.

For each window size W in {300, 400, 500, 600, 700, 800}, this script:
  1. Reads every healthy and broken-tooth CSV from the Gearbox Dataset folder.
  2. Applies a non-overlapping window of size W to each of the 4 sensor channels.
  3. Extracts 11 statistical features per sensor (44 total) for each window.
  4. Appends the load value and binary class label (0=healthy, 1=faulty).
  5. Saves the resulting tabular dataset to data/processed/features_W<W>.csv.

Statistical features extracted per sensor window (11 total):
  Mean, RMS, Standard Deviation, Variance, Skewness, Kurtosis,
  Peak-to-Peak (P2P), Crest Factor, Shape Factor, Margin Factor, Impulse Factor

  Margin Factor = max(|x|) / mean(sqrt(|x|))^2
  (same definition as Hassan et al., Machines 2026)
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

# ── Paths (all relative to this script) ──────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Gearbox Dataset")
OUT_DIR     = os.path.join(BASE_DIR, "data", "processed")
WINDOWS     = [300, 400, 500, 600, 700, 800]

FEATURE_NAMES = [
    "mean", "rms", "std", "var", "skew",
    "kurt", "p2p", "crest", "shape", "margin", "impulse"
]


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract 11 statistical features from a (W, 4) window array.
    Returns a 1-D array of length 44 (11 features × 4 sensors),
    ordered as: [S1_feat1, ..., S1_feat11, S2_feat1, ..., S4_feat11].
    """
    eps = 1e-10
    mean_v  = np.mean(window, axis=0)
    rms_v   = np.sqrt(np.mean(window ** 2, axis=0))
    std_v   = np.std(window, axis=0, ddof=1)
    var_v   = np.var(window, axis=0, ddof=1)
    skew_v  = skew(window, axis=0, bias=False)
    kurt_v  = kurtosis(window, axis=0, bias=False)
    max_v   = np.max(window, axis=0)
    min_v   = np.min(window, axis=0)
    p2p_v   = max_v - min_v
    maxabs  = np.max(np.abs(window), axis=0)
    meanabs = np.mean(np.abs(window), axis=0)
    meansqrt = np.mean(np.sqrt(np.abs(window)), axis=0)

    rms_s     = np.where(rms_v == 0,      eps, rms_v)
    mabs_s    = np.where(meanabs == 0,    eps, meanabs)
    msqrt_s   = np.where(meansqrt == 0,   eps, meansqrt)

    crest_v   = maxabs / rms_s
    shape_v   = rms_v  / mabs_s
    margin_v  = maxabs / (msqrt_s ** 2)   # Margin Factor (Hassan et al.)
    impulse_v = maxabs / mabs_s

    # Stack → (11, 4) → flatten → (44,)  [sensor-major order]
    features = np.vstack((
        mean_v, rms_v, std_v, var_v, skew_v,
        kurt_v, p2p_v, crest_v, shape_v, margin_v, impulse_v
    ))
    return features.T.flatten()


def process_file(filepath: str, label: int, load: float, W: int) -> list:
    """Return a list of feature-rows from one CSV file."""
    try:
        df   = pd.read_csv(filepath)
        cols = [c for c in df.columns if c.lower().startswith('a')]
        if len(cols) == 4:
            data = df[cols].values
        elif df.shape[1] >= 4:
            data = df.iloc[:, :4].values
        else:
            print(f"  [SKIP] {filepath}: unexpected shape {df.shape}")
            return []

        n_windows = len(data) // W
        rows = []
        for i in range(n_windows):
            window = data[i * W: (i + 1) * W]
            row    = np.concatenate((extract_features(window), [load, label]))
            rows.append(row)
        return rows
    except Exception as e:
        print(f"  [ERROR] {filepath}: {e}")
        return []


def get_load(fname: str) -> float:
    match = re.search(r'hz(\d+)\.csv', os.path.basename(fname).lower())
    return float(match.group(1)) if match else 0.0


# ── Column names ──────────────────────────────────────────────────────────────
def make_col_names() -> list:
    cols = []
    for s in range(1, 5):
        for fn in FEATURE_NAMES:
            cols.append(f"S{s}_{fn}")
    cols += ["load", "label"]
    return cols


COL_NAMES = make_col_names()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    healthy_files = sorted(glob.glob(os.path.join(DATASET_DIR, "Healthy",      "*.csv")))
    broken_files  = sorted(glob.glob(os.path.join(DATASET_DIR, "Broken Tooth", "*.csv")))

    print(f"Found {len(healthy_files)} healthy and {len(broken_files)} broken-tooth files.")
    print(f"Extracting {len(FEATURE_NAMES)} features per sensor × 4 sensors = "
          f"{len(FEATURE_NAMES) * 4} features per sample.")

    file_info = (
        [(f, 0, get_load(f)) for f in healthy_files] +
        [(f, 1, get_load(f)) for f in broken_files]
    )

    for W in WINDOWS:
        print(f"\nProcessing window size W = {W} samples ({W / 50000 * 1000:.1f} ms @ 50 kHz)…")
        all_rows = []
        for filepath, label, load in file_info:
            all_rows.extend(process_file(filepath, label, load, W))

        df_out = pd.DataFrame(all_rows, columns=COL_NAMES)
        out_path = os.path.join(OUT_DIR, f"features_W{W}.csv")
        df_out.to_csv(out_path, index=False)

        n_healthy = (df_out["label"] == 0).sum()
        n_faulty  = (df_out["label"] == 1).sum()
        print(f"  Saved {len(df_out)} samples "
              f"({n_healthy} healthy / {n_faulty} faulty) → {out_path}")

    print("\nFeature extraction complete.")
    print(f"Feature columns: {COL_NAMES[:5]} … ({len(COL_NAMES)} total incl. load, label)")


if __name__ == "__main__":
    main()
