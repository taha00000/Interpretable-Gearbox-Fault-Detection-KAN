"""
01_data_preparation.py
----------------------
Validates the SpectraQuest Gearbox Dataset folder structure and prints a
summary of all files before the rest of the pipeline is run.

Expected layout (relative to this script):
    Gearbox Dataset/
        Healthy/       *.csv   (10 files, one per load: 0%, 10%, ..., 90%)
        Broken Tooth/  *.csv   (10 files, one per load)

Run this first to confirm everything is in place.
"""

import os
import glob
import re
import pandas as pd

# ── Paths (relative to this script) ──────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Gearbox Dataset")
HEALTHY_DIR = os.path.join(DATASET_DIR, "Healthy")
BROKEN_DIR  = os.path.join(DATASET_DIR, "Broken Tooth")

def get_load(fname: str) -> float:
    match = re.search(r'hz(\d+)\.csv', os.path.basename(fname).lower())
    return float(match.group(1)) if match else -1.0

def check_folder(label: str, folder: str) -> list:
    if not os.path.isdir(folder):
        print(f"  [ERROR] Folder not found: {folder}")
        return []
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    print(f"\n  {label} — {len(files)} files found in: {folder}")
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f, nrows=5)
            n_rows = sum(1 for _ in open(f)) - 1   # fast line count
            rows.append({
                "File"   : os.path.basename(f),
                "Load %"  : int(get_load(f)),
                "Rows"   : n_rows,
                "Cols"   : df.shape[1],
            })
        except Exception as e:
            rows.append({"File": os.path.basename(f), "Error": str(e)})
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    return files

def main():
    print("=" * 60)
    print("  Dataset Validation — SpectraQuest Gearbox Fault Diagnosis")
    print("=" * 60)

    healthy_files = check_folder("HEALTHY", HEALTHY_DIR)
    broken_files  = check_folder("BROKEN TOOTH", BROKEN_DIR)

    total = len(healthy_files) + len(broken_files)
    print(f"\n  Total CSV files found : {total}  (expected 20)")

    if len(healthy_files) == 10 and len(broken_files) == 10:
        print("\n  [OK] Dataset structure looks correct.")
        print("  You can now run:  python 02_feature_extraction.py")
    else:
        print("\n  [WARNING] Expected 10 healthy + 10 broken files.")
        print("  Please check your Gearbox Dataset folder layout.")

    print("=" * 60)

if __name__ == "__main__":
    main()
