"""
calculate_twonn_id.py
---------------------
Calculates the TwoNN Intrinsic Dimension (ID) of the extracted 44-dimensional features.
It computes the ID for both the entire dataset and the strictly healthy dataset
across all window sizes (W=300 to 1000).

The TWO-NN estimator relies on the ratio of the distance to the second nearest 
neighbor over the distance to the first nearest neighbor.
"""

import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
WINDOWS = [300, 400, 500, 600, 700, 800, 900, 1000]

def calculate_twonn_id(X):
    """
    Computes the TWO-NN intrinsic dimension estimate.
    F(mu) = 1 - mu^(-d) => d = N / sum(ln(mu)) where mu = r2 / r1
    """
    # Find 3 nearest neighbors (k=0 is the point itself, k=1 is 1st NN, k=2 is 2nd NN)
    nn = NearestNeighbors(n_neighbors=3, metric='euclidean', n_jobs=-1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    # Filter identical duplicate points (r1 == 0) to avoid division by zero
    valid_idx = r1 > 1e-10
    r1 = r1[valid_idx]
    r2 = r2[valid_idx]
    
    if len(r1) == 0:
        return np.nan
        
    mu = r2 / r1
    
    # MLE of intrinsic dimension
    N = len(mu)
    id_estimate = N / np.sum(np.log(mu))
    return id_estimate

def main():
    print("======================================================")
    print(" TWO-NN Intrinsic Dimension Estimation")
    print("======================================================")
    
    for W in WINDOWS:
        filepath = os.path.join(PROCESSED_DIR, f"features_W{W}.csv")
        if not os.path.exists(filepath):
            print(f"Skipping W={W}, file not found.")
            continue
            
        df = pd.read_csv(filepath)
        feat_cols = [c for c in df.columns if c not in ['load', 'label']]
        
        df_healthy = df[df['label'] == 0]
        
        # Standardizing features is generally essential for distance-based ND metrics
        scaler_all = StandardScaler()
        X_all = scaler_all.fit_transform(df[feat_cols].values)
        
        scaler_healthy = StandardScaler()
        X_healthy = scaler_healthy.fit_transform(df_healthy[feat_cols].values)
        
        id_all = calculate_twonn_id(X_all)
        id_healthy = calculate_twonn_id(X_healthy)
        
        print(f"Window Size W={W}:")
        print(f"  All Data    (N={len(X_all)}): Original Dim = {len(feat_cols)}, TwoNN Intrinsic Dim = {id_all:.2f}")
        print(f"  Healthy Only(N={len(X_healthy)}): Original Dim = {len(feat_cols)}, TwoNN Intrinsic Dim = {id_healthy:.2f}\n")

if __name__ == "__main__":
    main()
