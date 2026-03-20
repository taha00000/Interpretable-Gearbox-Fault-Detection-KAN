import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

def extract_features(window_data):
    """
    Extracts 10 statistical features for an array of data (1D or 2D where columns are channels).
    Returns a numpy array of features.
    Features: Mean, RMS, Std, Var, Skewness, Kurtosis, P2P, Crest Factor, Shape Factor, Impulse Factor
    """
    # window_data shape: (W, 4)
    # compute features along axis 0
    mean_val = np.mean(window_data, axis=0)
    rms_val = np.sqrt(np.mean(window_data**2, axis=0))
    std_val = np.std(window_data, axis=0, ddof=1)
    var_val = np.var(window_data, axis=0, ddof=1)
    skew_val = skew(window_data, axis=0, bias=False)
    kurt_val = kurtosis(window_data, axis=0, bias=False)
    
    max_val = np.max(window_data, axis=0)
    min_val = np.min(window_data, axis=0)
    p2p_val = max_val - min_val
    
    max_abs = np.max(np.abs(window_data), axis=0)
    mean_abs = np.mean(np.abs(window_data), axis=0)
    
    # Avoid division by zero
    eps = 1e-10
    rms_safe = np.where(rms_val == 0, eps, rms_val)
    mean_abs_safe = np.where(mean_abs == 0, eps, mean_abs)
    
    crest_factor = max_abs / rms_safe
    shape_factor = rms_val / mean_abs_safe
    impulse_factor = max_abs / mean_abs_safe
    
    # Stack features to get shape (10, 4) then flatten to (40,)
    features = np.vstack((mean_val, rms_val, std_val, var_val, skew_val, kurt_val, 
                          p2p_val, crest_factor, shape_factor, impulse_factor))
    return features.T.flatten() # Order: Sensor1_f1..10, Sensor2_f1..10, etc.

def process_file(filepath, label, load, W):
    try:
        df = pd.read_csv(filepath)
        # Check if columns are a1, a2, a3, a4
        cols = [c for c in df.columns if 'a' in c.lower() or 'sensor' in c.lower()]
        if len(cols) != 4:
            # Maybe the file has no header but just 4 columns
            if df.shape[1] == 4:
                data = df.values
            else:
                data = df.iloc[:, :4].values
        else:
            data = df[cols].values
            
        num_windows = len(data) // W
        rows = []
        for i in range(num_windows):
            window = data[i*W : (i+1)*W]
            feats = extract_features(window)
            # Append load and label
            row = np.concatenate((feats, [load, label]))
            rows.append(row)
        return rows
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []

def main():
    dataset_dir = r"c:/Users/tahah/OneDrive/Desktop/Moving-window-based-feature-extraction-method-for-vibration-based-condition-monitoring-main/Gearbox Dataset"
    windows = [300, 400, 500, 600, 700, 800]
    out_dir = r"c:/Users/tahah/OneDrive/Desktop/Moving-window-based-feature-extraction-method-for-vibration-based-condition-monitoring-main/data/processed"
    os.makedirs(out_dir, exist_ok=True)
    
    # Find all CSVs
    healthy_files = glob.glob(os.path.join(dataset_dir, "Healthy", "*.csv"))
    broken_files = glob.glob(os.path.join(dataset_dir, "Broken Tooth", "*.csv"))
    print(f"Found {len(healthy_files)} healthy files and {len(broken_files)} broken files.")
    
    # Extract load from filename (e.g., h30hz0.csv -> 0, b30hz90.csv -> 90)
    import re
    def get_load(fname):
        match = re.search(r'hz(\d+)\.csv', fname.lower())
        if match:
            return float(match.group(1))
        return 0.0

    file_info = []
    for f in healthy_files:
        file_info.append((f, 0, get_load(f))) # label 0 for healthy
    for f in broken_files:
        file_info.append((f, 1, get_load(f))) # label 1 for faulty
        
    for W in windows:
        print(f"Processing window size W = {W}...")
        all_rows = []
        for filepath, label, load in file_info:
            rows = process_file(filepath, label, load, W)
            all_rows.extend(rows)
            
        df_out = pd.DataFrame(all_rows)
        # Create column names
        feature_names = ['mean', 'rms', 'std', 'var', 'skew', 'kurt', 'p2p', 'crest', 'shape', 'impulse']
        col_names = []
        for s in range(1, 5):
            for fn in feature_names:
                col_names.append(f'S{s}_{fn}')
        col_names.extend(['load', 'label'])
        df_out.columns = col_names
        
        out_path = os.path.join(out_dir, f"features_W{W}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"Saved {len(df_out)} samples to {out_path}.")

if __name__ == "__main__":
    main()
