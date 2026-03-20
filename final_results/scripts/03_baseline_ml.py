import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

def get_models():
    return {
        'DT': DecisionTreeClassifier(random_state=42),
        'RF': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42),
        'NB': GaussianNB(),
        'KNN': KNeighborsClassifier(),
        'GBC': GradientBoostingClassifier(random_state=42),
        'LR': LogisticRegression(random_state=42, max_iter=1000)
    }

def evaluate_models_for_window(filepath):
    """
    Evaluates all 7 models on a given dataset file (one window size).
    Returns a dict with model names as keys and dict of metrics (acc, prec, rec) as values.
    """
    df = pd.read_csv(filepath)
    # The last columns are 'load' and 'label', everything else is features
    X = df.drop(columns=['label', 'load']).values
    y = df['label'].values
    
    models = get_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {m: {'acc': [], 'prec': [], 'rec': []} for m in models}
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Since it's binary classification (0=healthy, 1=faulty)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            
            results[name]['acc'].append(acc)
            results[name]['prec'].append(prec)
            results[name]['rec'].append(rec)
            
    # Average across folds
    avg_results = {}
    for name in models:
        avg_results[name] = {
            'acc': np.mean(results[name]['acc']),
            'prec': np.mean(results[name]['prec']),
            'rec': np.mean(results[name]['rec'])
        }
    return avg_results

def main():
    data_dir = r"c:/Users/tahah/OneDrive/Desktop/Moving-window-based-feature-extraction-method-for-vibration-based-condition-monitoring-main/data/processed"
    windows = [300, 400, 500, 600, 700, 800]
    
    # Store results: metric -> DataFrame(rows: models, columns: window sizes)
    acc_table = pd.DataFrame(index=get_models().keys(), columns=windows)
    prec_table = pd.DataFrame(index=get_models().keys(), columns=windows)
    rec_table = pd.DataFrame(index=get_models().keys(), columns=windows)
    
    for W in windows:
        filepath = os.path.join(data_dir, f"features_W{W}.csv")
        if not os.path.exists(filepath):
            print(f"File {filepath} not found, skipping...")
            continue
            
        print(f"Evaluating Baseline ML for W={W}...")
        results = evaluate_models_for_window(filepath)
        
        for name, metrics in results.items():
            acc_table.loc[name, W] = metrics['acc'] * 100
            prec_table.loc[name, W] = metrics['prec'] * 100
            rec_table.loc[name, W] = metrics['rec'] * 100
            
    # Save results first before printing
    out_dir = r"c:/Users/tahah/OneDrive/Desktop/Moving-window-based-feature-extraction-method-for-vibration-based-condition-monitoring-main/results"
    os.makedirs(out_dir, exist_ok=True)
    acc_table.to_csv(os.path.join(out_dir, "baseline_accuracy.csv"))
    prec_table.to_csv(os.path.join(out_dir, "baseline_precision.csv"))
    rec_table.to_csv(os.path.join(out_dir, "baseline_recall.csv"))
    
    print("\n--- Accuracy (%) ---")
    print(acc_table.to_markdown())
    
    print("\n--- Precision (%) ---")
    print(prec_table.to_markdown())
    
    print("\n--- Recall (%) ---")
    print(rec_table.to_markdown())

if __name__ == "__main__":
    main()
