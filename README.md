# Interpretable Gearbox Fault Detection via Kolmogorov-Arnold Networks

This repository contains the official code implementation for the comparative evaluation of purely interpretable **Kolmogorov-Arnold Networks (KAN)** against traditional machine learning classifiers on statistical vibration features for gearbox condition monitoring.

## 🚀 Novelty & Contribution
1. **First Standalone Pure KAN Application:** Applied specifically to tabular statistical features for gearbox fault detection (not embedded in a black-box CNN hybrid).
2. **Explainable AI (XAI) via B-Splines:** Utilizes KAN's visually interpretable 1D spline activation functions to ground learned features back into physical mechanics.
3. **Data-Driven Dimensionality Reduction:** KAN-driven automatic feature pruning isolates the minimal deployable fault diagnosis rule.
4. **Comprehensive Benchmarking:** Direct 5-fold cross-validation comparison of KAN against 7 traditional ML classifiers (DT, RF, SVM, NB, KNN, GBC, LR) and an equivalent MLP architecture using the SpectraQuest Gearbox Dataset across six varying temporal window sizes (W = 300 to 800).

## 📁 Repository Structure
```
├── data/
│   └── processed/                # Moving window generated CSV datasets (W=300 to 800)
├── efficient_kan/                # High-performance KAN PyTorch implementation
├── final_results/                # Consolidated tables, paper plots, and evaluation metrics
├── Gearbox Dataset/              # Raw SpectraQuest healthy and broken tooth vibration data
├── results/                      # Output CSV metrics (Accuracy, Precision, Recall, L1 Norms)
│
├── 02_feature_extraction.py           # Sliding non-overlapping window physical statistical extraction pipeline
├── 03_baseline_ml.py                  # Scikit-learn validation of traditional ML techniques
├── 04_kan_training.py                 # PyTorch & KAN cross-validation with grid/spline evaluations
├── 05_interpretability_and_pruning.py # XAI visual edge mapping and minimal-feature node pruning
└── requirements.txt                   # Dependency list
```

## 🛠 Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/Interpretable-Gearbox-Fault-Detection-KAN.git
   cd Interpretable-Gearbox-Fault-Detection-KAN
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Pipeline:**
   - Execute `python 02_feature_extraction.py` to process the sliding windows.
   - Run `python 03_baseline_ml.py` to bench the standard non-interpretable models.
   - Run `python 04_kan_training.py` to evaluate the KAN/MLP architecture.
   - Execute `python 05_interpretability_and_pruning.py` to extract and visualize the explainable activations and perform L1 norm feature reduction.
