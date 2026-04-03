# Unsupervised Gearbox Fault Detection: Detailed Methodology

This document systematically breaks down the end-to-end pipeline utilized to process vibration data and evaluate 9 distinct unsupervised/semi-supervised anomaly detection algorithms. It provides a theoretical and practical overview of how each algorithm identifies structural faults solely by studying healthy operating conditions.

---

## Part 1: Step-by-Step Pipeline Execution

### Step 1: Feature Extraction (`02_feature_extraction.py`)
The pipeline begins by reading raw 50kHz vibration signals from the SpectraQuest Gearbox Dataset.
- **Action:** A sliding moving window (configurable from W=300 to W=1000 samples) sweeps across the continuous temporal data.
- **Extraction:** For each window, 11 specific statistical features (Mean, RMS, Variance, Skewness, Kurtosis, Peak-to-Peak, Crest Factor, Shape Factor, Margin Factor, and Impulse Factor) are mathematically calculated across all 4 sensor channels, yielding a highly concentrated **44-dimensional feature vector** representing that specific timestamp's physical state.

### Step 2: Strict Unsupervised Data Splitting (`unsupervised_01_data_split.py`)
To emulate a realistic predictive maintenance scenario, the model must *never* see faulty data during training.
- **Action:** The feature vectors are parsed. The **Training Set** is heavily filtered to contain **100% healthy samples** (representing normal, pristine gearbox operation).
- **Validation/Testing Sets:** The remaining data, alongside all instances of the "Broken Tooth" structural faults, are randomly shuffled into sets used exclusively to validate thresholds and compute final accuracy. 

### Step 3: Baseline Detection Computation (`unsupervised_02_baseline_ml.py`)
Non-parametric and traditional Machine Learning solutions are fitted.
- **Action:** The algorithms learn the boundaries and densities of the 44-dimensional healthy training set. During inference on the Test set, they output a "deviation score" indicating how far a sample sits from the learned norm.

### Step 4: Foundation Deep Learning (`unsupervised_03_deep_learning.py`)
Neural networks and geometric memory banks are established to learn high-level representations.
- **Action:** The PyTorch Autoencoder learns physiological reconstruction, while the PatchCore algorithm builds a literal "memory bank" of standard operations. Both output geometric distances as their anomaly score.

### Step 5: Advanced Horizon Architectures (`unsupervised_05_advanced_dl.py`)
State-of-the-art architectures historically applied to machine vision are refactored for 1D signal analytics.
- **Action:** Highly complex paradigms (Probabilistic generative evaluation, latent hypersphere constraints, and knowledge distillation) establish incredibly precise boundaries defining normal conditions.

### Step 6: Unified Evaluation (`unsupervised_04_evaluation.py`)
- **Action:** The anomaly scores from all 9 executed frameworks are compiled. We threshold the validation sets to maximize the `F1-Score` and record the final `ROC-AUC` curve topologies to quantify exact robustness.

---

## Part 2: Detailed Algorithm Breakdown

### 1. Local Outlier Factor (LOF)
**Category:** Baseline / Density-Based Machine Learning
- **Theory:** LOF operates on the assumption that anomalies are structurally isolated. It computes the local density of a sample by measuring the distances to its $k$-nearest neighbors.
- **Application:** If a test sample's density is significantly lower than the density of the healthy samples it surrounds itself with in the feature space, it is aggressively flagged as a fault.
- **Results:** Achieved the highest non-parametric results (99.6% AUC at W=1000).

### 2. One-Class SVM (OC-SVM)
**Category:** Baseline / Boundary-Based Machine Learning
- **Theory:** Utilizing a Radial Basis Function (RBF) kernel, OC-SVM projects the healthy data into a higher mathematical dimension where it attempts to draw the tightest possible hyperplane isolating the healthy data from the origin.
- **Application:** Any test sample falling on the wrong side of this learned separator margin is classified as anomalous.

### 3. Isolation Forest
**Category:** Baseline / Tree-Based partitioning
- **Theory:** Isolation Forests build randomized decision trees. Normal data points, which are densely clustered, require many splits to be isolated. Anomalies, which reside in sparse extremities, are isolated in very few splits.
- **Application:** The path length from the root node to the terminating leaf acts as an anomaly score. Shorter path = high fault probability.

### 4. Dense Autoencoder (AE)
**Category:** Deep Learning / Reconstruction Loss
- **Theory:** An hourglass-shaped neural network composed of an Encoder (compressing the 44 features to an 8-dimensional bottleneck) and a Decoder (attempting to upscale back to 44).
- **Application:** Fed only healthy data, the network becomes exceptionally good at reconstructing standard vibrations. When a broken-tooth signal is inputted, the network fails to reconstruct the unfamiliar chaotic vibrations. The resulting **Mean Squared Error (MSE)** acts as the anomaly score.

### 5. 1D PatchCore (Memory-Bank)
**Category:** Deep Representation / Coreset Nearest-Neighbor
- **Theory:** PatchCore creates a library of "what normal looks like." To save memory, it uses a *K-Center Greedy Algorithm* to store only the most distinct, representative 10% of healthy features, avoiding massive redundancy.
- **Application:** During testing, the sample simply computes its Euclidean distance against the core memory bank. Because it only memorizes healthy geometry, it yielded extremely fast and reliable anomaly flags.

### 6. Variational Autoencoder (VAE)
**Category:** Advanced Deep Learning / Probabilistic Generative
- **Theory:** Unlike the deterministic Dense AE, a VAE forces its bottleneck to conform to a smooth Probability Distribution (specifically a Multivariate Gaussian). 
- **Application:** It replaces rigid point estimates with probabilistic spaces by optimizing the Evidence Lower Bound (ELBO). Faulty data falls outside the acceptable probability contours of the latent manifold, yielding exponentially high reconstruction penalties.

### 7. Deep Support Vector Data Description (Deep SVDD)
**Category:** Advanced Deep Learning / Latent Hypersphere Constriction
- **Theory:** The deep learning evolution of OC-SVM. A neural network learns a proprietary, non-linear transformation that forces all healthy data points to cluster as tightly as possible around a single static center point $c$ in latent space.
- **Application:** The loss function strictly penalizes distance from $c$. Any test sample whose neural projection lands far outside the optimal radius geometry is flagged instantly as a gearbox fault.

### 8. Teacher-Student Knowledge Distillation
**Category:** Advanced Deep Learning / Regression Distillation Representation
- **Theory:** A complex paradigm requiring two identical networks. The *Teacher* network is initialized with random weights and permanently frozen (making it an obscure but consistent feature extractor). The *Student* network trains to perfectly mimic the Teacher’s arbitrary output.
- **Application:** Crucially, the student only learns to mimic the teacher on *healthy* data. When faulty data arrives, the student encounters an alien feature distribution and produces an output vastly misaligned with the Teacher. The **Regression Mismatch (MSE)** dictates the anomaly score, resulting in extraordinary precision (99.4% AUC at W=1000).

### 9. Self-Supervised Denoising Autoencoder (SSL-DAE)
**Category:** Advanced Deep Learning / Pretext Task SSL
- **Theory:** Self-Supervised Learning introduces artificial complications to force the network to learn deeper semantic meanings. Gaussian noise is heavily injected into the healthy feature sets. The network's singular task is to clean the signal and reproduce the pristine, original healthy target.
- **Application:** By learning *how* to denoise healthy mechanical structures, the network implicitly memorizes the standard waveform rules of the gearbox. Subjecting an actual faulty signal to the network yields massive divergence from these rules, registering the second highest Deep Learning scores (98.0% AUC at W=1000).
