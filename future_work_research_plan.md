# Research Plan: Unsupervised Gearbox Fault Detection

This research plan is formulated based on the proposed future work to transition from supervised fault detection to an **unsupervised/one-class learning** paradigm using moving window segmentation. The core focus is on early defect identification using only healthy samples for training.

## 1. Objectives
*   **Implement Moving Window Segmentation:** Generate robust data segments from continuous time-series data to capture dynamic gearbox behaviors.
*   **Unsupervised / One-Class Modeling:** Train models exclusively on "healthy" gearbox data to establish a baseline of normal working patterns.
*   **Anomaly-Based Fault Detection:** Detect faults by measuring deviations (e.g., reconstruction error, distance metrics) from the learned normal patterns.
*   **Explore Lightweight Architectures:** Evaluate traditional ML and lightweight Deep Learning (DL) models for detection robustness and computational efficiency.

## 2. Phase-by-Phase Implementation Plan

### Phase 1: Data Preprocessing & Moving Window Segmentation
*   **Define Window Parameters:** Determine optimal window size (e.g., covering at least one full dynamic cycle of the gearbox) and overlap percentage (e.g., 50% for temporal continuity).
*   **Segmentation Implementation:** Write a script to slice the continuous vibration signals into discrete, overlapping windows.
*   **Dataset Splitting:** 
    *   **Training Set:** Exclusively healthy samples.
    *   **Validation Set:** Mix of healthy and faulty samples to tune the anomaly detection threshold.
    *   **Testing Set:** Mix of unseen healthy and faulty samples (for final evaluation).
*   **Feature Extraction:** Adapt existing extraction logic to process segments dynamically (extracting time-domain, frequency-domain features, and Margin Factor per window), or feed raw segments directly into DL models.

### Phase 2: Baseline Unsupervised Machine Learning
*   **Model Selection:** 
    *   One-Class Support Vector Machine (OC-SVM)
    *   Isolation Forest
    *   Local Outlier Factor (LOF)
*   **Training:** Train conventional ML models on the extracted features of the healthy training set.
*   **Anomaly Scoring:** Compute deviation distances/scores for test samples. Establish an empirical threshold on the validation set for identifying anomalous (faulty) windows.
*   **Deliverables:** Scripts for the training and testing pipelines of traditional one-class baseline models.

### Phase 3: Advanced Deep Learning & Memory-Bank Approaches (PatchCore)
*   **Autoencoder (AE) Architectures:** 
    *   *Dense Autoencoder:* Trained on the hand-crafted features.
    *   *1D Convolutional Autoencoder (1D-CNN AE):* Trained directly on the raw segmented time-series data to bypass manual feature extraction and discover deep features.
*   **PatchCore (Memory-Bank Anomaly Detection):**
    *   Adapt the PatchCore algorithm (highly effective in visual anomaly detection) for 1D time-series applications.
    *   Extract deep feature representations from a backbone network (e.g., a pretrained 1D-CNN) using the healthy segments.
    *   Construct a nominal feature memory bank (coreset) to efficiently store the learned "healthy" representations.
*   **Unsupervised KAN (Optional Novelty):** Investigate formulating the Kolmogorov-Arnold Networks (KAN) into an Autoencoder structure for interpretable unsupervised modeling.
*   **Training & Scoring:** Train the networks/extractors on normal data. Use **Reconstruction Error** (for AEs) or **Nearest-Neighbor Coreset Distance** (for PatchCore) as the anomaly score. Samples exceeding the baseline deviation threshold are classified as structural faults.
*   **Deliverables:** Implementation of PyTorch-based lightweight DL Autoencoders and the 1D PatchCore anomaly detection algorithm.

### Phase 4: Horizon Deep Learning Approaches
*   **Variational Autoencoders (VAE):** Implement probabilistic reconstruction using ELBO to serve as statistically grounded anomaly thresholds.
*   **Deep SVDD:** Learn a neural transformation that maps healthy features into a dense hypersphere. Deviations outside the hypersphere radius indicate faults.
*   **Teacher-Student Distillation:** Train a student network to mimic a frozen random teacher purely on healthy data. Feature mismatches on test sets are structural anomaly flags.
*   **Self-Supervised Learning (SSL):** Develop a Denoising Autoencoder using Gaussian noise injection, tasking the network to recover the pristine vibration data characteristics.

### Phase 4: Evaluation & Early Fault Detection Analysis
*   **Evaluation Metrics:** 
    *   Area Under the Receiver Operating Characteristic curve (ROC-AUC) for threshold-independent assessment.
    *   Precision, Recall, F1-Score, and False Positive Rate (FPR) based on the derived anomaly threshold.
*   **Early Detection Validation:** Evaluate the models on dataset samples representing incipient (very early-stage) faults to validate the method's sensitivity.
*   **Robustness Analysis:** Assess the models' resilience to noise and changing operational speeds/loads.
*   **Deliverables:** Evaluation scripts generating performance metrics, ROC curves, and visual plots comparing the distribution of anomaly scores between healthy and varying degrees of faulty data.

## 3. Anticipated Outcomes & Contributions
1. **Eliminated Reliance on Labeled Fault Data:** Demonstrates the ability to monitor gearbox health realistically, as industrial settings rarely have extensive labeled failure data.
2. **Robust Anomaly Detection:** Validates that unsupervised techniques can rival or complement supervised learning for fault identification.
3. **Publication/Extension Potential:** Creates a strong foundation for a follow-up publication focusing on unsupervised real-time anomaly detection in predictive maintenance.
