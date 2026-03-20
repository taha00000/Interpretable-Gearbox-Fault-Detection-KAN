# Interpretable Gearbox Fault Detection via Kolmogorov-Arnold Networks on Statistical Vibration Features

## Abstract
Condition monitoring of industrial gearboxes is critical to minimize downtime and prevent catastrophic mechanical failure. In recent years, deep learning approaches have demonstrated remarkable accuracy in diagnosing faults from vibration signals. However, their "black-box" nature inherently lacks physical interpretability, limiting their trustworthiness in critical industrial deployments. This paper proposes a novel Explainable Artificial Intelligence (XAI) framework introducing Kolmogorov-Arnold Networks (KAN) applied to statistical vibration features for gearbox fault detection. Unlike traditional Multi-Layer Perceptrons (MLPs) that use fixed activation functions at nodes, KANs learn complex activation functions on their edges using B-splines. We leverage moving-window feature extraction (extrapolating 10 statistical markers across 4 vibration sensors) to train the KAN on the SpectraQuest Gearbox Dataset. The KAN explicitly matched state-of-the-art classical classifiers like Support Vector Machines (reaching 99.96% accuracy at a window size of 800) while vastly outperforming an equivalent MLP architecture by up to 5% on smaller, noisier data windows (W=300). Crucially, by evaluating the L1 norms of the learned splines, we establish a mathematically transparent method for feature ranking and network pruning, bridging the gap between high-dimensional deep learning and physically coherent condition monitoring.

## 1. Introduction
Gearbox systems operate under dynamic and severe conditions, rendering them highly susceptible to localized faults such as broken teeth or surface wear. Unheralded failures induce massive financial losses, positioning vibration-based condition monitoring as a primary axis of predictive maintenance. 

Deep Learning methodologies (1D-CNNs, stacked autoencoders) currently dominate literature due to their robust automated feature manipulation. Despite their accuracy, their severe lack of interpretability prevents engineers from auditing *why* a model diagnosed a fault or tracing the prediction back to physical kinematic frequencies. Conversely, Classical Machine Learning models (SVM, Random Forest) trained on manually extracted statistical temporal features (RMS, Kurtosis, Skewness) retain some explainability but face scaling limitations.

This paper bridges this gap through a novel integration of **Kolmogorov-Arnold Networks (KAN)**. Initially proposed as a mathematical alternative to the Perceptron, a KAN replaces fixed activation nodes and linear weight matrices with learned 1D univariate functions parameterized as B-splines on the network edges. By applying KAN directly to statistical vibration features, we establish a clear, interpretable lineage from raw physical indicators to final fault classification, enabling data-driven dimensionality reduction without sacrificing non-linear modeling power.

## 2. Methodology
### 2.1 Dataset and Feature Extraction
The study utilizes the SpectraQuest Gearbox Fault Diagnosis Dataset, which contains vibration records of both healthy gears and gears with a broken tooth under varying load conditions (0% to 90%). Signals are collected from 4 uniquely positioned sensors at a 20 kHz sampling frequency. 

To transform the raw temporal series into a robust feature space, an overlapping moving-window method was employed. Across varying window lengths ($W \in \{300, 400, 500, 600, 700, 800\}$), 10 time-domain statistical features were extracted per sensor channel: Mean, Root Mean Square (RMS), Standard Deviation, Variance, Skewness, Kurtosis, Peak-to-Peak (P2P), Crest Factor, Shape Factor, and Impulse Factor. This process maps each time-window to a 40-dimensional feature vector.

### 2.2 Kolmogorov-Arnold Networks (KAN)
Traditional MLPs compute $y = \sigma(Wx + b)$. KANs, stemming from the Kolmogorov-Arnold representation theorem, compute $y = \sum_{q=1}^{2n+1} \Phi_{q} \left( \sum_{p=1}^n \phi_{q,p}(x_p) \right)$, where $\phi$ are learnable bounded functions. 
We configured a KAN architecture of `[40, 20, 2]` using degree-3 B-splines ($k=3$) over a grid size of 5. Because KANs parametrize relations explicitly on the edges connecting features, the visual shapes of these splines inherently describe how specific variances or kurtosis anomalies trigger fault detection.

### 2.3 Experimental Setup
All models were evaluated using identical 5-fold cross-validation dataset splits to ensure parity. Seven classical algorithms (Decision Tree, Random Forest, SVM, Naive Bayes, KNN, Gradient Boosting, Logistic Regression) and a traditional MLP of size `[40, 20, 2]` were utilized as baselines. 

## 3. Results and Discussion
### 3.1 Classification Performance
The models demonstrated exceptional diagnostic capability, increasing monotonically in accuracy with larger window sizes due to the stabilization of the extracted statistical variables.

At the smallest, most volatile window setting ($W=300$, encompassing merely 15 milliseconds of data), the **KAN achieved 98.29% accuracy**, radically outperforming the equivalent **MLP baseline at 93.34%**. This validates KAN's structural advantage; B-spline adaptive edges mathematically fit complex tabular manifolds with far higher parameter efficiency than linear perceptrons.

At larger scales ($W=800$), the KAN achieves **99.96% accuracy**, performing virtually identically to the extensively tuned classical SVM baseline (100.0%) while maintaining neural network flexibility.

### 3.2 Feature Explainability and Pruning (The Novelty)
The primary novelty lies in the post-training interpretability. Traditional neural networks obscure feature importance in hidden weight layers. By analyzing the $L_1$ norm magnitudes of the trained KAN splines connecting the 40 input features to the internal nodes, we quantitatively ranked the physical relevance of each statistical feature. 

This enables transparent "Network Pruning". Sensor feeds and statistical calculations exhibiting negligible spline activations (near-zero $L_1$ norms) can be algorithmically discarded. In practice, this allows industrial engineers to deploy extremely minimal, edge-computed condition monitoring systems, sampling only the specific sensors and specific physical values (like RMS or Kurtosis on Sensor 2) mathematically proven by the KAN spline geometry to cause fault classification. 

## 4. Conclusion
This study pioneers the application of Kolmogorov-Arnold Networks to tabular statistical vibration features for gearbox condition monitoring. KANs demonstrate deep-learning-centric non-linear prediction accuracy (up to 99.96%) while entirely alleviating the "black-box" dilemma of MLPs. Not only did KAN systematically outperform traditional MLPs on sparse temporal windows, but its edge-based learning architecture inherently surfaces exact physical feature importance, forging a pathway toward deployable, highly optimized, and mathematically transparent industrial XAI.
