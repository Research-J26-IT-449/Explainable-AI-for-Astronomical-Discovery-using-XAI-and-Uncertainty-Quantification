# Report 07: Baseline Comparisons and Benchmark Analysis

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Baseline Framework

To demonstrate scientific rigor for the IT4010 PP1 assessment, the deep learning architectures were benchmarked against three classical and statistical baselines on the exact same zero-leakage test set ($N=150$):

1. **Majority-Class Zero-Rule Classifier (`DummyClassifier`):** Always predicts the most frequent training class (Class K). Represents the naive lower bound of performance in an imbalanced dataset.
2. **L2-Regularized Logistic Regression:** Linear classification baseline trained on the 4,000-dimensional continuous flux features.
3. **Random Forest Classifier (100 Trees, max depth 12):** Non-linear ensemble baseline evaluating decision-tree feature thresholding.
4. **Standard 1D-CNN:** Baseline convolutional neural network.
5. **1D-ResNet:** Deep residual network with skip connections.
6. **1D-CNN-BiLSTM:** Proposed hybrid architecture.

---

## 2. Comparative Benchmark Results

| Model / Paradigm | Model Category | Test Accuracy | Macro F1-Score | Balanced Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Majority Class (Zero-Rule)** | Statistical Baseline | 64.67% | 0.1571 | 20.00% |
| **Standard 1D-CNN** | Deep Learning Baseline | 32.00% | 0.1699 | 29.17% |
| **1D-ResNet** | Deep Residual DL | 46.00% | 0.3116 | 42.69% |
| **Random Forest (100 Trees)** | Classical Non-Linear ML | 84.00% | 0.5758 | 56.87% |
| **Logistic Regression (L2)** | Classical Linear ML | 84.67% | 0.5368 | 52.39% |
| **1D-CNN-BiLSTM (Ours)** | **Hybrid Deep Learning** | **77.33%** | **0.6406** | **74.15%** |

---

## 3. Key Scientific Insights

1. **The Balanced Accuracy Paradox:**
   - Both Logistic Regression and Random Forest obtain high raw accuracy ($84.0\% - 84.7\%$) by predicting almost exclusively Classes K and G, resulting in poor balanced accuracy ($52.4\% - 56.9\%$) and near-zero recall on minority classes A and B.
   - In contrast, the **1D-CNN-BiLSTM** achieves the highest **Balanced Accuracy (74.15%)** and highest **Macro F1-score (0.6406)** of any model tested, demonstrating superior balance across the temperature spectrum.
2. **Why CNN-BiLSTM Outperforms Standard CNN/ResNet:**
   - Stellar spectra possess both local morphological absorption features (convolutions capture multi-pixel line shapes) and broad physical dependencies (pseudocontinuum tilt and multi-element ionization balances across the entire $1.51 - 1.70\,\mu\text{m}$ passband).
   - The bidirectional LSTM enables long-range relational reasoning across disparate wavelengths, enabling the network to differentiate subtle temperature transitions that pure local convolutions miss.
