# Report 17: Final Metric Verification and Statistical Auditing

**Project ID:** J26-IT-449  
**Researcher:** P.K. Edirisooriya (Student ID: IT23330696)  
**Date:** October 20, 2026  
**Module:** 1 — Stellar Spectra Classification & Explainable AI  
**Assessment:** Progress Presentation 1 (PP1)  

---

## 1. Executive Summary
This report documents the independent recomputation of all performance metrics for the best-performing deep learning model (**1D-CNN-BiLSTM**, checkpoint `models/corrected_split/1d_cnn_bilstm.pth`) evaluated on the strictly isolated, zero-leakage test set (`processed/corrected_clean/test.npz`, $N=150$).

All metrics were recomputed directly from the forward-pass logits and ground-truth labels. The previously reported test accuracy of **77.33%** and balanced accuracy of **74.15%** have been **independently confirmed and reproduced to exact precision**.

---

## 2. Global Performance Metrics

| Metric | Verified Value | Wilson 95% Confidence Interval |
| :--- | :--- | :--- |
| **Overall Accuracy** | **77.33%** (116 / 150) | [69.99%, 83.30%] |
| **Balanced Accuracy** | **74.15%** | [66.82%, 80.51%] |
| **Macro Precision** | **61.10%** | — |
| **Macro Recall** | **74.15%** | — |
| **Macro F1-Score** | **0.6406** | — |
| **Weighted Precision** | **80.60%** | — |
| **Weighted Recall** | **77.33%** | — |
| **Weighted F1-Score** | **0.7866** | — |

---

## 3. Class-by-Class Verification and Confusion Matrix

### 3.1 Class Ordering and Label Mapping
- Index 0: **Class A** (Teff: 7,500 – 10,000 K)
- Index 1: **Class B** (Teff: 10,000 – 30,000 K)
- Index 2: **Class F** (Teff: 6,000 – 7,500 K)
- Index 3: **Class G** (Teff: 5,000 – 6,000 K)
- Index 4: **Class K** (Teff: 3,500 – 5,000 K)

### 3.2 Verified Confusion Matrix
```
             Pred A   Pred B   Pred F   Pred G   Pred K    Total (Support)
True A           2        2        0        0        0          4
True B           0        2        0        0        0          2
True F           0        1        6        2        0          9
True G           0        0        6       28        4         38
True K           1        0        2       16       78         97
-----------------------------------------------------------------------
Total Pred       3        5       14       46       82        150
```

### 3.3 Class Breakdown Table

| Class | Support | Precision | Recall | F1-Score | Recall 95% Wilson CI |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | 4 | 66.67% (2/3) | 50.00% (2/4) | 0.5714 | [15.00%, 85.00%] |
| **B** | 2 | 40.00% (2/5) | 100.00% (2/2) | 0.5714 | [34.24%, 100.00%] |
| **F** | 9 | 42.86% (6/14) | 66.67% (6/9) | 0.5217 | [35.42%, 87.94%] |
| **G** | 38 | 60.87% (28/46) | 73.68% (28/38) | 0.6667 | [57.99%, 85.03%] |
| **K** | 97 | 95.12% (78/82) | 80.41% (78/97) | 0.8715 | [71.42%, 87.09%] |

---

## 4. Architectural Comparison on the Clean Split

To benchmark the 1D-CNN-BiLSTM against architectural alternatives and classical baselines under identical zero-leakage conditions:

| Model Architecture | Accuracy | Balanced Accuracy | Macro F1 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1D-CNN-BiLSTM (Proposed)** | **77.33%** | **74.15%** | **0.6406** | **Best Deep Model** |
| Random Forest (100 trees) | 84.00% | 56.87% | 0.5758 | Fails on rare classes |
| Logistic Regression (L2) | 84.67% | 52.39% | 0.5368 | Fails on rare classes |
| Majority Class Classifier | 64.67% | 20.00% | 0.1571 | Trivial baseline |
| 1D-ResNet | 46.00% | 42.69% | 0.3116 | Overfits clean split |
| Standard 1D-CNN | 32.00% | 29.17% | 0.1699 | Severe degradation |

**Scientific Conclusion:** While classical baselines achieve deceptively high nominal accuracy by assigning almost all test stars to majority Class K, **1D-CNN-BiLSTM achieves superior Balanced Accuracy (74.15% vs 56.87%)**, successfully distinguishing minority stellar types.
