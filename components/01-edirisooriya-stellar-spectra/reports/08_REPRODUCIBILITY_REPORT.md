# Report 08: Reproducibility and Experiment Configuration Report

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Experimental Environment & System Specifications

| Component | Specification |
| :--- | :--- |
| **Operating System** | Microsoft Windows 11 Home / Pro (x86_64) |
| **Python Runtime** | Python 3.14.6 (64-bit) |
| **PyTorch Framework** | PyTorch 2.14.1+cpu |
| **Scientific Stack** | Astropy 7.1.0, NumPy 2.2.3, SciPy 1.15.2, Pandas 2.2.3, Scikit-Learn 1.6.1 |
| **Explainable AI** | SHAP 0.47.0 (`shap.GradientExplainer`) |
| **Web UI** | Streamlit 1.42.0 |
| **Verification Suite** | Pytest 9.1.1 |

---

## 2. Seed and Partition Parameters

- **Random Seed:** `42` (Fixed across dataset partitioning, numpy generators, PyTorch CPU seeds, and scikit-learn baselines).
- **Partition Ratios:** Train = 70.0% (700 samples), Validation = 15.0% (150 samples), Test = 15.0% (150 samples).
- **Stratification Strategy:** Two-stage `StratifiedShuffleSplit` indexed with `random_state=42`.
- **Primary Data Directory:** `C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\module1_stellar_spectra\`
- **Zero-Leakage Partition Directory:** `processed/corrected_clean/{train.npz, val.npz, test.npz}`

---

## 3. Training Hyperparameters

- **Optimizer:** `AdamW` ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$)
- **Weight Decay:** $10^{-4}$
- **Initial Learning Rate:** $10^{-3}$ ($0.001$)
- **Batch Size:** 16
- **Epochs:** 25 with best validation loss checkpoint restoration
- **Loss Function:** `nn.CrossEntropyLoss()`
