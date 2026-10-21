# Report 20: Final Explainable AI Faithfulness and Stability Validation

**Project ID:** J26-IT-449  
**Researcher:** P.K. Edirisooriya (Student ID: IT23330696)  
**Date:** October 20, 2026  
**Module:** 1 — Stellar Spectra Classification & Explainable AI  

---

## 1. Executive Summary
This report provides formal statistical validation of the Explainable AI (Gradient SHAP) framework along two key technical dimensions:
1. **Faithfulness:** Does masking the top-attributed wavelengths degrade model classification confidence significantly faster than random perturbations (Area Over Perturbation Curve, AOPC)?
2. **Stability:** Are the attribution explanations robust against observational noise typical of astronomical surveys (Jaccard Rank Similarity)?

---

## 2. Faithfulness Evaluation (AOPC) & Statistical Significance

### 2.1 AOPC Metrics
- **Evaluated Cohort:** 50 correctly classified test spectra across diverse spectral types.
- **Perturbation Steps ($k$):** [0, 50, 100, 200, 300, 500, 800, 1200] bins replaced with continuum mean (1.0).
- **Mean AOPC Score (Top-$k$ SHAP):** **0.7507** ($\pm 0.1898$)
- **Mean AOPC Score (Random Baseline):** **0.1192** ($\pm 0.1280$)
- **Faithfulness Advantage Ratio:** **6.30x**

### 2.2 Formal Statistical Hypothesis Testing
We formulated the null hypothesis:
H0: mu_top <= mu_rand
vs the alternative:
H1: mu_top > mu_rand

- **Paired t-test:** t = 19.4374, p = 1.12e-24
- **Wilcoxon Signed-Rank Test:** W = 0.0, p = 1.78e-15

**Conclusion:** The null hypothesis is rejected with extreme statistical confidence (p < 1e-14). The Gradient SHAP attributions are mathematically faithful to the model's inner decision logic.

---

## 3. Stability Evaluation Under Noise (Jaccard Overlap)

Evaluated across 25 stars with top-150 attribution bins under realistic APOGEE noise levels:

| Survey Regime | Noise Level (sigma) | Mean Jaccard Similarity | Std Dev | Stability Rating |
| :--- | :--- | :--- | :--- | :--- |
| **High SNR (SNR=100)** | 0.01 | **0.8069** (80.69%) | 0.0628 | Excellent |
| **Nominal Survey (SNR=50)** | 0.02 | **0.7768** (77.68%) | 0.0635 | Strong |
| **Low SNR (SNR=25)** | 0.04 | **0.6853** (68.53%) | 0.0539 | Acceptable |

**Conclusion:** Gradient SHAP exhibits high stability ($>77\%$) across realistic APOGEE signal-to-noise ratios, proving that explanations are robust to photon noise.
