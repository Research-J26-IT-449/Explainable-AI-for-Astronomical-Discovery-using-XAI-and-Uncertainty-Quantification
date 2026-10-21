# Report 21: Minority Class Reliability and Statistical Uncertainty

**Project ID:** J26-IT-449  
**Researcher:** P.K. Edirisooriya (Student ID: IT23330696)  
**Date:** October 20, 2026  
**Module:** 1 — Stellar Spectra Classification & Explainable AI  

---

## 1. Context and Class Imbalance Reality
In the SDSS/APOGEE DR17 infrared survey, the vast majority of targets are cool giants and main-sequence stars (spectral classes G and K). Early-type hot stars (classes O, B, and A) are rare because:
1. Hot stars emit primarily in the ultraviolet and optical, with minimal near-infrared flux.
2. The APOGEE targeting selection focused heavily on Galactic red giants.

As a result, the clean dataset of 1,000 spectra naturally reflects this astrophysical imbalance:
- **Train (700):** A: 19, B: 8, F: 42, G: 179, K: 452
- **Validation (150):** A: 4, B: 2, F: 9, G: 38, K: 97
- **Test (150):** A: 4, B: 2, F: 9, G: 38, K: 97

---

## 2. Statistical Analysis of Minority Classes A and B

### 2.1 The Small Sample Size Risk
In the test set:
- **Class B has only 2 samples.** Both were predicted correctly by the model ($	ext{Recall} = 100.0\%$, 2 / 2).
- **Class A has only 4 samples.** 2 were predicted correctly ($	ext{Recall} = 50.0\%$, 2 / 4).

### 2.2 Wilson Score 95% Confidence Intervals

| Class | Test Count ($N$) | Correct Hits | Nominal Recall | 95% Wilson Score Interval |
| :--- | :--- | :--- | :--- | :--- |
| **A** | 4 | 2 | 50.00% | **[15.00%, 85.00%]** |
| **B** | 2 | 2 | 100.00% | **[34.24%, 100.00%]** |
| **F** | 9 | 6 | 66.67% | **[35.42%, 87.94%]** |
| **G** | 38 | 28 | 73.68% | **[57.99%, 85.03%]** |
| **K** | 97 | 78 | 80.41% | **[71.42%, 87.09%]** |

### 2.3 Scientific Interpretation for PP1 Assessment
- **Critical Caveat:** While the nominal recall for Class B is 100.0%, its 95% Wilson confidence interval extends down to **34.24%**. 
- **Academic Stance:** We **do not claim** complete empirical mastery of B-type stars. The 2 test instances demonstrate that the model can identify extreme high-temperature profiles without catastrophic false-negative classification, but statistically rigorous generalization claims require future synthetic spectral augmentation.
- **Balanced Metric Adoption:** The choice to report **Balanced Accuracy (74.15%)** and **Macro F1 (0.6406)** guarantees that performance is evaluated equally across all classes rather than inflated by the majority class K.
