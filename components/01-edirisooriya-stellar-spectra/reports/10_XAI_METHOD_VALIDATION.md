# Report 10: XAI Method Validation and Directionality Verification

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Directionality vs Absolute Magnitude in Spectroscopy

In Explainable AI, signed SHAP values indicate whether a feature increases or decreases the log-odds of a specific target class:
- $\phi_i > 0$: Wavelength bin $i$ drives the prediction toward target class $c$.
- $\phi_i < 0$: Wavelength bin $i$ drives the prediction away from target class $c$.

### Clarification on AAAS (Absolute Average Attribution Score)
In the existing script `04_nist_line_matching.py`, the metric computed is:
$$\text{AAAS}(\lambda) = \frac{1}{M} \sum_{m=1}^M |\phi_m(\lambda)|$$

**Crucial Research Disclosure:**
AAAS reflects **feature salience (importance magnitude)**, *not* physical emission vs absorption. An absorption line (where flux drops below 1.0) can have a strongly positive SHAP attribution if its presence confirms a cool star, or a negative attribution if its absence implies a hot star. In all reports and presentations, AAAS is defined strictly as **importance magnitude**.
