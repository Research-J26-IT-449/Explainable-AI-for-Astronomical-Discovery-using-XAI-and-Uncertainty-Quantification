# Report 11: Scientific Atomic-Line Validation and Physical Interpretability

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Research Motivation & Core Question

A central premise of this research is: **Do deep learning models classify stellar spectra by discovering genuine physical atomic absorption transitions, or do they rely on diffuse continuum slopes and non-physical detector artifacts?**

To answer this objectively, we implemented `04_scientific_validation.py` to identify peak attribution wavelengths and evaluate their physical correspondence to NIST atomic line databases.

---

## 2. Literature Reference Database

Thirty-one high-priority near-infrared stellar absorption transitions known in the APOGEE $H$-band ($15,100 - 17,000$ Å) were compiled from NIST Atomic Spectra Database and high-resolution spectroscopic surveys (Shetrone et al. 2015; Smith et al. 2013). These include:
- **Hydrogen Brackett Series:** H Br-14 (15,264.7 Å), H Br-13 (15,560.7 Å), H Br-12 (16,113.7 Å), H Br-11 (16,811.1 Å).
- **Iron Peaks (Fe I):** 15,194.5 Å, 15,207.5 Å, 15,490.3 Å, 15,648.5 Å, 15,964.4 Å, 16,153.2 Å.
- **Alpha & Light Elements:** Si I (15,960.1 Å, 16,060.0 Å), Mg I (15,745.0 Å, 15,770.1 Å, 16,397.9 Å), Al I (16,718.9 Å, 16,763.4 Å), C I, N I, O I, Ti I, Ca I.

---

## 3. Peak Detection & Matching Methodology

1. Prominent local maxima in the global mean absolute SHAP curve were identified using `scipy.signal.find_peaks` with an 85th percentile prominence threshold and minimum inter-peak spacing of 15 bins (~7 Å).
2. Fifty-seven (57) distinct attribution peaks were detected across the 4,000-bin grid.
3. Each peak was matched against the nearest reference line using an astrophysical matching window $\Delta\lambda \le 3.5$ Å (accounting for line broadening and 0.475 Å/pixel dispersion).
