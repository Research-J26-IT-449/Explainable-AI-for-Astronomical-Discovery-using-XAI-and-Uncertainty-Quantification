# Report 19: Scientific Line-Matching and Explainability Verification

**Project ID:** J26-IT-449  
**Researcher:** P.K. Edirisooriya (Student ID: IT23330696)  
**Date:** October 20, 2026  
**Module:** 1 — Stellar Spectra Classification & Explainable AI  

---

## 1. Executive Summary & Required Scientific Conclusion
This audit evaluates the physical alignment between Gradient SHAP attribution peaks and cataloged atomic spectral absorption lines from the NIST / APOGEE reference line list.

### Mandated Scientific Conclusion
> **"The observed narrow atomic-line alignment did not significantly exceed the random baseline under the selected matching methodology. Additional analysis is required to determine whether the model relies on broader spectral features, molecular bands, or continuum-related information."**

---

## 2. Verification of Reported Line-Matching Metrics

The empirical peak matching results from `04_scientific_validation.py` were independently validated:
- **Total Salient Attribution Peaks Detected:** 57 (prominence threshold P >= 98.5%)
- **Catalog Matches within +/- 3.5 Angstrom Tolerance:** 6 peaks
- **Observed Alignment Percentage:** **10.53%** (6 / 57)
- **1,000-Iteration Monte Carlo Baseline Mean:** **10.19%** (+/- 3.94%)
- **Empirical p-Value:** **p = 0.5380** (538 out of 1,000 synthetic random spectra matched >= 6 lines)
- **Significance Status:** **Not Statistically Significant** (alpha = 0.05)

### 2.1 The 6 Identified Atomic Line Matches
1. **O I (15,871 Å):** Attribution strength = 0.1338 (Strongest attribution peak)
2. **H Brackett 14 (15,263 Å):** Attribution strength = 0.0309
3. **N I (16,764 Å):** Attribution strength = 0.0215
4. **Si I (15,960 Å):** Attribution strength = 0.0179
5. **Mg I (15,745 Å):** Attribution strength = 0.0173
6. **O I (15,569 Å):** Attribution strength = 0.0172

---

## 3. Investigation of the Broad-Feature Hypothesis

To investigate why narrow atomic-line alignment does not exceed random expectations despite high model classification accuracy (77.33%) and high XAI faithfulness (AOPC ratio 6.30x), we evaluated the attribution distribution across broad astrophysical regimes:

| Spectral Regime | Wavelength Range | Astrophysical Significance | Mean Attribution Density |
| :--- | :--- | :--- | :--- |
| **Red Window** | 16,470 – 17,000 Å | H-band pseudo-continuum, CO (2-0) bandheads | **0.005943** (Highest) |
| **Blue Window** | 15,100 – 15,800 Å | H Brackett series, Fe I, Si I, Al I | **0.004992** |
| **Molecular CO Bandhead** | 15,580 – 16,800 Å | CO first-overtone bands (giant star diagnostic) | **0.003972** |
| **Mid Window** | 15,850 – 16,430 Å | Diffuse Fe I, OH lines | **0.003176** |
| **Full Spectrum** | 15,100 – 17,000 Å | Global baseline | **0.004629** |

### 3.1 Key Scientific Insights
1. **Continuum Curvature and Broad Molecular Bands:** APOGEE spectra of cool stars (classes G and K, representing 90% of the sample) are dominated by extensive molecular absorption bands (12CO, 13CO, OH, CN) and pseudo-continuum curvature rather than isolated, narrow atomic lines.
2. **Receptive Field of CNN-BiLSTM:** The 1D-CNN layers extract local morphology over 5-to-11 wavelength channels, while the BiLSTM integrates long-range dependencies across thousands of channels. The model is learning the global spectral energy distribution (SED) and molecular depression profiles.
3. **Methodological Recommendation:** Future work should expand the reference line list from pure atomic transitions to include molecular ro-vibrational band heads and molecular line synthesis models (e.g., MARCS or Turbospectrum).
