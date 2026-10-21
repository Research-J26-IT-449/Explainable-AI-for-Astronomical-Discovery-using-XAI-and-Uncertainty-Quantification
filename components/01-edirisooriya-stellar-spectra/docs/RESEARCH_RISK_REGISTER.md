# Research Risk Register: Module 1 — Stellar Spectra XAI

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Risk Register Matrix

| Risk ID | Risk Description | Severity | Likelihood | Mitigation Strategy Implemented | Current Status | Verification Evidence |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **RSK-01** | **Historical Data Leakage:** Nested indexing slice bug causing train-test overlap. | **CRITICAL** | High | Identified during audit; engineered verified zero-leakage split (`processed/corrected_clean/`); retrained all models. | **RESOLVED** | Report 04, `processed/corrected_clean/test_indices.npy` |
| **RSK-02** | **Severe Class Imbalance:** APOGEE infrared survey dominated by cool giants (K: 64.6%, G: 25.2%). | **HIGH** | High | Prioritize Balanced Accuracy and Macro-F1 over raw accuracy; transparently disclose minority class sample sizes. | **MITIGATED** | Report 03, Report 06, `app.py` |
| **RSK-03** | **Spurious Line Alignment:** Believing high line alignment without statistical control. | **HIGH** | Medium | Implemented 1,000-trial Monte Carlo random baseline test; derived empirical p-value ($p = 0.5380$). | **MITIGATED** | Report 11, Report 12, `04_scientific_validation.py` |
| **RSK-04** | **Unvalidated XAI Attributions:** Explanations failing to reflect actual model reasoning. | **HIGH** | Medium | Evaluated Area Over Perturbation Curve (AOPC); verified 6.30x faithfulness advantage over random perturbations. | **RESOLVED** | Report 13, `05_xai_faithfulness.py` |
| **RSK-05** | **Explanation Instability:** Attributions shifting drastically under survey noise. | **MEDIUM** | Medium | Evaluated Jaccard similarity across SNR 100, 50, 25; verified $>77.6\%$ region overlap at survey standards. | **RESOLVED** | Report 14, `06_xai_stability.py` |
| **RSK-06** | **Misleading Proxy Labels:** Calling ASPCAP TEFF classes true Harvard MK types. | **HIGH** | Low | Strictly standardized terminology to 'Effective-Temperature Proxy Class' across all reports, slides, and Q&A. | **MITIGATED** | Report 03, Presentation Slide 4 |
| **RSK-07** | **Model Overfitting / Memorization:** Networks memorizing discrete noise spikes. | **MEDIUM** | Medium | Applied 101-pixel continuum normalization, L2 weight decay ($10^{-4}$), and early stopping on validation loss. | **RESOLVED** | Report 05, Report 08 |
| **RSK-08** | **Corrupted/Truncated FITS Input:** Network crashes on interrupted FITS downloads. | **MEDIUM** | Medium | Enforced multi-extension HDU validation; strictly rejected files with $<10$ HDUs. | **RESOLVED** | Report 02, `01_data_preprocess.py` |
