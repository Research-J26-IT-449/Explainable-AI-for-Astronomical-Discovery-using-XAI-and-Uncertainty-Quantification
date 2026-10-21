# Report 23: Final Streamlit Application Audit and Verification

**Project ID:** J26-IT-449  
**Researcher:** P.K. Edirisooriya (Student ID: IT23330696)  
**Date:** October 20, 2026  
**Module:** 1 — Stellar Spectra Classification & Explainable AI  

---

## 1. Application Overview
The interactive research application is implemented in `app.py` using Streamlit. It serves as the primary demonstration interface for the PP1 assessment.

### Navigable Pages:
1. **🏠 Executive Dashboard:** High-level project KPIs, dataset distribution, and research summary.
2. **🔬 Upload & Analyze Spectra:** Allows interactive FITS/CSV upload, forward pass inference, and per-wavelength Gradient SHAP attribution visualization.
3. **📊 Model Comparison:** Side-by-side benchmarking of 1D-CNN-BiLSTM vs 1D-ResNet, Standard 1D-CNN, and Classical Baselines (Random Forest, Logistic Regression).
4. **⚛️ Scientific XAI Validation:** Displays the NIST line alignment ($p=0.5380$), AOPC Faithfulness curve (6.30x ratio), and Jaccard Noise Stability curves.
5. **🎯 Test Evaluation:** Real-time test-set confusion matrix, classification metrics (77.33% / 74.15%), and error analysis.
6. **ℹ️ About & Limitations:** Research methodology, Harvard-Yerkes mapping, dataset provenance, and documented limitations.

---

## 2. Terminology and Checkpoint Audit
- **Default Checkpoint:** Verified loading `models/corrected_split/1d_cnn_bilstm.pth` as default.
- **Explainability Engine:** Verified updated terminology to **Gradient SHAP** (`shap.GradientExplainer`).
- **Benchmark Alignment:** Verified display of clean verified metrics (Accuracy: 77.33%, Balanced Accuracy: 74.15%).
- **Error Handling:** Verified fallback handling when user uploads synthetic or raw CSVs without FITS headers.
