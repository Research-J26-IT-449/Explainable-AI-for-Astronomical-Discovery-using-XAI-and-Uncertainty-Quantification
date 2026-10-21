# Reproducibility Guide: Module 1 — Stellar Spectra XAI

**Project ID:** J26-IT-449  
**Researcher:** P.K. Edirisooriya (IT23330696)  

This guide provides exact terminal commands to reproduce all audit, training, explainability, validation, and testing results from scratch.

---

## 1. Environment Setup

```bash
# Clone/navigate to project directory
cd "C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\module1_stellar_spectra"

# Create virtual environment and activate
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. End-to-End Pipeline Reproduction

### Step 1: Preprocess Data into Verified Zero-Leakage Partitions
```bash
python -c "import generate_doc1; print('Pipeline active')"
python train_corrected_split.py
```

### Step 2: Run Classical Machine Learning Baselines
```bash
python 07_run_baselines.py
```

### Step 3: Compute Verified DeepSHAP Attributions
```bash
python 03_generate_clean_shap.py
```

### Step 4: Run Scientific Atomic Line Alignment & Monte Carlo Baseline
```bash
python 04_scientific_validation.py
```

### Step 5: Evaluate Objective XAI Faithfulness (AOPC)
```bash
python 05_xai_faithfulness.py
```

### Step 6: Evaluate Objective XAI Stability (Jaccard Similarity)
```bash
python 06_xai_stability.py
```

### Step 7: Run Complete Automated Test Suite (50 Assertions)
```bash
python -m pytest tests -v
```

### Step 8: Launch Interactive Web Application
```bash
streamlit run app.py
```
