# Module 1: Stellar Spectra Classification & Physical Transition Discovery

**Project ID:** J26-IT-449 (SLIIT)  
**Research Focus:** Explainable AI for Astronomical Discovery using XAI and Uncertainty Quantification

---

## 1. Overview & Data Provenance

- **Original Data:** APOGEE DR17 `apStar` FITS near-infrared stellar spectra downloaded directly via HTTP from the official SDSS Science Archive Server (`https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/`).
- **Dataset Size:** **767 verified APOGEE FITS spectra** in `raw_spectra/` across 12 diverse galactic fields (`000+02`, `000+04`, `000+60`, `000+45`, `001+01`, `030+00`, `120+12`, `180+04`, `M13`, `M15`, `M92`, `M3`, `M67`).
- **Dataset Splits:** Stratified Shuffle Split into:
  - **Train:** 536 spectra
  - **Validation:** 115 spectra
  - **Test:** 116 spectra
- **Wavelength Grid:** Near-Infrared H-band (~15,100 Å to 17,000 Å) resampled onto 4,000 uniform log-wavelength bins.
- **Label Formulation & Distribution:**
  - Effective temperature (`RV_TEFF`) from FITS headers mapped into Harvard spectral proxy classes:
    - **M:** $< 3500\text{ K}$ (0 samples)
    - **K:** $3500 - 5000\text{ K}$ (507 samples)
    - **G:** $5000 - 6000\text{ K}$ (174 samples)
    - **F:** $6000 - 7500\text{ K}$ (48 samples)
    - **A:** $7500 - 10000\text{ K}$ (25 samples)
    - **B:** $10000 - 30000\text{ K}$ (13 samples)
    - **O:** $\ge 30000\text{ K}$ (0 samples)
  - **Scientific Limitation:** These are *temperature-based proxy classes*, NOT independently verified Harvard spectral types.

---

## 2. Model Architectures & Benchmark

Three separate 1D Deep Learning architectures were implemented and trained across 30 epochs:
1. **1D-CNN-BiLSTM** (`models/1d_cnn_bilstm.pth`, `models/best_model.pth`): Hybrid convolutional feature extractor followed by Bidirectional LSTM layers.
2. **1D-ResNet** (`models/1d_resnet.pth`): Deep residual network with 1D convolutions, batch normalization, and skip connections.
3. **Standard 1D-CNN** (`models/standard_1d_cnn.pth`): Multi-layer 1D CNN baseline with max pooling and dropout.

### Performance Summary (Test Set, N=116)
| Model | Val Acc (Best) | Test Acc | Macro F1 | Balanced Acc |
| :--- | :---: | :---: | :---: | :---: |
| **1D-CNN-BiLSTM** *(Best)* | **85.22%** | **86.21%** | **0.5984** | **0.6445** |
| 1D-ResNet | 87.83% | 50.00% | 0.3917 | 0.5369 |
| Standard 1D-CNN | 79.13% | 50.00% | 0.3316 | 0.3565 |

*Evaluation Details for Best Model (`1D-CNN-BiLSTM`):*
- Test Set Size: 116
- Accuracy: 0.8621 (100 correct out of 116)
- Macro F1: 0.5984
- Balanced Accuracy: 0.6445
- Confusion Matrix ($5 \times 5$):
  ```
  [[ 4  0  0  0  0]
   [ 2  0  0  0  0]
   [ 1  0  3  3  0]
   [ 0  0  1 23  2]
   [ 0  0  0  7 70]]
  ```

---

## 3. Explainable AI (XAI) & NIST Line Matching

- **Method:** DeepSHAP (`shap.GradientExplainer`) evaluated across the test spectra.
- **Physical Association:** SHAP importance peaks are compared with atomic/molecular transition lines from the NIST Atomic Spectra Database across the APOGEE H-band (~1.51 - 1.70 $\mu\text{m}$).
- **Key Candidate Lines (Ranked by AAAS - Absolute Average Attribution Score):**
  - **Al I 16763 Å** (AAAS = 0.019343)
  - **N I 16764 Å** (AAAS = 0.019029)
  - **Mg I 16397 Å** (AAAS = 0.014606)
  - **Mg I 15745 Å** (AAAS = 0.014306)
  - **Mg I 15770 Å** (AAAS = 0.013251)
  - **Al I 16718 Å** (AAAS = 0.008884)
  - **H Brackett 11 (16811 Å)** (AAAS = 0.007733)
  - **Fe I 15964 Å** (AAAS = 0.006631)
  - **Si I 15960 Å** (AAAS = 0.006609)
- **Scientific Limitation:** These matches represent *candidate attribution regions*, not confirmed spectroscopic discoveries.

---

## 4. Web Application & Dashboard

An interactive Streamlit application (`app.py`) is provided:
- **Interactive UI:** Upload any spectrum (`.csv`, `.npz`, `.fits`), run inference, inspect confidence distributions, and visualize spectrum overlays with NIST spectral lines.
- **Model Comparison:** Live inspection of the 3 trained architectures.
- **Test-Set Audit:** Live validation of confusion matrix invariants and per-class precision/recall metrics.

To run the application:
```bash
streamlit run app.py
```
Or double-click `run_app.bat` on Windows.

---

## 5. Execution & Reproducibility Instructions

### 1. Environment Setup
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Preprocess Spectra
```bash
python 01_data_preprocess.py
```

### 3. Model Training
```bash
python train_three_models.py
```

### 4. Evaluation & Metric Computation
```bash
python eval_metrics.py
```

### 5. SHAP Attribution & Line Matching
```bash
python 03_shap_attributions.py --model_path models/best_model.pth --test_dir processed --out_path shap_attributions.npz
python 04_nist_line_matching.py --shap_path shap_attributions.npz --out_path line_matches.csv
python visualize_shap.py
```

### 6. Automated Pipeline Verification (Pytest)
```bash
pytest tests/test_pipeline.py -v
```
All 34 test assertions pass, validating label boundaries, confusion matrix sums, diagonal accuracy matching, and metric consistency.
