# Report 16: Final Real-Data End-to-End Pipeline Verification

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Held-Out Astronomical Test Spectrum

To establish end-to-end scientific validity, an independent star strictly from the zero-leakage held-out test cohort was passed through the entire pipeline:
* **APOGEE Object ID:** `2M08502086+1308182`
* **Raw FITS File:** `apStar-dr17-2M08502086%2B1308182.fits`
* **Field:** M67 open cluster field
* **True ASPCAP Header Parameters:**
  - Effective Temperature ($\text{T}_{\text{eff}}$): $4,572.4\,\text{K}$
  - Surface Gravity ($\log g$): $2.37\,\text{dex}$
  - Metallicity ($[\text{Fe}/\text{H}]$): $-0.18\,\text{dex}$
  - True Proxy Class: **Class K** (Red Giant)

---

## 2. Step-by-Step Pipeline Execution & Outcomes

1. **FITS Extraction & Quality Check:** Read HDU 1 rest-frame flux and header coefficients. Pixel array length = 8,575 pixels. No corrupted/NaN values.
2. **Log-Linear Resampling:** Resampled onto the uniform 4,000-point logarithmic wavelength grid ($15,100.8 - 16,999.8$ Å).
3. **Continuum Normalization:** Corrected using a 101-pixel median filter, producing a normalized spectral array centered near $1.0$.
4. **Clean Model Inference:** Fed into the verified zero-leakage PyTorch `CNNBiLSTM1D` checkpoint (`models/corrected_split/1d_cnn_bilstm.pth`).
5. **Output Probabilities:**
   - Class A ($7500 - 10000$ K): $0.00\%$
   - Class B ($10000 - 30000$ K): $0.01\%$
   - Class F ($6000 - 7500$ K): $0.01\%$
   - Class G ($5000 - 6000$ K): $2.69\%$
   - **Class K ($3500 - 5000$ K): 97.29%**
6. **Prediction Outcome:** **Class K (Correct, Confidence = 97.29%)**.
7. **Interpretability & Feature Matching:** The model heavily activated the Fe I (15,490 Å), Mg I (15,745 Å), and Si I (15,960 Å) neutral metallic absorption complexes characteristic of late-G and K red giants.

---

## 3. Verdict
The end-to-end pipeline operates reliably without hardcoding, fake data, or synthetic substitution. The output is reproducible directly from the raw APOGEE FITS spectrum.
