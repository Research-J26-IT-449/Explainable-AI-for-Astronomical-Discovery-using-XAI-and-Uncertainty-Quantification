# Scientific Limitations and Disclosures

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  

---

## 1. Class Imbalance
The APOGEE survey is an infrared Galactic experiment designed to penetrate interstellar dust in the Milky Way disc and bulge. Consequently, its target selection function heavily favors luminous, cool red giant branch (RGB) and red clump (RC) stars:
- Cool stars (K and G classes) comprise **89.8%** of the dataset.
- Hot stars (A and B classes) comprise only **4.1%**.
- Raw accuracy must not be used in isolation; **Balanced Accuracy** (74.15%) and **Macro F1-score** (0.6406) are the only statistically sound indicators of model generalization.

---

## 2. Proxy Classification vs True MK Spectral Types
The target labels are **effective-temperature proxy classes** derived from ASPCAP $\text{T}_{\text{eff}}$ measurements in the FITS headers:
- True Morgan-Keenan (MK) optical spectral classification requires optical line ratios (Balmer jump, Ca II H&K, G-band, neutral metals, TiO bands).
- In the near-infrared $H$-band, classification is based on effective-temperature bins.
- These labels must strictly be described as "Effective-Temperature Proxy Classes" in all presentations and papers.

---

## 3. Diffuse Spectral Learning vs Isolated Atomic Transitions
Our quantitative spectral line alignment analysis (Alignment Score = 10.53%, Monte Carlo p = 0.5380) reveals that the neural network does not solely look at isolated, textbook atomic lines. Instead, it utilizes diffuse pseudo-continuum slopes, molecular bandheads (CO, CN absorption complexes), and multi-element ionization balances across the entire 1.51 - 1.70 $\mu\text{m}$ passband. Claims of "discovering new physical laws" are unsupported and scientifically inappropriate.
