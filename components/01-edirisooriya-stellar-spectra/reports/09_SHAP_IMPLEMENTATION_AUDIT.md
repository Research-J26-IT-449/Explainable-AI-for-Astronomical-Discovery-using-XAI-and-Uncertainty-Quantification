# Report 09: SHAP Implementation Audit

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Explainability Framework Inspection

In `03_shap_attributions.py` and `03_generate_clean_shap.py`, model interpretability is implemented using `shap.GradientExplainer`.

### Mathematical Foundations
GradientExplainer models the expectation of the gradients of the model's output with respect to the input over a background reference distribution:

$$\phi_i(x) = (x_i - \mathbb{E}[x_i]) \times \int_{0}^1 \frac{\partial f(x_0 + \alpha (x - x_0))}{\partial x_i} d\alpha$$

where $x_0 \sim \mathcal{D}_{\text{background}}$ represents reference baseline spectra.

---

## 2. Technical Audit of Implementation Decisions

1. **Architecture Compatibility:**
   - Standard DeepLIFT / TreeSHAP cannot process PyTorch recurrent layers (`nn.LSTM`).
   - `shap.GradientExplainer` relies on native PyTorch autodiff backward passes through the unrolled recurrent sequence. It executes cleanly on the `CNNBiLSTM1D` architecture without NaN propagation or runtime failures.
2. **Background Distribution Selection:**
   - A representative cohort of 30 randomly selected spectra from the training partition was used as the background reference.
   - Using real stellar spectra as background ensures the baseline distribution respects realistic continuum and noise levels, avoiding out-of-distribution baseline artifacts.
3. **Attribution Tensor Dimensions:**
   - Raw output: $(150, 4000, 5)$ float32 tensor (150 test samples $\times$ 4,000 spectral bins $\times$ 5 classes).
   - Mean absolute attribution (AAAS):
     $$\text{AAAS}(\lambda) = \frac{1}{N} \sum_{k=1}^N |\phi_k(\lambda)|$$
   - Dimension: 4,000 bins corresponding to wavelengths $15,100.8 - 16,999.8$ Å.
