# Report 14: XAI Stability Evaluation and Jaccard Region Overlap Report

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Research Motivation: Explanation Robustness in Astronomy

Astronomical observations naturally contain photon-counting Poisson noise, detector readout noise, and atmospheric telluric residuals. If a deep learning explanation method fluctuates wildly under minor observational noise, it cannot be trusted by astrophysicists as a reliable tool for scientific discovery.

In `06_xai_stability.py`, we evaluated the **robustness and stability** of DeepSHAP explanations under realistic Gaussian noise matching typical APOGEE signal-to-noise ratios ($\text{SNR} \in \{100, 50, 25\}$).

---

## 2. Quantitative Stability Metric: Region-Based Jaccard Overlap

Because single pixel bins can experience minor discrete boundary shifting, important wavelengths were grouped into continuous regions (top 150 bins with an 8-pixel dilation window, $\approx 3.8$ Å). The stability between original explanation regions $A$ and noise-perturbed explanation regions $B$ was measured using the Jaccard similarity index:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

### Experimental Findings:

| Simulated Survey Condition | Added Flux Noise ($\sigma$) | Mean Jaccard Similarity $J(A, B)$ | Standard Deviation ($\pm \sigma_J$) | Robustness Status |
| :--- | :---: | :---: | :---: | :---: |
| **High Quality ($\text{SNR} \approx 100$)** | $\sigma = 0.01$ | **0.8069** | $\pm 0.0628$ | **EXCELLENT STABILITY** |
| **Standard APOGEE ($\text{SNR} \approx 50$)** | $\sigma = 0.02$ | **0.7768** | $\pm 0.0635$ | **HIGH STABILITY** |
| **Low Signal ($\text{SNR} \approx 25$)** | $\sigma = 0.04$ | **0.6853** | $\pm 0.0539$ | **MODERATE STABILITY** |

### Key Scientific Conclusions:
- At typical survey signal-to-noise levels ($\text{SNR} = 50 - 100$), the Jaccard overlap exceeds **$77\% - 81\%$**, demonstrating that DeepSHAP consistently identifies the same physical wavelength regimes even when spectra are perturbed by noise.
- As noise quadruples to $\text{SNR} \approx 25$, stability gracefully degrades to $68.5\%$, confirming the network relies on persistent spectral morphology rather than brittle noise spikes.
