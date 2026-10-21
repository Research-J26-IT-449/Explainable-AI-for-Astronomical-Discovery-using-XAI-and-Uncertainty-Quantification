# Report 03: Proxy Spectral Label Validation Report

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Definition and Source of Proxy Labels

In stellar astrophysics, true spectral classification (the Morgan-Keenan or Harvard MK system) is traditionally determined by visual inspection or automated matching of optical absorption line ratios (e.g., Balmer lines, He I, He II, Ca II H&K, G-band, neutral metals, TiO bands).

In this research project, spectra are drawn from the near-infrared APOGEE survey. The target labels are **effective-temperature derived proxy spectral classes**, mapped from ASPCAP spectroscopic effective temperatures ($\text{T}_{\text{eff}}$) extracted from the FITS table headers.

### Label Mapping Boundaries
The project adopts standardized empirical effective temperature boundaries for main sequence and giant stars:

$$\text{Class} = \begin{cases} 
\text{M} & \text{if } 0 \le \text{T}_{\text{eff}} < 3500\,\text{K} \\ 
\text{K} & \text{if } 3500 \le \text{T}_{\text{eff}} < 5000\,\text{K} \\ 
\text{G} & \text{if } 5000 \le \text{T}_{\text{eff}} < 6000\,\text{K} \\ 
\text{F} & \text{if } 6000 \le \text{T}_{\text{eff}} < 7500\,\text{K} \\ 
\text{A} & \text{if } 7500 \le \text{T}_{\text{eff}} < 10000\,\text{K} \\ 
\text{B} & \text{if } 10000 \le \text{T}_{\text{eff}} < 30000\,\text{K} \\ 
\text{O} & \text{if } \text{T}_{\text{eff}} \ge 30000\,\text{K} 
\end{cases}$$

---

## 2. Active Sample Class Distribution

Across the 1,000 active preprocessed APOGEE spectra, the distribution across proxy classes is:

| Proxy Class | Temperature Range (K) | Sample Count | Percentage (%) | Observed $\text{T}_{\text{eff}}$ Range (K) |
| :---: | :---: | :---: | :---: | :---: |
| **K** | $3500 \le \text{T}_{\text{eff}} < 5000$ | 646 | 64.6% | $3,500.0 - 4,999.4$ |
| **G** | $5000 \le \text{T}_{\text{eff}} < 6000$ | 252 | 25.2% | $5,003.0 - 5,989.2$ |
| **F** | $6000 \le \text{T}_{\text{eff}} < 7500$ | 61 | 6.1% | $6,005.1 - 7,489.6$ |
| **A** | $7500 \le \text{T}_{\text{eff}} < 10000$ | 25 | 2.5% | $7,626.4 - 9,739.0$ |
| **B** | $10000 \le \text{T}_{\text{eff}} < 30000$ | 16 | 1.6% | $10,016.8 - 27,712.4$ |
| **Total** | — | **1,000** | **100.0%** | — |

*Note: Classes M and O have 0 samples in this selection due to APOGEE targeting criteria and the rarity of extremely hot O-stars in the selected fields.*

---

## 3. Scientific Implications and Disclosures for PP1

1. **Severe Natural Class Imbalance:** Classes K and G represent **89.8%** of the entire dataset. In contrast, hot classes (A and B) represent only **4.1%** combined. This is an unavoidable physical property of infrared galactic surveys (which preferentially target cool red giants that penetrate dust extinction).
2. **Impact on Evaluation:** Overall test accuracy will be predominantly driven by performance on K and G stars. Models that fail on B stars can still achieve $>89\%$ raw accuracy. Therefore, **Macro F1-score** and **Balanced Accuracy** are the only scientifically honest metrics for this task.
3. **Disclosure Rule:** The terminology "Effective-Temperature Proxy Class" must be maintained across all PP1 presentation slides and defense answers. It must never be claimed that the model is predicting true Harvard optical spectral types.
