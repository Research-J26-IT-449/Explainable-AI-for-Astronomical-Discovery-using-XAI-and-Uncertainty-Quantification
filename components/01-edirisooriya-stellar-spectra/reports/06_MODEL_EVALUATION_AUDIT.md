# Report 06: Model Evaluation and Verification Audit

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Executive Evaluation Comparison

This report details the rigorous evaluation of the three deep learning architectures across both the **Historical Benchmark** (with the discovered 102-sample train-test leakage documented in Report 04) and the **Corrected Zero-Leakage Benchmark** (completely clean held-out test cohort of 150 independent stars).

### Summary Results Table

| Model Architecture | Split Condition | Test Accuracy | Macro F1-Score | Balanced Accuracy | Test Loss |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1D-CNN-BiLSTM (Best)** | **Historical (Leaked)** | 91.33% | 0.7008 | 0.7274 | 0.4182 |
| **1D-CNN-BiLSTM (Best)** | **Corrected (Clean)** | **77.33%** | **0.6406** | **0.7415** | 0.5897 |
| **1D-ResNet** | Historical (Leaked) | 65.33% | 0.3703 | 0.5148 | 1.1023 |
| **1D-ResNet** | Corrected (Clean) | 46.00% | 0.3116 | 0.4269 | 1.1124 |
| **Standard 1D-CNN** | Historical (Leaked) | 64.67% | 0.1571 | 0.2000 | 1.3421 |
| **Standard 1D-CNN** | Corrected (Clean) | 32.00% | 0.1699 | 0.2917 | 1.8773 |

---

## 2. Deep Dive: 1D-CNN-BiLSTM Generalization

On the verified clean, zero-leakage test set ($N=150$ independent stars), the **1D-CNN-BiLSTM** achieves:
- **Test Accuracy:** **77.33%** (116 / 150 correct)
- **Macro F1-Score:** **0.6406**
- **Balanced Accuracy:** **0.7415**

### Corrected Split Confusion Matrix:
```
           Pred A  Pred B  Pred F  Pred G  Pred K    Total
True A          2       2       0       0       0        4
True B          0       2       0       0       0        2
True F          0       1       6       2       0        9
True G          0       0       6      28       4       38
True K          1       0       2      16      78       97
Total           3       5      14      46      82      150
```

### Analysis of Per-Class Behavior:
1. **Majority Class Recognition:** Class K recall is $78 / 97 = 80.4\%$. Class G recall is $28 / 38 = 73.7\%$.
2. **Intermediate Class Recognition:** Class F recall is $6 / 9 = 66.7\%$.
3. **Minority Hot Star Detection:**
   - Both True Class B stars were correctly detected (Recall = 100.0%, 2/2).
   - Class A recall is 50.0% (2/4), with the remaining 2 predicted as Class B (physically adjacent hot star regime).
4. **Balanced Accuracy ($74.15\%$):** Demonstrates that the hybrid CNN-BiLSTM does not collapse into predicting only the dominant K-type class, but actively captures distinctive absorption profiles across all 5 classes.
