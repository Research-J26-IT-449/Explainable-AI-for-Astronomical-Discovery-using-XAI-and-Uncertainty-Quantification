# Report 15: Automated Test Suite Audit Report

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Test Execution Summary

The automated verification suite was executed using `pytest 9.1.1` under Python 3.14.6:

```
Platform: win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.7.0
Root Directory: C:\Users\pasan\OneDrive\Desktop\RESEARCH PLAN\module1_stellar_spectra
Collected Test Files: 3
  1. tests/test_extended_audit.py (15 tests)
  2. tests/test_label_boundaries.py (1 test)
  3. tests/test_pipeline.py (34 tests)

Total Assertions / Tests: 50
Passed: 50 (100.0%)
Failed: 0 (0.0%)
Skipped: 0
Execution Time: 2.63 seconds
```

---

## 2. Test Coverage Matrix

| Test Suite / Module | Assertions | Target Verification | Outcome |
| :--- | :---: | :--- | :---: |
| **`TestZeroLeakageDataset`** | 4 | Train/Val/Test counts (700/150/150), 4000 feature length, 0 overlap, finite values | **PASSED** |
| **`TestModelArchitectures`** | 4 | Forward pass tensor dimensions `(B, 5)`, NaN check, checkpoint loading | **PASSED** |
| **`TestScientificLineValidation`** | 3 | Peak detection non-zero, alignment score bounds, random baseline presence | **PASSED** |
| **`TestXAIEvaluation`** | 2 | AOPC top-k $>0$, Faithfulness advantage ratio $>1.0\times$, Jaccard stability $\ge 0.70$ | **PASSED** |
| **`TestBaselineBenchmarks`** | 2 | Benchmark JSON presence, CNN-BiLSTM balanced accuracy supremacy | **PASSED** |
| **`test_label_boundaries.py`** | 1 | Full manifest temperature to proxy class consistency across all 1,004 stars | **PASSED** |
| **`TestLabelBoundaries`** | 8 | Boundary thresholds for classes M, K, G, F, A, B, O; edge inclusivity | **PASSED** |
| **`TestManifest`** | 5 | Column completeness (`objid`, `teff`, `label`), no nulls, known class enum, $N > 62$ | **PASSED** |
| **`TestDatasetSplits`** | 5 | Array pair length match, feature length 4000, test size matches metrics | **PASSED** |
| **`TestEvaluationConsistency`** | 9 | Matrix sum equals test size (150), diagonal accuracy equals reported accuracy (91.33%) | **PASSED** |
| **`TestSHAPOutput`** | 3 | Key presence, 4000-bin axis, all values finite | **PASSED** |
| **`TestLineMatching`** | 4 | Column presence, non-empty, AAAS $\ge 0$, wavelengths in APOGEE $H$-band | **PASSED** |

**Total Pass Rate: 50 / 50 (100.0%)**
