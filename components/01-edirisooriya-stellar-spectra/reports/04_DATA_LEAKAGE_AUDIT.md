# Report 04: Data Leakage and Dataset Splitting Audit

**Project ID:** J26-IT-449  
**Individual Component:** SHAP-Driven Feature Discovery in Stellar Spectra  
**Researcher:** P.K. Edirisooriya (IT23330696)  
**Date:** October 20, 2026  

---

## 1. Discovery of Historical Data Leakage

During Phase 2 of this technical audit, an in-depth examination of `01_data_preprocess.py` identified a subtle but impactful Python array indexing bug during dataset partitioning.

### Code Inspection: The Indexing Flaw
In the original preprocessing script, a two-stage stratified split was written as follows:
```python
# Stage 1: split master dataset into train_val and test
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=42)
train_idx, test_idx = next(strat_split.split(X, y))

# Stage 2: split the train portion into train and validation
val_split_adj = val_split / (1.0 - test_split)
strat_split_val = StratifiedShuffleSplit(n_splits=1, test_size=val_split_adj, random_state=42)
train_idx, val_idx = next(strat_split_val.split(X[train_idx], y[train_idx]))

# Mask assignment bug:
train_mask = np.zeros(len(y), dtype=bool)
val_mask = np.zeros(len(y), dtype=bool)
test_mask = np.zeros(len(y), dtype=bool)
train_mask[train_idx] = True  # <-- train_idx is 0..849 from sub-split!
val_mask[val_idx] = True      # <-- val_idx is 0..849 from sub-split!
test_mask[test_idx] = True    # <-- test_idx is 0..999 from master split!
```

### Mathematical Mechanism of the Bug
- In scikit-learn, `split(X[train_idx], y[train_idx])` outputs indices relative to the **sub-array** (indices $0, 1, \dots, 849$).
- These indices were assigned directly to boolean masks of length 1,000 without mapping back via `train_val_idx[sub_train_idx]`.
- As a consequence, `test_idx` (which pointed to indices across the full range $0 \dots 999$) intersected with the lower range indices of `train_mask`.

### Measured Overlap in Historical Split:
* **Train & Validation Overlap:** 0 samples
* **Train & Test Overlap:** **102 samples** (out of 150 test samples = 68.0% leakage)
* **Validation & Test Overlap:** **29 samples** (out of 150 test samples = 19.3% leakage)
* **True Held-Out Test Samples:** Only 19 samples were truly unseen.

This mathematical finding explains why the historical 1D-CNN-BiLSTM model achieved **91.33% test accuracy** ($137 / 150$). The model had seen 102 of the 150 test spectra during training.

---

## 2. Corrected Zero-Leakage Group-Aware Splitting

To rectify this flaw and ensure the research meets the highest scientific standards, a corrected splitting routine was constructed:

```python
# Stage 1: Master split into train_val (85%) and test (15%)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(sss1.split(X, y))

# Stage 2: Partition train_val into train (70%) and val (15%)
val_frac = 0.15 / 0.85
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
sub_train, sub_val = next(sss2.split(train_val_idx, y[train_val_idx]))

# Correct index mapping:
clean_train_idx = train_val_idx[sub_train]
clean_val_idx = train_val_idx[sub_val]
clean_test_idx = test_idx
```

### Verification of the Corrected Split:
* Total unique stars: 1,000
* **Train Set:** 700 spectra (Class 0: 17, Class 1: 12, Class 2: 43, Class 3: 176, Class 4: 452)
* **Validation Set:** 150 spectra (Class 0: 4, Class 1: 2, Class 2: 9, Class 3: 38, Class 4: 97)
* **Test Set:** 150 spectra (Class 0: 4, Class 1: 2, Class 2: 9, Class 3: 38, Class 4: 97)
* **Train $\cap$ Validation:** **0 samples (0.0% overlap)**
* **Train $\cap$ Test:** **0 samples (0.0% overlap)**
* **Validation $\cap$ Test:** **0 samples (0.0% overlap)**

Every star in the test set is completely held out.

---

## 3. Policy for PP1 Reporting

In academic research, detecting and correcting an evaluation bug before formal submission demonstrates exceptional research rigor and maturity. The project will report both sets of results:
1. **Historical Run (Documented Audit Baseline):** 91.33% accuracy, clearly noting the 102-sample overlap bug discovered during the code audit.
2. **Corrected Clean Run (True Scientific Benchmark):** Retrained on the verified zero-leakage split, providing the true generalization capability of the models.
