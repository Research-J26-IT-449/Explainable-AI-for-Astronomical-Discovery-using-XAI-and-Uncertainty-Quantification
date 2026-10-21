# Model Card: 1D-CNN-BiLSTM for Stellar Spectra Classification

**Project ID:** J26-IT-449  
**Model Name:** `CNNBiLSTM1D`  
**Checkpoint Path:** `models/corrected_split/1d_cnn_bilstm.pth`  
**Author:** P.K. Edirisooriya (IT23330696)  

---

## 1. Model Details
- **Architecture:** Hybrid 1D-Convolutional Bidirectional Long Short-Term Memory Network.
- **Input Dimensions:** `(Batch_Size, 4000)` continuous float32 normalized flux.
- **Output Dimensions:** `(Batch_Size, 5)` class logits corresponding to classes A, B, F, G, K.
- **Components:**
  - Conv Block 1: `Conv1D(1, 32, kernel_size=11, padding=5)` + BatchNorm + ReLU + MaxPool1D(4).
  - Conv Block 2: `Conv1D(32, 64, kernel_size=7, padding=3)` + BatchNorm + ReLU + MaxPool1D(4).
  - Recurrent Sequence: `BiLSTM(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)`.
  - Classifier: `Linear(256, 128)` + ReLU + Dropout(0.4) + `Linear(128, 5)`.

---

## 2. Performance Metrics (Zero-Leakage Benchmark)
- **Held-Out Test Accuracy:** **77.33%** (116 / 150 correct)
- **Balanced Accuracy:** **74.15%**
- **Macro F1-Score:** **0.6406**
- **Class Recall:**
  - Class K: 80.4% (78 / 97)
  - Class G: 73.7% (28 / 38)
  - Class F: 66.7% (6 / 9)
  - Class A: 50.0% (2 / 4)
  - Class B: 100.0% (2 / 2)

---

## 3. Explainability & Interpretability
- **Attribution Method:** `shap.GradientExplainer` (DeepSHAP).
- **Faithfulness:** Top-k attribution masking yields 6.30x faster confidence degradation than random masking (AOPC = 0.7507).
- **Stability:** >77.6% Jaccard explanation overlap under standard APOGEE survey noise (SNR >= 50).
