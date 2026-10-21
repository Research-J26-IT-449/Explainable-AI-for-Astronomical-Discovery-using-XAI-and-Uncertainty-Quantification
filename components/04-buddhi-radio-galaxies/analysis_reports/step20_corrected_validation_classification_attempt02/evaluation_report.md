# Step 20 — Fixed-checkpoint corrected validation classification

**CLASSIFICATION EVALUATION COMPLETED**. DEVELOPMENT VALIDATION results; not independent TEST results.

Checkpoint: `outputs/models/mirabest_development_exactdedup_v1.0.0/corrected_seed42_01/models/resnet_cnn_best.pth`; selected epoch 30.
Checkpoint SHA-256: `d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c`.
Manifest: `data/manifests/mirabest_development_exactdedup_v1.0.0/manifest.json`; SHA-256 `4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91`.
Protocol: MiraBest-Development-ExactDedup-v1.0.0. One pass over 116 unique validation images in saved membership order, 51 FR-I / 65 FR-II; quarantined records excluded. Reader is scoped to those 116 rows, not the full corrected TRAIN+VALIDATION cohort. Maximum eight raw images per read; 15 bounded calls. Tensor batches 32,32,32,20; no shuffle/workers.

Command from repository root:
```powershell
.venv\Scripts\python.exe -B analysis_reports/step20_corrected_validation_classification/evaluate_fixed_checkpoint.py --execute-validation
```

Python 3.12.10, Torch 2.14.0+cpu, scikit-learn 1.9.1; CPU, 10 Torch threads. Safe weights_only checkpoint load, strict unchanged ResNetCNN state load, model.eval(), requires_grad=False and torch.inference_mode(). Original 150x150 grayscale uint8 input -> float32 /255 -> mean/std 0.5 normalization. No rotations, flips, resizing, external-upload rejection, OOD recalibration or other augmentation. Seed 42; no optimizer, scheduler or model selection. All model parameters and BatchNorm buffers exactly match before/after; gradients remain absent.

Overall accuracy: **0.870689655172 (87.068966%)**, 101 correct / 116; 15 errors. Exactly reproduces Step 18 aggregate accuracy. Validation loss 0.344764322042; Step 18 0.344764322042.

Confusion matrix (rows=ground truth, columns=prediction; FR-I, FR-II):
```
[[38, 13], [2, 63]]
```

| Class | Support | Precision | Recall | F1 | Errors by true class |
|---|---:|---:|---:|---:|---:|
| FR-I | 51 | 0.950000000 | 0.745098039 | 0.835164835 | 13 |
| FR-II | 65 | 0.828947368 | 0.969230769 | 0.893617021 | 2 |

| Average | Precision | Recall | F1 |
|---|---:|---:|---:|
| macro | 0.889473684 | 0.857164404 | 0.864390928 |
| weighted | 0.882168784 | 0.870689655 | 0.867918215 |

Balanced accuracy (equal mean of class recalls): 0.857164404.
Standard sklearn.metrics implementations with explicit labels [0,1]; zero_division=0 sets undefined precision/recall/F1 to zero. Both class supports are nonzero; no undefined class metrics arise here. Full precision is retained in JSON and CSV.

## Scientific interpretation

True-class errors: FR-I 13/51, FR-II 2/65. FR-I precision/recall are 0.950000/0.745098; FR-II precision/recall are 0.828947/0.969231. These distinguish missed true-class cases (recall) from false assignments among predicted class cases (precision); counts and rates should both be reported. The figure/table show the observed asymmetry without attributing it to physical morphology or a causal mechanism.
FR-II contributes 65/116 (56.03%) and FR-I 51/116 (43.97%). Overall accuracy and support-weighted metrics reflect this imbalance; macro scores and balanced accuracy weight classes equally. This is not evidence that balancing/retraining would improve results. No such experiment was conducted.

This reproduces saved checkpoint development performance; it is not independent validation. The same 116-image cohort selected the checkpoint by minimum loss across 30 epochs. Small cohort, one training seed, and validation selection limit generalization claims. Exact-content development deduplication does not establish astronomical source independence, near-duplicate absence, external-survey robustness or TEST performance. Quarantined conflicting-label root cause remains unresolved. No superiority, discovery, physics alignment or XAI conclusion is drawn.

## Evidence and integrity

validation_predictions.csv preserves all 116 image identifiers, labels, probabilities and correctness in manifest order. metrics.json contains sklearn scores and Step 18 comparison; bounded_read_log.json records every selected row request; preflight_provenance.json records environment, command, source/artifact hashes and exact validation scope. Figures: confusion_matrix.png and per_class_metrics.png. completion_integrity.json is the final success marker with output checksums and verified preservation checks.
No TRAIN, TEST or quarantined image was decoded. Archived label operands were skipped; only verified manifest labels were interpreted. The original split TEST suffix was not parsed. Whole-archive/split provenance hashing and gzip/internal-checksum operations necessarily read opaque serialized bytes; no claim is made of zero OS-level reads of serialized TEST bytes. All 41 protected artifacts and all snapshotted existing research/frontend/source files remain unchanged; Git index, HEAD, branch and prior status are unchanged. No staging, commit, push, retraining, XAI, TEST evaluation or additional configuration execution. The existing .pytest_cache permission warning during git status is unrelated to this evaluation.

Next step: researcher review of these development metrics and prediction evidence. No additional model evaluation or XAI begins automatically.
