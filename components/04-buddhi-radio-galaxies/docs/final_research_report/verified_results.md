# Verified result inventory

Evidence date: 21 October 2026. Verification means saved-artifact inspection and arithmetic, not inference replication. Figures are referenced in place; results are not overwritten.

## Identities

| Item | Verified value |
|---|---|
| Corrected dataset protocol | MiraBest-Development-ExactDedup-v1.0.0 |
| Manifest SHA-256 | `4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91` |
| Corrected checkpoint | `outputs/models/mirabest_development_exactdedup_v1.0.0/corrected_seed42_01/models/resnet_cnn_best.pth` |
| Checkpoint SHA-256 | `d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c` |
| TRAIN / VALIDATION | 552 (245 FR-I / 307 FR-II) / 116 (51 FR-I / 65 FR-II) |
| Quarantined global indices | 198, 323, 474, 587, 694 |
| Corrected XAI protocol | Corrected-GradCAM-Validation-v1.0.0 |
| Target layer | `model.blocks[2].convolution[3]` |

The canonical manifest hash and opaque checkpoint hash were independently recomputed. No checkpoint object was deserialized. The archive's provenance records SHA-256 `505e0b0761a76364a417695380d32d9f26cca7bc0a2608cad46ab338fa46a6b8` and MD5 `fa9d6ad8fdde95b0a19df11433e8bca3`; the historical split reference is SHA-256 `c4a548a66dd37505b8d663bfe6826258364b8118afde4928f5f83a6cdd263008`. Those provenance references do not imply a fresh image audit or TEST inspection. Actual before/after protected-file hashes are enumerated in [verification.json](verification.json).

## Classification

Saved source: [Step 20 metrics](../../analysis_reports/step20_corrected_validation_classification_attempt02/metrics.json) and [116-row CSV](../../analysis_reports/step20_corrected_validation_classification_attempt02/validation_predictions.csv).

| Quantity | Value |
|---|---:|
| Correct / evaluated | 101 / 116 |
| Accuracy | **87.06896551724138%**, displayed 87.068966% |
| Confusion matrix, rows true [FR-I, FR-II] | **[[38,13],[2,63]]** |
| FR-I precision / recall / F1 | 0.950000 / 0.745098 / 0.835165 |
| FR-II precision / recall / F1 | 0.828947 / 0.969231 / 0.893617 |
| Macro precision / recall / F1 | 0.889474 / 0.857164 / **0.8643909282207154** |
| Weighted precision / recall / F1 | 0.882169 / 0.870690 / 0.867918 |
| Balanced accuracy | 0.857164 |

The saved CSV independently reproduces membership order, support, unique indices, finite probabilities, probability sums, argmax predictions, confusion matrix and accuracy. Macro F1 is also reconstructed from the matrix in `verification.json`. This matches Step 18's aggregate accuracy; it does not constitute independent TEST evidence.

## Training and XAI

| Quantity | Recorded value | Evidence |
|---|---|---|
| Epochs / selected epoch | 30 / 30 | Step 18 history and report |
| Minimum validation loss | 0.3447643220424652 | Selected by validation loss |
| Highest observed accuracy epoch | 23, 88.793103% | Not the selected checkpoint |
| Training duration | Approximately 1,089.27 s / 18.15 min | Saved process timing |
| Original valid CAMs | 116 / 116 | Step 21 records |
| Mean Grad-CAM AOPC30 | **0.294510471100364** | Independently averaged saved records |
| Mean random AOPC30 | **0.05396011906980855** | Twenty controls averaged within each image, then 116 images |
| Mean brightness AOPC30 | **0.22225442241910118** | Independently averaged saved records |
| Negative CAM AOPCs | 8 | Preserved, not clipped |
| Transformed prediction flips | 1,553 / 3,480 repeated comparisons | Not independent-image count |
| Degenerate transformed comparisons | 1,181 | Unavailable Jaccard, not zero |
| XAI wall time / recorded maximum working set | ~329.91 s / 427,483,136 bytes | Historical telemetry, not a hardware guarantee |

Floating-point summation at machine precision gives 0.29451047110036394, 0.053960119069808535 and 0.2222544224191012 in the independent stdlib arithmetic check. These agree with the saved summary within 10⁻¹² and round to the requested six-decimal values. No new model scores were generated.

## Verification boundaries

All requested headline values are supported. Original-map validity does not imply transformed-map validity, physical localization, calibrated probabilities or external generalization. ECE, expert-mask physics alignment and corrected OOD results are not available. Some broader XAI analyses are deferred. All 41 existing protected hashes match; the current checked Step 21 implementation hash set matches its saved execution provenance. The complete review is not a fresh replication of historical experiments.
