# Step 19 — Corrected CNN validation analysis

Verdict: **VALIDATION ANALYSIS COMPLETED**. Saved-artifact analysis only; no training, validation inference, dataset image access, TEST evaluation or XAI execution.

## Verified experiment

Experiment `corrected_seed42_01`; protocol `MiraBest-Development-ExactDedup-v1.0.0`. TRAIN 552 (245 FR-I, 307 FR-II); VALIDATION 116 (51 FR-I, 65 FR-II). Global indices 198, 323, 474, 587 and 694 remain excluded; retained labels unchanged.
Membership SHA-256: `4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91`.
Checkpoint SHA-256: `d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c`.

Completed 30 epochs on CPU; process exit 0; elapsed 1089.275 seconds (18.15 minutes). Stopped at the configured maximum, not early stopping. Selected epoch **30**: validation loss **0.344764322042**, validation accuracy **87.068966%** (101/116 correct, 15 incorrect; counts derived from aggregate accuracy, not independent per-image predictions). Epoch 23 had the highest recorded accuracy, 88.793103% (103/116), but was not selected because its loss 0.355109068854 exceeded epoch 30's loss. Do not substitute accuracy-based selection.

Verified CSV history against metrics JSON and all 30 telemetry rows; finite values, consecutive epochs and count-compatible accuracies; checkpoint metadata against selected epoch/configuration; completion artifact checksums; recorded implementation/configuration hashes; corrected manifest publication checksums; Step 18 report and process/verification records. No conflicts found. Checkpoint read with torch.load(weights_only=True, map_location='cpu') for metadata only; no model constructed or executed. run_provenance.json deliberately retains its initial 'started' status; complete.json and process_finished.json establish completion. Its command_argv omits the interpreter's -B flag as expected from sys.argv; the external process record captures it.

## Configuration and convergence

Existing ResNetCNN, fresh initialization, seed 42, batch 32, Adam learning rate 0.0003 and weight decay 0.0001; cross-entropy; ReduceLROnPlateau patience 2/factor 0.5; early-stop patience 7; strict minimum validation-loss selection. TRAIN uses original ±15 degree rotations, horizontal/vertical flips and mean/std 0.5 normalization; VALIDATION uses deterministic tensor conversion and normalization.

Training loss declined from 0.682545 to 0.335367; training accuracy rose from 56.70% to 86.78%. Validation loss declined overall from 0.685757 to 0.344764, but with marked non-monotonic fluctuations. Epoch 6 peaked at validation loss 1.364185 and 43.97% accuracy. Later best losses at epochs 14, 17, 23, 28 and 30 indicate improvement of the validation-loss envelope, not smooth convergence.

The final epoch achieved a new minimum loss, lower than epoch 28's 0.354542 by 0.009777. Thus improvement was still observed at the endpoint; a stable plateau or optimal training duration is not established. This does not predict improvement from extra epochs and does not authorize extending or repeating the successful experiment.

## Overfitting assessment

At epoch 30, training loss is 0.335367 versus validation loss 0.344764 (gap 0.009397), and training accuracy 86.775362% versus validation 87.068966% (validation higher by 0.293603 percentage points). The late curve does not show sustained worsening validation loss alongside improving training loss, so it provides no clear evidence of persistent late-stage overfitting. This is not proof of absence: the cohort is small, repeatedly used for selection, and the curves are noisy. Training metrics are accumulated on augmented batches while weights evolve in training mode (including dropout/BatchNorm), whereas validation uses fixed data and evaluation mode. They are not directly matched unaugmented evaluations; validation accuracy above training is not itself an error.

## Learning-rate changes

Recorded LR is after validation and scheduler.step; a reduction applies to training from the next epoch.

| Scheduler update after epoch | Previous LR | New recorded LR | First training epoch using new LR |
|---:|---:|---:|---:|
| 4 | 0.0003 | 0.00015 | 5 |
| 8 | 0.00015 | 7.5e-05 | 9 |
| 11 | 7.5e-05 | 3.75e-05 | 12 |
| 20 | 3.75e-05 | 1.875e-05 | 21 |
| 26 | 1.875e-05 | 9.375e-06 | 27 |

## Metric jumps and internal consistency

Largest absolute consecutive validation-accuracy changes (all recorded, not smoothed):

| Epoch transition | Accuracy change (percentage points) | Validation loss change |
|---|---:|---:|
| 5 → 6 | -35.344828 | +0.850278614 |
| 4 → 5 | +24.137931 | -0.470058673 |
| 11 → 12 | +22.413793 | -0.330978320 |
| 7 → 8 | +18.965517 | -0.697714372 |
| 17 → 18 | -12.931034 | +0.187815808 |
| 22 → 23 | +10.344828 | -0.063655202 |

These are substantial fluctuations worth retaining in reporting. All agree across the saved history, telemetry and summary; accuracies correspond to integer counts for their cohort sizes. Scheduler changes and loss-based selection match the implementation. No bookkeeping contradiction was found, but aggregate history cannot identify which images/classes changed predictions or establish why. Optimization, stochastic augmentation and BatchNorm behavior are possible influences, not diagnosed causes. No additional experiment was performed.

## Available and missing classification metrics

Available: selected-checkpoint overall validation accuracy 87.068966%, validation cross-entropy 0.344764322042 and epoch-level train/validation history. The seven-file experiment inventory contains no per-image predictions, logits/probabilities, paired labels or confusion counts. The corrected manifest supplies ground-truth labels only; it cannot recover model predictions. The checkpoint contains state_dict, epoch, aggregate loss/accuracy and config only.

**NOT YET AVAILABLE:** confusion matrix; per-class precision, recall and F1; macro F1; weighted F1. The 101 correct / 15 incorrect totals and class supports do not uniquely determine these metrics. No balanced accuracy or class-specific error pattern is inferred. JSON uses null and explicit availability reasons, not fabricated zeros.

## Historical comparison

Historical saved selected validation accuracy is 85.714286% (102/119), versus corrected 87.068966% (101/116): a descriptive difference of +1.354680 percentage points. Historical selected epoch 22/loss 0.350147 differs from this run's epoch 30/loss 0.344764. This is not a controlled paired comparison or evidence of statistical superiority.

Original TRAIN/VALIDATION sizes were 554/119; corrected sizes are 552/116. The five-record exact-content group crossed TRAIN (323, 694) and VALIDATION (198, 474, 587), with conflicting labels. Removing all five changes training data and validation composition, including class balance; fresh training also changes the learned model and trajectory. The direction/magnitude of historical leakage bias cannot be quantified from these summaries or assumed to explain the accuracy difference. Historical metrics remain historical, not silently corrected. Exact-image disjointness within the audited corrected development cohort does not prove astronomical source independence, absence of near duplicates or external-survey generalization. Corrected-model TEST performance is unknown; historical TEST metrics do not apply to this checkpoint.

## Limitations and next evaluation

One seed/run; small validation cohort; validation-driven checkpoint selection; substantial epoch variability; final-epoch best with convergence unproven; missing per-image classification evidence; unresolved conflicting-label provenance; unverified source grouping/near duplicates; no independent external or corrected-model TEST evaluation. No statistical superiority or XAI claim is made.

Recommended next step, requiring separate explicit approval: one fixed-checkpoint, inference-only evaluation of exactly the 116 corrected VALIDATION records, with trusted deterministic preprocessing, bounded authorized-row access and no weight/BatchNorm updates. Save index, ground truth, prediction and class scores with checkpoint/manifest/configuration hashes; calculate confusion matrix and per-class/macro/weighted F1. Do not tune or retrain against these results. This would be development characterization, not independent validation. No TEST or XAI execution is included in that recommendation.

## Outputs and evidence

- training_loss_curve.png: genuine unsmoothed TRAIN/VALIDATION loss values.
- accuracy_curve.png: genuine unsmoothed accuracies, selected epoch and highest-accuracy marker.
- metric_summary.json: exact values, missingness reasons, LR changes, jumps and seven-file inventory.
- pre_analysis_integrity.json / integrity_verification.json: baseline and post-analysis hashes/Git state.
- analyze_saved_history.py: reproducible artifact-only analysis; no dataset imports or forward pass.

Source references: outputs/models/mirabest_development_exactdedup_v1.0.0/corrected_seed42_01/ (run_provenance.json, complete.json, epoch_telemetry.jsonl, metrics/resnet_training_history.csv, metrics/resnet_validation_metrics.json, models/resnet_cnn_best.pth); analysis_reports/step18_corrected_cnn_training/ (training_report.md, process_finished.json, verification.json); config/corrected_resnet_training_v1.0.0.yaml; data/manifests/mirabest_development_exactdedup_v1.0.0/ (manifest.json, provenance.json, validation_report.json); outputs/metrics/resnet_validation_metrics.json; src/models/train.py (training/evaluation modes and aggregate-only output definitions).

## Integrity

All 41 protected artifact hashes matched before and after analysis. All snapshotted existing source, configuration, protocol, frontend, checkpoint, metric, XAI and Step 18 files unchanged. Git branch/HEAD/index and status unchanged; only this new report directory was written. No dataset images or TEST records/labels accessed; no model forward pass, training, XAI, staging, commit, push or historical artifact rewrite. Checksum verification of protected files is opaque byte hashing, not dataset decoding. Git status emitted the preexisting .pytest_cache permission warning; it did not affect artifact verification.
