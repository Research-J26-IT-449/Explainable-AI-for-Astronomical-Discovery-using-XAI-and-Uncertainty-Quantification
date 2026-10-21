# Step 18 — Corrected CNN training execution report

Experiment: `corrected_seed42_01`. One authorized run; successful process exit code 0.

Started UTC: 2026-10-21T04:48:12.367866+00:00; finished UTC: 2026-10-21T05:06:21.752860+00:00.
Local start: 2026-10-21T10:18:12.367866+05:30; local finish: 2026-10-21T10:36:21.752860+05:30.
Total process wall time: 1089.275 seconds (18.15 minutes).

Command (repository root):
```powershell
.venv\Scripts\python.exe -B scripts/train_corrected_resnet.py --execute-training --run-id corrected_seed42_01
```

Python: 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]. PyTorch: 2.14.0+cpu. Device: CPU; 10 Torch threads; CUDA unavailable. No package changes.

Protocol: MiraBest-Development-ExactDedup-v1.0.0. TRAIN 552 (245 FR-I / 307 FR-II); VALIDATION 116 (51 FR-I / 65 FR-II). Quarantine: 198, 323, 474, 587, 694.
Corrected manifest SHA-256: `4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91`.

ResNetCNN: fresh seed-42 random initialization; residual widths 32/64/128; dropout 0.30; 339,746 parameters. No historical checkpoint initialization. Batch 32; Adam lr=0.0003, weight_decay=0.0001; CrossEntropyLoss; ReduceLROnPlateau(mode=min, patience=2, factor=0.5); maximum 30 epochs; early-stopping patience 7. Selection: strictly minimum validation loss. Original training transforms: random rotation ±15°, horizontal/vertical flips, tensor conversion, mean/std 0.5 normalization. Validation: deterministic tensor conversion and same normalization.

Completed epochs: 30; stopping reason: maximum_epoch_limit. Best epoch: 30; validation loss: 0.344764322042; corresponding accuracy: 0.870689655172 (87.07%).

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | LR after scheduler | Train+validation seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.682545100 | 0.567028986 | 0.685756877 | 0.560344828 | 0.0003 | 33.687 |
| 2 | 0.652284036 | 0.619565217 | 0.720630379 | 0.560344828 | 0.0003 | 33.350 |
| 3 | 0.613185621 | 0.664855072 | 0.792277318 | 0.560344828 | 0.0003 | 33.652 |
| 4 | 0.575954613 | 0.719202899 | 0.983964908 | 0.551724138 | 0.00015 | 36.268 |
| 5 | 0.550739715 | 0.728260870 | 0.513906234 | 0.793103448 | 0.00015 | 37.767 |
| 6 | 0.524656247 | 0.739130435 | 1.364184848 | 0.439655172 | 0.00015 | 37.173 |
| 7 | 0.478753738 | 0.777173913 | 1.339785440 | 0.439655172 | 0.00015 | 35.868 |
| 8 | 0.485831389 | 0.755434783 | 0.642071068 | 0.629310345 | 7.5e-05 | 35.244 |
| 9 | 0.453484847 | 0.793478261 | 0.712417154 | 0.603448276 | 7.5e-05 | 35.256 |
| 10 | 0.424042074 | 0.811594203 | 0.660736220 | 0.612068966 | 7.5e-05 | 36.524 |
| 11 | 0.427283758 | 0.793478261 | 0.821833056 | 0.534482759 | 3.75e-05 | 36.868 |
| 12 | 0.439802799 | 0.777173913 | 0.490854736 | 0.758620690 | 3.75e-05 | 34.786 |
| 13 | 0.395433987 | 0.826086957 | 0.467117959 | 0.767241379 | 3.75e-05 | 37.934 |
| 14 | 0.405539460 | 0.813405797 | 0.399618079 | 0.844827586 | 3.75e-05 | 35.028 |
| 15 | 0.408986753 | 0.822463768 | 0.476473936 | 0.741379310 | 3.75e-05 | 36.289 |
| 16 | 0.388649509 | 0.842391304 | 0.456916610 | 0.775862069 | 3.75e-05 | 35.617 |
| 17 | 0.394151221 | 0.807971014 | 0.381362724 | 0.870689655 | 3.75e-05 | 33.708 |
| 18 | 0.370553570 | 0.858695652 | 0.569178532 | 0.741379310 | 3.75e-05 | 33.763 |
| 19 | 0.381279130 | 0.838768116 | 0.388014600 | 0.836206897 | 3.75e-05 | 34.357 |
| 20 | 0.372916834 | 0.827898551 | 0.436373406 | 0.775862069 | 1.875e-05 | 35.989 |
| 21 | 0.369438958 | 0.833333333 | 0.387482232 | 0.827586207 | 1.875e-05 | 35.538 |
| 22 | 0.365965911 | 0.846014493 | 0.418764271 | 0.784482759 | 1.875e-05 | 36.853 |
| 23 | 0.360852222 | 0.838768116 | 0.355109069 | 0.887931034 | 1.875e-05 | 36.587 |
| 24 | 0.351274750 | 0.865942029 | 0.363180113 | 0.870689655 | 1.875e-05 | 39.562 |
| 25 | 0.358389374 | 0.849637681 | 0.355456877 | 0.870689655 | 1.875e-05 | 34.199 |
| 26 | 0.346845784 | 0.862318841 | 0.370623928 | 0.836206897 | 9.375e-06 | 34.519 |
| 27 | 0.361058599 | 0.844202899 | 0.366934595 | 0.836206897 | 9.375e-06 | 34.553 |
| 28 | 0.330837220 | 0.864130435 | 0.354541787 | 0.870689655 | 9.375e-06 | 39.022 |
| 29 | 0.344635469 | 0.864130435 | 0.421365695 | 0.801724138 | 9.375e-06 | 40.006 |
| 30 | 0.335366840 | 0.867753623 | 0.344764322 | 0.870689655 | 9.375e-06 | 44.415 |

Checkpoint: `outputs/models/mirabest_development_exactdedup_v1.0.0/corrected_seed42_01/models/resnet_cnn_best.pth`.
SHA-256: `d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c`.
Verified readable via torch.load(weights_only=True, map_location='cpu') and strict ResNetCNN state-dict loading, with finite parameters/buffers; no inference performed.

Artifacts: isolated run_provenance.json (configuration, manifest, environment, implementation hashes), complete.json (four native artifact checksums), models/resnet_cnn_best.pth, metrics/resnet_training_history.csv, metrics/resnet_validation_metrics.json, figures/resnet_training_curves.png and epoch_telemetry.jsonl. All checksums checked. The approved YAML remains execution_authorized:false; the separate explicit execution flag implements this user's bounded authorization without changing frozen configuration.

No training failures or restarts. Added operational telemetry only: new src/models/training_telemetry.py, its synthetic tests, and a scoped wrapper/source-hash addition in src/models/corrected_training.py. The original trainer bytes and mathematics were unchanged. Telemetry restores the temporary single-process epoch binding on exit. Durations cover training plus validation, exclude scheduler/checkpoint/plot IO, and are separate from native deterministic history. LR is recorded after scheduler updates. Focused synthetic verification: 94 passed, 0 failed/skipped/errors in 9.46 seconds.
Synthetic command: `.venv\Scripts\python.exe -B -m pytest -q -ra -p no:cacheprovider --basetemp analysis_reports/s18_tmp tests/test_training_telemetry.py tests/test_corrected_training.py tests/test_corrected_mirabest_protocol.py tests/test_mirabest_pickle_stream_reader.py`.

Preflight: all 41 protected hashes matched; manifest/counts/class labels/exclusions verified; new output namespace; about 39 GiB free; no preexisting corrected-training process. Authorized bounded raw-read smoke check used TRAIN index 0 and VALIDATION index 31. Training uses at most eight selected raw images per reader call, clears each raw batch, and retains only the tensor minibatch. Archived label operands are skipped; labels come from approved corrected metadata. TEST pixel arrays/labels were not decoded or interpreted. Opaque archive hashes/gzip/checksums necessarily read serialized bytes; this is not an OS-level claim that serialized TEST bytes were never read.

Integrity: all 41 protected artifacts remain unchanged; original archive, original split, corrected protocol artifacts, historical model/metrics/XAI, frontend and all other baseline files unchanged except the authorized corrected-training telemetry integration. Git branch/HEAD/index unchanged. No staging, commits or pushes. Initial process inventory required read-only elevated CIM access after sandbox Access denied; this was not a data failure. No training exceptions; stderr empty.

Limitations: these are validation-selected development results, not independent final generalization evidence. Exact-content disjointness does not establish source independence or exclude near duplicates. Quarantined conflicting-label root cause remains unresolved. Historical and corrected cohort metrics are not directly interchangeable. No TEST evaluation, Grad-CAM/XAI, calibration changes or duplicate-audit rerun occurred.

Recommended next step: researcher review of saved training/validation history and validation-only analysis planning for this new checkpoint. Any TEST or XAI execution requires separate approval; do not repeat successful training automatically.

Verdict: TRAINING COMPLETED — READY FOR VALIDATION ANALYSIS
