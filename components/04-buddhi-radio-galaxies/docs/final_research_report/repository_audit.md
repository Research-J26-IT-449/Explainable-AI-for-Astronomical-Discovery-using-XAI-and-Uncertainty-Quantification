# Repository audit preceding report preparation

Audit date: 21 October 2026. Branch: `feature/external-radio-input`. HEAD: `2d6e5883e4a6a16252efdb24e5e4f52f61aca026`. The index was empty; the working tree contained substantial existing modified and untracked research/UI work. HEAD alone does not identify every corrected-model implementation file. No branch, index or existing file was changed for this review.

## Evidence hierarchy

The corrected manifest, selected checkpoint identity, saved per-image predictions and saved per-image XAI records are the principal evidence. Their completion/provenance records and dated milestone reports provide context. Earlier summaries are historical snapshots. User-supplied numbers were treated as claims to verify, not as substitutes for artifacts.

| Area | Principal inspected evidence | Finding |
|---|---|---|
| Proposal/objectives | Root README; `docs/50_percent_progress.md`; documentation inventory | No official proposal or approved objective document located in the inspected current repository. Implemented aims can be reconstructed but cannot be called the approved wording. |
| Dataset correction | Step 14B evidence, Step 14C review, Step 15 proposal, corrected manifest/provenance | Five conflicting-label exact duplicates quarantined; 552 TRAIN and 116 VALIDATION retained. |
| Architecture/training | `src/models/resnet_cnn.py`, `src/data/preprocess.py`, corrected training configuration, Steps 18–19 | Custom lightweight residual CNN, fresh seed-42 training; minimum validation loss selects epoch 30. |
| Classification | Step 20 attempt02 predictions, metrics and integrity records | 101/116 correct; confusion matrix and headline accuracy/F1 reproduced from saved evidence. |
| XAI | Core modules; Step 21 attempt02 protocol, records, summary and interpretation | 116 valid original CAMs; primary deletion and implemented stability/proxy analyses completed, broader analyses not all completed. |
| Dashboard | `demo/`, `public_demo/`, Steps 22–24 | Gradio research-results presentation; public prediction disabled pending compatible OOD evidence. |
| Academic background | Existing documentation plus primary publisher/arXiv metadata | No complete proposal bibliography found; a targeted verified reference set accompanies the draft. |

## Conflicts and reconciliation

1. The progress document reports historical 85.71% validation and 83.19% TEST accuracy and describes XAI as future work. It does not describe the later corrected experiment. Its assertion of a single TEST run is not accepted as established execution provenance; the registry records uncertainty about historical run count and checkpoint bytes at evaluation time.
2. The original Phase 3B protocol identifies the historical checkpoint and 119 validation records. The separately versioned corrected XAI protocol identifies the corrected checkpoint and 116 records. These are separate cohorts/identities, not an undocumented amendment to the old protocol.
3. Frozen `execution_authorized: false` configuration values are distinguished from explicit bounded execution approvals. Previous approvals do not authorize new experiments.
4. MiraBest's published dataset description reports a larger collection than the local 792-record binary payload. The report describes the local verified cohort, not every record in later/public MiraBest releases.
5. Original cross-split content leakage was found after historical experimentation. The corrected protocol removes all five affected records without guessing their true labels. The cause of identical pixels with conflicting labels remains unknown.
6. All original corrected-model CAMs were valid, but 1,181 transformed comparisons were unavailable through degenerate CAMs. Original-map validity cannot be reported as universal explanation robustness.
7. Corrected XAI completion covers the explicitly executed scope. Secondary full deletion curves/replacement sensitivities and parameter-randomization experiments remain deferred; several other analyses are undefined or lack independent reference data.
8. Step 22 functional model/Grad-CAM integration does not establish live public inference availability. Step 23 found insufficient corrected-model OOD evidence; Step 24 preserves a results-only public presentation.
9. Execution-time implementation hashes in Step 21 preflight match the current files checked by `verification.json`. This does not make all uncommitted files part of HEAD or establish that every historical execution has complete provenance.
10. Step 21's completion ledger captured an operational stdout log before its final appended line. `final_integrity.json` documents this operational-log mismatch; `final_report_manifest.json` supplies a later ledger. Do not describe the earlier stdout snapshot as the final log checksum. No scientific record was rewritten here.

## Independent artifact checks

The verification uses no model execution: it hashes opaque checkpoint bytes, verifies the corrected manifest, checks the 116 saved prediction rows and their probability constraints, reconstructs the confusion matrix, averages saved per-image AOPCs (including the within-image 20-control average), and checks saved fixed-target metadata. All 41 protected artifact hashes match the existing baseline. Detailed expected/actual hashes and preservation checks are in `verification.json`.

No raw dataset images or locked TEST records were decoded or inspected. Historical TEST discussion is limited to already published aggregate documentation. No tests or scientific evaluations were run during report preparation.
