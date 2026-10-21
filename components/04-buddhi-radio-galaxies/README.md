# Component 4 — RadioMorph AI: Explainable Radio Galaxy Morphology Classification

**Responsible member:** Buddhi Gayashan  
**Group:** J26-IT-449 — Explainable AI for Astronomical Discovery using XAI and Uncertainty Quantification

This component is the radio galaxy morphology analysis contribution to the shared research repository. The original group README reserved this directory as a documentation placeholder. Selected implementation and evidence have now been migrated here; the original individual research project and backup remain separate and unchanged. Group integration remains subject to review. No other member's component is changed.

## Overview and objectives

RadioMorph AI studies binary Fanaroff–Riley morphology classification (FR-I=0, FR-II=1), genuine gradient-based explanations and reproducible research safeguards. Its implemented aims are to maintain a corrected development cohort, train a lightweight residual classifier, assess fixed-checkpoint classification and explanation behaviour, and communicate limitations through a local Gradio research dashboard. The official approved proposal was not found in the inspected individual repository; these are implemented aims, not an invented approval statement.

This is a **local research component**, not a public deployment. No deployment is authorized or performed. Live upload prediction remains disabled pending validated corrected-model OOD evidence. No alternative inference path bypasses that gate.

## Dataset and corrected preparation

The local MiraBest Zenodo v1 binary payload is identified by [record 4288837](https://zenodo.org/records/4288837). Raw data are not included. The separately versioned **MiraBest-Development-ExactDedup-v1.0.0** protocol quarantines global indices **198, 323, 474, 587, 694**, which formed one identical-content group with conflicting labels and historical TRAIN–VALIDATION overlap. No record is automatically relabelled.

| Corrected cohort | Total | FR-I | FR-II |
|---|---:|---:|---:|
| TRAIN | 552 | 245 | 307 |
| VALIDATION | 116 | 51 | 65 |

Retained membership order and labels are preserved. [Corrected metadata](data/manifests/mirabest_development_exactdedup_v1.0.0/README.md) and saved development-only duplicate-audit evidence are included; raw pixels and the locked TEST split are not. Exact-image disjointness within the audited development cohort does not establish astronomical source independence or near-duplicate absence. Do not regenerate historical splits.

Canonical manifest SHA-256:
`4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91`

Archive provenance: SHA-256 `505e0b0761a76364a417695380d32d9f26cca7bc0a2608cad46ab338fa46a6b8`; MD5 `fa9d6ad8fdde95b0a19df11433e8bca3`. These are recorded identities, not permission to redistribute data or access TEST.

## CNN architecture and training methodology

`src/models/resnet_cnn.py` implements a custom lightweight 339,746-parameter residual CNN: a 32-channel stem, three residual blocks with 32/64/128 channels, adaptive 2×2 pooling and a 512→64→2 classifier with dropout 0.30. This is not a pretrained ImageNet ResNet-18.

The corrected run used fresh random initialization, seed 42, Adam, learning rate 0.0003, weight decay 0.0001, batch size 32, cross-entropy loss, maximum 30 epochs, early-stopping patience 7 and ReduceLROnPlateau (patience 2, factor 0.5). Minimum VALIDATION loss selected epoch 30, loss 0.3447643220. Settings are in `config/corrected_resnet_training_v1.0.0.yaml`; root `config.yaml` also retains historical configuration and attribution, not an alternative corrected training specification.

Inputs are grayscale 150×150 float tensors, normalized with mean/std 0.5. Training retains ±15° random rotation and horizontal/vertical flips; validation is deterministic without augmentation. The corrected adapter reads at most eight selected raw images per call while assembling tensor batches. It never requires bulk materialization of mixed-split image arrays.

## Grad-CAM and evaluation

`src/xai/gradcam.py` uses pre-softmax class-logit gradients, a temporary activation hook and `torch.autograd.grad()` at **`model.blocks[2].convolution[3]`**. It produces native 19×19 nonnegative CAMs and aligned 150×150 display heatmaps, without updating model parameters or BatchNorm buffers. Degenerate explanations are explicit, not fabricated.

The corrected protocol explains the original predicted class. Primary probability AOPC averages score decreases at exact 5%,10%,15%,20%,25%,30% deletion budgets with border-median replacement. Native tile rankings are compared against brightness and exactly twenty seeded random controls per image. Negative values are retained. Stability inverse-aligns continuous maps before support-aware top-k Jaccard. Intensity-derived emission masks are proxies, not expert anatomical labels. See [corrected XAI protocol](analysis_reports/step21_corrected_gradcam_evaluation_attempt02/corrected_xai_protocol.md).

The historical `Phase3B-XAI-v1.0.0` protocol binds the old checkpoint and 119-image cohort. It remains distinct from **Corrected-GradCAM-Validation-v1.0.0** and its 116 images. Frozen `execution_authorized: false` values remain unchanged. Old bounded approvals do not authorize new experiments.

## Historical recorded research results

These are previously completed results, **not new evaluations executed during migration**. Saved Step 20 CSV evidence and original Step 21 per-image records were checked arithmetically during migration; see [verification](research_evidence/result_verification.json).

| Quantity | Recorded result |
|---|---:|
| Corrected validation images | 116 |
| Accuracy | 87.068966% (101/116) |
| Macro F1 | 0.864391 |
| Confusion matrix (true rows / predicted columns: FR-I, FR-II) | [[38,13],[2,63]] |
| Genuine original Grad-CAM explanations | 116 |
| Mean Grad-CAM probability AOPC30 | 0.294510 |
| Mean random-control AOPC30 | 0.053960 |
| Mean brightness-control AOPC30 | 0.222254 |

Evidence: [classification metrics](analysis_reports/step20_corrected_validation_classification_attempt02/metrics.json), [prediction CSV](analysis_reports/step20_corrected_validation_classification_attempt02/validation_predictions.csv), [XAI aggregate summary](analysis_reports/step21_corrected_gradcam_evaluation_attempt02/quantitative_summary.json), [interpretation](analysis_reports/step21_corrected_gradcam_evaluation_attempt02/scientific_interpretation.md), [training history](research_evidence/training/resnet_training_history.csv).

Corrected checkpoint (not distributed):
`outputs/models/mirabest_development_exactdedup_v1.0.0/corrected_seed42_01/models/resnet_cnn_best.pth`

Checkpoint SHA-256:
`d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c`

## OOD and uncertainty boundaries

Energy and feature-distance rejection, input guards, calibration provenance and fail-closed inference are implemented. Historical calibration is tied to a different checkpoint and cannot be reused. Corrected-model calibration and independent OOD acceptance evidence are unavailable; no ordinary softmax score or temperature scaling is claimed to solve that gap. Softmax outputs are model confidence, not established calibrated correctness probabilities. This component does not claim comprehensive uncertainty quantification or external-survey generalization.

## Local setup and installation

Use Python 3.12 and run commands **from this component directory**, not the group repository root. Scripts derive their component root from their own location; `src` is a component-local Python package, not the shared integration API. Avoid mixing multiple components' `src` packages in one process.

```powershell
cd components/04-buddhi-radio-galaxies
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For POSIX systems use `python3.12 -m venv .venv` and `.venv/bin/python` in place of the Windows interpreter. No dependency installation or upgrade occurred during migration. Research requirements retain their original broad constraints; the installed verification versions are recorded under `research_evidence/migration_checks/verification.json`. That record is not a fully locked transitive environment or a CUDA compatibility guarantee.

## Automated checks

```powershell
.venv\Scripts\python.exe -B -m pytest -q -ra -p no:cacheprovider
```

`tests/conftest.py` explicitly blocks real-data tests and the excluded private radio-image gallery. Skips are unavailable integration checks, not passes. All other assertions retain their scientific definitions. Tests use synthetic fixtures and temporary disposable artifacts; no dataset or checkpoint download occurs. Historical and newly executed test outcomes are kept separate in the migration report.

For focused XAI checks:

```powershell
.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider tests/test_gradcam.py tests/test_xai_faithfulness.py tests/test_xai_stability.py tests/test_xai_morphology_audit.py tests/test_xai_sanity_checks.py tests/test_xai_statistics.py
```

## Local dashboard

```powershell
$env:GRADIO_ANALYTICS_ENABLED = "False"
.venv\Scripts\python.exe -B demo/app.py
```

Open the local loopback URL printed by Gradio (normally `http://127.0.0.1:7860`). Do not use sharing/tunnels or deploy this repository. The default presentation displays saved corrected-model metrics, aggregate confusion matrix and limitations; live uploads remain disabled. Individual source-image examples are not distributed because redistribution clearance is unresolved. `public_demo/` and its builder are retained historical source/test dependencies, not deployment instructions or authorization for this group migration.

## Component structure

```text
04-buddhi-radio-galaxies/
  README.md, requirements.txt, config.yaml, pytest.ini
  config/               frozen historical/corrected settings
  src/                  data, CNN, inference, OOD and XAI modules
  scripts/              controlled research entry points (not executed by migration)
  tests/                synthetic tests and explicit availability boundaries
  demo/                 local results-only Gradio interface and aggregate assets
  docs/                 protocols, registry and provisional academic report
  data/manifests/       corrected membership/provenance metadata only
  analysis_reports/     selected lightweight historical evidence at original paths
  research_evidence/    migration inventory, checks, relocated training summaries
  public_demo/          retained legacy source/test dependency; no deployment
```

## Scientific limitations and reproducibility

Validation was used for checkpoint selection and is not an independent TEST result. One corrected seed was trained. Exact deduplication does not verify source independence. Stability experiments include noise-induced prediction flips, approximately 49% clipping, and degenerate transformed CAMs. Eight negative CAM AOPCs are preserved. Emission-proxy overlap is not verified jet/lobe/core localization or physical causality. Corrected parameter-randomization execution and expensive secondary deletion sensitivities remain deferred. Four underspecified secondary controls remain explicitly unavailable. Expert-mask physics alignment, ECE with frozen bins, corrected OOD validation and external generalization are not established.

Original archives, TEST data, raw image exports, checkpoints, intermediate heatmaps, full run records/receipts and machine-local logs remain with the researcher. Their hashes and missing-evidence boundaries are documented; source availability alone does not make real experiments reproducible without those authorized artifacts. Do not weaken provenance checks to run with substitutes. Obtain the exact artifacts and a new approval before any real experiment.

See [migration inventory and verification report](docs/migration_report.md), [machine-readable inventory](research_evidence/migration_inventory.json), [final academic draft](docs/final_research_report/README.md) and [known limitations](docs/final_research_report/limitations_and_requirements.md). The academic draft remains provisional until the official proposal, template, bibliography and permissions are supplied. Historic absolute paths in evidence are archival descriptions, not executable configuration paths. The group root's placeholder summary remains a historical setup document; this component README records the migration status without changing shared files.
