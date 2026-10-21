# Step 23 ? Corrected-model OOD calibration feasibility

**Verdict: BLOCKED ? CALIBRATION EVIDENCE REQUIRED.**

## Exact original blocker

The dashboard requires outputs/ood/mirabest_development_exactdedup_v1.0.0/ood_calibration.pt.
It does not exist. src/inference/corrected_demo.py requires matching corrected
checkpoint identity, checked manifest, exact corrected TRAIN/VALIDATION memberships,
feature statistics, thresholds and preprocessing/split provenance before allowing
an uploaded-image prediction or Grad-CAM. Historical fallback is deliberately absent.

Corrected checkpoint SHA-256: d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c

Corrected manifest SHA-256: 4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91

Both match approval. Existing outputs/ood/ood_calibration.json and
external_compatibility_calibration.json bind historical checkpoint
d5e1f126c13c2b7bebb95e35f41f0bc6397dd0ea9f4fbfcfbc55648609ab5bc7, not the corrected checkpoint. Neither may be
renamed or copied to enable the new model. No incompatible artifact was loaded.

## Data feasibility and provenance

Authorized corrected TRAIN 552 and VALIDATION 116 exist for ID statistics and
threshold fitting. Neither cohort is OOD; its labels must not be changed to OOD.
The calibration folder data/ood/calibration/ contains only .gitkeep. Seven external
image files exist under data/ood/test/: building_01.jpg, car_01.jpg, dog_01.jpg,
landscape_01.jpg, optical_astronomy_01.jpg, person_01.jpg, screenshot_01.png. They
were inspected as filename/size metadata only; no image bytes decoded or scored.
Their names alone are not verified provenance, independent domain labels, source
independence or an untouched evaluation partition.

Existing outputs/ood/ood_evaluation.json and docs/research_execution_registry.md
already record seven external examples in historical-model evaluation. That is
prior evaluation exposure, not corrected-model calibration evidence or an
independently reserved release package. Historical records are not automatically
invalid for every future descriptive use, but cannot silently be relabeled as a
new untouched calibration/evaluation split. The registry records incomplete external
selection history/provenance. No new independent split or release criteria were
found in the relevant data/config/calibration/gate sources inspected here.

Synthetic files in data/ood/sanity/ (noise, uniform, gradient, checkerboard) are
engineering checks, not genuine astronomical OOD evidence. External compatibility
calibration is explicitly historical-model synthetic positive-only calibration;
it does not establish natural-image false acceptance or independent survey accuracy.
No OOD data, labels, independence claims or sample counts were invented.

## Established reusable method versus missing evidence

src/ood/ood_detector.py defines energy E=-logsumexp(z) and a diagonal feature
distance sqrt(sum((f-mu)^2/variance)), using 512 pooled features. Higher scores are
more OOD-like. Fit deterministic ID feature mean/variance on TRAIN only, variance
floor 1e-6; choose each threshold from the 0.99 quantile of VALIDATION ID scores.
Accept only when BOTH checks pass. Preprocessing is trusted grayscale 150x150,
float32 ToTensor, mean/std 0.5, no augmentation or public-upload rejection in ID fitting.

This ID-only quantile procedure does not mathematically require OOD examples to
compute thresholds. However, computing such an artifact alone cannot establish
OOD rejection, false acceptance, AUROC/FPR95, or satisfy this task's release-evidence
condition. A 99th-percentile selection rule is not a predeclared independently
verified deployment acceptance criterion. Fitting thresholds on VALIDATION and
reporting acceptance on those same images would be in-sample, not an independent
ID evaluation; no TEST holdout may be substituted. Softmax confidence or temperature
scaling is not a replacement OOD detector.

No numerical release thresholds, minimum AUROC, maximum false-accept/reject rate,
or approved independent calibration/evaluation partition was found. None was
invented after looking at scores. All new corrected calibration metrics are
**NOT AVAILABLE / null** because no calibration or evaluation was executed.
Historical score/performance numbers are not transferred to the corrected model.

## Compatible implementation work still needed if evidence is supplied

The mathematical detector is reusable. scripts/calibrate_ood_detector.py is NOT a
safe corrected-cohort command as-is: it points at historical checkpoint/splits,
loads the mixed-split dataset eagerly and parses the original split including
TEST indices. src/ood/calibration.py expects top-level train/validation/test split
lists and validates all three; it cannot directly validate the nested, development-
only corrected manifest. The future corrected inference wrapper also delegates
to that validator before its additional membership checks. Therefore dropping in
an arbitrary new PT file is not sufficient even if its checkpoint hash matches.

A minimal separate corrected-manifest calibration/validation adapter must use the
existing bounded selected-row reader, pin the corrected checkpoint/manifest, fit
only allowed TRAIN/VALIDATION ID data, and avoid TEST parsing/materialization.
Preserve historical serializers/artifacts. This is identified required implementation,
not a reason to weaken provenance checks. No code change was made now because the
conditional data/evaluation prerequisites failed before experiment execution.

## Frozen procedure prerequisites and actual execution

If the researcher supplies a qualifying package, record its identities/hashes and
separate threshold-selection from independent evaluation before scoring. Retain
the existing two score directions, 99th-percentile rule and AND acceptance; report
ID acceptance/false rejection, genuine OOD rejection/false acceptance, separate
energy/distance AUROC and FPR at the predefined TPR (existing 95% convention),
with counts and unavailable cases. Acceptance criteria must be fixed before
examining evaluation outcomes. Never iterate on the same holdout until it passes.
Finite OOD samples cannot guarantee rejection of every unknown image.

For this task: **calibration did not run**, no new model/data inference was performed,
no genuine research images were decoded, and no calibration artifacts or thresholds
were created. Public uploads remain disabled/fail closed. Prediction plus Grad-CAM
on genuinely accepted public images is **not verified** because the required gate
cannot be satisfied. This is an evidence blocker, not a claim that the model or
already-tested Grad-CAM implementation is broken.

## Minimal functional verification

34 focused corrected-demo/dashboard tests passed in 6.33 seconds: 0 failed, skipped
or errors. Exact command is in focused_test_results.json. Fixtures are synthetic,
cover safe loading/hash checks, calibration identity/membership checks, class
mapping, guard rejection, finite outputs, public fail-closed behavior and UI wiring.
Genuine OOD rejection was not tested: no qualifying authorized evaluation package
was available. Prior Step22 real-checkpoint/synthetic-Grad-CAM component evidence
was preserved and not mistaken for a calibrated public-path acceptance result.

A real loopback Gradio smoke test served HTTP 200 at port 59092. Its unmocked
production callback received only a synthetic grayscale ramp and returned
UNSUPPORTED, confidence absent, no class probabilities/heatmaps, and the explicit
corrected-calibration-unavailable reason. The server was closed afterwards. This
verifies startup and graceful rejection, not accepted-image end-to-end prediction.
No historical gate bypass, calibration fabrication or dummy live prediction occurred.

## Files and launch

Only analysis_reports/step23_corrected_ood_calibration/ was created, containing:
baseline.json, local_dashboard_smoke.py, dashboard_smoke.json,
calibration_feasibility.json, focused_test_results.json, feasibility_report.md,
and integrity_verification.json. No implementation/source/dashboard file changed.
No new calibration artifact or metrics masquerading as measured results exists.

Launch from repository root:

```powershell
.venv\Scripts\python.exe -B demo/app.py
```

Default address http://127.0.0.1:7860 (or next available Gradio port). The UI loads,
but accepted corrected-model upload prediction/Grad-CAM remains unavailable.

## One next researcher action

Supply one researcher-approved evaluation package with provenance/hash manifests, independently reserved ID and genuine OOD evaluation memberships disjoint from threshold selection, and predefined numerical dashboard-release acceptance criteria.

This requests one fixed package/decision, not permission to repeatedly tune on
existing examples or access TEST. No arbitrary required sample count is invented.

## Integrity and scientific limitations

All 41 protected artifact hashes match. Historical/corrected checkpoints, original
and corrected manifests, Step18?22 results, frozen protocols, dashboard and unrelated
working-tree files remain byte-identical to the task snapshot. Git branch
feature/external-radio-input, HEAD 2d6e5883e4a6a16252efdb24e5e4f52f61aca026 and index/status remain unchanged.
No staging, commit, push, reset, clean, stash, branch switch, retraining, classification
rerun, XAI evaluation, TEST sample access, or unrelated audit occurred. Saved historical
aggregate reports were read, not TEST records/labels/pixels. Protected-file hashing is
opaque integrity checking, not image decoding. The existing .pytest_cache warning
from git status is unrelated to dataset integrity.

Source references: src/inference/corrected_demo.py; src/ood/ood_detector.py;
src/ood/calibration.py; scripts/calibrate_ood_detector.py;
scripts/run_ood_evaluation.py (inspected only, never executed);
outputs/ood/ood_calibration.json; outputs/ood/external_compatibility_calibration.json;
outputs/ood/ood_evaluation.json (historical aggregate only);
docs/research_execution_registry.md OOD-01/TEST-OOD-01;
analysis_reports/step22_final_ui_integration/integration_report.md.
