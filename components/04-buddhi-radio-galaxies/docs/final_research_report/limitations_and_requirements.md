# Limitations and unresolved requirements

## Scientific limitations that must remain in the final report

1. **Development validation, not independent generalization.** The same validation cohort supported checkpoint selection and subsequent analysis. Corrected TEST and external-survey performance are unknown. Historical model metrics cannot be transferred to the corrected checkpoint.
2. **One corrected training seed.** Bootstrap intervals across available images do not quantify variation across independently retrained models or surveys. No superiority claim is supported by comparing changed historical/corrected cohorts.
3. **Integrity correction has a defined scope.** The quarantine resolves the confirmed five-record exact-content group within audited TRAIN+VALIDATION membership. It does not explain conflicting-label origins, verify all other labels, detect every near duplicate, establish source independence or make a claim about TEST cleanliness.
4. **Grad-CAM is coarse and conditional.** A 19×19 map upsampled to 150×150 is not an independent high-resolution annotation. Target/layer choice and interpolation matter. Visual appeal is not faithfulness, causality or astronomical correctness.
5. **Deletion is an intervention on image intensities.** Border-median replacement can create OOD inputs. AOPC is policy- and score-dependent; negative values remain valid observations. The corrected study lacks a validated OOD diagnostic for these perturbations.
6. **Transformation behaviour is weak in important cases.** Prediction flips, substantial clipping and degenerate CAMs are retained. Availability differs across conditions; stronger-noise available-case means cannot be interpreted as improved robustness. Intensity perturbations are not calibrated radio-flux experiments.
7. **No expert physics annotations.** Intensity masks are proxies, not verified cores, jets, lobes or hotspots. No annotation-based physics Jaccard or direct black-hole detection is claimed. This report did not inspect new image pixels for qualitative conclusions.
8. **Partial broader XAI coverage.** Corrected original explanations and executed primary analyses are complete for 116 records; secondary full curves/AUC, replacement sensitivities and parameter-randomization execution are deferred. Implementation tests do not fill this empirical gap.
9. **Four undefined auxiliary analyses remain visible.** Secondary 19×19 stability (`native_grid_pooling_partial_support_ties_unspecified`); random emission baseline (`random_native_to_input_projection_unspecified`); sanity brightness control (`sanity_brightness_definition_unspecified`); sanity random control (`sanity_random_distribution_grid_seeds_replicates_projection_unspecified`). They are unavailable, not zero or successful.
10. **Live public inference unavailable.** Compatible corrected-model calibration and independent OOD evaluation/acceptance evidence are absent. Softmax confidence is not established as calibrated correctness probability. The results-only dashboard must retain its gate.
11. **Provenance and platform limits.** Current checked execution hashes match, but corrected work includes uncommitted files. HEAD alone is insufficient as a complete experiment identity. Historical TEST-run count and original checkpoint bytes at evaluation time are incompletely proven. The documented Step 21 operational-log snapshot issue is not concealed. CPU telemetry and cooperative checks are not universal resource guarantees.
12. **Publication status.** Public deployment was not verified. Local-source-image redistribution rights and final code/data attribution must be established; no model weights, dataset images or private artifacts should be released on the strength of this report.

## Requirements before university submission

- Obtain the official approved proposal, objectives and university report template. Reconcile the provisional aims with approved wording and identify deviations honestly.
- Confirm author identity, individual contributions, supervisor details, required declarations, ethics/data-use wording, citation style, word count and required front matter. No approval or ethics clearance is inferred.
- Complete a verified review of directly relevant radio-CNN literature and reconcile the proposal bibliography. Inspect full texts before making detailed comparative claims.
- Verify MiraBest release creator/date/license/attribution metadata and permissions for representative source images. Dataset public availability alone is not redistribution permission.
- Have the researcher review every numerical and qualitative interpretation, figure caption and scientific limitation. No claims of named physical structures are inserted without evidence.

## Future research requiring separate authorization

Independent source-level assessment, expert masks, external evaluation, any locked-TEST decision, corrected OOD calibration, additional training and deferred XAI experiments require explicit new authorization and appropriate predefined methods. These are recommendations, not automatically approved next actions. Essential report writing can proceed now with the limitations above; no extra experiment is required merely to produce the draft.

## Unresolved questions

What caused identical pixels with conflicting labels? Are distinct filenames repeated observations of the same astronomical source? How do results vary with a fresh training seed or survey? Do explanations depend adequately on learned parameters for the corrected checkpoint? How much instability is attributable to clipping versus other model/input properties? Which regions would independent astronomers annotate? Which OOD data and acceptance criteria are suitable for release? Existing evidence does not resolve these questions, and this report does not invent their answers.
