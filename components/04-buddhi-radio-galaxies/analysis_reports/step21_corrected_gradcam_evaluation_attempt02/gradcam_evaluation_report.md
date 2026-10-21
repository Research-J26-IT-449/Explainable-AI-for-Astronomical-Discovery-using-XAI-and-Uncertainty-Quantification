# Corrected-model Grad-CAM evaluation report

**Verdict: XAI PARTIALLY COMPLETED — LIMITATIONS DOCUMENTED.**
All essential endpoints of Corrected-GradCAM-Validation-v1.0.0 completed for the full 116-image corrected validation cohort. This does not complete every historical Phase 3B secondary analysis.

## Identity, authorization and verification

Protocol: Corrected-GradCAM-Validation-v1.0.0. Dataset protocol: MiraBest-Development-ExactDedup-v1.0.0.
Checkpoint: `outputs/models/mirabest_development_exactdedup_v1.0.0/corrected_seed42_01/models/resnet_cnn_best.pth`; SHA-256 `d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c`.
Manifest: `data/manifests/mirabest_development_exactdedup_v1.0.0/manifest.json`; SHA-256 `4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91`.
The separately versioned protocol changes model/cohort identity only where explicitly approved, and documents essential/secondary scope before execution. Historical protocol/YAML and artifacts were never edited. Applicable mathematical modules were reused unchanged. Historical tie-breaking namespaces remain explicit mathematical conventions, not misrepresented execution provenance.

Command (repository root):
```powershell
.venv\Scripts\python.exe -B analysis_reports/step21_corrected_gradcam_evaluation_attempt02/evaluate_corrected_xai.py --execute
```

Preflight passed. Synthetic integration checks passed without real data/model forward calls; 188 focused Grad-CAM/faithfulness/stability/morphology tests passed (0 failed, 6.02 seconds). Safe weights_only CPU checkpoint load; strict ResNetCNN architecture; eval mode; parameters frozen; original validation preprocessing unchanged. Exactly 116 validation rows decoded, one at a time (maximum one selected raw image per call, within the eight-image bound). No TRAIN/TEST/quarantined pixels decoded; archived label values skipped; labels came from the corrected manifest.

## Genuine explanation coverage

116/116 unique records completed; class supports 51 FR-I/65 FR-II, 101 correct/15 incorrect. Original CAM statuses: {'valid': 116}. All predictions matched Step 20; maximum probability difference 1.1920929e-07 (tolerance 1e-5 for float32 single-image versus batch computation). No separate classification pass, model selection, retraining or tuning occurred.
Target model.blocks[2].convolution[3]; original predicted-class pre-softmax logit; temporary activation hook and autograd.grad. Native nonnegative CAM 19x19; bilinear 150x150, align_corners=False; original geometry and normalization preserved. All weights/BatchNorm buffers stayed exactly equal after every image; parameter gradients remained absent. All three saved map arrays were rechecked for shapes, finiteness and valid ranges.

## Primary probability AOPC30

Border-median replacement; native tile rankings fixed from original input; exact nested 5/10/15/20/25/30% budgets; exactly 20 seeded random controls (1000–1019), averaged per image. AOPC is mean original-minus-deleted fixed-class probability; logit/margin curves are also saved. Negative values are retained. Deletion inputs may be out of distribution; no historical OOD calibration was loaded and no perturbed score was filtered as publicly supported.

| Endpoint | Available / 116 | Mean | Median | FR-I mean | FR-II mean | Correct mean | Incorrect mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| cam_probability_aopc30 | 116 | 0.294510 | 0.244269 | 0.430278 | 0.187986 | 0.329636 | 0.058000 |
| random_probability_aopc30 | 116 | 0.053960 | 0.044879 | 0.049733 | 0.057277 | 0.055150 | 0.045947 |
| brightness_probability_aopc30 | 116 | 0.222254 | 0.160654 | 0.418114 | 0.068580 | 0.259660 | -0.029608 |
| CAM_minus_random_probability_AOPC30 | 116 | 0.240550 | 0.164779 | 0.380545 | 0.130708 | 0.274486 | 0.012053 |
| CAM_minus_brightness_probability_AOPC30 | 116 | 0.072256 | 0.071735 | 0.012163 | 0.119406 | 0.069976 | 0.087608 |

Negative CAM probability AOPC observations: 8. Zero/negative deletion effects are not clipped or omitted. Positive paired contrasts favor CAM sensitivity over that control; negative contrasts do not. Neither direction proves astronomical correctness.

Paired class-stratified image bootstrap: 10,000 replicates, seed42, 95% percentile intervals. These are exploratory, conditional on metric availability and source-independence assumptions. No confirmatory p-values or significance claims were introduced.

- CAM_minus_brightness_probability_AOPC30: overall estimate [0.07225604868126274], overall 95% interval [[0.05130564737893338, 0.09340115768953358]]; equal-class macro estimate [0.06578453364774105], macro interval [[0.044871146645051596, 0.08673768405154601]]; coverage {'class_counts': [51, 65], 'conditional_on_metric_availability': False, 'excluded': [], 'image_ids': [31, 35, 37, 38, 50, 51, 52, 53, 55, 60, 68, 73, 94, 105, 112, 117, 136, 142, 144, 147, 157, 161, 162, 163, 165, 166, 168, 170, 171, 176, 181, 209, 215, 221, 227, 237, 255, 256, 266, 287, 288, 289, 291, 299, 301, 303, 304, 306, 318, 326, 334, 355, 366, 371, 372, 381, 390, 393, 404, 406, 413, 414, 419, 428, 431, 437, 440, 445, 447, 453, 463, 469, 470, 473, 490, 500, 501, 507, 509, 533, 542, 545, 549, 550, 568, 574, 597, 603, 606, 617, 618, 620, 623, 627, 633, 637, 638, 640, 641, 646, 652, 656, 668, 669, 671, 690, 693, 695, 706, 714, 748, 749, 754, 763, 767, 770], 'reasons': [], 'status': 'available', 'warnings': []}.
- CAM_minus_random_probability_AOPC30: overall estimate [0.2405503520305554], overall 95% interval [[0.1962033404170533, 0.28231970005473195]]; equal-class macro estimate [0.2556266901584075], macro interval [[0.20670978112948826, 0.3009331673413194]]; coverage {'class_counts': [51, 65], 'conditional_on_metric_availability': False, 'excluded': [], 'image_ids': [31, 35, 37, 38, 50, 51, 52, 53, 55, 60, 68, 73, 94, 105, 112, 117, 136, 142, 144, 147, 157, 161, 162, 163, 165, 166, 168, 170, 171, 176, 181, 209, 215, 221, 227, 237, 255, 256, 266, 287, 288, 289, 291, 299, 301, 303, 304, 306, 318, 326, 334, 355, 366, 371, 372, 381, 390, 393, 404, 406, 413, 414, 419, 428, 431, 437, 440, 445, 447, 453, 463, 469, 470, 473, 490, 500, 501, 507, 509, 533, 542, 545, 549, 550, 568, 574, 597, 603, 606, 617, 618, 620, 623, 627, 633, 637, 638, 640, 641, 646, 652, 656, 668, 669, 671, 690, 693, 695, 706, 714, 748, 749, 754, 763, 767, 770], 'reasons': [], 'status': 'available', 'warnings': []}.

## Stability and robustness

Thirty independent predefined transformations per image, including 20 noise realizations. Fixed original class is explained even when prediction flips. Continuous maps inverse-aligned before top-k selection in common valid support; support <80% unavailable. Primary top10% Jaccard and 5%/20% sensitivities use existing tie policy. This Jaccard is explanation stability, not expert-reference astronomical localization. Noise repeats are averaged within image/level before statistical summaries; comparisons below are diagnostic counts, not independent images.

| Condition | Comparisons | Prediction flips | Unavailable comparisons | Mean clipped fraction |
|---|---:|---:|---:|---:|
| rotation -5 | 116 | 1 | 0 | 0.000000 |
| rotation -2 | 116 | 1 | 0 | 0.000000 |
| rotation 2 | 116 | 1 | 0 | 0.000000 |
| rotation 5 | 116 | 4 | 0 | 0.000000 |
| rotation -15 | 116 | 3 | 0 | 0.000000 |
| rotation 15 | 116 | 7 | 0 | 0.000000 |
| horizontal_flip None | 116 | 8 | 0 | 0.000000 |
| vertical_flip None | 116 | 8 | 0 | 0.000000 |
| noise 0.005 | 1160 | 760 | 459 | 0.491878 |
| noise 0.01 | 1160 | 760 | 722 | 0.492186 |
| brightness 0.98 | 116 | 0 | 0 | 0.000000 |
| brightness 1.02 | 116 | 0 | 0 | 0.000051 |

Image-level primary top10% Jaccard summaries:

| Endpoint | Available images | Mean | FR-I mean | FR-II mean | Correct mean | Incorrect mean |
|---|---:|---:|---:|---:|---:|---:|
| stability.brightness.0.98.top0.1 | 116 | 0.990592 | 0.989691 | 0.991299 | 0.991452 | 0.984802 |
| stability.brightness.1.02.top0.1 | 116 | 0.990660 | 0.990377 | 0.990882 | 0.991452 | 0.985330 |
| stability.horizontal_flip.None.top0.1 | 116 | 0.676351 | 0.683305 | 0.670895 | 0.687113 | 0.603890 |
| stability.noise.0.005.top0.1 | 78 | 0.452004 | 0.640174 | 0.253930 | 0.450975 | 0.471041 |
| stability.noise.0.01.top0.1 | 46 | 0.567976 | 0.633129 | 0.258498 | 0.561409 | 0.712444 |
| stability.rotation.-15.top0.1 | 116 | 0.762108 | 0.753756 | 0.768661 | 0.776124 | 0.667734 |
| stability.rotation.-2.top0.1 | 116 | 0.891493 | 0.864023 | 0.913045 | 0.894718 | 0.869772 |
| stability.rotation.-5.top0.1 | 116 | 0.841818 | 0.818320 | 0.860255 | 0.852943 | 0.766909 |
| stability.rotation.15.top0.1 | 116 | 0.749782 | 0.727199 | 0.767502 | 0.763204 | 0.659413 |
| stability.rotation.2.top0.1 | 116 | 0.882067 | 0.842826 | 0.912855 | 0.890683 | 0.824049 |
| stability.rotation.5.top0.1 | 116 | 0.835986 | 0.810341 | 0.856107 | 0.844244 | 0.780377 |
| stability.vertical_flip.None.top0.1 | 116 | 0.734696 | 0.656714 | 0.795882 | 0.733916 | 0.739948 |

All/correct/incorrect/class stability counts and reason codes: {'all': {'comparisons': 3480, 'flips': 1553, 'images': 116, 'unavailable': {'transformed_degenerate_cam': 1181}}, 'class0': {'comparisons': 1530, 'flips': 278, 'images': 51, 'unavailable': {'transformed_degenerate_cam': 241}}, 'class1': {'comparisons': 1950, 'flips': 1275, 'images': 65, 'unavailable': {'transformed_degenerate_cam': 940}}, 'correct': {'comparisons': 3030, 'flips': 1287, 'images': 101, 'unavailable': {'transformed_degenerate_cam': 940}}, 'incorrect': {'comparisons': 450, 'flips': 266, 'images': 15, 'unavailable': {'transformed_degenerate_cam': 241}}}. Raw records retain preserving/flipped status and Jaccards separately; no flips or degeneracies were silently dropped. Noise clipping is not a physically calibrated radio-flux perturbation. Stability alone does not establish informative or faithful attribution.

## Intensity-proxy overlap and physics limitations

Existing median/MAD/quantization-floor masks were independently built from image intensity (not CAMs), with inclusive/primary/restrictive thresholds and brightness/uniform baselines. These are processed-intensity proxies, not independent validated cores, jets, lobes or hotspots. Annotation-based physics Jaccard is NOT AVAILABLE. Proxy IoU must not be relabeled as such.

| Primary proxy metric | Available | Mean | FR-I mean | FR-II mean |
|---|---:|---:|---:|---:|
| emission_attribution_mass | 116 | 0.090169 | 0.158507 | 0.036550 |
| area_adjusted_enrichment | 116 | 0.078583 | 0.144299 | 0.027021 |
| top10_emission_iou | 116 | 0.074599 | 0.099598 | 0.054985 |

## Deterministic representative figures

Lowest global saved Step20 index per ground-truth/correctness stratum, frozen before CAM generation. No selection by attractive heatmap, confidence or metric result.

- class0_correct: index 35; representatives/index_35.png.
- class0_incorrect: index 31; representatives/index_31.png.
- class1_correct: index 50; representatives/index_50.png.
- class1_incorrect: index 437; representatives/index_437.png.

Figures show original fixed-scale intensity, normalized CAM and alpha .45 overlay, and are labeled exploratory. Core/jet/lobe interpretations require independent morphology annotations and imaging context. Intended language: radio emission structures produced by AGN activity, indirectly linked to supermassive black holes. This is not black-hole detection. Qualitative observations after figure inspection are recorded separately; no expert localization or causal interpretation is inferred from the numbers.

## Deferred/unavailable analyses

- annotation_physics_jaccard: independent_validated_region_annotations_not_available.
- confirmatory_tests: corrected_development_analysis_exploratory_only.
- corrected_ood_diagnostics: historical_calibration_not_validated_for_corrected_checkpoint.
- ece: no_frozen_confidence_bin_definition.
- full_deletion_auc_and_replacement_sensitivities: deferred_secondary_tenfold_deletion_workload.
- parameter_randomization: deferred_secondary_3480_additional_cam_calls.
- random_emission_baseline: random_native_to_input_projection_unspecified.
- sanity_brightness_map: sanity_brightness_definition_unspecified.
- sanity_random_map: sanity_random_distribution_grid_seeds_replicates_projection_unspecified.
- secondary_stability_19x19: native_grid_pooling_partial_support_ties_unspecified.

Full 21-point/AUC and replacement sensitivities are compatible but deferred secondary resource scope (ten-fold deleted-input workload); parameter randomization would add 3480 gradient explanations. Neither was performed. This is not an assertion of mathematical incompatibility or completion. Four inherited convention gaps remain unresolved. ECE bins were not invented; corrected OOD diagnostics were not replaced with historical calibration.

## Runtime, integrity and limitations

Process wall time 329.909s (5.50min), from 2026-10-21T05:30:58.357831+00:00 to 2026-10-21T05:36:28.280854+00:00; exit code 0. Per-image median 2.712s, range 1.957–3.841s. Maximum process-lifetime peak working set 427483136 bytes (0.398GiB); maximum sampled working set 424407040 bytes. Family totals (exclusive seconds): {'emission_proxy': 17.441934799997398, 'original_gradcam': 29.783613500000683, 'primary_aopc': 102.171974400002, 'representative_export': 0.8367881000003763, 'stability': 164.18027880000045}. Setup, aggregation and uninstrumented overhead are not included in these family sums. 2GiB/900s cooperative checks passed; blocking operations cannot be interrupted immediately. Operational telemetry is separate from deterministic scientific records.

All 41 protected artifacts, existing snapshotted files, checkpoints, manifests, historical protocols, Step18–20 results and Git index/HEAD unchanged. No TRAIN/TEST image access, training, new classification evaluation, UI edit or staging/commit/push. Provenance reads include opaque serialized archive/split hashes, not TEST label/pixel interpretation.

Completion snapshot checksum differences: {'execution_stdout.log': {'snapshot': 'dcc2ebab91e42716fae2042735faa638e6f545804998a928aad7d85e72462a12', 'final': '239ec145a50be525b7879d74897f4f812d07d0fc71aefd7b91274d8981fb13c9'}}. The operational stdout log received the final completion line after the engine snapshot. Its prefix hash was verified against that snapshot; all deterministic scientific artifact checksums match. final_integrity.json records final post-process hashes without altering the original completion evidence. This is a log-publication ordering limitation, not a scientific record change.

One validation-selected model/seed, small error groups, validation reuse, unverified source independence/near duplicates, coarse positive-only CAM, deletion distribution shift, normalization/threshold/tie dependence and clipping constrain interpretation. No external or TEST generalization, physical correctness, model superiority or causality is established.

The essential corrected evaluation scope is complete; broader frozen-secondary coverage remains incomplete. Before research completion: researcher review of class/correctness differences, transparently report all limitations/deferred analyses and decide whether further sanity/sensitivity work is required. Any additional experiment, expert annotation exercise, TEST evaluation or UI integration requires separate authorization.
