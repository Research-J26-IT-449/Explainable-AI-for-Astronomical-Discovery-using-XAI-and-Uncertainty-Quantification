# Corrected-GradCAM-Validation-v1.0.0

Approved identity changes: researcher Step 21B. This is a separate exploratory
development protocol, not an amendment to, or completion of, historical
Phase3B-XAI-v1.0.0. Freeze this file and protocol.json before pixel access.

## Population and fixed model

Use all 116 VALIDATION entries, in membership order, from
data/manifests/mirabest_development_exactdedup_v1.0.0/manifest.json.
SHA-256: 4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91.
Dataset protocol: MiraBest-Development-ExactDedup-v1.0.0. Supports 51 FR-I (0)
and 65 FR-II (1). Exclude 198,323,474,587,694. No TRAIN/TEST image access.
Do not decode TEST labels or parse its original manifest suffix. Checksum reads
of opaque archive/split bytes do not imply zero OS reads of serialized bytes.

Checkpoint: outputs/models/mirabest_development_exactdedup_v1.0.0/corrected_seed42_01/models/resnet_cnn_best.pth.
SHA-256: d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c.
Existing ResNetCNN; strict weights_only CPU loading, all modules eval, parameters
requires_grad=False. Preserve weights, BatchNorm buffers and absent gradients.
No optimizer, training, checkpoint selection, corrected OOD calibration or TEST.

## Grad-CAM and preprocessing

Reuse src/xai/gradcam.py unmodified. Target model.blocks[2].convolution[3];
pre-softmax original predicted-class logit; temporary forward hook and autograd.grad;
spatial mean gradient channel weights, weighted sum, ReLU. Save raw 19x19 CAM,
unnormalized bilinear 150x150 attribution (align_corners=False), and safely
min-max-normalized display heatmap with core valid/degenerate status. Degenerate
maps are retained; do not fabricate attribution rankings or replace images.

Trusted original 150x150 grayscale uint8 /255 float32, Normalize(0.5,0.5), no
augmentation, resize, contrast remapping or public-upload guard. No external guard
changes. Original model-input tensor uses the existing validation transform;
perturbation tensors use existing faithfulness.normalize_intensity. Fix the original
predicted class for all interventions, including transformed prediction flips.
Compare the explanation forward's prediction/confidence with saved Step 20
records (probability tolerance 1e-5 for single-image vs batch float32 computation).
Prediction mismatch or probability discrepancy beyond tolerance aborts with evidence.
Do not run a separate classification evaluation.

## Primary faithfulness

Reuse faithfulness.prepare_rankings, evaluate_rankings, aopc, paired_contrasts.
Original native raw CAM descending; brightness=mean original intensity per tile.
Tile boundaries floor(150*j/19), j=0..19; within-tile row-major order, exact nested
budgets floor(22500*k/20), k=1..6 (5,10,15,20,25,30%). Compute rankings once.
Unnormalized replacement: median of the outer five-pixel border (primary).
Exactly 20 random rankings/image, seeds 1000..1019,
PCG64(SeedSequence([seed,index])). Ties retain the inherited SHA-256 namespace
Phase3B-XAI-v1.0.0|tile-tie|42|index|tile_id, then tile ID: this namespace is a
mathematical convention, not historical model/cohort provenance.

AOPC is mean(original score minus deleted score) over the six steps. Primary
score is fixed-class softmax probability; secondary diagnostics target logit and
binary logit margin. Retain negatives. Keep all 20 raw seed curves together per
image and average within-image before summaries. Compare CAM minus mean random
and CAM minus brightness, preserving image pairing and availability denominators.
Deletion can be OOD; retain its classifier scores without claiming public support.
Historical OOD calibration is not loaded or attributed to the corrected model.

## Stability and robustness

Reuse all 30 stability.protocol_transformations independently from the original:
rotations -5,-2,+2,+5 degrees; sensitivity -15,+15; horizontal/vertical flips;
Gaussian noise sigma .005/.01, seeds 2000..2009 per level with
PCG64(SeedSequence([seed,index])); brightness .98/1.02 secondary display sensitivity.
Clip intensities [0,1] and record clipped fraction. Rotations: centered bilinear,
fixed 150x150, border-median fill. Inverse-align continuous unnormalized CAMs
before top-k binarization; common valid support only; unavailable below 80%.
Primary top-10% Jaccard; sensitivities 5%/20%; exactly ceil(q*valid_pixels).
Pixel ties retain inherited SHA-256 Phase3B-XAI-v1.0.0|pixel-tie|42|index|pixel_id,
then pixel ID. Reuse compare_explanations without new pooling/alignment rules.
Retain fixed target across flips; record probability/margin changes, valid support,
both CAM statuses and missingness. Report preserving/flipped cases separately.
Average noise seeds within each image/level for image-level summaries, retaining
all seed observations and available/expected counts. Brightness is not calibrated
radio-flux robustness. Jaccard here measures explanation stability, not expert
astronomical localization.

## Intensity proxies and independent references

Reuse morphology_audit.audit_emission_alignment with unnormalized upsampled CAM.
Outer-five-pixel median b and sigma=1.4826*MAD; image > b+max(a*sigma,
d*(max(image)-b),1/255). Primary (a,d)=(3,.05), inclusive (2,.02), restrictive
(5,.10). Report area, emission/background attribution mass, enrichment=mass-area
and proxy top-10% IoU with brightness/uniform controls. Empty/full masks and
degenerate attribution retain existing reason codes. These are intensity-derived
proxies, never validated core/jet/lobe/hotspot masks. No independent region
annotation set is verified here; expert-reference physics Jaccard unavailable.

## Statistics, scope and incomplete analyses

Image is the statistical unit, not pixels, seeds or transformations. Use existing
statistics.image_record, aggregate_repeats and summarize for counts/reasons, mean,
median, IQR, sample SD, true classes and correct/incorrect strata. For the two
paired AOPC contrasts, reuse class-stratified paired image bootstrap (10,000,
seed42; 2.5/97.5 percentile, linear) with metric-specific complete-case coverage.
Intervals condition on availability, not model retraining/survey variation.
This corrected development analysis is exploratory; no new confirmatory tests
or superiority claims. No outcome-driven amendments or heatmap tuning.

The essential scope is all original explanations, primary border-median AOPC
with matched controls, full 30-transform 150x150 stability and proxy diagnostics.
Full 21-point deletion/AUC and black/Gaussian replacement sensitivities are
compatible but deferred secondary work: 116*22*20*3=153120 deleted-input calls
versus 116*22*6=15312 primary calls, a ten-fold deletion workload. This choice is
fixed before execution, not based on results; no cohort or primary metric changes.
Six-stage/five-seed parameter randomization is also deferred secondary work
(3480 additional gradient explanations and disposable model resets). Its absence
limits claims about dependence on learned parameters; this is not complete frozen
Phase 3B coverage. Do not mislabel these as scientifically invalid definitions.

Retain four approved unresolved reasons: native_grid_pooling_partial_support_ties_unspecified;
random_native_to_input_projection_unspecified; sanity_brightness_definition_unspecified;
sanity_random_distribution_grid_seeds_replicates_projection_unspecified.
ECE unavailable: no compatible frozen confidence-bin definition was found; do not
introduce bins here. Historical OOD diagnostics unavailable for this checkpoint;
no recalibration. Mark each unavailable/deferred endpoint explicitly, never zero.

## Representative selection, resources and reproducibility

Select the lowest global Step 20 index in each ground-truth x correctness stratum:
correct FR-I, correct FR-II, incorrect FR-I, incorrect FR-II. Freeze IDs before
CAM generation; retain even degenerate maps. No selection by attractive heatmaps.
Show original intensity, fixed-colormap normalized heatmap, alpha .45 overlay;
no stretching/cropping. Label figures exploratory, not validated scientific
explanations. Any qualitative interpretation must follow inspection of these
saved figures, not fabricated astronomical annotations.

CPU; seed42; keep installed Torch settings and original architecture/preprocessing.
Selected-row reader allowlist contains only the 116 corrected validation rows;
at most eight raw payloads per read. Process one image at a time; retain no
dataset-wide raw cache. Save only per-image native/display maps, scalar records,
selected original/overlay figures and seed curves, not perturbation arrays.

Cooperative 2 GiB working-set /900-second per-image safety checkpoints before/
after families, each deletion scorer call and each transformed CAM. Reliable
memory measurement required; fail closed otherwise. Blocking model/OS operations
cannot be interrupted instantly. Abort after first error, provenance/state drift
or resource violation. No silent replacement, reduced cohort or restart.
Separate timing/memory from scientific records. Save incomplete failure evidence,
never mark cohort successful without 116 complete unique records and final hashes.
Exclusive outputs; no overwrite/resume/repeated experiment in this attempt.
Record source/config/protocol/manifest/checkpoint hashes, software, exact command,
seed conventions, selected IDs and protected-artifact/Git checks before/after.

Limitations: validation-selected model, small error strata, one seed, coarse CAM,
positive-only gradients/interpolation, deletion distribution shift, clipping,
threshold/tie sensitivity, unverified source independence/near duplicates and no
external/TEST validation. Do not claim causality, physics correctness, direct
black-hole detection or superiority. Use: radio emission structures produced by
AGN activity, indirectly linked to supermassive black holes.
