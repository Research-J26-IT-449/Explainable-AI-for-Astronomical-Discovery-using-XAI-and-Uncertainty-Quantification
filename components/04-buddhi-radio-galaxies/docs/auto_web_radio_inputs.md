# AUTO / WEB RADIO IMAGE (EXPERIMENTAL)

AUTO is an experimental Gradio choice. Upload an opaque, single-frame image
containing genuine radio morphology. Web origin alone does not establish radio
provenance. The original MiraBest and stricter external-radio modes remain
available; `predict_image` still defaults to `mirabest` for API reproducibility.
Use `predict_image(image, input_mode="auto_web")` to request AUTO explicitly.

## Architecture

1. Decode PIL input; reject corruption, unsupported numeric scaling, NaN/Inf,
   transparency, animations, and dimensions outside 16–4096 pixels per axis.
   Palette PNGs are decoded with their palette and alpha before processing.
   Both external paths share the mode decoder: L/RGB retain their pixels;
   opaque RGBA becomes RGB; opaque LA and mode 1 become L; P uses its palette;
   CMYK becomes RGB. I values in [0,255] retain their values. Unsigned 16-bit
   modes (including endian variants), and I values above 255 within [0,65535],
   use fixed `round(value / 257)` to 8-bit L. No data-dependent stretching,
   cropping, or intensity inversion occurs. Signed/overrange I and float F fail
   closed. Quantization can lose faint levels and does not establish radio validity.
   This decoding layer feeds canonical L/RGB candidates into the unchanged
   calibrated policy; calibration metadata and thresholds are not rewritten.
2. Apply whole-image radio-map representation checks. Quiet-background, sparse
   positive-emission structure is required, without requiring symmetry or a
   centered source. Reject blank/uniform, dense or photographic structure,
   substantial boundary content, negative-background structure, and excessive
   high-frequency diagnostic texture. These are deterministic policy limits,
   not semantic identification of radio galaxies. All numeric policy values
   are stored in the separate calibration metadata.
3. For grayscale or effectively achromatic inputs, preserve the original
   grayscale pixel intensities. For 150x150 inputs, geometry is untouched.
   If the original frozen OOD checks pass, AUTO uses exactly the original
   classifier tensor, probabilities, and label. No confidence rule or new gate
   changes this accepted original route. If original OOD rejects, AUTO requires
   the separate external gate, just like other external candidates; original
   mode itself continues to reject under its original rules.
4. Before rejecting non-achromatic RGB, attempt strict single-hue pseudocolor
   detection (details below). Confident pseudocolor is represented only by fixed
   HSV Value = maximum RGB and must still pass the separate external gate.
   Otherwise retain the approximately affine, positive shared spatial
   structure across **every varying RGB channel**. The source luminance, each
   varying channel, and maximum-channel image must each pass the representation
   checks. A sparse channel cannot rescue dense/independently textured channels.
   Channel coherence is measured over the union of their emission regions so
   a large empty background cannot dilute disagreement between sparse structures.
   Candidate images are luminance, individual R/G/B intensities, and maximum
   channel intensity. Constant channels are omitted; exact duplicate tensors
   are counted once. This supports simple shared-morphology color encodings,
   not arbitrary nonlinear palettes or arbitrary optical/radio composites.
5. Symmetrically pad non-square candidates using the rounded median of all four
   source borders, preserving all original content. Resize only when needed
   using PIL bilinear to 150x150. No crop, source selection, inversion, smoothing,
   or intensity stretching enters the classifier. Convert via torchvision
   ToTensor float32 and Normalize(mean=0.5, std=0.5).
6. Require separately calibrated energy **and** feature-distance gates for all
   plausible external candidates, never heuristic-only fallback. Classify all
   candidates with the locked ResNet. Any label disagreement returns AMBIGUOUS,
   even if one candidate is more confident or another fails its gate. If they
   agree but any candidate fails a gate or has confidence below the fixed 0.60
   policy minimum, return UNSUPPORTED. Otherwise report the unanimous label
   with mean candidate probabilities, never highest-confidence selection.
7. UI reports class/confidence, handling route, candidate names, and the
   experimental limitation. Rejection/ambiguity has no classification scores.
   Numeric gate diagnostics are available in the structured API result, not
   the ordinary UI.

The strict `external_radio` mode is unchanged by AUTO and still requires the
original frozen OOD checks. See [external_radio_inputs.md](external_radio_inputs.md).

## Separate calibration

Run `python scripts/calibrate_external_compatibility.py`. This writes only:

- `outputs/ood/external_compatibility_calibration.pt`
- `outputs/ood/external_compatibility_calibration.json`

These generated local artifacts are ignored by the repository's existing
`outputs/ood/` rule; they must be generated/transferred explicitly for deployment.
If absent, stale, incompatible, or corrupted, the external candidate path fails
closed. Compatible original-route images can still use the original calibration.

The verified local loader is used with `download=False`. Only the existing
554 TRAIN and 119 VALIDATION indices are selected, without reading labels to
select positives. The final test split is inspected as **indices only** to
enforce disjointness, and is never selected for calibration. The two exported
held-out demo PNGs are used only in explicit consistency regressions. The
seven reserved external OOD images are neither read nor used to select policy.

Each allowed sample produces thirteen fixed variant families: original; centered
180x150 and 150x180 canvases; bilinear 150→96→150 degradation; contrast ×0.9
and ×1.1; gamma 0.9 and 1.1; affine RGB encoding `(p, 0.75p, 0.5p)`; and
grayscale JPEG quality 95; and red/orange, blue/cyan, and green pseudocolor
encodings whose maximum RGB is exactly the original uint8 intensity.
No crops, rotation, label-dependent choices, or
automatic brightness optimization are used. Mild clipping/quantization occurs
only in explicitly defined **calibration** variants, never as runtime intensity
remapping. The policy is fixed before evaluating the requested test regressions.

The same candidate extraction/guards apply to variants. Rejected variants are
excluded with per-family counts recorded. TRAIN candidates define the 512-element
feature mean and diagonal variance (floor 1e-6) from the locked ResNet spatial
pool. VALIDATION candidates independently define upper energy and standardized
feature-distance thresholds at the predeclared 99th percentile. No optimizer,
weight update, runtime fitting, or final test threshold selection occurs.

The artifact includes exact TRAIN/VALIDATION indices, counts, split-file path
and SHA256, checkpoint path and SHA256, verified dataset manifest SHA256,
feature dimension/layer, transformation definitions, seed 42, preprocessing
version/configuration, representation rules, consistency policy, and accepted/
rejected variant counts. Runtime validates the model, policies, statistics, and
current split bytes against this provenance. Original OOD artifacts are never
written; no original threshold is changed.

## Single-hue pseudocolor policy (v2)

This is a constrained representation detector, not proof of radio provenance.
It uses fixed rules before any classifier scores are computed:

- At least 80% of pixels have Value <= 0.10; at least 95% of border pixels do too.
- Maximum-RGB intensity must pass the existing radio emission, occupancy, border,
  background, and diagnostic texture checks without changing any of their limits.
- Meaningful emission has Value greater than max(0.05, 0.20 * its 99.5th percentile).
  At least 95% of these pixels must have saturation >= 0.60.
- Circular, Value-weighted hue resultant must be >= 0.95; at least 95% of emission
  weight must lie within 30 degrees of the dominant circular hue. This handles
  red hue wraparound while rejecting broad multi-hue content.
- Within emitting pixels, RGB channel means grouped by integer Value must be
  nondecreasing (2/255 rounding tolerance). Channel RMS deviations from those
  Value-group means must be <= 0.02. Independent spatial channel structure cannot
  masquerade as a color lookup table merely by occupying the same hue family.

The sole recovered representation is original maximum RGB, without a brightness
stretch, histogram adjustment, or confidence-based channel choice. Geometric
padding and ToTensor/Normalize are unchanged. Confident recovery does not permit
heuristic-only acceptance; the separate calibrated gate and confidence rule remain
mandatory. Existing affine-candidate disagreement still returns AMBIGUOUS.

New deterministic calibration LUTs, with p = original uint8 intensity, are:

- Red/orange: (p, round(p * (0.10 + 0.45*p/255)), 0).
- Blue/cyan: (0, round(p * (0.20 + 0.50*p/255)), p).
- Green: (round(0.20*p), p, round(0.10*p)).

Each is monotonic and max RGB = p exactly. Only the separate external statistics
and thresholds are regenerated from TRAIN/VALIDATION; original MiraBest OOD,
weights, splits, and reported metrics stay unchanged. Detector rules and the 99th
percentile choice are fixed before reading the development-image predictions.

## Development examples, excluded from untouched final evaluation

The previously tested 3C31 and Cygnus A web images are **development examples**.
Both were rejected as `UNSUPPORTED` by AUTO under the fixed final policy: 3C31
for inconsistent color-channel structure and Cygnus A for dense or insufficient
compatible emission. No threshold, pseudocolor rule, or other criterion was tuned
to force either image to be accepted. This is a domain-shift limitation: genuine
published radio images can differ materially from the representation supported by
the MiraBest-trained classifier and its conservative external compatibility gate.
Do not later describe these files as untouched final external evaluation samples.
Run `scripts/check_pseudocolor_development.py --three-c31 PATH --cygnus-a PATH`
only after fixing the policy/calibration. It records their file SHA256, paths,
known labels, honest outcomes, and `eligible_for_untouched_final_evaluation=false`
in `outputs/ood/pseudocolor_development_checks.json`. Do not adjust thresholds
or repeatedly tune transforms to force their expected labels.

## Scientific limits

Positive-only synthetic calibration measures compatibility with transformed
MiraBest, **not** a false-accept rate for natural/optical images and not
external-survey accuracy. Candidate agreement is correlated evidence from one
image, not independent confirmation of correctness. A shared optical structure
or drawing can resemble a radio map and pass these checks. Conversely genuine
radio maps with extended/full-frame emission, different backgrounds, weak signal,
nonlinear color palettes, annotations, or different survey resolution/sensitivity
can be rejected. Arbitrary web images remain unsuitable for scientific FR claims.

Tests use synthetic unrelated styles for rejection and sentinel TEST indices for
calibration discipline. They do not establish semantic rejection of every person,
car, dog, landscape, optical galaxy, screenshot, or drawing. Independent external
radio positives and unrelated negatives are required for a descriptive evaluation
before making stronger reliability claims. Reported final MiraBest metrics are
unchanged and do not describe AUTO performance.
