# Scientific interpretation of completed corrected XAI results

This is an exploratory, validation-selected, single-model study; all values below
come from the completed saved records. No additional model call was performed.

## Faithfulness and class/correctness strata

Mean primary probability AOPC30: CAM 0.294510, random 0.053960, brightness
0.222254 (n=116 for each). CAM-minus-random mean 0.240550, descriptive paired
95% bootstrap interval [0.196203, 0.282320]. CAM-minus-brightness mean 0.072256,
interval [0.051306, 0.093401]. The cohort means indicate stronger score sensitivity
to CAM-ranked deletion under this one predefined replacement than the matched
controls. They do not prove causality, physical correctness, or external validity.
Eight negative CAM AOPCs remain in the dataset. No confirmatory test was executed.

FR-I versus FR-II CAM AOPC means: 0.430278 versus 0.187986. However the gain over
brightness is 0.012163 versus 0.119406 respectively. Raw class-level sensitivity
and added value over intensity ranking are different observations; neither
establishes that one class's explanations are intrinsically better. Correct
versus incorrect CAM AOPC means: 0.329636 (101 images) versus 0.058000 (15 images).
The corresponding gains over random are 0.274486 versus 0.012053. These groups
also differ in class composition and original confidence; the study does not
isolate correctness as the cause of those differences. Only two misclassified
FR-II cases exist; do not infer a general subgroup law from them.

## Stability, missingness and noise clipping

Mean inverse-aligned top10% Jaccard is 0.891493 for -2 degrees and 0.882067 for
+2 degrees (116 available images each). Horizontal/vertical flips yield
0.676351/0.734696. These quantify explanation agreement under the frozen
transformations, not verified morphological localization or all-purpose robustness.

Noise is a major limitation: at EACH sigma (.005 and .01), all 1160 saved
transformed predictions are FR-I. Consequently all 76 originally predicted FR-II
images change class for every one of the ten seeds per level: 760 flips/1160 at
each sigma. This includes 630 flips on true FR-II and 130 on true FR-I, per level.
All explanation targets nevertheless remain the original predicted class. Across
the entire 30-transform schedule there are 1553 flips/3480 comparisons, including
1520 under noise. These are repeated within-image diagnostics, not 1553 independent
failing sources. Do not label this model generally noise-robust.

There are 459 unavailable noise Jaccards at sigma .005 and 722 at .01 because
transformed CAMs are degenerate (1181 total); none are silently set to 0 or 1.
Image-level noise top10% Jaccard means are 0.452004 (78 images with at least one
available repeat) and 0.567976 (46 images), respectively. They condition on different
available subsets and repeat counts. The larger second number is NOT evidence
that stronger noise improves stability. Missingness must accompany every summary.

Mean clipped pixel fractions are about 49.19% and 49.22% for these noise levels.
Clipping at zero in background-dominated processed images changes the perturbation
distribution; these tests do not represent symmetric instrumental noise in a
calibrated radio map. The saved records show output/attribution behavior under the
specified transformation, not its physical cause. Distinguishing background,
normalization, learned-feature, or physical-noise explanations requires additional
separately approved work; none was undertaken or silently substituted here.

## Emission proxies and qualitative explanations

For the primary intensity-derived mask, mean attribution mass is 0.090169,
mean mask area fraction 0.011586590, and mean area-adjusted enrichment 0.078583.
Thus attribution is enriched relative to mask area, but about 90.98% of mean
positive attribution mass lies outside this particular thresholded mask. Coarse
positive CAM tails and a small thresholded emission region are relevant limitations;
these records do not diagnose their cause. Mean proxy top10% IoU is 0.074599.
It is not expert-annotation Jaccard and is not a validated astrophysical quality
score. Report area, mass and enrichment together rather than a favorable overlap
number alone. FR-I/FR-II mean mass is 0.158507/0.036550; these depend on intensity
masks and attribution distribution, not verified jet/lobe identities.

qualitative_observations.md records actual inspection of the four predetermined
figures. Correct examples broadly associate CAMs with visible emission, while
incorrect examples also yield valid maps with offsets/background coverage. Four
selected figures cannot establish cohort-level correctness or physical grounding.
Use the phrase: radio emission structures produced by AGN activity, indirectly
linked to supermassive black holes. No black-hole detection or expert localization
claim is made.

## Completion and remaining work

All 116 original explanations, primary matched-control AOPC, defined 150x150
stability conditions and intensity-proxy variants are complete. Full deletion/AUC,
replacement sensitivities and parameter randomization remain deferred secondary
analyses, not completed or scientifically invalid ones. The four inherited
undefined conventions remain explicitly unavailable. ECE has no inherited bin
definition; independent-mask physics Jaccard and corrected-model OOD diagnostics
are unavailable. Historical calibration was not reused.

Before final research/UI integration, review the substantial noise flips/clipping,
report missingness and subgroup limitations, and decide whether deferred sanity
checks/sensitivities are necessary for the desired claims. No extra training,
classification pass, TEST, XAI run or UI modification is authorized by this report.
The defensible verdict is XAI PARTIALLY COMPLETED ? LIMITATIONS DOCUMENTED, with
the new protocol's essential scope complete and broader secondary coverage incomplete.
