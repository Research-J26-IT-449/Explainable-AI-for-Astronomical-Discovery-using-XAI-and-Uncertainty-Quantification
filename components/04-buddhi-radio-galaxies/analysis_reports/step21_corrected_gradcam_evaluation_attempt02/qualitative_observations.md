# Exploratory visual observations — predetermined examples

These observations follow inspection of the four saved representative figures.
Selection used the lowest saved Step 20 global index in each ground-truth ×
correctness stratum, before attribution generation. No alternative image or CAM
was substituted. All observations refer to displayed processed intensity, not
calibrated flux or independently annotated astrophysical regions.

| Index / stratum | Visible intensity and attribution |
|---|---|
| 35, correct FR-I; predicted FR-I, confidence about 0.9324 | The original shows a compact bright central region with weaker surrounding/curved emission. CAM is broadly concentrated around that region, with less detailed structure than the intensity image. It does not separately resolve or validate a core or jet. |
| 31, incorrect FR-I; predicted FR-II, confidence about 0.5492 | The original shows an elongated diagonal pair/connected emission pattern. The predicted-FR-II CAM has separated maxima offset toward the upper-right and lower-left, extends into darker surroundings, and has reduced attribution near part of the central bright pattern. Visual mismatch is visible but does not establish its cause or prove an implementation error. |
| 50, correct FR-II; predicted FR-II, confidence about 0.9928 | The original has two bright components arranged approximately vertically. The broad CAM covers those components and intervening/surrounding area. This supports only the observation of spatial association with visible emission, not validated lobe/hotspot localization or a causal explanation. |
| 437, incorrect FR-II; predicted FR-I, confidence about 0.6101 | The original has a brighter upper-left component and weaker lower-right emission. The strongest CAM region spans an area between/below the upper component and toward the lower component; a weaker patch also appears in darker background. It is not a precise segmentation of the emission and does not establish a physical reason for the FR-I prediction. |

The two correct examples show broad attribution near visible emission; the two
incorrect examples also produce nondegenerate maps. Their differences cannot be
generalized to all FR-I/FR-II or correct/incorrect cases from four figures.
Use full-cohort scalar summaries and explicit missingness for those comparisons.
Smooth overlays are interpolations of a 19×19 positive CAM and add no spatial
evidence. A display-normalized bright patch indicates relative attribution within
one image, not a calibrated attribution magnitude across images.

No independent expert core, jet, lobe or hotspot annotations are available in this
evaluation. No physics-alignment Jaccard is claimed. The correct scientific context
is **radio emission structures produced by AGN activity, indirectly linked to
supermassive black holes**; neither these images nor Grad-CAM directly detect a
black hole. No map was edited to resemble an expected structure.
