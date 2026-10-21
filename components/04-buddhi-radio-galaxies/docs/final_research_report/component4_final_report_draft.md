# RadioMorph AI: AGN-Aware Radio Source Classification Using CNN and Explainable AI

**Individual research report — provisional draft**  
Project J26-IT-449 — Explainable AI for Astronomical Discovery, Component 4  
Evidence cut-off: 21 October 2026

Institution, programme, candidate details, supervisor details, declarations, submission date and formatting: **to be confirmed against the official university template**. This draft does not certify institutional approval. The repository's formal research proposal was not located; the aims below describe the implemented contribution rather than quoting approved objectives.

## 1. Abstract

RadioMorph AI investigates binary Fanaroff–Riley radio-source classification using a lightweight residual convolutional neural network and gradient-based visual explanations. The work combines dataset-integrity controls, reproducible model training, fixed-checkpoint classification analysis and Grad-CAM evaluation. An exact-content audit identified five records with identical decoded pixels but conflicting labels, including TRAIN–VALIDATION overlap. A separately versioned correction quarantined all five while preserving historical evidence, leaving 552 training and 116 development-validation images. A freshly initialized ResNetCNN was trained with seed 42 and selected using minimum validation loss.

The selected checkpoint achieved 101 correct classifications from 116 validation records: accuracy 87.068966% and macro F1 0.864391. Its confusion matrix, with FR-I and FR-II as ordered rows and columns, was [[38,13],[2,63]]. Genuine predicted-class Grad-CAM explanations were generated for all 116 records. Mean probability AOPC through 30% deletion was 0.294510 for Grad-CAM ranking, 0.053960 for seeded random ranking and 0.222254 for brightness ranking. These findings support sensitivity to the selected deletion ordering under the specified replacement policy, rather than physical causation or verified anatomical localization.

Transformation experiments revealed important weaknesses, including prediction changes, clipping under intensity-noise perturbations and degenerate transformed explanations. No independent expert-mask alignment, corrected-model TEST evaluation, external-survey validation or validated corrected-model OOD gate was established. The Gradio dashboard therefore presents verified saved research results while keeping public-upload prediction disabled. The contribution is a reproducible development-cohort classification and explanation study with explicit evidence boundaries, not a claim of deployment-ready astronomical discovery. [E01–E20]

## 2. Introduction

### 2.1 Research context

Fanaroff–Riley morphology provides a useful framework for describing extragalactic radio sources. The original classification relates radio morphology to the relative distribution of bright emission; it is not a measurement of black-hole properties from a small image alone. The foundational paper provides the historical basis for the FR-I/FR-II distinction. [Fanaroff and Riley, 1974](https://academic.oup.com/mnras/article/167/1/31P/2604898)

For this component, the scientific object is the morphology of radio emission structures produced by AGN activity, indirectly linked to supermassive black holes. The implemented system predicts an image label from a grayscale cutout. It does not detect a black hole directly, measure jet energetics or establish a causal account of source evolution. Its labels and images must be interpreted within the selected dataset and annotation process.

CNN classification can make repeated image assessment computationally practical, but a high development score alone does not establish trustworthy behaviour. Content leakage can inflate apparent validation performance, while visual explanations may be attractive without accurately identifying the information driving a decision. This project therefore treats data integrity, predictive performance and explanation behaviour as separate requirements. [E01, E04, E11–E17]

### 2.2 Problem statement and implemented aims

The practical problem is to classify FR-I versus FR-II radio-source images while providing an inspectable, class-specific visualization and preserving reproducible evidence. The implemented aims are: to prepare a deterministic corrected development cohort; train and select a lightweight residual classifier; measure its fixed-checkpoint validation performance; generate genuine Grad-CAM maps and assess specified perturbation/stability behaviour; and communicate results through a responsible research dashboard. These aims require reconciliation with the official proposal before final submission.

The central research questions are whether the selected model discriminates the two labels on the corrected development-validation cohort, whether deleting regions ranked by Grad-CAM changes the fixed target score more than specified controls, and how explanations and predictions behave under predefined transformations. A further engineering question concerns how uncertainty and unavailable evidence should constrain public presentation. No hypothesis of external-survey superiority or independent TEST generalization is supported by this study.

### 2.3 Scope and contributions

The contribution includes a versioned quarantine-based dataset protocol, a reproducible fresh-training workflow, saved per-image classification evidence, gradient-based explanations with perturbation diagnostics, and a results-only interface. Historical experiments remain preserved and distinct. One corrected seed and one selected checkpoint are considered; this is not a model search, a multi-survey benchmark or an assessment of uncertainty across repeated training. [E01–E10, E19–E22]

## 3. Literature Review

### 3.1 Radio morphology and dataset context

The FR distinction predates deep learning and concerns observable radio morphology. It supplies the label taxonomy used here, rather than a justification for treating every binary prediction as a physical diagnosis. The original work is the relevant starting point for the class names. [Fanaroff and Riley, 1974](https://academic.oup.com/mnras/article/167/1/31P/2604898)

Porter and Scaife describe MiraBest as a morphologically classified radio-galaxy dataset assembled for machine learning. Their published description concerns 1,256 sources derived from survey observations and prior classifications. The local binary archive verified by this project contains 792 records; that count must not be substituted for the publication's collection size or assumed to describe all MiraBest releases. Dataset edition, confidence filtering and archive identity are necessary parts of a reproducible comparison. [Porter and Scaife, 2023](https://arxiv.org/abs/2305.11108); [MiraBest Batched Dataset record](https://zenodo.org/records/4288837). [E01–E03]

### 3.2 Residual networks

He and colleagues introduced residual learning to improve the optimization of deep image-recognition networks. The relevant architectural principle is learning a residual transformation alongside a shortcut. RadioMorph AI uses a compact custom residual model, not an unmodified ImageNet ResNet-18 or the original paper's benchmark model. Residual connections motivate the design; their presence does not establish equivalent capacity, pretraining or performance. [He et al., 2015](https://arxiv.org/abs/1512.03385). [E05]

### 3.3 Gradient-based visual explanation

Grad-CAM uses gradients of a target score with respect to convolutional feature maps to construct a coarse class-specific localization map. This makes it suitable for examining a trained CNN without modifying its learned weights. The resulting visualization remains conditional on the chosen layer and class. Upsampling improves display resolution but cannot recover spatial detail absent from the native feature grid. [Selvaraju et al., 2016](https://arxiv.org/abs/1610.02391). [E10]

### 3.4 Evaluating explanations rather than appearances

Samek and colleagues describe region perturbation as a way to evaluate ordered image importance rather than relying only on visual appeal. The present study uses an explicitly defined deletion score, but its exact grid, fill value and controls are repository-specific decisions. Its numerical results should not be equated with a different paper's benchmark or with annotation-based localization. [Samek et al., 2015](https://arxiv.org/abs/1509.06321). [E11–E13]

Adebayo and colleagues propose sanity checks for saliency methods, emphasizing the importance of testing dependence on learned model information. An implemented randomization module is not evidence that a corrected-model randomization experiment was executed. The corrected study deferred that analysis; parameter dependence is consequently an unresolved empirical question for this checkpoint. [Adebayo et al., 2018](https://arxiv.org/abs/1810.03292). [E18]

### 3.5 Confidence and research gap

Guo and colleagues show that neural-network confidence can be miscalibrated. Softmax output is therefore reported here as model confidence rather than verified correctness probability. Calibration of class confidence also differs from demonstrating rejection of unknown inputs. Neither a temperature-scaling claim nor a valid corrected-model OOD release gate can be inferred from classification accuracy. [Guo et al., 2017](https://arxiv.org/abs/1706.04599). [E19]

This targeted review supports a study combining classification, integrity and explanation diagnostics. It is not a systematic literature review and does not establish novelty relative to every radio-galaxy CNN. The official proposal bibliography and a verified comparative review of directly related radio-classification studies remain submission requirements. [E23]

## 4. Research Methodology

### 4.1 Evidence-led experimental design

The research separates historical baseline evidence from a corrected development experiment. The corrected model uses MiraBest-Development-ExactDedup-v1.0.0, followed by a separately versioned Corrected-GradCAM-Validation-v1.0.0 evaluation. Checkpoint and manifest hashes bind the cohort to the selected model. Saved results, not fresh model execution, were inspected when preparing this report. [E01, E06, E08, E10]

### 4.2 Dataset integrity and correction

The local original development split comprised 554 TRAIN and 119 VALIDATION records. The exact-duplicate audit covered these 673 records and found one five-record identical-content group with ten duplicate pairs and six conflicting-label pairs. The affected global indices were 198, 323, 474, 587 and 694. Their split and label assignments are listed in Appendix A. All five were quarantined rather than automatically relabelled or selecting an unsupported representative. [E02–E04]

Remaining membership order and labels were preserved. The corrected TRAIN cohort contains 552 images: 245 FR-I and 307 FR-II. Corrected VALIDATION contains 116: 51 FR-I and 65 FR-II. Saved content-hash evidence establishes exact-content disjointness within this audited development scope after exclusion. It does not establish near-duplicate absence, independence of astronomical sources, label correctness elsewhere or cleanliness of the locked TEST cohort. No TEST records were used to choose this correction. [E01–E04]

### 4.3 Preprocessing and data access

Inputs are single-channel grayscale 150×150 images. Deterministic classifier preprocessing produces float tensors and applies normalization with mean 0.5 and standard deviation 0.5, equivalent to mapping intensity x in [0,1] to 2x−1. Validation uses no augmentation, cropping or new contrast transformation. Training retains the existing random rotation up to ±15°, horizontal flip and vertical flip. Trusted research data do not pass through public-upload blank/near-uniform rejection rules. [E05–E07]

The corrected adapter obtains labels from verified corrected metadata and restricts pixel materialization to authorized membership. Its bounded reader processes at most eight raw images per read. A tensor mini-batch of 32 is therefore not a raw archive batch of 32: selected raw chunks are transformed and released while constructing the training batch. The access mechanism avoids unrestricted bulk deserialization of mixed-split pixel arrays. These are implementation safeguards and saved execution claims, not an independently certified operating-system isolation boundary. [E07]

### 4.4 Model and training procedure

The model is a custom 339,746-parameter ResNetCNN. It uses a 32-channel convolutional stem, three residual blocks with 32, 64 and 128 output channels, and projection shortcuts where dimensions change. Adaptive average pooling to 2×2 feeds a 512-to-64-to-2 classifier with ReLU and dropout 0.30. The output is two pre-softmax logits for FR-I=0 and FR-II=1. [E05]

Fresh random initialization was used, rather than loading the historical selected checkpoint. Training used seed 42, CPU, batch size 32, cross-entropy loss and Adam with learning rate 0.0003 and weight decay 0.0001. The maximum was 30 epochs. ReduceLROnPlateau monitored validation loss with patience 2 and factor 0.5; early stopping used patience 7. The best checkpoint was selected by minimum validation loss, not maximum accuracy. This protocol was fixed before the completed run. [E06]

### 4.5 Fixed-checkpoint classification

The selected checkpoint was strictly loaded into the existing architecture in evaluation mode. One corrected-validation inference pass used deterministic preprocessing and inference mode, preserving model parameters and BatchNorm state. Saved per-image evidence records global index, true and predicted label, both probabilities and correctness. Support, probability constraints, uniqueness and membership were verified. Metrics use standard precision, recall and F1 definitions, with zero-division handling set to zero. [E08–E09]

For class c, precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is the harmonic mean of precision and recall. Macro averages weight the two classes equally; weighted averages weight them by support. Balanced accuracy is the average class recall. This distinguishes class-sensitive performance from overall accuracy under the observed 51/65 validation imbalance.

### 4.6 Grad-CAM generation

The target layer is `model.blocks[2].convolution[3]`, whose activations have native spatial size 19×19. For fixed original predicted class c and activation channel k, the spatially averaged logit gradient is

αᶜₖ = (1/Z) ΣᵢΣⱼ ∂yᶜ/∂Aₖᵢⱼ,

and the raw map is Lᶜ = ReLU(Σₖ αᶜₖAₖ). A temporary forward hook captures activations; `torch.autograd.grad()` obtains gradients without accumulating parameter gradients. A separate gradient-enabled forward pass is necessary even though weights remain frozen. Bilinear interpolation produces the 150×150 display map; safe normalization supplies [0,1] values. Zero/constant maps are explicitly degenerate. The existing API requires evaluation mode and removes hooks on failure as well as success. [E10]

The corrected evaluation retains the original predicted class for attribution and subsequent scoring, including after a perturbation changes the model prediction. It checks explanation-run predictions against Step 20 evidence. Explanations are not adjusted to resemble expected astronomical structures. [E10–E12]

### 4.7 Faithfulness and uncertainty summaries

Native CAM tiles partition the 150×150 image using bⱼ=floor(150j/19), j=0…19. Ranking uses raw nonnegative CAM values, deterministic ties and exact nested pixel budgets at 5%, 10%, 15%, 20%, 25% and 30%. Pixels are replaced in unnormalized intensity space by the median of the outer five-pixel border, followed by unchanged model normalization. Brightness ranking and twenty seeded random rankings provide controls; random seeds are 1000–1019, with PCG64 initialized from seed and dataset index. [E11]

The primary probability score is

AOPC₃₀(x) = (1/6) Σₘ₌₁⁶ [p_c(x) − p_c(x_deleted,m)].

It is a probability difference, not a percentage of correctly explained pixels. Negative values are retained. Random controls are averaged within image before image-level aggregation. Secondary logit/margin diagnostics do not replace the primary endpoint. Saved class-stratified paired percentile-bootstrap summaries use 10,000 replicates and seed 42; intervals are descriptive and conditional on availability. They do not measure uncertainty across retraining or new surveys. No new inferential calculation was performed for this draft. [E11–E13]

### 4.8 Stability and emission proxies

Independent transformations comprise rotations ±2° and ±5°, sensitivity rotations ±15°, horizontal/vertical flips, Gaussian noise σ=0.005 and 0.01 with seeds 2000–2009, and secondary brightness gains 0.98 and 1.02. Rotation uses bilinear sampling and border-median fill. Continuous transformed CAMs are inverse-aligned before top-k thresholding. Primary top-10% Jaccard uses common valid support, deterministic ties and an 80% minimum support; 5% and 20% are sensitivities. Repeated transformations are grouped within images, not independent statistical subjects. [E14–E15]

The independent intensity proxy uses outer-border median b and robust dispersion σ_b=1.4826 median(|border−b|). Its primary mask is E=x>b+max(3σ_b,0.05(max(x)−b),1/255). Inclusive and restrictive variants use 2σ_b/2% and 5σ_b/10%. Attribution mass inside E, area fraction and mass-minus-area enrichment are calculated, with proxy IoU as a secondary quantity. These masks are not expert-labelled jets, lobes, cores or hotspots. No annotation-based physics-alignment Jaccard is available. [E16–E18]

## 5. System Design and Implementation

### 5.1 Research pipeline

The pipeline links immutable archive/split provenance to corrected membership, bounded authorized data access, fresh training, fixed-checkpoint evidence and corrected XAI outputs. Distinct versioned paths prevent corrected artifacts from overwriting historical checkpoints, results or XAI protocols. Source hashes, saved configuration, run metadata and per-image records provide stronger execution evidence than a project title or a bare Git revision. [E01, E06–E10, E22]

| Layer | Main implementation | Responsibility |
|---|---|---|
| Corrected protocol | `src/data/corrected_protocol.py` | Metadata validation, quarantine and deterministic provenance |
| Data adapter | `src/data/corrected_training_data.py` | Authorized bounded access and preserved labels |
| Model/training | `src/models/resnet_cnn.py`, `corrected_training.py` | Residual architecture, reproducible training and selection |
| Explanation | `src/xai/gradcam.py` | Class-logit gradient-based CAM generation |
| Evaluation | `faithfulness.py`, `stability.py`, `morphology_audit.py`, `statistics.py` | Defined metrics and availability handling |
| Presentation | `demo/app.py`, `dashboard.py`, `research_results.py` | Gradio dashboard and saved research presentation |

The existing parameter-randomization implementation is a tested core capability; its presence is not counted as a completed corrected-model experiment.

### 5.2 Dashboard and safe public presentation

RadioMorph AI preserves a professional Gradio interface, exposes the corrected model identity and displays saved accuracy, macro F1, confusion matrix and AOPC comparison. Public packaging excludes unapproved model weights, datasets and per-image radio figures whose redistribution clearance has not been established. Existing private local examples do not imply permission to publish those source images. [E19–E21]

Public-upload predictions remain disabled because compatible corrected-model OOD evidence is unavailable. Missing calibration, incompatible provenance or unsupported input must not produce FR-I/FR-II probabilities or fabricated heatmaps. Historical-model calibration cannot be transferred to the corrected checkpoint. This is a responsible system limitation rather than a hidden completed feature. The current interface is a research-results demonstration, not an unrestricted live astronomical classifier. [E19–E21]

### 5.3 Reproducibility and verification boundaries

Saved training provenance records Python 3.12.10, PyTorch 2.14.0+cpu and CPU execution. Seeds and deterministic settings improve repeatability but do not establish bitwise equivalence across all platforms. Test reports support mathematical and engineering checks; they are not scientific replications. For example, Step 22 records 124 focused passing tests and Step 24 records 61. These counts refer to separate saved checks, are not summed into a unique test total, and were not rerun for this report. [E06, E20–E22]

## 6. Results and Discussion

### 6.1 Training convergence

The corrected run completed 30 epochs in approximately 18.15 minutes. Epoch 30 achieved the minimum validation loss, 0.3447643220, and was selected. Its validation accuracy was 87.068966%. The highest recorded accuracy occurred earlier, at epoch 23 (88.793103%); choosing that checkpoint after reviewing accuracy would change the predefined selection criterion. [E06]

Learning-rate reductions were recorded after epochs 4, 8, 11, 20 and 26. History rates are post-scheduler values and apply to the subsequent training epoch. Validation loss continued to reach its best value at the limit, so the record does not demonstrate a settled plateau. Late training provides no clear persistent divergence supporting strong overfitting claims, but neither does it rule out selection optimism. No additional training was conducted to test whether more epochs would help. [E06]

Existing figures: [loss curve](../../analysis_reports/step19_corrected_validation_analysis/training_loss_curve.png) and [accuracy curve](../../analysis_reports/step19_corrected_validation_analysis/accuracy_curve.png). Figure filenames are checked in the companion evidence review; no curves are recomputed here.

### 6.2 Classification performance

The fixed checkpoint reproduced the training record's 101/116 aggregate result. Rows below are ground truth and columns are predictions. [E08–E09]

| True class | Predicted FR-I | Predicted FR-II | Support |
|---|---:|---:|---:|
| FR-I | 38 | 13 | 51 |
| FR-II | 2 | 63 | 65 |

| Metric | FR-I | FR-II | Macro | Support-weighted |
|---|---:|---:|---:|---:|
| Precision | 0.950000 | 0.828947 | 0.889474 | 0.882169 |
| Recall | 0.745098 | 0.969231 | 0.857164 | 0.870690 |
| F1 | 0.835165 | 0.893617 | **0.864391** | 0.867918 |

Overall accuracy is **87.068966%**; balanced accuracy is **85.716440%**. FR-I accounts for 13 of the 15 errors and has lower recall. Conversely, FR-I predictions have higher precision. FR-II's higher support and recall increase overall accuracy relative to equal-class recall. Accuracy alone would conceal this asymmetry. The [saved confusion matrix](../../analysis_reports/step20_corrected_validation_classification_attempt02/confusion_matrix.png) and CSV supply the underlying evidence.

The historical validation figure of 85.71% cannot be used to claim a statistically verified improvement. Membership changed, leaked/conflicting records were removed, and training/model identity differs. Corrected validation was itself used for checkpoint selection and is not an independent test of generalization. Historical TEST results do not establish the corrected model's TEST performance. [E03, E08–E09, E24]

### 6.3 Grad-CAM and deletion analysis

All 116 original predicted-class CAMs were valid and consistent with saved classification predictions. The cohort mean AOPC values were independently checked by averaging saved per-image records, without model execution. [E10–E13]

| Ranking | Mean probability AOPC₃₀ | Images |
|---|---:|---:|
| Grad-CAM | **0.294510** | 116 |
| Mean seeded random control | **0.053960** | 116 |
| Brightness | **0.222254** | 116 |

The mean paired differences are 0.240550 versus random and 0.072256 versus brightness. Saved descriptive 95% percentile intervals for these sample-weighted paired means are [0.196203,0.282320] and [0.051306,0.093401], respectively. They are not claims of external superiority. Eight images have negative Grad-CAM AOPC: deletion increased the target probability on average. These cases remain in the results. [E12–E13]

The FR-I and FR-II mean CAM AOPCs are 0.430278 and 0.187986. Means for correctly and incorrectly classified records are 0.329636 and 0.058000. These are descriptive strata of the selected model, not independent experiments or evidence that one class is physically better explained. Different score margins, image properties and model decisions can affect deletion responses. Border-median deletion can also create OOD inputs; no corrected OOD study validates their distributional status. [E12, E19]

### 6.4 Stability, missingness and intensity noise

For the primary aligned top-10% comparison, mean Jaccard was approximately 0.891493 at −2° and 0.882067 at +2°, each with 116 available images. Horizontal and vertical flip means were approximately 0.676351 and 0.734696. These values indicate transformation-dependent explanation behaviour, not invariant physical understanding. [E14–E15]

Across 3,480 repeated transformed comparisons, 1,553 prediction flips were recorded and 1,181 comparisons were unavailable because the transformed CAM was degenerate. These are repeated comparisons from 116 images, not thousands of independent observations. The original predicted class remained the explanation target despite class flips. [E15]

Noise was particularly problematic. At both predefined noise levels, saved transformed predictions were FR-I for all 1,160 realizations per level. Each level had 760 flips relative to the original predictions. Approximately 49% of image pixels were clipped on average. The σ=0.005 and 0.01 image-level Jaccard summaries had only 78 and 46 available images, respectively, with additional within-image missingness. A higher available-case mean at the stronger noise level therefore cannot be interpreted as improved robustness. Nor can clipped display-intensity noise or brightness gains establish physically calibrated radio-flux robustness. [E15]

### 6.5 Emission proxies and representative figures

For the primary intensity mask, mean emission attribution mass was 0.090169, area-adjusted enrichment 0.078583 and top-10% proxy IoU 0.074599. The mask occupies a small area; enrichment and mass answer different questions. Much attribution lies outside this particular mask, and neither enrichment nor overlap validates localization of named astrophysical structures. Expert-mask physics alignment is **not available**. [E16–E18]

Representative selections were fixed by the lowest dataset index in each ground-truth-by-correctness group: 35 (correct FR-I), 31 (incorrect FR-I), 50 (correct FR-II) and 437 (incorrect FR-II). Their figures are in the saved `representatives/` directory referenced in the evidence map. Selection did not depend on attractive heatmaps. This draft does not independently inspect raw pixels or assert that highlighted regions are verified jets, lobes or cores. Individual figures require redistribution review before publication. [E17]

### 6.6 Completeness and practical limitations

Essential corrected-model explanation generation, primary deletion and implemented stability/proxy analyses are complete for 116 images. The broader frozen suite is not fully executed: secondary full deletion/AUC and alternative replacements were deferred for resources, as were corrected-checkpoint parameter-randomization runs. Native-grid stability and certain auxiliary random/brightness controls lack specified conventions. ECE and expert-mask physics alignment are unavailable, and historical OOD results are not corrected-model results. [E18–E19]

Saved corrected XAI execution took approximately 329.91 seconds on CPU, with a maximum recorded working-set measure of 427,483,136 bytes. These are measurements from that run, not guaranteed requirements or instantaneous peak bounds for every future environment. Cooperative resource checks cannot interrupt a blocking model operation immediately. [E22]

## 7. Conclusion and Future Work

RadioMorph AI demonstrates a reproducible corrected-development workflow for FR-I/FR-II classification and genuine gradient-based visual explanation. Its selected model achieved 87.068966% validation accuracy and 0.864391 macro F1. Primary deletion analysis showed a larger mean target-probability decrease for Grad-CAM ranking than for the specified random and brightness controls. These conclusions are restricted to the corrected cohort, selected checkpoint and implemented perturbation definition.

The integrity correction is scientifically important: preserving historical evidence while quarantining conflicting duplicates prevents unsupported relabelling and makes cohort changes explicit. Nevertheless, exact-content separation does not prove source independence. Validation-based selection, one training seed, absent independent corrected-model TEST results, strong noise sensitivity and incomplete explanation checks restrict interpretation. The system does not establish astronomical discovery, physical causation or calibrated external prediction.

Future work should first recover the official proposal/template and complete attribution and release permissions. Scientific extensions require separate approval: resolve conflicting-label provenance, assess source grouping, obtain independent reference annotations and external data, complete predefined randomization/sensitivity checks where appropriate, and define a compatible corrected-model OOD calibration/evaluation design before enabling public inference. Such work must preserve the original evidence and avoid choosing new thresholds or methods on protected evaluation outcomes. No future experiment is authorized by this report.

## 8. References

The following entries were checked against primary publisher/arXiv metadata. Preprint years are used explicitly where the final venue details were not audited. Full-text appraisal and the official proposal bibliography remain to be completed; see [reference verification](reference_verification.md).

1. Fanaroff, B. L., and Riley, J. M. (1974). The Morphology of Extragalactic Radio Sources of High and Low Luminosity. *Monthly Notices of the Royal Astronomical Society*, 167(1), 31P–36P. [Publisher](https://academic.oup.com/mnras/article/167/1/31P/2604898), [DOI](https://doi.org/10.1093/mnras/167.1.31P).
2. Porter, F. A. M., and Scaife, A. M. M. (2023). MiraBest: A Dataset of Morphologically Classified Radio Galaxies for Machine Learning. arXiv:2305.11108. [Primary record](https://arxiv.org/abs/2305.11108).
3. He, K., Zhang, X., Ren, S., and Sun, J. (2015). Deep Residual Learning for Image Recognition. arXiv:1512.03385. [Primary record](https://arxiv.org/abs/1512.03385).
4. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D. (2016). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. arXiv:1610.02391. [Primary record](https://arxiv.org/abs/1610.02391).
5. Samek, W., Binder, A., Montavon, G., Bach, S., and Müller, K.-R. (2015). Evaluating the visualization of what a Deep Neural Network has learned. arXiv:1509.06321. [Primary record](https://arxiv.org/abs/1509.06321).
6. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., and Kim, B. (2018). Sanity Checks for Saliency Maps. arXiv:1810.03292. [Primary record](https://arxiv.org/abs/1810.03292).
7. Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. arXiv:1706.04599. [Primary record](https://arxiv.org/abs/1706.04599).
8. MiraBest Batched Dataset. Zenodo record 4288837. [Dataset record](https://zenodo.org/records/4288837), DOI:10.5281/zenodo.4288837. **Creator/date/license metadata must be verified before final bibliography and redistribution; title/identifier were located, but direct retrieval timed out.**

Project-specific result citations [E01–E24] refer to the companion [evidence mapping table](evidence_mapping.md), not academic publications. Repository evidence is not substituted for external scientific validation.

## 9. Appendices

### Appendix A: Corrected identities and quarantined records

Corrected membership SHA-256: `4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91`.

Corrected checkpoint SHA-256: `d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c`.

| Global index | Historical development split | Label |
|---|---|---|
| 198 | VALIDATION | FR-I (0) |
| 323 | TRAIN | FR-II (1) |
| 474 | VALIDATION | FR-I (0) |
| 587 | VALIDATION | FR-II (1) |
| 694 | TRAIN | FR-II (1) |

These metadata do not resolve the true label of the identical content. No quarantined record was used in corrected training or validation.

### Appendix B: Evidence and reproducibility package

The companion documents enumerate every major local claim, verified headline result, unavailable analysis and missing reference. `verification.json` records artifact-only arithmetic, source-hash comparisons and before/after preservation checks. No tests, inference, training, dataset decoding or calibration were executed for report preparation. Historical completion status and test counts are attributed to saved records.

### Appendix C: Submission requirements

Confirm university layout, word limit, citation style, author details, actual contribution statement, approved objectives, supervisor information, required ethics/data-use declarations and permission for individual source figures. Do not submit placeholder approvals as facts. This report intentionally excludes any claim that TEST was run exactly once, that the corrected model is externally validated, or that Grad-CAM establishes causal astrophysical localization.
