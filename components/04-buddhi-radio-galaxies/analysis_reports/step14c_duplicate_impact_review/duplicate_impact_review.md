# Step 14C ? Duplicate root-cause and research impact review

Date: 2026-10-20 (Asia/Colombo client date). Repository branch: `feature/external-radio-input`; HEAD: `2d6e5883e4a6a16252efdb24e5e4f52f61aca026`.

**READY FOR RESEARCHER DECISION.** Evidence supports a dataset limitation requiring an explicit research decision, not an implementation repair. Recommendation: pause additional XAI execution pending acknowledgment; preserve all frozen artifacts. No further run is authorized by this review.

## Confirmed findings and index interpretation

Saved Step 14B outputs and their checksums agree: 673 TRAIN/VALIDATION records, 669 unique hashes, one byte-confirmed five-record group, ten duplicate pairs and six conflicting-label pairs. Its content SHA-256 is `f2c580e4e323dd2999048b32cbc58c0d3826a853a09f24fb320eee81d4cf5151`. These are global, zero-based dataset indices, not split-local positions.

The reader orders eight `data_batch_1`?`data_batch_8` members of 88 rows; global index = 88 ? (batch number ? 1) + member row. Each listed index matches that formula. `TransformedSubset.__getitem__` maps a split-local position through the saved global index list. Archive member naming is not the frozen research split definition.

| Global index | Frozen split | Label | Archive member / zero-based row | Saved original filename |
|---:|---|---:|---|---|
| 198 | validation | 0 | `batches/data_batch_3` / 22 | `BestCat_proc/Cropped/100_036.061_+000.563.png` |
| 323 | train | 1 | `batches/data_batch_4` / 59 | `BestCat_proc/Cropped/200_020.485_-000.490.png` |
| 474 | validation | 0 | `batches/data_batch_6` / 34 | `BestCat_proc/Cropped/100_016.732_+000.523.png` |
| 587 | validation | 1 | `batches/data_batch_7` / 59 | `BestCat_proc/Cropped/200_033.655_+000.710.png` |
| 694 | train | 1 | `batches/data_batch_8` / 78 | `BestCat_proc/Cropped/200_035.137_+000.468.png` |

Label mapping is FR-I=0, FR-II=1 in `config.yaml`, `src/data/split_dataset.py`, saved dataset-validation metadata and both XAI run manifests. The two TRAIN members are FR-II; VALIDATION contains two FR-I and one FR-II. Cross-split pairs comprise two same-label and four conflicting-label pairs. Within-VALIDATION has three pairs, two with conflicting labels; within-TRAIN has one same-label pair. Category group counts overlap and must not be added.

## Root-cause evidence versus hypotheses

Confirmed mechanism: `stratified_split_indices()` stratifies individual indices by label using `train_test_split`; it neither hashes images nor groups duplicates. Disjoint indices therefore do not imply disjoint content. The loader reads archive label lists and preserves row order; the bounded reader associates the saved labels/filenames with those rows. Distinct saved filenames with coordinate-like fields do not establish distinct images or verified astronomical sources. Existing checksums establish identity with the pinned artifact, not scientific label correctness.

Not established: why the archived rows have identical bytes; whether upstream export/cropping, blank or placeholder content, catalog association, label assignment or another process caused it; which label is correct; whether any filename denotes the same astrophysical object. No pixel statistics or visual review was performed. Nothing here establishes a reader bug or rules out every possible parsing defect independently: byte confirmations reuse the same pinned reader. Current evidence does not locate the earliest upstream stage of duplication or prove local archive corruption. Archive provenance is inherited from the verified Step 14B run; this review did not reopen the archive.

## Prior XAI sample overlap

Run `323bb0f19f5b8768`: [31, 50].
Run `73452038829db01e`: [35, 37, 38, 51, 52, 53, 55, 94].

Combined ten indices: [31, 35, 37, 38, 50, 51, 52, 53, 55, 94]. Intersection with the duplicate group: **empty**. Each namespace's image-record filenames match its manifest. This checks index overlap, not a new full cross-run integrity audit. The three affected VALIDATION records remain among the 109 unevaluated validation indices. The ten engineering records retain their engineering purpose; their model was trained/selected using the affected frozen cohort, so no-overlap does not remove indirect research limitations. This group does not establish a cause for index 94 OOD rejection, noise flips, clipping or degenerate CAMs.

## Scientific impact

- Validation contains TRAIN-seen image content: three affected records (3/119 = 2.52%). TRAIN repeats the same content twice (2/554 records). Derived from the single saved group, TRAIN has 553 distinct hashes and VALIDATION 117; together they have 669. Repeated content violates a strict independent-image interpretation, including across ground-truth class strata.
- Accuracy may be helped by same-label leakage or harmed by conflicting labels. Neither direction nor the actual numerical effect is proven. With unchanged deterministic evaluation and identical input arrays, the three validation copies receive identical logits/predictions. With labels 0,0,1, one or two must be wrong for a binary prediction. This is a logical implication, not measured per-image performance. The three records' total accuracy weight is 3/119; that is not an estimate of retraining or model-selection bias.
- `src/models/train.py` chooses the lowest validation-loss checkpoint, schedules on validation loss and applies early stopping. Affected observations can influence these decisions; removing them post hoc does not undo historical selection. Saved ResNet metadata reports best epoch 22, loss 0.35014713111043977 and accuracy 0.8571428571428571. These remain historical measurements of the frozen cohort, not corrected estimates. No revised accuracy was calculated.
- Future full-cohort XAI results would weight identical content repeatedly and assign it to conflicting ground-truth strata. Image-level bootstrap/sign-flip procedures cannot automatically establish independence here. Fixed predicted-class targeting remains mathematically defined, but ground-truth correctness strata can differ for identical inputs. Duplicate-aware summaries would be new sensitivity analyses requiring a separately documented decision, not a silent frozen-protocol change.
- XAI plausibility does not validate labels, generalization, faithfulness or astrophysical structure. No conclusion about locked TEST content, duplicate overlap, or historical TEST bias is available: none was investigated.

## Resolution options

| Option | Defensible use | Required limitations / cost |
|---|---|---|
| Preserve frozen dataset/split/model and disclose findings | Historical reproducibility and conditional exploratory characterization of this exact frozen model/cohort | Preserve all 119 records and endpoints if continuing unchanged; prominently report leakage, conflicting labels, repeated content and independence limitations. Do not call it duplicate-free independent validation or silently amend inference. New execution needs separate approval. |
| Separately version a corrected future dataset protocol | Prospective cleaner generalization and XAI research | Researcher-approved upstream provenance/label adjudication first; never infer truth from majority label or drop an arbitrary copy. Define grouping, label-conflict/quarantine rules and near-duplicate/source-identity scope before selection. Create new dataset/split/provenance identifiers and future model-selection/training plan without overwriting locked research artifacts. Any TEST access requires its own explicit governance; it is not authorized here. |

## Minimum recommended next action and XAI proceed decision

Acknowledge the finding and approve a written interpretation of the intended research claim before spending on more XAI evaluation. Preserve Step 14B evidence and existing engineering records. No immediate split edit, relabeling, retraining, checkpoint replacement or rerun is warranted by metadata alone.

Remaining XAI evaluation **can technically proceed under the unchanged frozen protocol only as disclosed, conditional exploratory characterization**, following an explicit researcher decision and separate bounded execution authorization. It cannot resolve leakage or support a claim of independent, duplicate-free scientific validation. If independent validation is required for the research objective, defer that claim and plan a separately versioned corrected protocol. Existing image-level uncertainty assumptions must not be presented as verified after this finding. This review recommends a decision hold, not an automatic continuation or an assertion that the software is blocked.

## Evidence references and provenance

All paths below are repository-relative:

- `analysis_reports/step14b_real_duplicate_audit/run_20261019_221005/results/audit_report.json`: complete cohort, member provenance, labels, batch counters and byte-confirmation status.
- Same directory: `duplicate_groups.json`, `image_hashes.csv`; parent directory: `integrity_report.json`, `preflight.json`, `audit_summary.md`. All three saved result SHA-256 values were rechecked successfully against the integrity report.
- `data/splits/split_indices.json`: saved TRAIN/VALIDATION memberships; current SHA-256 `c4a548a66dd37505b8d663bfe6826258364b8118afde4928f5f83a6cdd263008` matches audit provenance. JSON parsed for TRAIN/VALIDATION lookup only; no TEST image/data inspection.
- `config.yaml`, `outputs/metrics/dataset_validation.json`: class mapping and dataset source. Dataset source recorded as MiraBest Zenodo v1 basic binary, Zenodo record 4288837; no external retrieval performed.
- `src/data/mirabest_pickle_stream_reader.py`: `_member_start`, row extraction and label/filename pairing; `scripts/run_duplicate_audit_train_val.py`: canonical hashes and pairwise byte confirmation.
- `src/data/download_mirabest.py`: archive row/label loading order; `src/data/split_dataset.py`: global index views and index-level stratification; `src/data/preprocess.py`: unchanged deterministic evaluation transform.
- `src/models/train.py`, `outputs/metrics/resnet_validation_metrics.json`: checkpoint selection and historical validation metadata.
- `outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/run.json` and `73452038829db01e/run.json`: prior evaluated indices and unchanged class mapping.

Saved archive SHA-256: `505e0b0761a76364a417695380d32d9f26cca7bc0a2608cad46ab338fa46a6b8`; MD5: `fa9d6ad8fdde95b0a19df11433e8bca3`. These are prior verified identities, not a new raw-archive read. Current checkpoint SHA-256 matches saved audit identity `d5e1f126c13c2b7bebb95e35f41f0bc6397dd0ea9f4fbfcfbc55648609ab5bc7` via binary hashing only; no checkpoint deserialization/inference.

## Safety and remaining uncertainties

No dataset image was opened, decoded or materialized; no TEST image/data access; no audit, model, training, XAI or test execution. No frozen split, label, checkpoint, source, protocol or prior evidence changed. All 41 protected artifacts were binary-hash checked and matched the existing protected manifest. Existing-file snapshot and Git index/HEAD/status checks accompany report creation; only this new report is written. Upstream root cause and label truth remain unresolved; a future separately authorized provenance review is required before any correction.
