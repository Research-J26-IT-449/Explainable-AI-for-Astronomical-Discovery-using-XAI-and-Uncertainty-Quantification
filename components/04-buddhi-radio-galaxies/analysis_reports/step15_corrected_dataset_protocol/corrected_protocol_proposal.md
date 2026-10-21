# Step 15: Corrected MiraBest dataset protocol proposal

**Status: DRAFT FOR RESEARCHER DECISION ? NOT APPROVED OR IMPLEMENTED.**
Date: 2026-10-20, Asia/Colombo (client date). Branch: `feature/external-radio-input`; implementation reference: `2d6e5883e4a6a16252efdb24e5e4f52f61aca026`.

## 1. Evidence summary

**Confirmed:** Step 14B completed one exact-content audit of 554 TRAIN and 119 VALIDATION records. Its three result checksums were verified against the saved integrity report for this proposal. The 673 records contain 669 unique content hashes and one five-record, byte-confirmed group:

| Global index | Original split | Original label |
|---:|---|---|
| 198 | VALIDATION | FR-I (0) |
| 323 | TRAIN | FR-II (1) |
| 474 | VALIDATION | FR-I (0) |
| 587 | VALIDATION | FR-II (1) |
| 694 | TRAIN | FR-II (1) |

The group has ten pairs: one within TRAIN, three within VALIDATION, six cross-split; six pairs have conflicting labels. Pair categories for label conflict overlap split categories. Content hash: `f2c580e4e323dd2999048b32cbc58c0d3826a853a09f24fb320eee81d4cf5151`.

Indices are global, zero-based positions in the pinned dataset, not split-local positions. Saved member/row/filename provenance is in Step 14B; no image decoding was repeated. Step 14C established no overlap with the ten completed Phase 3B indices `[31,35,37,38,50,51,52,53,55,94]`.

The historical frozen split also includes a locked 119-record TEST partition. That count is supplied historical context, not a cohort inspected or used for design here. Only TRAIN and VALIDATION arrays were selectively parsed from the split manifest; the TEST array, its labels and pixels were not inspected. No raw dataset/archive or model checkpoint was opened.

The execution registry predates the completed audit: its statements that the full duplicate audit is pending and the entry point is synthetic-only are historical, superseded by Step 14B/14C. The registry is preserved unchanged; its old claims are not current execution status. Existing historical provenance gaps and exploratory XAI limitations remain relevant.

## 2. Scientific problem definition

Disjoint record indices did not prevent identical content from appearing across development splits. Index-level label stratification does not group identical images. Two TRAIN examples share the same content and label; validation contains that content under both classes. Why the bytes are identical and which labels are correct remain unresolved.

A corrected development protocol must prevent cross-split exact-content overlap, explicitly quarantine or adjudicate label conflicts, produce deterministic membership without performance-based choices, and preserve all original bytes and historical metrics. It must never infer label truth from majority vote, a model prediction, filename prefix, heatmap plausibility or an arbitrary representative. Checksums establish identity, not astronomical correctness.

Scope is exact decoded-content equality within the already audited TRAIN/VALIDATION universe. No assertion about TEST, near duplicates, astronomical source independence or external-survey validity follows.

## 3. Comparison of correction strategies

Counts below are prospective metadata arithmetic, not created splits or experimental outcomes. Original development class counts verified from the saved audit are TRAIN 245/309 and VALIDATION 53/66 (FR-I/FR-II).

| Strategy | Membership and class balance | Model selection / reproducibility | Comparability and future execution |
|---|---|---|---|
| A. Exclude all five | Keep original assignments of all others: TRAIN 552 = 245 FR-I + 307 FR-II; VALIDATION 116 = 51 FR-I + 65 FR-II; total 668 unique hashes. No replacement samples or rebalancing. | Fully deterministic and no label adjudication guess. Small reduction in data; potentially valid records are quarantined along with questionable ones. Selection must be repeated for a future corrected model. | Smallest transparent change; cohorts still differ from history. Requires separately approved new training/selection/evaluation for corrected-model claims. |
| B. Retain one after provenance resolution | Total 669 retained records. Keeping an originally TRAIN representative gives TRAIN 553 / VALIDATION 116; keeping an originally VALIDATION representative gives 552 / 117. Class counts depend on its adjudicated label. | Blocked until both content suitability and label provenance are resolved. A deterministic representative rule must be frozen beforehand; choice cannot depend on model/heatmap performance. | Saves one example relative to A at greater adjudication cost. Same future training/selection requirement; not a justification for automatic relabeling. |
| C. Content-group-aware splitting | Exclude all unresolved conflict groups or retain adjudicated groups in one split only. Counts vary if assignments are regenerated. With the current exclusions and no reassignment, C reduces to A. | Strong general policy: every content group must stay within one split. A new group-stratified split requires fixed grouping, target-count/rounding/optimization rules and seed before use; none should be improvised now. | Useful for expanded datasets and later resolved groups. Repartitioning adds historical comparability changes and requires new training and evaluation. |

For B, lowest global index after adjudication would be a possible deterministic representative rule (198, originally VALIDATION), not a scientific quality criterion. It is not recommended or adopted. A correct label alone would not establish that the shared content is suitable for morphology research.

## 4. Recommended strategy and justification

**Proposed decision: A, with C's content-disjointness invariant.** Quarantine the entire unresolved group, change no retained labels and preserve all remaining TRAIN/VALIDATION memberships. This avoids choosing a preferred class or representative and avoids a new split search after seeing historical results. The five records are excluded from future eligible manifests, not erased from the archive. Quarantine is a conservative eligibility decision, not a claim that every record is corrupt or mislabeled.

Proposed protocol name: **MiraBest-Development-ExactDedup-v1.0.0**. Lifecycle: `proposed`, `execution_authorized: false`. It is a separate development-cohort protocol, not an amendment to the frozen Phase3B-XAI-v1.0.0 document. Reintroduction or further exclusions require a new reviewed manifest/protocol version, not mutable v1 membership.

## 5. Proposed corrected split rules

1. Require exact pinned archive, parent split, saved audit result and reader/audit-source identities. Use the saved 673-record audit as the sole candidate universe; do not use a whole-dataset loader or TEST data for design.
2. Verify parent TRAIN and VALIDATION sets are unique and disjoint and match audit membership/metadata. Obtain only those two lists. Other parent fields are opaque and not copied into corrected manifests.
3. Define quarantine Q = `{198,323,474,587,694}` and require that it equals the complete saved conflicting-content group. Never add an unrelated record to restore old counts.
4. Corrected TRAIN = original TRAIN minus Q; corrected VALIDATION = original VALIDATION minus Q. Serialize each list in ascending global-index order. No random draw, reshuffle, balancing, sample weighting change or new selection seed is required for this membership rule.
5. Group all retained records by the saved canonical image-content SHA-256. Require every retained group to be a singleton for this specific v1 cohort and require no content hash across the two splits. An unexpected extra duplicate/conflict is a failure requiring review, not automatic exclusion or reassignment.
6. Expected counts are exactly 552 TRAIN and 116 VALIDATION, 668 retained records/hashes; quarantine 5. Expected class counts: TRAIN 245/307; VALIDATION 51/65. FR-I proportions change from 44.22% to 44.38% in TRAIN, and 44.54% to 43.97% in VALIDATION. These small shifts do not warrant unapproved balancing.
7. Group ID is the canonical content hash, never a label-containing identity. Preserve original filename and candidate filename-coordinate keys only as unverified metadata; do not infer units or source matching thresholds.
8. The corrected protocol contains no new TEST partition and makes no TEST cleanliness claim. No TEST list, class distribution, pixels or predictions may inform correction or tuning. Future TEST governance requires independent explicit approval; existing TEST is historically evaluated and must not be called a newly untouched holdout.

## 6. Required provenance and acceptance tests

### Proposed manifest format (future files only)

Use a canonical UTF-8 JSON manifest with sorted object keys, ascending-index record arrays, compact separators, no NaN/Infinity, a final LF, and a pinned schema identifier `MiraBest-development-exactdedup-manifest-v1`. Canonical byte SHA-256 is the manifest identity; store it in a separate checksum sidecar to avoid self-hashing. Numeric labels are integer 0/1; hashes are lowercase hexadecimal. A human-readable report can accompany it; no executable serialization or image arrays belong in the manifest.

Suggested future namespace: `data/manifests/mirabest_development_exactdedup_v1.0.0/`, with `manifest.json`, `manifest.sha256`, `quarantine.json` and `README.md`. This proposal creates none of these. The manifest contains only permitted development records, not TEST entries. Quarantine JSON is canonical/checksummed separately and referenced by its hash. Operational approval/execution receipts remain separate from deterministic scientific manifests.

Header fields: protocol/schema versions and lifecycle; purpose and scope; parent dataset/archive reference, SHA-256 and MD5; parent split reference/hash; audit result and per-file hashes; canonical content-hash definition; reader/audit implementation hashes; source Git commit plus dirty-worktree source hashes; exact exclusion rule; expected split/class counts; class mapping; retained record list; quarantine reference/hash; preprocessing reference/hash for future execution; verified versus inherited provenance flags; creation tool/environment/command/time and explicit approval reference when actually available. Use `not_recorded` for absent approval/timestamp data, not invented history. Separate execution metadata if it would otherwise make deterministic membership serialization differ.

Record fields: global `dataset_index`, `original_split`, `corrected_split`, unchanged `ground_truth_label`, canonical content hash, shape `[150,150]`, dtype `uint8`, archive member/row, original filename and optional unverified candidate source key. Quarantine records additionally preserve original metadata, full group membership and reason `unresolved_conflicting_label_exact_content_group`; no replacement/corrected label is assigned.

The parent image hash is SHA-256 over `AGN-XAI-RAW-IMAGE-SHA256-v1\0`, dtype marker `<u1\0`, big-endian shape dimensions and 22,500 C-order uint8 content bytes. Pin the exact existing implementation/hash rather than approximate this in a new reader. Existing byte equality confirmation is inherited evidence; no claim of a second independent decoder verification is made.

### Required future acceptance tests (not run now)

- Synthetic unit tests for exclusions, inherited labels, set membership, sort order, duplicate IDs, absent metadata, conflicting groups, invalid dimensions/dtype, malformed/mismatched hashes and deterministic canonical serialization.
- Metadata-only derivation must yield the exact sets, counts and 245/307 and 51/65 composition above, with no dropped or added retained indices. Same inputs reproduce byte-identical membership manifests across repeated calls/environments; operational timestamps remain separate.
- No retained hash repeats within or across splits; all five quarantine members are excluded completely. Group identities omit labels. A hash-candidate mismatch or incomplete audit result fails closed.
- Parent audit completion/status, file checksums, archived source identities and saved pair confirmations must verify before metadata reuse. A future authorized decoder path must match the existing pinned representation; checksum verification is not permission for pixel decoding.
- Explicit closed development allowlist rejects any index outside the saved 673-record universe before loading; retained interfaces additionally reject the quarantined five. Synthetic forbidden-row sentinels prove rejection before pixel materialization, batch bounds, no bulk unpickle and release between batches. Runtime approval gates remain mandatory.
- No TEST records, labels or pixels are needed for manifest generation or synthetic tests. Do not relax data isolation because the cohort is corrected. Any later audit/decoding requires separate authorization.
- Preserve raw archive, parent split, old checkpoints, calibration, metrics, protocol and approvals byte-for-byte; all future outputs use new no-overwrite namespaces and checked provenance. Test that old/new model, split, calibration and result identities cannot be silently mixed.
- Acceptance includes a human provenance review and recorded researcher approval. Passing software tests alone does not adjudicate label truth, source independence or scientific validity.

## 7. Impact on existing model and XAI findings

Keep frozen model/checkpoint, historical metrics and all original evidence exactly as recorded. Label them `original_frozen_554_119`, with the known duplicate/conflict limitation. No corrected historical accuracy or repaired model claim can be produced merely by deleting validation rows: the old model trained on the affected data and validation drove checkpoint choice, scheduling and early stopping.

A future corrected model must be trained/selected anew under separately approved conditions. For a controlled comparison, propose the same selected architecture, original hyperparameters/preprocessing and seed 42, but fresh initialization, new weights and new validation-based selection; freeze those training choices before execution. Training sample count changes and stochastic optimization changes prevent attribution of a score difference solely to one mechanism. Repeated-seed investigation, if wanted, needs its own prespecified scope; none is authorized.

Old-model scoring on 116 retained validation records, if ever authorized, is only a historical-model subset sensitivity analysis, not leakage-free retraining or fresh validation. The validation cohort has already informed development; even a corrected cohort is internal development validation, not an independently untouched benchmark. Do not compare historical 119 and corrected 116 scores as a paired same-cohort performance test.

The ten existing XAI records remain original-model engineering evidence, not corrected-model results, despite no direct overlap. Do not pool old-model and new-model records. A corrected XAI study needs a separately versioned evaluation protocol/allowlist, new model-bound output namespace and planned statistics with explicit metric availability. Existing APIs enforcing 119 validation records/provenance cannot simply be pointed at 116; a future scoped implementation proposal and synthetic tests are required. Do not mutate the frozen Phase 3B protocol to make it fit.

If OOD diagnostics are part of the corrected study, old calibration remains valid only for its original model/split provenance. A new checkpoint/cohort requires a separately approved new calibration artifact: corrected TRAIN feature statistics and corrected VALIDATION thresholds only, no TEST. Never overwrite or silently reuse the old calibration with new weights.

All four unavailable XAI analyses, intensity-proxy limits, noise clipping/flips, degenerate explanations, negative AOPC effects and original-input OOD reporting gaps remain visible; dataset correction is not evidence that those mechanisms are resolved. Statistical independence beyond exact content remains unverified.

## 8. Future execution requirements

Order, each subject to explicit approval:

1. Researcher selects strategy and accepts scope/limitations, or requests provenance adjudication before choosing. This proposal itself grants no approval.
2. Authorize metadata-only manifest implementation plus synthetic acceptance tests and a narrow source/file list; freeze the approved membership/provenance document separately. Do not regenerate original splits or rerun the duplicate audit as a hidden prerequisite.
3. Review and approve the new manifests, source changes, data isolation and safe reader behavior; preserve all parent evidence. Unexplained metadata/source mismatch stops preparation.
4. Separately authorize bounded corrected TRAIN/VALIDATION loading and new training/model selection with new checkpoint/results paths; record exact commands, software, seeds, model-state and artifact hashes. No TEST evaluation is included.
5. Independently authorize any new calibration and corrected-model validation evaluation with predetermined endpoints and model-selection policy. Do not tune protocol choices after seeing preferred corrected scores.
6. Freeze a new XAI evaluation protocol bound to the corrected cohort/checkpoint, approve synthetic integration and a bounded pilot, then decide separately on larger execution. No transfer of historical receipts/approvals.
7. Report original and corrected studies separately, their denominators/selection histories, and any explicitly prespecified common-cohort sensitivity comparison. Retain unavailable statuses and avoid new confirmatory claims from post hoc comparisons.

No training command or real-execution approval receipt is issued by this proposal. Future compute/runtime requirements require separate estimation; the duplicate-audit duration is not a training-time estimate.

## 9. Remaining risks and limitations

Upstream content duplication and label correctness are unresolved. Quarantine may remove legitimate morphology examples; keeping one may retain a placeholder or invalid image. Exact hashing misses near duplicates and multiple observations of the same source. Filename fields are not validated astronomical identities. The saved byte-confirmation procedure shares its reader with hashing, so independent upstream/decoder validation is still absent. Existing provenance records do not prove every historical command or model-selection attempt.

The corrected development cohort cannot establish TEST cleanliness, source-independent generalization or external-survey performance. Prior exposure to validation and historical TEST results cannot be undone by a new filename/version. Any future untouched external holdout requires separate prospective governance, not retrospective claims. No numerical improvement, fairness/balance benefit, XAI faithfulness gain or causal mechanism is established here.

## 10. Explicit researcher approval checkpoint

**Final verdict: READY FOR RESEARCHER PROTOCOL DECISION.**

Requested decision: approve or revise proposed **MiraBest-Development-ExactDedup-v1.0.0**, Strategy A plus group-disjointness acceptance, with exactly 552 TRAIN and 116 VALIDATION records, full five-record quarantine, unchanged retained labels/memberships and no TEST use. Approval of methodology must be distinguished from approval of later implementation, data loading, retraining, evaluation, calibration or XAI execution. The researcher must explicitly authorize each future execution scope.

Until then: no corrected manifest is created, no model is changed, and no remaining real XAI evaluation is automatically resumed.

### Evidence and safety record

Primary references (repository-relative):

- `analysis_reports/step14b_real_duplicate_audit/run_20261019_221005/results/audit_report.json`, `image_hashes.csv`, `duplicate_groups.json`; parent `preflight.json`, `integrity_report.json`, `audit_summary.md`.
- `analysis_reports/step14c_duplicate_impact_review/duplicate_impact_review.md`.
- `data/splits/split_indices.json` (TRAIN/VALIDATION arrays only), `config.yaml`, `outputs/metrics/dataset_validation.json` (development provenance/class mapping; no TEST evaluation record consulted).
- `docs/research_execution_registry.md` (historical provenance/authorization context, not a current audit-completion ledger).

Inherited frozen provenance: archive SHA-256 `505e0b0761a76364a417695380d32d9f26cca7bc0a2608cad46ab338fa46a6b8`, MD5 `fa9d6ad8fdde95b0a19df11433e8bca3`; parent split SHA-256 `c4a548a66dd37505b8d663bfe6826258364b8118afde4928f5f83a6cdd263008`; selected historical checkpoint SHA-256 `d5e1f126c13c2b7bebb95e35f41f0bc6397dd0ea9f4fbfcfbc55648609ab5bc7`. Archive/checkpoint identities are inherited, not newly read for this step. Source/audit and report hashes remain in saved Step 14B provenance. Current parent split hash and saved audit output hashes verified; no raw dataset/archive, checkpoint or TEST record was opened.

Only this planning document is created. Original evidence/source/configuration and Git index/HEAD were checked unchanged across document creation. No image decoding, tests, duplicate audit, model training/inference, XAI, staging, commit, push, reset, clean or stash operation occurred.
