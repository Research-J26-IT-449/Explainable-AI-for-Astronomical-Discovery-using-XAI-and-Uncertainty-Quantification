# Step 14B real exact-duplicate audit

Execution/integrity verdict: **PASS**. This is not a duplicate-free dataset verdict.

Started preflight: 2026-10-19T22:10:05.596521+05:30. Branch: `feature/external-radio-input`; HEAD: `2d6e5883e4a6a16252efdb24e5e4f52f61aca026`.

## Exact command

```powershell
C:\Users\User\Desktop\research\J26-IT-449-IT23308602-AGN-XAI\.venv\Scripts\python.exe -B scripts/run_duplicate_audit_train_val.py --execute-real-audit --project-root C:\Users\User\Desktop\research\J26-IT-449-IT23308602-AGN-XAI --output-dir C:\Users\User\Desktop\research\J26-IT-449-IT23308602-AGN-XAI\analysis_reports\step14b_real_duplicate_audit\run_20261019_221005\results
```

## Findings

Exactly one invocation completed: 554 unique TRAIN and 119 unique VALIDATION images; 669 distinct content hashes. All 85 primary batches completed in ascending index order (84 x 8, then 1). Four two-image byte-confirmation batches added 2 TRAIN and 6 VALIDATION materializations: 681 total materializations, 673 unique images. No missing indices, incomplete batches, stderr, or rejected hash collisions.

| Category | Groups | Pairs |
|---|---:|---:|
| within_train | 1 | 1 |
| within_validation | 1 | 3 |
| cross_train_validation | 1 | 6 |
| conflicting_labels | 1 | 6 |

There is one overall group, five affected images, and ten overall duplicate pairs. Group categories overlap; label-conflict pairs also overlap split categories. Six pairs have conflicting labels.

| Index | Split | Label |
|---:|---|---:|
| 198 | validation | 0 |
| 323 | train | 1 |
| 474 | validation | 0 |
| 587 | validation | 1 |
| 694 | train | 1 |

All five images were byte-confirmed equal. No cause for their identical content is established by this audit. No dataset/split repairs were attempted.

## Provenance and integrity

- archive SHA-256: `505e0b0761a76364a417695380d32d9f26cca7bc0a2608cad46ab338fa46a6b8`
- split SHA-256: `c4a548a66dd37505b8d663bfe6826258364b8118afde4928f5f83a6cdd263008`
- checkpoint SHA-256: `d5e1f126c13c2b7bebb95e35f41f0bc6397dd0ea9f4fbfcfbc55648609ab5bc7`
- Archive MD5: `fa9d6ad8fdde95b0a19df11433e8bca3`
- Audit source SHA-256: `353d307bd4eaaad19017946da4b468981130031578986c041660c863c7831c88`
- Bounded reader SHA-256: `981da0039a69d0e743093cd3122a07b0b873f0984274f21d19656fc7ef06a9f8`
- All 41 protected artifacts matched the saved protected manifest before and after execution.
- Existing source/output/unrelated file snapshots, Git index bytes, HEAD, branch, and status were unchanged. No staging/commit/push.
- Output checksums and all 24 integrity checks passed; see integrity_report.json.

## Runtime, resources and limits

Elapsed invocation time: 36.766 seconds. Maximum retained decoded batch payload: 180,000 bytes (8 x 22,500); reader individual pixel buffer maximum: 22,500 bytes.
Observed directly launched-process peak working set: 4554752 bytes. Only the directly launched process was sampled. Windows virtual-environment launchers can spawn an interpreter child; full process-tree peak memory was not established and must not be inferred from this measurement. No rerun performed.

## Safety and scientific limits

- Exact decoded-image equality only; no near-duplicate or astronomical source-independence conclusion.
- TEST metadata and opaque archive checksums were inspected; reader instrumentation supports zero TEST image extraction/materialization, not an OS-level independent access trace.
- No model checkpoint was deserialized or used for inference.
- Zero rejected collisions is derived from successful completion of mandatory byte confirmation; collision mismatch would abort the audit.
- TEST pixel materialization/extraction counters were zero in all 89 extraction calls. This is code/instrumentation evidence, not an independent operating-system access trace.
- No training, classifier inference, XAI, or checkpoint deserialization occurred. Only this authorized duplicate audit ran.
- Exact duplicates and conflicting labels require separate scientific review; do not infer astronomical source identity or general independence.

## Evidence files

- preflight.json: command, environment, frozen pins, split counts and original file snapshots.
- stdout.log / stderr.log: exact captured audit console streams.
- execution_telemetry.json: single-invocation exit code, duration and memory measurement.
- results/image_hashes.csv: all 673 image metadata/hash records.
- results/duplicate_groups.json: byte-confirmed duplicate members.
- results/audit_report.json: full completion report, batch instrumentation and two artifact checksums.
- integrity_report.json: post-execution verification, counts and all three result-file checksums.
