# Step 20 approved schema fix and verification

Root cause: the new script expected name-to-index mapping, while the frozen manifest correctly uses index-string-to-name mapping. Only that assertion was corrected to {"0": "FR-I", "1": "FR-II"}; reversing the single substitution reproduces the original script SHA-256. No scientific convention or manifest changed.

Seven synthetic AST-based schema checks passed (0 failed/skipped/errors, 0.07 seconds). They execute the actual assertion using synthetic mappings without importing the evaluator or accessing checkpoint/dataset data. The real preflight passed.

The first attempt reports remain byte-identical. This fresh sibling attempt directory contains an exact copy of the corrected script and independent outputs; no existing reports or results were overwritten.

Exactly one complete inference pass evaluated the 116 authorized validation rows. Post-execution CSV checks independently reconstruct [[38,13],[2,63]], verify all probabilities, argmax, membership order, supports and correctness flags. Completion checksums, original archive, manifest, checkpoint, all 41 protected artifacts, prior snapshotted files and Git state passed integrity checks. Verification used saved predictions only, not further model execution.

FR-I has more errors both in count (13 versus 2) and rate (25.49% versus 3.08%). Its precision is higher (95.00% versus 82.89%), but recall lower (74.51% versus 96.92%): more FR-I cases are assigned FR-II. These are observed development-cohort patterns, not diagnosed physical causes. The larger FR-II class raises support-weighted emphasis relative to equal-class macro summaries.

See evaluation_report.md and metrics.json for the genuine classification results and development-validation limitations.
