# RadioMorph AI component migration report

## Scope and preservation

Only `components/04-buddhi-radio-galaxies/` is changed. The existing Unicode branch is preserved; no other component or shared group document is edited. Original project and backup are not migration destinations and remain untouched. This report documents implementation migration, not new scientific experimentation.

## Inclusion and exclusion

[Inventory](../research_evidence/migration_inventory.json) lists exact-copy evidence additions and directory decisions. Source, tests, frozen protocol/configuration and scientific definitions retain original bytes. Selected corrected membership metadata, development duplicate hashes, classification predictions/metrics, XAI summaries, methodology and training history are included. No raw data, original TEST image exports, trained models, CAM arrays or individual radio-image overlays are included. Notebooks are exploratory/scaffold files with no unique required implementation and no recorded outputs; they remain local. The full experiment archive is not claimed to be included.

The copied final-report `verification.json` is explicitly a distribution derivative: host worktree inventories are removed and its original SHA-256 is recorded. Scientific result values are not edited. Historical absolute paths in recorded commands remain historical evidence; executable source paths resolve from the component. Empty/private artifact references do not imply those files were published.

## Migration-related additions

- Component-scoped ignore rules prevent raw/serialized data, checkpoints, runtime files and caches from staging.
- Component-scoped Git attributes preserve provenance-significant bytes, including CRLF source files; no scientific code is normalized.
- An availability-only pytest hook marks tests needing excluded real data or private figures as blocked. Mathematical assertions and scientific endpoints are unchanged.
- README retains component ownership/group integration and replaces the obsolete placeholder-only status with actual migrated scope.
- Existing legacy packaging source remains solely for compatibility; no public deployment or publication is authorized.

## Saved result checks

[Result verification](../research_evidence/result_verification.json) reconstructs [[38,13],[2,63]], 101/116 accuracy and macro F1 from the original saved CSV, and averages existing per-image AOPCs. These checks are arithmetic on saved records, not inference. The source originals and scientific evidence remain unchanged. Corrected model calibration is still unavailable and the live gate is not bypassed.

## Newly executed checks

Actual environment, commands, outcomes, skips and error diagnostics are recorded in [verification](../research_evidence/migration_checks/verification.json), [pytest output](../research_evidence/migration_checks/pytest_output.txt) and [JUnit](../research_evidence/migration_checks/pytest.xml). Syntax compilation is in-memory, not experiment execution. Original-project test counts are not substituted for migration test results. A local-only smoke check, if completed, is recorded separately. No packages are installed or upgraded.

## Reproducibility gaps

The exact raw archive, historical split, selected weights and full saved execution records must be supplied through authorized local storage before real-data reproduction. Some historical documentation links deliberately point to researcher-held files not packaged here. They are missing-evidence references, not broken runnable import dependencies. Do not infer source-independent generalization, independent TEST performance, expert physics localization or completed broader XAI coverage. Publication rights and any future release remain separate decisions.

## Git boundary

Final staging must contain only this component. Git blob hashes must equal staged working-tree bytes and preserved source copies. The pushed commit must be verified against the existing remote branch using a normal non-force push. Final scope/integrity checks are recorded under `research_evidence/migration_checks/`; no other component may change.

## Initial test findings and narrow compatibility changes

The first migrated run recorded 824 passed, 14 skipped and 15 failures. Thirteen failures required intentionally excluded real artifacts; they are now explicit blocked integration checks, not asserted passes. Two tests in `tests/test_auto_radio.py` assumed the earlier Gradio Interface list API. Only those UI test adapters were updated to locate the existing Blocks controls and eight-output callback; alpha preservation, mode choices and rejection assertions remain. Scientific source modules and model behaviour were not changed. The initial output remains under `migration_checks/initial_*`; corrected availability is documented in the final run.

## Final verification outcomes

- 109 Python files compiled in memory successfully.
- Full migrated suite: **826 passed, 27 explicitly blocked/skipped, 0 failures, 0 errors** (853 collected). The final pytest invocation exited 0. Missing real-artifact checks are not reported as passes.
- Existing Python 3.12.10 / PyTorch 2.14.0 CPU environment used; `pip check` exited 0 with no broken requirements. No install/upgrade.
- Required model/data/XAI/inference/demo imports passed. Loopback-only Gradio HTTP smoke returned 200, displayed recorded metrics, rejected live analysis and stopped the server. No model checkpoint or raw data was loaded for that smoke.
- Saved metrics and aggregate assets match original evidence; all 41 protected originals and the original index/HEAD remain unchanged. No other group component file changed.
- No credential patterns or raw/model binaries found in migration candidates; every candidate is below 2 MiB. This is a bounded static audit, not proof that no conceivable secret exists. Historical academic attribution/provenance is retained.

Do not sum initial/final test counts as independent scientific observations. Synthetic tests can construct disposable random models and tiny toy training fixtures; no real research training, scientific inference, XAI experiment or deployment was performed.

Diagnostic exports omit host paths/hostnames and normalize transcript-only whitespace; they are documentary derivatives, not modified research outputs. Source and frozen documents retain their original bytes, including Markdown hard breaks and CRLF. The staged whitespace check permits intentional Markdown end-of-line spaces while checking blank EOF and spaces before tabs; no Git configuration is changed.
