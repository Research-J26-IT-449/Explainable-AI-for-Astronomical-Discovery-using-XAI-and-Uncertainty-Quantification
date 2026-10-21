# Step24 ? Final results-only demo readiness

**Verdict: PUBLIC RESEARCH DEMO PARTIALLY READY.**

## Delivered presentation

The existing RadioMorph AI design, navigation, layout and CSS are preserved.
Live upload/mode/analyze controls are disabled; obsolete active-analysis wording
was replaced. Public callbacks reject direct calls with UNSUPPORTED and no scores
or maps. There is no second inference path. Lazy research imports allow the
public package to run without PyTorch, private modules, model weights or datasets.
The public entry forces safe mode even when private flags request otherwise.

Displayed saved outputs: corrected development model information; validation
accuracy87.068966%, macroF1 0.864391, cohort116; confusion matrix[[38,13],[2,63]];
116 explanations generated; mean AOPC30 CAM0.294510, random0.053960,
brightness0.222254, plus scientific limitations. AOPC is a comparison table of
saved scores, not newly calculated research metrics. The confusion-matrix PNG
is a byte-identical copy of the genuine Step20 aggregate figure. Public JSON
extracts genuine Step20/21 recorded values; provenance.json records source hashes.
No metrics, explanations, annotations or research figures were recomputed/invented.

MiraBest source-image/overlay redistribution rights are not established by local
evidence. Accordingly, those images are excluded from the deployable package;
public gallery text explains why. A private opt-in local gallery can display the
four existing saved representative figures (35,31,50,437), checked by original
saved hashes, without generating CAMs. The public entry always excludes that
gallery. No dataset license is inferred from citation or public availability.

## Launch and deployment

Safe local public preview from repository root:

```powershell
.venv\Scripts\python.exe -B public_demo/app.py
```

Open http://127.0.0.1:7860. Existing demo/app.py also defaults to results-only mode.
Private local examples (never expose this mode publicly): set
RADIOMORPH_PUBLIC_DEMO=0 and RADIOMORPH_SHOW_LOCAL_EXAMPLES=1, then launch
`.venv\Scripts\python.exe -B demo/app.py`. This only displays saved examples;
the UI input controls remain disabled and the original inference gate is intact.

Final export: public_bundle_final/ under this report directory. Exactly12 reviewed
files: public entry, README, requirements, Dockerfile, bundle manifest; four demo
source/CSS files; aggregate JSON/PNG/provenance. The earlier public_bundle/ is
preserved as initial-build evidence, not recommended for publication. The builder
refuses existing destinations and copies only its explicit allowlist.

Minimal runtime pins match locally tested installed versions: Python3.12,
Gradio6.29.0, Pillow12.3.0. See public_demo/README.md for fresh-environment setup.
No package install/upgrade occurred. The root research requirements are unchanged.
Transitive dependencies and the Docker base digest are not fully locked.

Hugging Face Docker Spaces configuration is prepared: sdk:docker, app_port:7860,
nonroot container, 0.0.0.0:7860 inside container; local Python uses loopback.
Official references: https://huggingface.co/docs/hub/spaces-sdks-docker and
https://huggingface.co/docs/hub/spaces-config-reference . After approval, use only
the final allowlisted bundle as release/Space contents, not this research worktree.
No Docker build, clean-environment install, upload or hosted deployment was executed;
therefore hosted deployment success is NOT claimed. A fresh build and actual hosted
verification remain future publication steps.

## Tests and local HTTP evidence

61 focused tests passed in9.43seconds,0 failed/skipped/errors. Exact command in
focused_test_results.json. Checks cover saved metrics/figure equality, no private
public images, private gallery references, disabled controls, direct rejection,
asset tampering, bundle allowlist/no-overwrite, public-entry enforcement and existing
UI/guard/Grad-CAM integration. A stale label assertion failed initially and was
updated; no functional failure was hidden. Repeat test passes are not extra tests.

The FINAL bundle actually served the root page and confusion-matrix asset over
loopback HTTP, each200, at port55931. It worked with Torch unavailable, without
research-module imports. Direct unmocked callback remained disabled. Server closed
after verification. No browser visual inspection or model prediction is claimed.
See http_smoke_final.json. Scoped git diff --check exited0 (existing CRLF warnings).

The export check found4 runtime pyc cache files after the smoke test. Only those
new generated cache files and empty directories were removed after verifying paths
within the final bundle. All allowlisted file hashes stayed unchanged. Final
export now matches exactly its12-file manifest with no caches/private artifacts.
Known-token-pattern and hardcoded Windows-user-path scans of all bundled text
returned no findings. This is a scoped check, not a universal secret guarantee.

## Publication checklist

Included: reviewed source/CSS, aggregate results/figure/provenance, public runtime
entry, pinned direct requirements, Docker configuration and publication README.
Excluded: all source images/overlays, per-image records/heatmap arrays, datasets,
original/corrected split manifests, checkpoints/calibration artifacts, notebooks,
research scripts/modules, analysis_reports, .git, env files, caches and credentials.
No unapproved weights are uploaded or included.

Before publishing: obtain researcher/institution release approval; confirm source
code license and result/aggregate-asset ownership/attribution. The repository has
no root LICENSE; no license was invented. Preserve citations to MiraBest Zenodo v1
https://doi.org/10.5281/zenodo.4288837 and Grad-CAM https://arxiv.org/abs/1610.02391 .
Citations are not redistribution permission. Keep image examples excluded unless
separate permission is established. Rebuild/retest after any approved release edit,
then clean-build and verify actual hosting manually. No publication authorization
is implied by this package.

Corrected OOD calibration is NOT required to publish a results-only presentation.
It IS required before future live inference, together with independent ID/OOD
release evidence, fixed acceptance criteria and corrected-manifest-aware loading
as described in Step23. This public entry stays results-only even if someone later
adds model/calibration files; enabling inference requires separately reviewed code.

## Scientific limitations and preservation

These are validation-selected development results, not independent TEST or external
validation. Source independence/near duplicates are unverified. Noise-induced
prediction flips, clipping and degenerate transformed CAMs remain disclosed.
Expert-reference physics alignment and corrected OOD evaluation are unavailable;
randomization and broader sensitivities remain deferred. CAM is coarse positive
attribution, not physical causation, expert morphology annotation or black-hole
detection. Softmax confidence is not calibrated. No real-time inference is claimed.

All41 protected artifact hashes match. Both checkpoints, original/corrected
manifests, Step18?23 results, frozen protocols, CSS and inference/XAI core are
unchanged. Only README.md, demo/app.py, demo/dashboard.py and the UI-label assertions
in tests/test_demo_dashboard.py changed among prior files. New presentation,
packaging/test/report paths are listed in changed_file_summary.json. Earlier
unrelated modifications were preserved. Git branch/HEAD/index unchanged; no stage,
commit, push, reset, clean, stash or switch. No TEST sample access, training,
research metric recalculation or new XAI experiment occurred.

Local research presentation: WORKS. Public deployment: prepared and locally verified,
not container/hosted verified or approved for publication. Live uploads: DISABLED.
Stop here; no automated publication or further experiment.
