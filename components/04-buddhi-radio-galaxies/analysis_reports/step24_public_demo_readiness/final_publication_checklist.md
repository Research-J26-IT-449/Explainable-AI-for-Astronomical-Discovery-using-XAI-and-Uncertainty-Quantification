# Final public release checklist

- [x] Final source/aggregate-only export: public_bundle_final (12 files).
- [x] Live controls and public callbacks disabled; no inference bypass.
- [x] Saved metrics/aggregate confusion figure displayed with provenance.
- [x] Dataset images/overlays and model weights excluded.
- [x] Local standalone HTTP/asset verification; 61 focused tests pass.
- [x] Source/assets match bundle hashes; runtime caches removed from export.
- [x] No detected common secret patterns or local Windows-user paths in bundle.
- [x] All41 protected artifacts and Step18?23 results unchanged.
- [ ] Researcher/institution release approval; confirmed code/result license/authority.
- [ ] Confirm aggregate-asset attribution/redistribution terms before publishing.
- [ ] Separate permission before adding any source-image/overlay examples.
- [ ] Fresh-environment/container build, then manual hosted verification after approval.

Only the final bundle should be considered for release, never the research repository.
MiraBest: https://doi.org/10.5281/zenodo.4288837 . Grad-CAM: https://arxiv.org/abs/1610.02391 .
Corrected OOD evidence is needed for FUTURE LIVE INFERENCE, not for publication of
this results-only demo. No deployment occurred and publication is not auto-approved.
