---
title: RadioMorph AI Research Results
emoji: 🔭
colorFrom: cyan
colorTo: purple
sdk: docker
app_port: 7860
---

# RadioMorph AI — results-only public research demonstration

**No live image inference.** This application presents saved development results
from the corrected MiraBest development ResNetCNN. Corrected-model OOD validation
is unavailable; uploads, predictions and live Grad-CAM are disabled. No weights,
datasets, original source images or private manifests are required or bundled.

## Local setup and launch (prepared bundle)

Use Python 3.12. Create a virtual environment and install only this bundle's
requirements.txt. Dependencies are pinned to the versions tested locally; this
does not pin every transitive dependency or Docker base-image digest.

```text
python -m venv .venv
# Activate .venv for your operating system, then:
python -m pip install -r requirements.txt
python -B app.py
```

Open http://127.0.0.1:7860. No model, archive or calibration download occurs.
The public entry enforces results-only mode even if an environment variable
requests private examples. It does not expose a second inference path.

## Hugging Face Spaces / Docker (manual, after approval)

The bundle has a Docker Spaces README with sdk: docker and app_port: 7860.
See https://huggingface.co/docs/hub/spaces-sdks-docker and
https://huggingface.co/docs/hub/spaces-config-reference .
After the publication checklist is approved, use only the generated allowlisted
bundle as a new Space's contents. Do not publish the whole research repository.
The container binds port 7860 on 0.0.0.0; local Python defaults to loopback.
No hosted build or deployment has been executed or verified in this task.

## Results and methods

116 corrected validation images; accuracy 87.068966%, macro F1 0.864391;
confusion matrix [[38,13],[2,63]] (rows true, columns predicted; FR-I, FR-II).
This validation set selected the checkpoint; these are not independent TEST results.
Precomputed Grad-CAM used predicted-class logit gradients at blocks.2.convolution.3,
native 19x19 and aligned 150x150 display. Mean probability AOPC through 30% deletion:
CAM 0.294510, random 0.053960, brightness 0.222254. Fixed border-median replacement
and 20 seeded random controls per image were used. Negative effects were retained.
No metrics are recomputed when the app launches. The bundled aggregate PNG is an
exact copy of a saved figure; provenance.json records source/copy hashes.

## Responsible use and limitations

Research presentation only, not a survey classifier or scientific decision service.
Confidence is not calibrated. Perturbations may be OOD; noise tests showed prediction
instability, clipping and degenerate transformed CAMs. Expert-mask physics alignment
and corrected OOD validation are unavailable; randomization/sensitivity analyses
remain deferred. Grad-CAM does not prove causality, physical correctness, verified
core/jet/lobe localization, source independence or direct black-hole detection.
No independent TEST or external-survey performance is established.

## Attribution and publication checklist

Research project: J26-IT-449-IT23308602-AGN-XAI, AGN-Aware Radio Source Structure
Classification using CNN and XAI. MiraBest dataset source:
https://doi.org/10.5281/zenodo.4288837 . Grad-CAM method:
https://arxiv.org/abs/1610.02391 . These citations are not redistribution permission.

The repository contains no root code LICENSE. Researcher/institution approval and
an appropriate code license/release authority must be confirmed before publication.
Dataset-image/overlay redistribution rights are not established from local evidence;
therefore none are in this bundle. Do not copy private local galleries into it.
Review ownership/permission for aggregate figures and reported results too.

Included: reviewed dashboard source/CSS, public entry, pinned minimal requirements,
Dockerfile, this README, aggregate result JSON/PNG and provenance hashes.
Excluded: raw images, MiraBest overlays, per-image records, datasets/splits,
checkpoints, calibration artifacts, notebooks, research directories, .git, caches,
environment files, credentials and API tokens. No license, secret or token is invented.

Live prediction requires separate evidence-based corrected OOD calibration and
independent release evaluation. It is not enabled by this publication package.
