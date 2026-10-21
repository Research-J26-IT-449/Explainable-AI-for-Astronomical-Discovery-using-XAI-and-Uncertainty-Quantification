# Explainable AI for Astronomical Discovery using XAI and Uncertainty Quantification

**Research group:** J26-IT-449

## Project overview

This repository provides a shared collaboration and integration structure for four independent research components exploring explainable AI (XAI) and uncertainty quantification in astronomical analysis. Each component is developed and documented by its responsible member. Shared integration is coordinated separately so that component work remains traceable and independently reviewable.

This README describes the planned project organization. It does not report completed experiments or scientific findings. Component implementation and results should be reported only when supported by verified code, data provenance, and evaluation evidence.

## Research components

| Component | Research area | Responsible member | Repository folder |
|---|---|---|---|
| 1 | Stellar spectra analysis | Edirisooriya | `components/01-edirisooriya-stellar-spectra/` |
| 2 | Galaxy morphology analysis | Rajapaksha | `components/02-rajapaksha-galaxy-morphology/` |
| 3 | Exoplanet detection | Sewwandi | `components/03-sewwandi-exoplanet-detection/` |
| 4 | Radio galaxy morphology analysis (RadioMorph AI) | Buddhi Gayashan | `components/04-buddhi-radio-galaxies/` |

The RadioMorph AI folder is currently a documentation placeholder. Its individual project is separate from this group repository. No code, datasets, model weights, or other research artifacts are being migrated as part of this repository setup.

## Repository structure

```text
.
├── README.md
├── CONTRIBUTING.md
├── .gitignore
├── components/
│   ├── 01-edirisooriya-stellar-spectra/
│   ├── 02-rajapaksha-galaxy-morphology/
│   ├── 03-sewwandi-exoplanet-detection/
│   └── 04-buddhi-radio-galaxies/
├── integration/
└── docs/
```

Each component folder is the home for that member's component-specific code and documentation. The `integration/` folder is for shared integration planning and notes. The `docs/` folder is for group-level documentation. Add files only as the work is actually prepared; avoid placeholder claims that imply implementation or evaluation has been completed.

## Branching strategy

- **`main`** — stable, reviewed group integration.
- **`developer`** — shared development and integration branch.
- **`edirisooriya`** — Component 1 development.
- **`rajapaksha`** — Component 2 development.
- **`sewwandi`** — Component 3 development.
- **`buddhi`** — Component 4 development.

Members should develop primarily on their assigned branch and within their component folder. Send component work to `developer` through a pull request. After review and integration checks, promote the shared integration state from `developer` to `main` through a pull request. Do not treat a branch name as evidence that its work is complete.

## Collaboration workflow

1. Start from the latest appropriate branch and create changes on your assigned member branch.
2. Keep component-specific work in its assigned folder; coordinate shared changes through `developer`.
3. Use focused changes and meaningful commit messages.
4. Open a pull request to `developer` for component or integration work.
5. Review the changes, record relevant checks, and resolve conflicts carefully before merging.
6. Promote reviewed integration work from `developer` to `main` with a pull request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for repository contribution and reproducibility guidance.

## Research reproducibility

For each experiment, record the question, dataset/source and version, inclusion or preprocessing steps, software and dependency versions, configuration, random seeds where applicable, evaluation protocol, and outputs needed to reproduce the analysis. Keep raw data and large artifacts out of Git unless storage, licensing, redistribution rights, and repository policy have been reviewed. Separate planned work, work in progress, and verified results, and cite the evidence behind reported claims.

## Current project status

**Repository setup:** The shared repository structure and collaboration documentation are being established.

**Research status:** This repository setup does not establish that any component implementation, experiment, or result is complete. Component-specific progress must be documented and verified by its contributors.
