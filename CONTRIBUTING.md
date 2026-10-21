# Contributing

This repository supports four independent research components and a shared integration area. Contributions should remain traceable, reviewable, and reproducible.

## Development and integration workflow

1. Each member develops on their assigned branch: `edirisooriya`, `rajapaksha`, `sewwandi`, or `buddhi`.
2. Work primarily inside the assigned component folder:
   - Edirisooriya: `components/01-edirisooriya-stellar-spectra/`
   - Rajapaksha: `components/02-rajapaksha-galaxy-morphology/`
   - Sewwandi: `components/03-sewwandi-exoplanet-detection/`
   - Buddhi Gayashan: `components/04-buddhi-radio-galaxies/`
3. Coordinate shared integration changes through the `developer` branch.
4. Write meaningful commit messages that describe the change.
5. Open a pull request from a member branch to `developer`. Explain the purpose, scope, evidence, and any checks performed.
6. Review code and documentation before merging. Run the relevant checks and record their results.
7. Resolve merge conflicts carefully. Preserve the intent of each contribution, inspect the resulting diff, and rerun relevant checks.
8. Promote reviewed integration work from `developer` to `main` with a pull request.

**Recommended flow:** Member branch → Pull Request → `developer` → Review and integration testing → Pull Request → `main`.

## Credentials and sensitive information

Never commit passwords, API keys, access tokens, private keys, personal credentials, or populated local environment files. Use local environment variables or an approved secrets manager. A safe configuration template may be committed as `.env.example`, but it must contain placeholders only.

## Data and model artifacts

Do not upload large datasets, generated artifacts, or model checkpoints until repository storage limits, data licensing, redistribution rights, privacy requirements, and the group's storage approach have been reviewed. Prefer documenting an authorized data source and reproducible retrieval or preparation steps. Do not place private or restricted data in this public repository.

## Reproducibility and provenance

Document dataset identifiers or sources, versions, preprocessing, dependency versions, configuration, random seeds where applicable, evaluation methods, and commands needed to reproduce results. Distinguish planned work and work in progress from verified results. Scientific claims must be supported by recorded evidence; a merged change alone does not validate a result.

## Pull request checklist

- [ ] The change is scoped to the relevant component or coordinated integration work.
- [ ] No credentials, restricted data, or unreviewed large artifacts are included.
- [ ] Documentation records the relevant sources, configuration, and provenance.
- [ ] Checks and their actual outcomes are described.
- [ ] Claims about implementation or scientific findings are supported by evidence.
