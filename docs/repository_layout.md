# Repository Layout and Privacy Boundary

## Public GitHub Repository

The public repository contains only reusable research code, tests,
configuration, anonymized manuscripts, public figures, and claim-boundary
documentation.

Public paths:

- `src/`
- `tests/`
- `configs/`
- `docs/spec/`
- `docs/figures/`
- `docs/paper_b/`
- `scripts/`

## Local Private Materials

The local `private/` directory is ignored by Git and contains:

- author-identifying manuscript versions;
- cover letters and journal compliance checklists;
- reviewer and submission workflow notes;
- private repository administration scripts;
- submission archives and compiled PDFs;
- machine-specific experiment notebooks.

Do not move these files back into public paths. Run
`python scripts/audit-public-release.py` before creating any public release.

## Artifacts

`artifacts/` is local-first and ignored by default. Public evidence should be
regenerated from documented commands or deposited separately with appropriate
provenance and licensing.

## History Safety

Removing a file from the current working tree does not remove it from Git
history. Public releases must be created with
`scripts/export-public-repository.py` and initialized as a new repository.
