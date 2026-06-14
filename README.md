# ReflexLM

ReflexLM is a research implementation of a bounded Native Synaptic Interface
(NSI) runtime for structured terminal, process, filesystem, and time signals.
The repository focuses on falsifiable mechanism evidence rather than production
autonomy or unrestricted software-agent claims.

## Public Scope

- Structured receptors and state-vector encoding.
- Salience, inhibition, prediction-error, persistence, and bounded action heads.
- Controlled command-selection and runtime-recovery benchmarks.
- Explicit ablations, negative evidence, safety boundaries, and reproducibility
  gates.
- An anonymized bounded-mechanism manuscript workspace under `docs/paper_b/`.

Private submission workflows, reviewer materials, author metadata, local
experiment artifacts, model weights, and machine-specific research notes are
intentionally excluded from the public repository.

## Layout

| Path | Purpose |
| --- | --- |
| `src/reflexlm/` | Runtime, models, baselines, data utilities, and CLI tools |
| `tests/` | Unit, mechanism, negative-control, and reproducibility tests |
| `configs/` | Versioned experiment and model configuration |
| `docs/spec/` | Research scope, preregistration, and claim-boundary documents |
| `docs/paper_b/` | Anonymized bounded-mechanism manuscript sources |
| `docs/figures/` | Editable figure sources and generated public figures |
| `scripts/` | Public experiment, rendering, audit, and export utilities |

See [Repository Layout](docs/repository_layout.md) for the public/private
boundary.

## Install

Python 3.12 or newer is required.

```bash
python -m pip install -e .[dev]
```

Optional local-LLM dependencies:

```bash
python -m pip install -e .[dev,llm]
```

## Validate

Run the fast public release checks:

```bash
python scripts/audit-public-release.py
python -m pytest -q tests/test_schema.py tests/test_oracle.py tests/test_dataset_generation.py
```

Run the complete suite when sufficient time and local resources are available:

```bash
python -m pytest
```

The full suite contains long-running integration and reproduction tests.

## Paper Workspace

Artifact-backed tables:

```bash
python scripts/render-paper-b-tables.py
```

Editable figures:

```bash
python scripts/render-paper-figures.py
```

Build the anonymized manuscript from `docs/paper_b/`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Public Release

Do not publish the current Git history because earlier local commits may contain
removed submission metadata or personal information. Create a clean,
history-free snapshot and initialize a new public repository from it:

```bash
python scripts/audit-public-release.py
python scripts/export-public-repository.py --output ../reflexlm-public
```

Review the exported tree, then initialize Git inside that new directory.

## Claim Boundary

Current evidence supports bounded NSI command-selection and homeostatic
persistent-state mechanism arguments under controlled tasks. It does not
support exact cross-runtime internal microdynamics, unbounded semantic memory,
open-ended repair, unrestricted shell use, production autonomy, consciousness,
AGI, or an epoch-making architecture claim.

## License

Project-authored code is available under [LICENSE-CODE-MIT](LICENSE-CODE-MIT).
Source-derived datasets and traces retain their upstream provenance and license
constraints; see [DATA_LICENSE_NOTICE.md](DATA_LICENSE_NOTICE.md).
