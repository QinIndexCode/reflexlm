# Paper B: Bounded Mechanism Manuscript

This directory contains the anonymized, LaTeX-first public manuscript workspace
for the bounded NSI mechanism paper.

## Contents

- `main.tex`: anonymized manuscript source.
- `references.bib`: bibliography.
- `tables/`: artifact-backed LaTeX tables.
- `figures/`: editable and rendered public figures.

Author metadata, cover letters, reviewer correspondence, compliance checklists,
submission portals, and compiled submission packages are intentionally kept
outside the public repository.

## Build

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Generated LaTeX products and PDFs are ignored by Git.

## Regenerate Public Materials

```powershell
python scripts/render-paper-figures.py
python scripts/render-paper-b-tables.py
```

## Claim Boundary

The manuscript may argue bounded native nervous command selection and bounded
HMAC-authenticated homeostatic persistent-state transfer under controlled
runtime tasks. It must not claim exact cross-runtime internal homeostatic
microdynamics, unbounded semantic memory, production autonomy, unrestricted
shell use, open-ended repair, independent replication, or epoch-making
architecture status.

## Operational Mechanism Definition

Paper B treats the evaluated mechanism as a bounded state-frame-to-action-head
policy package:

- Input: terminal, process, filesystem, time, goal, safety, and candidate
  command/file fields.
- Learned outputs: internal target, route, action type, command slot, file slot,
  confidence, and inhibition signals.
- Runtime contract: actions are serialized only after native-head selection and
  must pass safety, allowlist, stale-state, and candidate-validity gates.
- Out of scope: free-form shell synthesis, GUI control, robotics, arbitrary
  patch generation, production autonomy, and semantic long-term memory.

Homeostatic persistence is likewise operational: it covers authenticated
threshold/adaptation state transfer for bounded runtime control, not broad
personal memory or exact biological homeostasis.
