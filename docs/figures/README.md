# Editable figure pipeline

Paper-facing diagrams are kept as editable source files under `docs/figures/src`.
The current sources use Mermaid (`.mmd`) and Graphviz DOT (`.dot`) syntax plus
simple layout comments consumed by `scripts/render-paper-figures.py`.

The renderer also writes draw.io-compatible `.drawio` XML under
`docs/figures/drawio` and copies those editable files into `docs/paper_b/figures`.
This follows the `Agents365-ai/drawio-skill` workflow: grid-aligned shapes,
wide spacing, orthogonal rounded edges, and editable source as the authority.
If a local draw.io desktop CLI is available, these `.drawio` files can be
opened and exported with embedded editable XML; the repo does not require that
CLI for tests.

Static exports use short edge tags such as `E1` anchored directly on the routed
connection segment and a connection-label legend below the diagram. Full edge
descriptions remain in the draw.io source. The renderer now chooses label
positions from edge-owned route segments with explicit foreign-route clearance,
so a tag is not considered valid if its box overlaps another edge or sits in a
visually ambiguous low-clearance corridor.
The renderer audits node overlap, edge-label overlap, edge paths crossing
non-endpoint cards, and relation paths that overlap or cross each other away
from shared endpoints. A figure is not considered paper-ready unless those
layout audits pass. Lane figures also reserve a left title band; nodes that
intrude into that band fail the audit. Static tags rotate with the owning edge
segment to reduce diagonal-line drift in exported PNG/PDF figures.

## AI prototype layer

The renderer also writes GPT-image-compatible visual prototype prompts under
`docs/figures/ai_prototypes/prompts`. These prompts are for composition,
hierarchy, color, spacing, and visual metaphor exploration only. They are not
authoritative paper figures and must not introduce new mechanisms, sealed
feedback, production-autonomy claims, or hidden/candidate/gold markers.

Accepted workflow:

1. Generate one or more raster prototypes from the prompt.
2. Select the useful layout ideas only.
3. Recreate the final diagram in the editable Mermaid/DOT/draw.io source.
4. Re-run `python scripts/render-paper-figures.py` and use the generated SVG/PDF
   output in LaTeX.

Do not treat raw AI-generated raster diagrams as final evidence artwork unless
they have been recreated as editable vector source and pass the layout audit.

Render command:

```powershell
python scripts/render-paper-figures.py
```

The command writes SVG, PDF, and PNG exports to `docs/figures/export`, writes
editable `.drawio` files to `docs/figures/drawio`, and copies all paper-facing
outputs into `docs/paper_b/figures` for LaTeX inclusion and later manual edits.

The source files and `.drawio` files, not the rendered exports, are the
authoritative editable diagrams. ASCII arrow diagrams in notes are not
paper-ready figures.
