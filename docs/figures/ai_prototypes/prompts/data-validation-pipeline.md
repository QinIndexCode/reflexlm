# AI Prototype Prompt: Data and Validation Pipeline

- Source figure: `docs/figures/src/data_validation_pipeline.mmd`
- Intended model family: `gpt-image-2`
- Status: visual prototype only; not final evidence artwork.

## Prompt

Create a clean editorial architecture-diagram prototype for an academic paper.

Target model: gpt-image-2.
Figure title: Data and Validation Pipeline.
Figure subtitle: Non-sealed construction and gates precede one-way sealed final evaluation.
Canvas: wide landscape, high resolution, white background, vector-like shapes.
Visual style: Nature/ACM paper-ready systems diagram, restrained color palette, high contrast, spacious grouping, no decorative sci-fi effects.

Layer/lane structure:
- data: Data construction from y=8.2 to y=10.1 [input]
- audit: Audit and split gates from y=6 to y=8.2 [evidence]
- train: Training and postflight from y=3.75 to y=6 [model]
- release: Package and final evaluation from y=1.55 to y=3.75 [output]
- boundary: Sealed boundary from y=0.35 to y=1.55 [boundary]

Required nodes and roles:
- collect: Public read-only traces; or preregistered synthetic-safe data [input]
- normalize: Normalize and scrub; remove hidden gold candidate markers [control]
- split: Repo-disjoint split; train validation holdout [control]
- audit: Data health audit; hashes leakage graded difficulty [evidence]
- train: Non-sealed train; fixed config and seeds [model]
- post: Postflight gates; source-overlap native controls [evidence]
- package: Package only after gates; no sealed tuning [output]
- sealed: Sealed final eval; one-way evidence only [boundary]

Required directed relations:
- collect -> normalize: raw traces
- normalize -> split: clean rows
- split -> audit: effective hashes
- audit -> train: only if passed
- train -> post: measured metrics
- post -> package: if deltas pass
- package -> sealed: final evaluation only

Design requirements:
- Preserve the exact mechanism boundary: bounded command selection only, not production autonomy or open-ended repair.
- Use grouped lanes or layered regions if useful: observation/receptors, latent state, routing/native heads, controls/ablations, claim boundary.
- Prefer semantic icons only when they clarify the node role; do not add extra system components.
- Use short callouts for relations, or edge-number tags with a compact legend, but keep all labels clearly separated from nodes and lines.
- Avoid overlapping labels, crossing-heavy wiring, and ambiguous label ownership.
- Leave enough whitespace for later Draw.io/vector recreation.
- The image is a visual prototype; final paper artwork will be manually/vector recreated from editable source files.

## Negative Prompt

No new architecture modules. No sealed-evaluation feedback loop. No candidate_0/candidate_1 markers, gold labels, hidden hints, shell autonomy, robot mascots, screenshots, code blocks, dense tiny text, or unverifiable performance claims.

## Boundary Note

AI raster output is a visual prototype only and is non-authoritative. Use it only to choose composition, visual hierarchy, and spacing before updating editable Mermaid/DOT/draw.io sources.
