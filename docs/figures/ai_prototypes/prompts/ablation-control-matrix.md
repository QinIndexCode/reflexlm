# AI Prototype Prompt: Ablation and Control Matrix

- Source figure: `docs/figures/src/ablation_control_matrix.mmd`
- Intended model family: `gpt-image-2`
- Status: visual prototype only; not final evidence artwork.

## Prompt

Create a clean editorial architecture-diagram prototype for an academic paper.

Target model: gpt-image-2.
Figure title: Ablation and Control Matrix.
Figure subtitle: Each control is measured before deltas are interpreted; zero scores are not automatic proof.
Canvas: wide landscape, high resolution, white background, vector-like shapes.
Visual style: Nature/ACM paper-ready systems diagram, restrained color palette, high contrast, spacious grouping, no decorative sci-fi effects.

Layer/lane structure:
- ablations: Ablation mechanisms from y=8 to y=10.1 [control]
- baselines: Shortcut and text baselines from y=5 to y=8 [evidence]
- interpretation: Delta interpretation from y=2.6 to y=5 [output]
- boundary: Claim boundary from y=0.5 to y=2.6 [boundary]

Required nodes and roles:
- full: Full package; heads + NSI + cache + bounded route [model]
- nonsi: No-NSI; latent contribution removed [control]
- native: Native-head-only; cache or latent removed [control]
- cont: Continuation-only; heads removed [control]
- text: Prompt-only and ReAct; text-loop baselines [control]
- source: Source-overlap baseline; measured shortcut pressure [control]
- zero: Zero-result audit; classify evaluability first [boundary]
- deltas: Measured delta table; accuracy gaps + evaluability flags [evidence]
- gates: Claim gates; full must beat measured controls [output]

Required directed relations:
- full -> deltas: target evidence
- nonsi -> deltas: latent delta
- native -> deltas: package delta
- cont -> deltas: cache/head delta
- text -> deltas: text-loop delta
- source -> deltas: shortcut pressure
- zero -> deltas: interpretability audit
- deltas -> gates: predeclared thresholds

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
