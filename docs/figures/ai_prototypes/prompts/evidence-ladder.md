# AI Prototype Prompt: Evidence Ladder and Claim Boundary DAG

- Source figure: `docs/figures/src/evidence_ladder.dot`
- Intended model family: `gpt-image-2`
- Status: visual prototype only; not final evidence artwork.

## Prompt

Create a clean editorial architecture-diagram prototype for an academic paper.

Target model: gpt-image-2.
Figure title: Evidence Ladder and Claim Boundary DAG.
Figure subtitle: Positive evidence can support bounded claims only while preserved failures constrain overclaiming.
Canvas: wide landscape, high resolution, white background, vector-like shapes.
Visual style: Nature/ACM paper-ready systems diagram, restrained color palette, high contrast, spacious grouping, no decorative sci-fi effects.

Layer/lane structure:
- evidence: Positive evidence ladder from y=8.2 to y=10.1 [evidence]
- synthesis: Claim synthesis from y=5 to y=8.2 [output]
- boundary: Negative evidence and unsupported scope from y=0.8 to y=5 [boundary]

Required nodes and roles:
- p2m: Phase2M-v2; relation-key public split [evidence]
- p2p: Phase2P; sealed cross-model transfer [evidence]
- p2q: Phase2Q; public trace breadth [evidence]
- p2r: Phase2R; dynamic pytest traces [evidence]
- p2s: Phase2S; public repair multiseed [evidence]
- paperb: Paper B claim; bounded native nervous command selection [output]
- neg: Negative evidence; Phase2I K L failures preserved [control]
- unsupported: Unsupported scope; production autonomy epoch-making claims [boundary]

Required directed relations:
- p2m -> p2p: sealed transfer after non-sealed gate
- p2p -> p2q: public breadth
- p2q -> p2r: dynamic evidence
- p2r -> p2s: repair-pressure reproduction
- p2s -> paperb: supports bounded claim
- neg -> paperb: limits overclaim
- paperb -> unsupported: explicit boundary

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
