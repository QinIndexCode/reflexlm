# AI Prototype Prompt: Native Nervous Interface Architecture

- Source figure: `docs/figures/src/nsi_architecture.mmd`
- Intended model family: `gpt-image-2`
- Status: visual prototype only; not final evidence artwork.

## Prompt

Create a clean editorial architecture-diagram prototype for an academic paper.

Target model: gpt-image-2.
Figure title: Native Nervous Interface Architecture.
Figure subtitle: Bounded command selection only; not production autonomy or open-ended repair.
Canvas: wide landscape, high resolution, white background, vector-like shapes.
Visual style: Nature/ACM paper-ready systems diagram, restrained color palette, high contrast, spacious grouping, no decorative sci-fi effects.

Layer/lane structure:
- observation: Observation layer from y=8 to y=10.1 [input]
- latent_lane: Latent state layer from y=6.55 to y=8 [state]
- action_lane: Routing and action layer from y=4.55 to y=6.55 [model]
- controls_lane: Controls and ablations layer from y=2.25 to y=4.55 [control]
- boundary_lane: Claim boundary layer from y=0.15 to y=2.25 [boundary]

Required nodes and roles:
- receptors: Structured receptors; terminal process filesystem time [input]
- latent: NSI latent state; salience prediction error route bias [state]
- cache: Continuation memory; bounded prior runtime context [state]
- cortex: Debug Cortex route; semantic command selection [model]
- heads: Native heads; action route slot confidence [model]
- schema: Bounded motor schema; no JSON motor target [output]
- controls: Controls and ablations; no-NSI native-only no-cache text baselines [control]
- boundary: Claim boundary; bounded command selection not production autonomy [boundary]

Required directed relations:
- receptors -> latent: runtime-visible state
- latent -> cache: fused state
- cache -> cortex: continuation-conditioned evidence
- cortex -> heads: debug route features
- heads -> schema: bounded decision
- controls -> heads: mechanism deltas
- boundary -> controls: claim gating

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
