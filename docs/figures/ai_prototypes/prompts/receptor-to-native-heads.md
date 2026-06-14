# AI Prototype Prompt: Receptor-to-Native-Head Decision Path

- Source figure: `docs/figures/src/receptor_to_native_heads.mmd`
- Intended model family: `gpt-image-2`
- Status: visual prototype only; not final evidence artwork.

## Prompt

Create a clean editorial architecture-diagram prototype for an academic paper.

Target model: gpt-image-2.
Figure title: Receptor-to-Native-Head Decision Path.
Figure subtitle: Observable runtime evidence is converted into bounded native action fields.
Canvas: wide landscape, high resolution, white background, vector-like shapes.
Visual style: Nature/ACM paper-ready systems diagram, restrained color palette, high contrast, spacious grouping, no decorative sci-fi effects.

Layer/lane structure:
- evidence: Runtime evidence from y=8.25 to y=10.1 [input]
- scrub: Leakage controls from y=6.55 to y=8.25 [control]
- latent: Latent and routing state from y=4.8 to y=6.55 [state]
- decision: Native decision heads from y=2.65 to y=4.8 [model]
- boundary: Serialization boundary from y=0.55 to y=2.65 [boundary]

Required nodes and roles:
- trace: Runtime trace; stderr changed files watched paths [input]
- scrub: Leakage scrub; no gold hidden or candidate markers [control]
- features: Receptor features; observable state only [input]
- nsi: NSI latent; identity relation stage signals [state]
- route: Route gate; Debug Cortex only when needed [model]
- slots: Command-slot head; same-intent candidates [model]
- action: Native action tuple; action route slot confidence [output]
- nojson: Serialization after decision; not model JSON output [boundary]

Required directed relations:
- trace -> scrub: public or synthetic-safe input
- scrub -> features: visible fields
- features -> nsi: latent fusion
- nsi -> route: route bias
- route -> slots: debug command selection
- slots -> action: selected bounded command
- action -> nojson: runtime wrapper

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
