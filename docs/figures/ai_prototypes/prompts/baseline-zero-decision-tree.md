# AI Prototype Prompt: Baseline-Zero Interpretability Decision Tree

- Source figure: `docs/figures/src/baseline_zero_decision_tree.mmd`
- Intended model family: `gpt-image-2`
- Status: visual prototype only; not final evidence artwork.

## Prompt

Create a clean editorial architecture-diagram prototype for an academic paper.

Target model: gpt-image-2.
Figure title: Baseline-Zero Interpretability Decision Tree.
Figure subtitle: A zero-valued baseline must be classified before it can support a stronger mechanism claim.
Canvas: wide landscape, high resolution, white background, vector-like shapes.
Visual style: Nature/ACM paper-ready systems diagram, restrained color palette, high contrast, spacious grouping, no decorative sci-fi effects.

Layer/lane structure:
- observation: Observed result from y=8.2 to y=10.1 [evidence]
- capability: Capability and evaluability checks from y=5.8 to y=8.2 [control]
- outcome: Permitted interpretation from y=2.4 to y=5.8 [output]
- boundary: Boundary statement from y=0.5 to y=2.4 [boundary]

Required nodes and roles:
- zero: Control score is zero [evidence]
- access: Did control have; required capability? [control]
- expected: Expected zero; mechanism-dependency evidence only [output]
- evaluable: Not evaluable; exclude from performance delta [output]
- sanity: Is there graded sanity; where control can score? [control]
- valid: Valid zero failure; usable only with caveats [output]
- suspicious: Suspicious zero; redesign or add sanity subset [evidence]
- boundary: Strong claims require explained zeros; and measured nonzero sanity controls [boundary]

Required directed relations:
- zero -> access: classify
- access -> expected: no capability
- access -> sanity: capability exists
- access -> evaluable: task excludes control
- sanity -> valid: yes
- sanity -> suspicious: no
- suspicious -> boundary: blocks claim

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
