# Paper strengthening review before external deposit

Updated on 2026-05-22 after pausing Zenodo upload.

## Verdict

Do not upload the Zenodo dataset yet. The local Scientific Data package is technically close, but the overall paper set is not yet strong enough to treat the deposit as final. The main weakness is not the amount of data; it is manuscript framing and claim separation.

The current evidence can support a conservative Data Descriptor and a bounded mechanism paper. It still cannot support production autonomy, open-ended repair, unrestricted shell use, independent external reproduction, or epoch-making architecture claims.

## Current paper split

| Paper | Target | Current strength | Main risk |
| --- | --- | --- | --- |
| Paper A | Scientific Data Data Descriptor | Locally viable after archive and validator hardening | Could be rejected if it reads like an architecture-results paper or if deposit/license metadata are incomplete. |
| Paper B | Bounded native nervous-interface mechanism paper | Evidence is materially stronger after Phase2M-v2, Phase2P, Phase2Q, and Phase2R | Current `paper_draft.md` is too long, historically layered, and hard for reviewers to audit quickly. |
| Paper C | Open-ended repair / architecture generalization | Not ready | Requires Phase2S-style sandboxed repair, modern coding-agent baselines, rollback/safety metrics, and independent reproduction. |

## Why the current manuscript is still not strong enough

1. The main draft is a chronological experiment log, not yet a reviewer-optimized paper.
2. Several old negative sections remain useful but need to be compressed into a negative-evidence table.
3. The strongest current claim is bounded command selection, but the title and architecture language can still invite reviewers to expect open-ended agent behavior.
4. Repeated `1.000` results on bounded tasks are strong only if the paper foregrounds leakage controls, source-overlap pressure, repo-disjoint holdout, and ablations.
5. Sealed v3 is now saturated. Further improvements on that benchmark should not be used to strengthen the story.
6. Independent external reproduction remains absent.
7. Modern coding-agent baseline comparisons remain absent for open-ended repair.
8. The Scientific Data draft and mechanism paper still share too much narrative context; they should be separated before any public deposit is treated as final.

## Minimum strengthening before Zenodo

Complete these before external data deposition:

- Freeze Paper A as a Data Descriptor only.
- Rewrite Paper B around the mechanism, not around the full experiment chronology.
- Add a compact evidence matrix from Phase2M-v2 through Phase2R.
- Add a negative-evidence table covering Phase2I, Phase2K, Phase2L, and failed Phase2M smoke variants.
- Add a reviewer-facing leakage and shortcut-control table.
- Add a reviewer-facing claim boundary table: supported, partially supported, unsupported.
- Add a reproducibility table mapping each central claim to exact artifacts, scripts, hashes, and validation commands.
- Keep Zenodo DOI, anonymous reviewer URL, and final data license as external blockers until the manuscript split is stable.

## Paper B recommended structure

1. Problem: text-only agent loops lack native state receptors and bounded motor control.
2. Mechanism: structured receptors, NSI latent, native heads, continuation state, bounded command schema.
3. Threat model: no gold hints, no candidate markers, no sealed feedback, measured baselines, repo-disjoint splits.
4. Data ladder: synthetic reflex, semantic-required, public repo relation-key, public trace breadth, dynamic pytest traces.
5. Results: one table for Phase2M-v2, Phase2P, Phase2Q, and Phase2R; old phases summarized as negative evidence.
6. Ablations: full, no-NSI, native-head-only/no-cache, continuation-only, prompt-only, ReAct, source-overlap.
7. Limits: no open-ended repair, no production autonomy, no independent reproduction, no modern agent baseline dominance.
8. Next test: Phase2S sandboxed open-ended repair.

## Phase2S design standard

Phase2S should not be another command-slot benchmark. It should test whether the mechanism helps when an agent must:

- inspect runtime evidence,
- edit code,
- run tests,
- decide whether to rollback,
- stop safely,
- avoid unauthorized writes,
- beat a modern agent-style loop under preregistered gates.

If Phase2S fails, the project remains a bounded mechanism contribution. If Phase2S passes, it can support a separate architecture/generalization paper, not a retroactive expansion of Paper A.

## Current action order

1. Keep GitHub private while the manuscript split is unstable.
2. Refactor Paper A and Paper B narrative boundaries.
3. Build the reviewer-facing evidence/negative-evidence tables.
4. Only then prepare Zenodo draft upload and anonymous review access.
5. After Paper A deposit is stable, resume Phase2S design and execution.
