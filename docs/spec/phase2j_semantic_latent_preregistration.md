# Phase2J Semantic Latent Preregistration

Phase2J is a separate mechanism experiment, not a retry of the frozen Phase2I architecture.

## Rationale

Phase2I showed that the pairwise/native-head path can be repaired for continuation and command-slot ranking, but it did not isolate NSI-latent necessity. At the frozen Phase2I decision point, the registered latent interface did not carry command or slot identity, so same-intent semantic command-slot ambiguity was not identifiable from that Phase2I mechanism alone. Phase2J may add non-label command identity latent fields, but that is a mechanism-scope change and must not be backfilled into the Phase2I claim.

## Allowed Next Action

The only allowed training action after preregistration, readiness, data health, and pretrain gates pass is a non-sealed Phase2J smoke run. No full training, packaging, or sealed-v3 evaluation is allowed before the smoke postflight passes.

## Hard Boundaries

- Phase2I remains a bounded result and must not be upgraded by Phase2J preregistration.
- Sealed v2/v3 traces remain evaluation-only and must not be used for training, tuning, sampling, or failure-feedback design.
- If command or slot identity is added to NSI latent, that is a mechanism-scope change and must be reported as a separate Phase2J claim.
- Command or slot identity in NSI latent must come from runtime-observed receptor evidence or a general static-analysis signal. It must not be derived from gold labels, target slots, test names, answer keys, or sealed failure analysis.
- Full training requires a prior non-sealed smoke run with recorded split hashes, latent-necessity audit, source-overlap baseline, same-intent ambiguity coverage, and train/val intent coverage.
- Package and sealed evaluation require a passed prepackage gate after full training.

## Required Machine Check

Run `python -m reflexlm.cli.check_phase2j_preregistration` with the frozen Phase2I decision report and a Phase2J proposal JSON before any training command is issued.

Then run `python -m reflexlm.cli.audit_phase2j_implementation_readiness`. Passing preregistration is not sufficient: current code must also expose non-label command identity latent fields and non-sealed Phase2J train/validation profiles before data generation. This readiness audit never grants training permission; it only decides whether non-sealed data generation may start.

After non-sealed data generation and head-split materialization, run `python -m reflexlm.cli.audit_phase2j_data_health` and then `python -m reflexlm.cli.audit_phase2j_pretrain_gate`. The pretrain gate may allow only `run_nonsealed_phase2j_smoke_training_only`; it always keeps full training, packaging, and sealed evaluation blocked until later smoke/full gates pass.

After smoke training, run `python -m reflexlm.cli.audit_phase2j_smoke_postflight`. Passing the raw validation threshold is not enough for full training. The smoke must also beat the recorded source-overlap command-slot baseline; otherwise the evidence only shows that the non-sealed split is learnable or feature-solvable, not that the Phase2J latent mechanism adds causal value.
