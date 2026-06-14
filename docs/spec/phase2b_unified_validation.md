# Phase 2B Unified 7B Validation Protocol

## Purpose

Phase 2B validates the paper architecture claim: an LLM can be organized as a nervous-interface system with receptors, synaptic state, reflex routing, cortex processing, and bounded motor output.

This protocol replaces the old multi route-adapter interpretation. The previous route-expert run remains a negative control because route labels used for adapter training did not match the route signals used at hybrid inference time.

Status update: this branch is now a negative-control JSON motor path, not the forward validation path. It still asked Qwen7B to emit JSON text as the primary motor output, which conflicts with the Phase 2C requirement that motor actions come from explicit heads and runtime serialization. New large-model work should use `docs/spec/phase2c_native_nervous_validation.md`.

## Model Contract

- Use one `Qwen/Qwen2.5-7B-Instruct` base model.
- Use one unified LoRA adapter trained on `nsi_state_v2`.
- Keep NSI reflex output as a low-level synaptic signal and direct low-risk reflex path.
- Do not switch adapters by route at inference time.
- Emit only `{action, command, file_target}` over the fixed motor schema.

## Iteration Discipline

Each iteration must name the paper subclaim being tested before training:

- reflex latency and cost: salience gate should reduce unnecessary 7B calls.
- state correctness: receptor and synaptic state should reduce stale-state or hallucinated actions.
- semantic recovery: cortex path should improve debug, file-refresh, and routine-recovery cases.
- capacity: LoRA rank, alpha, sequence length, and learning rate should explain remaining failures if interface and data are adequate.

Training choices may be selected from validation-set evidence and failure analysis only. The fixed test split is reserved for final acceptance.

Because 8GB 7B QLoRA full-matrix runs can exceed the expected iteration budget, Phase 2B includes a fast matrix mode. Fast mode samples only the training and validation SFT files, balances by task, route, and action metadata, and still evaluates on the unchanged held-out test split. Passing fast mode is not final paper evidence by itself; it is a parameter-selection step before full-matrix confirmation.

## Acceptance

The Phase 2B claim is accepted only if `configs/phase2b_unified_qwen7b.yaml` gates pass and every number is traceable to run directories, raw episode logs, summaries, comparisons, and config hashes.

The machine-checkable gate is `python -m reflexlm.cli.check_phase2b_gates`. It must consume same-split evaluation JSON for:

- unified `nsi_state_v2` Qwen7B adapter candidates;
- Qwen7B prompt-only baseline;
- Qwen7B ReAct baseline;
- small-model reflex-only baseline.
- Phase 2B generalization audit from `python -m reflexlm.cli.analyze_phase2b_generalization`.
- Phase 2B overfit audit from `python -m reflexlm.cli.analyze_phase2b_overfit`.

The checker reports fixed completion and safety, improvement against the strongest 7B text baseline, model-call and token-cost reduction, route-sensitive gain against reflex-only, and whether low-level reflex latency/call counts remain bounded. Missing same-split baselines make the Phase 2B claim incomplete rather than passed.

The generalization audit is a hard prerequisite for fast-mode interpretation. It requires zero train/test episode overlap, zero `task_type + scenario_template` overlap, zero exact prompt overlap, zero prompt+target overlap, and zero hidden-marker leakage in prompts. Action, command, and file-slot overlap are reported but not treated as automatic failure because the bounded motor schema and allowlists intentionally reuse action primitives.

The overfit audit is a separate hard prerequisite. It reports train/validation loss gaps, semantic nearest-neighbor similarity between held-out test prompts and training prompts, target/action reuse, command-slot reuse, and file-slot reuse. Exact leakage or excessive semantic nearest-neighbor similarity fails the Phase 2B claim. High action or slot reuse without exact leakage is a caveat that must be discussed because it can indicate a narrow synthetic motor schema rather than robust external validity.

If the unified 7B run fails, the paper conclusion must say that low-level reflex evidence is supported while large-model nervous-interface integration remains unresolved.
