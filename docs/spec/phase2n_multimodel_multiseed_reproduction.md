# Phase2N Multi-Model / Multi-Seed Reproduction Preregistration

## Evidence Boundary

Phase2M-v2 relation-key is the current frozen evidence boundary. It supports a bounded claim: in this controlled terminal/process/filesystem/time setting, the full `NativeNervousPolicyPackage` with native heads, NSI latent, Debug Cortex routing, continuation state, and bounded motor heads can beat no-NSI, native-head-only/no-cache, continuation-only, prompt-only, and ReAct on the sealed semantic-required benchmark. Phase2N adds Qwen2.5-7B sealed-final seed robustness after preregistration, but only for the same reference model family.

It does not yet support broad real-world debugging, production autonomy, free shell synthesis, JSON motor generation, sealed cross-model transfer, or broad real-repo generalization. Phase2N exists to test robustness gaps without using sealed-v3 failures or successes as training signals.

## Center Claim

The center claim stays narrow:

- A single native nervous-interface package can provide bounded semantic-required command selection through explicit native heads, runtime-visible NSI latent state, continuation memory, and Debug Cortex routing.
- The mechanism must remain inspectable through ablations and measured baselines.
- Stronger claims require preregistered cross-model, cross-seed, and cross-repo evidence.

## Non-Goals

- Do not tune from sealed v3.
- Do not introduce free shell generation, JSON SFT motor output, multi-adapter route experts, or open-ended production autonomy.
- Do not hardcode model-specific, repo-specific, test-name-specific, candidate-slot, or sealed-failure patterns.
- Do not treat small-model failures as evidence against the architecture until compatibility, capacity, and tokenization risks are separately audited.

## Model Ladder

The first reproduction ladder prioritizes small, locally feasible models before another 7B run.

| Tier | Model | Purpose | Claim Weight |
| --- | --- | --- | --- |
| Q0 | `Qwen/Qwen2.5-0.5B-Instruct` | Pipeline and tokenizer/backbone compatibility; expected to be capacity-limited | plumbing only unless it passes all non-sealed gates |
| Q1 | `Qwen/Qwen2.5-1.5B-Instruct` | First meaningful small-model reproduction | claim-bearing if gates pass |
| Q3 | `Qwen/Qwen2.5-3B-Instruct` | Stronger small-model reproduction under local GPU constraints | claim-bearing if gates pass |
| Q7 | `artifacts/models/Qwen2.5-7B-Instruct` | Current reference model; repeat for seed robustness | reference claim-bearing |
| X1 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | first non-Qwen decoder-family robustness probe after loader-risk review | claim-bearing non-sealed evidence if gates pass |
| X2 | `HuggingFaceTB/SmolLM2-360M-Instruct` | second non-Qwen family probe under local GPU constraints after gated/open-model risk review | claim-bearing non-sealed evidence if gates pass |

Non-Qwen candidates are deliberately second-stage because LoRA target modules, tokenizer behavior, hidden size, and AutoModel compatibility can confound mechanism interpretation. If used, they must first pass a compatibility smoke and a written loader-risk note.

## Seed Ladder

Use fixed preregistered seeds:

- Smoke: `13`
- Robustness minimum: `13`, `29`, `47`
- Strong robustness extension if time permits: `13`, `29`, `47`, `71`, `101`

No seed may be selected after looking at sealed-v3 outcomes.

## Phase2N Execution Order

1. Write model registry and download manifests before training.
2. Pull Qwen small models into `artifacts/models/`.
3. Run model compatibility smoke: tokenizer load, backbone config load, native-head forward/training micro-run, and summary schema check.
4. Run non-sealed Phase2M-v2 relation-key smoke for Q0/Q1/Q3 with fixed config:
   - LoRA `r16/alpha32`
   - learning rate `1e-4`
   - max length `256`
   - pairwise disabled
   - additive latent fusion
   - command candidate encoder `features_only`
   - `max_train_records=128`
   - validation uses the frozen Phase2M-v2 relation-key validation split
5. Only if smoke passes, run full non-sealed reproduction:
   - `max_train_records=1024` cap for direct comparability with Phase2M-v2 current evidence; the frozen current train split contains `512` rows
   - `max_val_records=512` cap; the frozen current val split contains `256` rows
   - public holdout diagnostic on the frozen holdout split
6. Only if non-sealed full passes, package and optionally run sealed v3 as final evaluation.
7. Repeat successful model tier over the preregistered seed set.
8. Summarize mean, standard deviation, bootstrap confidence interval, pass rate, worst seed, and failure categories.

## Gates

### Compatibility Gate

- model files are downloaded under `artifacts/models/`
- tokenizer loads without custom test-specific code
- `AutoConfig` reports a valid hidden size
- native-head adapter construction does not require hardcoded target-module changes
- a 4-row train/val micro-run writes a valid training summary

### Smoke Gate

- validation command-slot accuracy `>= 0.85`
- model minus source-overlap `>= 0.10`
- low-level Qwen calls target `0`
- pairwise disabled
- no JSON motor target
- summary records base model, seed, split hashes, config hash, candidate encoder, latent fusion, and throughput

### Full Non-Sealed Gate

- full minus native-head-only `>= 0.10`
- full minus source-overlap `>= 0.15`
- public holdout command-slot accuracy `>= 0.85`
- public holdout model minus source-overlap `>= 0.10`
- allowlist hallucination `0`
- low-level Qwen calls `0`

### Multi-Seed Gate

- all required seeds complete without OOM or artifact/hash mismatch
- pass rate is reported, not hidden
- worst-seed metrics are reported
- mean and bootstrap confidence intervals are reported
- failures are frozen as evidence rather than retried with post-hoc tuning

### Sealed Gate

Sealed v3 is final evaluation-only. It may run only after non-sealed full gates pass for a preregistered model/seed. A sealed failure freezes transfer failure evidence and must not feed back into data generation, sampling, hyperparameters, model choice, or seed choice.

## Phase2O Architecture Innovation Audit

The project should not claim "epoch-making architecture" as a conclusion until it passes a separate architecture audit. The audit asks whether the result is a real architectural mechanism rather than a benchmark-specific recipe.

Required evidence:

- Mechanism separation: native heads, NSI latent, continuation memory, Debug Cortex routing, and bounded motor heads each have registered ablations.
- Baseline pressure: source-overlap, candidate-feature, prompt-only, ReAct, no-NSI, native-head-only/no-cache, continuation-only, wrong-cache, and cache-erased controls are measured by code.
- Robustness: at least three seeds and at least three model scales/families, with failure rates reported.
- Externality: repo-disjoint public holdout, then larger public trace collection, before any broad real-repo claim.
- Reproducibility: all splits, hashes, model ids, package manifests, and gates are captured before sealed evaluation.
- Safety boundary: no free shell generation or production autonomy claim unless separately preregistered and evaluated.

If any item fails, the correct paper posture is "bounded native nervous-interface mechanism evidence", not "general LLM agent architecture breakthrough".

## Current Phase2N Status

- Qwen2.5-0.5B and Qwen2.5-1.5B were downloaded and passed compatibility micro-runs.
- Qwen2.5-1.5B completed full non-sealed reproduction for seeds `13`, `29`, and `47`; all three seeds passed the relation-key full gate and public holdout gate.
- Qwen2.5-3B was downloaded and passed the compatibility micro-run.
- Qwen2.5-3B completed full non-sealed reproduction for seeds `13`, `29`, and `47`; all three seeds passed the relation-key full gate and public holdout gate.
- Qwen2.5-7B reference now has three non-sealed relation-key seeds: the original Phase2M-v2 reference seed `13`, plus Phase2N reproduction seeds `29` and `47`; all three seeds passed the relation-key full gate and public holdout gate.
- SmolLM2-1.7B was rejected before training because the model pull stalled under Xet HTTP fallback with incomplete weight bytes; this is preserved as loader-risk evidence, not a model failure.
- Gemma-2-2B was rejected before training because the repository is gated and the local environment lacks authorized HuggingFace access; this is preserved as access/loader-risk evidence, not a mechanism failure.
- Mistral-7B-v0.1 was rejected before compatibility because the Xet-backed large-weight download stalled even after installing `hf_xet` and excluding duplicate `.bin` weights; this is preserved as loader/download infrastructure evidence, not a mechanism failure.
- OpenLLaMA-3B-v2 was rejected before compatibility because the large weight download remained stalled at 0 bytes even after enabling `hf_transfer`; this is preserved as loader/download infrastructure evidence, not a mechanism failure.
- TinyLlama-1.1B passed the non-Qwen loader-risk review, compatibility micro-run, smoke, and full non-sealed reproduction for seeds `13`, `29`, and `47`; all three seeds passed the relation-key full gate and public holdout gate.
- SmolLM2-360M passed the non-Qwen loader-risk review, compatibility micro-run, smoke, and full non-sealed reproduction for seeds `13`, `29`, and `47`; all three seeds passed the relation-key full gate and public holdout gate.
- Qwen2.5 1.5B/3B/7B results are same-family non-sealed reproduction evidence. TinyLlama and SmolLM2 add two non-Qwen cross-family non-sealed reproduction probes.
- Qwen2.5-7B now has preregistered sealed-final transfer evidence for seeds `13`, `29`, and `47`: all three full packages complete `64 / 64`, prompt-only and ReAct complete `0 / 64`, native-head-only/no-cache and continuation-only complete `0 / 64`, no-NSI completes at most `14 / 64`, allowlist hallucination is `0`, and low-level Qwen calls are `0`.
- Phase2P extends sealed-final transfer after separate preregistration. Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-7B, TinyLlama-1.1B, and SmolLM2-360M each pass seeds `13`, `29`, and `47` on sealed `external_trace_v3_semantic_required`; all `15 / 15` gates pass, full completion minimum is `1.000`, native-head-only/no-cache and continuation-only maxima are `0.000`, no-NSI maximum is `0.6875`, minimum `full_minus_no_nsi` is `0.3125`, allowlist hallucination is `0`, and low-level Qwen calls are `0`.
- This is now sealed cross-model transfer evidence for the bounded semantic-required Debug Cortex/native nervous-interface mechanism. It is still not evidence for production autonomy, open-ended software repair, unrestricted shell generation, broad public real-repo generalization, independent external reproduction, or epoch-making architecture status.

## Immediate Next Actions

1. Preserve the Phase2O architecture-innovation audit verdict after Phase2P: current evidence is stronger bounded claim-bearing mechanism evidence, not an epoch-making architecture proof.
2. Add any larger open non-Qwen model only after a loader-risk note and compatibility smoke; do not treat loader/download/access failures as mechanism failures.
3. Expand public read-only trace breadth and independent reproduction hardening before any broad real-repo or top-tier architecture claim.
4. Add production-autonomy and open-ended debugging benchmarks only as separate preregistered studies with sandbox, authorization, rollback, incident, and safety gates.
