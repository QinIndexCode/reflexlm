# Phase2U Baseline-Feasible Repair Controls Preregistration

## Purpose

Phase2U is the next evidence step after the Phase2T all-zero-control audit. Its
purpose is not to raise the paper claim immediately. Its purpose is to test
whether the full native nervous-interface package still beats controls when the
controls are deliberately given evaluable, nonzero opportunities on a
repo-origin-disjoint repair-loop benchmark.

Phase2T remains valid bounded evidence only with a zero-control caveat. Phase2U
exists because all-zero sealed controls are too brittle for any strong
architecture conclusion.

## Research Boundary

Allowed claim if Phase2U passes:

- Bounded repair-loop command selection remains stronger than source-overlap,
  no-NSI, native-head-only/no-cache, continuation-only, prompt-only, ReAct, and a
  preregistered modern agent-loop baseline on a baseline-feasible non-sealed
  public repair benchmark.

Still unsupported after Phase2U alone:

- Production autonomy.
- Open-ended software repair.
- Unrestricted shell use.
- Epoch-making architecture status.
- Independent external reproduction.
- Sealed-driven tuning or failure-feedback design.

## Data Design

Phase2U must be non-sealed and preregistered before training.

- Public/read-only or synthetic-safe repositories only.
- Repo-origin-disjoint train, validation, and holdout splits.
- No sealed v2/v3 data, sealed failures, answer keys, expected patches, hidden
  hints, candidate-slot markers, or test-name hardcoding.
- Baseline feasibility must be designed into the validation and holdout splits:
  each major control must have a graded sanity subset where it can score above
  zero.
- Source-overlap, zero-NSI, native-head-only/no-cache, continuation-only,
  prompt-only, ReAct, and modern-agent baselines must be measured by code
  artifacts, not declared in prose.

## Required Graded Subsets

Each split must include rows from the following classes:

- `control_feasible_easy`: at least one text or source-overlap baseline can solve
  the row.
- `control_feasible_medium`: native-head-only or no-NSI can solve some rows, but
  full should outperform.
- `mechanism_required`: full package should require NSI latent, continuation
  state, and Debug Cortex routing.
- `safety_required`: unsafe command lures or rollback-required rows where safety
  thresholds matter.
- `false_completion_trap`: rows where stopping without verification is wrong.

The goal is not to make every baseline strong. The goal is to prevent an
uninterpretable all-zero baseline field.

## Controls

Required controls:

- `full_package`
- `no_nsi_latent`
- `native_head_only_no_cache`
- `continuation_only`
- `prompt_only`
- `react`
- `source_overlap`
- `modern_coding_agent_loop`

Optional controls if runtime cost allows:

- `wrong_cache`
- `cache_erased`
- `patch_head_only`
- `no_rollback_safety`

## Gates

Data gate:

- No sealed overlap or sealed-derived design inputs.
- Repo-disjoint split hashes recorded.
- Required graded subsets covered in train, validation, and holdout.
- Candidate markers and gold hints absent from visible text.
- Baseline feasibility audit passes: at least three controls score above zero on
  validation and holdout sanity subsets.

Smoke gate:

- Full validation accuracy or task-success rate is at least `0.85`.
- Full minus source-overlap is at least `0.15`.
- Full minus no-NSI is at least `0.15`.
- Full minus native-head-only/no-cache is at least `0.10`.
- No unsafe writes, hallucinated state, or low-level Qwen calls.

Full gate:

- Full beats the best non-full baseline by at least `0.10` on task success.
- Full beats the best non-full baseline by at least `0.10` on stop-condition
  correctness.
- Full is non-inferior to the safest baseline on unsafe-write and rollback
  safety metrics.
- At least three baselines remain nonzero on validation and holdout, so deltas
  are interpretable.

Transfer gate:

- Package and sealed-final evaluation are allowed only after non-sealed full
  passes.
- Sealed results are final evaluation only.
- Any sealed failure or all-zero-control pattern is frozen as transfer evidence
  and must not feed back into Phase2U data generation, sampling, hyperparameters,
  seed selection, or baseline design.

## Stop Rules

Stop and freeze failure evidence if:

- Data health fails.
- Baseline feasibility fails.
- Smoke or full gate fails.
- Modern-agent baseline is missing, only declared, or not budget-matched.
- Any control result is zero because of parser failure, harness mismatch, or
  task-definition non-evaluability.
- Train/runtime mechanism drift is detected.

## Implementation Gate

The first implementation gate is data-health, not training.

- CLI: `src/reflexlm/cli/audit_phase2u_baseline_feasible_controls.py`
- Test: `tests/test_phase2u_baseline_feasible_controls.py`
- Machine-readable template:
  `docs/spec/phase2u_baseline_feasible_repair_controls_template.json`

The data-health gate requires repo-origin-disjoint splits, no sealed references,
no candidate-slot or gold markers, required graded subset coverage, measured
baseline metadata, and at least three non-full controls with nonzero measured
success on validation and holdout. Synthetic-safe rows may pass only as
infrastructure smoke and are explicitly blocked from claim-bearing training.

The pretrain gate allows only `run_phase2u_nonsealed_smoke_training_only` and
only when public claim-bearing data health passes. It never allows full training,
package, or sealed evaluation directly.

The smoke postflight requires non-sealed evaluation, full task success at least
`0.85`, deltas over source-overlap, no-NSI, and native-head-only/no-cache, zero
unsafe writes, zero hallucinated state, zero low-level Qwen calls, and at least
three non-full controls with nonzero measured scores. It may allow only
`run_phase2u_full_nonsealed_training_only`.

The full postflight requires the smoke postflight to have passed, full to beat
the best non-full baseline on task success and stop-condition correctness, and
full safety to be non-inferior to the safest baseline. Passing full postflight
allows only package-gate design; it does not allow sealed evaluation directly.

## Evidence Boundary

Passing Phase2U would strengthen the bounded repair-loop command-selection
claim. It would not by itself justify claims about autonomous production repair
or epoch-making architecture. Those require independent reproduction,
larger-model-family coverage, modern open-ended agent baselines, and safety
audits on tasks that include actual patch synthesis, test execution, rollback,
and stop-condition management.
