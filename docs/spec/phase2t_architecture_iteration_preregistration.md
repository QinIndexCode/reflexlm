# Phase2T Architecture Iteration Preregistration

## Purpose

Phase2T is the next architecture iteration gate after the bounded Phase2S evidence.
It is not a paper-claim upgrade by itself. Its role is to test whether the Native
Synaptic Interface can move from controlled command selection into a sandboxed
repair-loop architecture with measurable gains over modern agent-loop baselines.

The current paper position stays bounded until Phase2T and follow-up reproduction
gates pass. Existing Phase2J/K/L/M/S results can support controlled mechanism
claims, but they do not prove production autonomy, open-ended debugging
generalization, or an epoch-making agent architecture.

## Research Boundary

Supported starting point:

- Controlled native-head and NSI-latent mechanisms can improve bounded software
  agent decisions under preregistered non-sealed and final sealed evaluations.
- Phase2S provides stronger bounded evidence, including public-repair style
  controls and cross-model same-family reproduction.
- Sealed evaluations remain final evaluation artifacts only.

Unsupported until Phase2T gates pass:

- Production autonomy.
- Free-form shell or unrestricted editing competence.
- Open-ended debugging generalization.
- Cross-family sealed transfer.
- Epoch-making architecture claims.

## Architecture Iteration

Phase2T must add repair-loop capabilities that are not reducible to command-slot
selection:

- `patch_proposal_head`: proposes bounded source edits from runtime-visible
  evidence.
- `test_selection_head`: chooses the next verification command under an
  allowlisted command policy.
- `rollback_safety_head`: triggers rollback when patches are unsafe or fail
  verification.
- `stop_condition_head`: emits a final stop only after objective verification.
- `verification_state_receptors`: encode pass/fail/test-output state after each
  command.
- `incident_timeout_receptors`: encode timeouts, crashes, sandbox faults, and
  tool failures.
- `progress_monitor_receptors`: distinguish progress, no-progress, and repeated
  failure loops.
- `bounded_edit_scope_policy`: enforces repository-local, task-local edits.

Training and runtime must share the same implementation. Any train/runtime drift
blocks claim upgrade.

## Data Policy

Phase2T data must be preregistered before training.

- Public or synthetic-safe repos only.
- Repo-origin-disjoint train, validation, and holdout splits.
- License and provenance metadata required for every public trace.
- No sealed v2/v3 data, sealed failures, hidden hints, gold labels, candidate
  markers, expected patches, benchmark-specific command hardcoding, or
  repo-specific path shortcuts.
- Baselines must be measured by code artifacts, not declared in prose.

## Benchmark Shape

The benchmark must exercise a real repair loop:

- Inspect evidence.
- Choose safe commands.
- Propose bounded edits.
- Run tests.
- Roll back unsafe or failed edits.
- Stop only when verification supports completion.

Required task families:

- Dependency or import mismatch.
- Localized unit assertion.
- Stale snapshot or fixture update.
- Configuration or environment marker.
- Multi-file traceback relation.
- Regression after partial repair.
- Safety-blocked command temptation.
- False-completion trap.

Required graded factors:

- Candidate count: 2, 3, 4.
- Evidence density: low, medium, high.
- Repair depth: one edit, two edits, stale-state refresh.
- Failure observability: direct traceback, indirect changed-file relation,
  ambiguous same-intent command.
- Ambiguity class: same-intent command, same-file read, stage transition,
  patch-location ambiguity.
- Safety pressure: none, unsafe-command lure, rollback-required.

## Controls And Baselines

Required controls:

- `full_package`
- `native_head_only_no_cache`
- `no_nsi_latent`
- `continuation_only`
- `prompt_only`
- `react`
- `modern_coding_agent_loop`
- `patch_head_only`
- `no_rollback_safety`

The modern coding-agent loop must have fixed model/provider, tool budget,
context policy, retry policy, edit permission, stop rule, and cost/command
budget before the full gate.

## Model And Seed Matrix

Minimum preregistered matrix:

- At least two model families.
- At least one non-Qwen family.
- At least three seeds per model.
- Same split hashes across models and seeds.
- Loader-risk review before using a new family.
- No claim upgrade from Qwen-only evidence.

## Metrics

Required metrics:

- Task success.
- Patch correctness.
- Test-pass recovery.
- Command count.
- Edit count.
- Rollback success.
- Unauthorized write count.
- False-completion rate.
- Stop-condition correctness.
- Low-level Qwen calls.
- Allowlist or state hallucination.
- Modern baseline cost.
- Time to verified repair.

## Gates

Data gate:

- Data health passes.
- Split hashes recorded.
- Repo disjointness verified.
- Baselines measured, not declared.
- Leakage and marker audits pass.

Smoke gate:

- No sealed feedback.
- Full package beats source-overlap and prompt/ReAct baselines on the
  preregistered validation split.
- No unsafe writes, false completions, hallucinated state, or low-level Qwen
  calls.

Full gate:

- Full beats best modern baseline by at least 10 percentage points on task
  success.
- Full beats best native ablation by at least 10 percentage points on task
  success.
- Full beats best modern baseline by at least 10 percentage points on patch
  correctness.
- Safety is non-inferior to the safest baseline and satisfies zero-tolerance
  thresholds.

Transfer gate:

- Holdout passes under repo-origin-disjoint public traces.
- Cross-family reproduction passes with confidence intervals.
- Independent reproduction package is archive-only runnable.

Claim gate:

- No production autonomy, open-ended debugging generalization, or epoch-making
  architecture claim is allowed unless all Phase2T gates pass and the result is
  reproduced across model families and seeds.

## Retrospective Status After Current Phase2T Artifacts

The current Phase2T artifact chain does not satisfy this preregistered
repair-loop claim gate. It provides bounded semantic-required command-selection
transfer plus a zero-control root-cause audit and non-sealed
baseline-feasibility sanity audit. It does not yet provide the registered modern
coding-agent loop baseline, patch-correctness gate, rollback-safety gate,
stop-condition gate, cross-family confidence intervals, or independent
archive-only reproduction required above.

Therefore Phase2T may be cited only as bounded command-selection evidence with
an all-zero-control caveat. It must not be cited as proof of repair-loop
superiority, production autonomy, open-ended debugging generalization, or an
epoch-making architecture.

## Stop Rules

Stop and freeze failure evidence if any of the following occurs:

- Data health fails.
- Sealed feedback is detected in design, training, or tuning.
- Smoke fails.
- Full gate delta fails.
- Safety threshold fails.
- Cross-family reproduction fails.
- Modern baseline is missing or only declared.
- Train/runtime mechanism drift is detected.

Failure evidence should guide the next non-sealed preregistration. It must not
be backfilled into sealed-derived training data.

## Next Implementation Gate

The first implementation step after this preregistration is public repair-loop
repository spec collection, not training. The spec gate must pass before any
dynamic trace collection:

- CLI: `src/reflexlm/cli/collect_phase2t_public_repair_loop_specs.py`
- Test: `tests/test_phase2t_public_repair_loop_specs.py`
- Initial public repo spec: `docs/spec/phase2t_public_repair_loop_repo_specs.json`
- Allowed next action after pass: `run_phase2t_dynamic_repair_trace_collection`

This gate validates public repo origin, pinned commit, license metadata,
repo-origin-disjoint train/val/holdout splits, task-family coverage, graded
difficulty coverage, repair-loop contract flags, and absence of sealed, gold,
candidate-slot, or expected-patch markers. It intentionally returns
`claim_bearing_training_ready=false`; training remains blocked until dynamic
trace collection and data-health gates pass.

The dynamic trace collector consumes the spec only after that gate passes:

- CLI: `src/reflexlm/cli/collect_phase2t_dynamic_repair_traces.py`
- Test: `tests/test_phase2t_dynamic_repair_traces.py`

The collector runs in disposable sandboxes, records patch/test/rollback/stop
artifacts, keeps source repositories read-only, and emits Phase2T repair-loop
schema rows. Its manifest also keeps `claim_bearing_training_ready=false`; a
separate data-health/pretrain gate is still required before training.

The data-health/pretrain gate is:

- CLI: `src/reflexlm/cli/audit_phase2t_dynamic_repair_traces.py`
- Test: `tests/test_phase2t_dynamic_repair_trace_audit.py`

It requires the collector manifest, train/val/holdout JSONL files, repair-loop
schema, architecture-target annotations, artifact presence, baseline metadata,
repo-disjoint splits, validation pressure-matrix coverage, zero low-level Qwen
calls, no unsafe writes, and no sealed or answer-marker leakage before smoke
training is allowed.

The native-head training conversion is:

- CLI: `src/reflexlm/cli/build_phase2t_head_dataset.py`
- Test: `tests/test_phase2t_head_dataset.py`

It reuses the validated public-repair command-identity latent conversion but
records Phase2T prompt style, repair-loop source trace metadata, and Phase2T
effective split hashes so Phase2S evidence is not silently backfilled into
Phase2T.
