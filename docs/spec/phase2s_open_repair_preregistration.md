# Phase2S open-ended repair preregistration draft

## Purpose

Phase2S is separate from the bounded command-selection dataset. Its purpose is to test the remaining unsupported claim: whether the bounded native nervous-interface package improves open-ended repository repair under sandboxed, auditable conditions.

## Non-negotiable boundary

- Do not use sealed v3 failures or successes to design Phase2S tasks.
- Do not train on sealed v3.
- Do not tune hyperparameters from sealed v3.
- Do not claim production autonomy before rollback, sandbox, incident, and safety gates pass.
- Do not hardcode task names, expected patches, file paths, or benchmark-specific commands.

## Required benchmark properties

- Real or public synthetic-safe repositories with reproducible failing tests.
- Agent must inspect evidence, choose commands, edit code, run tests, and decide when to stop.
- Every action must run in a disposable sandbox.
- Patch diffs, command logs, test outputs, rollback events, and safety blocks must be recorded.
- Holdout repositories must be disjoint from training and design examples.
- Baselines must include at least prompt-only, ReAct, a modern coding-agent style loop, native-head-only/no-cache, no-NSI, and continuation-only where applicable.
- Difficulty must be graded rather than binary-only: candidate count, evidence density, repair depth, failure observability, and ambiguity class must be recorded per task.

The preregistered factor levels are fixed before data generation:

- Candidate count: `2`, `3`, `4`.
- Evidence density: `low`, `medium`, `high`.
- Repair depth: `one_edit`, `two_edits`, `stale_state_refresh`.
- Failure observability: `direct_traceback`, `indirect_changed_file_relation`, `ambiguous_same_intent_command`.
- Ambiguity class: `same_intent_command`, `same_file_read`, `stage_transition`.

## Modern coding-agent baseline

The `modern_coding_agent_loop` baseline must be operationally specified before
the full gate. The specification must fix model or provider, tool budget,
context policy, retry policy, edit permission scope, stop rule, and cost or
command-budget accounting. It must use the same sandbox and write-scope safety
constraints as the full package. A declared baseline is not enough; metrics must
come from logged execution artifacts.

## Minimum task families

- Import or dependency mismatch.
- Localized failing unit assertion.
- Stale snapshot update.
- Configuration or environment-marker mismatch.
- Multi-file relation that requires changed-file and traceback evidence.

## Metrics

- Task success.
- Patch correctness.
- Test pass rate.
- Number of commands.
- Number of edits.
- Rollback success.
- Unauthorized write count.
- Low-level Qwen calls.
- Allowlist or state hallucination.
- Stop-condition correctness.

## Gates before any claim upgrade

- Data gate: repo-disjoint, no hidden/gold markers, no sealed overlap, reproducible failure seeds, open license or redistribution permission.
- Smoke gate: at least one complete repair loop succeeds without hardcoded commands or patch templates.
- Full non-sealed gate: full package beats the best measured baseline by at least `0.10` on task success and by at least `0.10` on patch correctness, with bootstrap confidence intervals and no claim upgrade on ties.
- Safety gate: zero unauthorized writes, zero allowlist or state hallucinations, zero low-level Qwen calls, rollback success `1.0`, false-completion rate `0.0`, and no safety regression against the best baseline.
- Transfer gate: held-out repositories pass without using failures for training feedback.
- Claim-upgrade gate: external public holdout, multi-seed, multi-model, independent reproduction, and unanimous read-only audit synthesis must all pass. Audit agreement alone cannot upgrade the claim without the metrics above.

## Stop conditions

- Any evidence of hardcoded task-specific behavior.
- Source repository mutation outside the sandbox.
- Hidden/gold/answer leakage.
- Full package fails to beat best baseline.
- Safety or rollback failure.

## Relationship to current paper

Phase2S is future work. It must not be used to inflate the bounded command-selection dataset claim. If Phase2S succeeds, it can support a separate research paper or a later dataset extension.

The machine-checkable preregistration template is `docs/spec/phase2s_preregistration_template.json`. It must pass `src/reflexlm/cli/check_phase2s_preregistration.py` before any Phase2S data generation or training.

## Current infrastructure smoke

The first Phase2S implementation stage adds a synthetic-safe open-repair smoke
generator and audit:

- `src/reflexlm/cli/generate_phase2s_open_repair_smoke.py`
- `src/reflexlm/cli/audit_phase2s_open_repair.py`
- `src/reflexlm/cli/collect_phase2s_public_repair_traces.py`
- `tests/test_phase2s_open_repair_smoke.py`

The local smoke generates `15 / 15 / 9` train/validation/holdout rows with real
sandboxed pytest-before-patch, pytest-after-patch, rollback, patch-diff, command
log, test-output, and sandbox-integrity artifacts. Its data-health audit passes,
but `claim_bearing_training_ready=false` because the rows are synthetic-safe
fixtures. The pretrain gate must therefore remain failed with
`do_not_train_phase2s_from_synthetic_smoke_only`. The next valid step is public
repo repair trace collection, not training on this smoke split.

The public-repo collector is a claim-bearing candidate data path, not a claim by
itself. It copies read-only public repos into disposable sandboxes, injects
bounded repair faults only inside the sandbox, records fail/patch/pass/rollback
artifacts, and requires the Phase2S data-health gate before any smoke training.
The audit now treats repo-disjointness as both `repo_id` disjointness and
`repo_url_or_origin` disjointness, so the same repository cannot be reused across
train/validation/holdout by renaming it.
