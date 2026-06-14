# Phase2AT Learned Bounded Patch Generation Preregistration

## Purpose

Phase2AT is the next claim-bearing stage after Phase2AR/Phase2AS. Phase2AR
supports bounded symbolic structural patch proposal on non-sealed public-repo
holdout tasks, but Phase2AS blocks any reinterpretation of that result as
learned patch generation. Phase2AT exists to test whether a native nervous
policy package can author bounded patch candidates under runtime-visible
constraints, not merely select recorded candidates or invoke a symbolic repair
rule.

## Claim Boundary

Supported only if the gates below pass:

- Native package proposes bounded patch candidates using an explicit learned
  patch candidate schema.
- Runtime execution applies the generated candidate in a repo sandbox, verifies
  the selected test, rolls back safely, and records all hashes.
- Non-full controls are nonzero and below ceiling, and the full package beats
  the best control by a preregistered margin.

Not supported by Phase2AT alone:

- Free-form patch generation outside the bounded schema.
- Open-ended debugging generalization.
- Production autonomy.
- Sealed cross-model transfer.
- Epoch-making architecture.

## Required Package Contract

The package manifest must pass
`audit_phase2at_learned_patch_generation_package_gate.py`:

- `patch_proposal_strategy = "learned_bounded_candidate"`
- `learned_patch_generation_enabled = true`
- `patch_candidate_schema_version = "phase2at.learned_bounded_patch_candidate.v1"`
- All open-repair control capabilities enabled:
  `patch_proposal_head`, `test_selection_head`, `rollback_safety_head`,
  `stop_condition_head`, `bounded_edit_scope_policy`,
  `progress_monitor_receptors`, `verification_state_receptors`
- `json_text_target = false`
- `native_head_calls_enabled = true`

Old packages with only open-repair control heads are not Phase2AT packages.
Symbolic runtime generators and recorded patch artifacts cannot be relabeled as
learned generation targets.

## Data Contract

Phase2AT data must be non-sealed and repo-origin-disjoint. Each row must provide
runtime-visible evidence only:

- failing test output or traceback summary
- changed/watched file relation
- bounded edit scope
- allowed patch operation family
- structural context required for the patch candidate schema

Each row must exclude:

- sealed v2/v3 content or failure feedback
- gold labels in visible text
- candidate slot markers
- exact test-name shortcuts
- recorded correct patch text as a generation target
- symbolic generator output as a generation target

## Learned Patch Candidate Schema V1

The learned output is a bounded operation descriptor, not arbitrary text:

- `target_path`
- `operation`
- `anchor`
- `before_fragment_hash`
- `after_fragment_template_id`
- `literal_or_symbol_payload`
- `safety_constraints`
- `verification_command_slot`

The runtime materializer may convert the descriptor to a diff only after
validating edit scope, anchor consistency, file hash, operation allowlist, and
rollback safety. This keeps Phase2AT within bounded native motor output rather
than JSON SFT or free shell autonomy.

## Gates

Data gate:

- no sealed overlap
- repo-origin-disjoint train/val/holdout
- visible text has no gold/candidate marker
- controls are nonzero and below ceiling
- split hashes and config hashes recorded

Training gate:

- training summary records package schema, split hashes, control baselines,
  throughput, low-level Qwen call target `0`, and the training contract fields
  required by the package gate
- recorded patch artifacts are not used as generation targets
- symbolic generator outputs are not used as generation targets

Execution gate:

- full success rate `>= 0.85`
- full minus best non-full control `>= 0.15`
- full minus symbolic restricted control `>= 0.10`
- hallucination count `0`
- unauthorized write count `0`
- low-level Qwen calls `0`
- rollback restores failure for every successful row

Stop conditions:

- package gate fails
- data gate fails
- controls are all zero or near ceiling
- full does not beat best control
- any sealed-derived feedback is detected

## Relationship To Current Evidence

Current Phase2AR/Phase2AS evidence remains valuable but bounded:

- Phase2AR: bounded symbolic structural repair proposal, multi-seed and same
  family cross-model reproduction on non-sealed public repo holdout.
- Phase2AS: boundary audit preventing learned/freeform overclaim.
- Phase2AT: required next evidence layer for learned bounded patch candidate
  generation.
