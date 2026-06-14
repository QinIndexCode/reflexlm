# Phase 2C Native Nervous-Model Validation

## Purpose

Phase 2C replaces the rejected Phase 2B JSON motor path. The goal is not to run a multi-agent or multi-expert orchestration stack. The goal is to validate a single-model architecture in which a shared LLM backbone receives native system-state latent input, maintains persistent runtime state, performs internal routing and inhibition, and emits explicit action-head outputs.

## Architecture Boundary

- Low-level NSI remains the receptor/reflex/inhibition/router reference.
- The 7B backbone is not responsible for every motor action.
- Debug/Semantic Cortex is invoked only when the synaptic target head escalates semantic work.
- The model does not generate JSON as its primary motor output.
- Runtime serializes the selected `action_type`, command slot, file slot, route, target, and confidence into the existing external motor schema.
- `test_failure_reflex` is an internal `ESCALATE_TO_DEBUG_CORTEX` target, not a low-level reflex action.
- `external_file_change_reflex` uses the stale-state receptor path and must emit `REFRESH_STATE` before file reads.

## Required Heads

Phase 2C requires explicit heads for:

- `internal_target`
- `action_type`
- `command_slot`
- `file_slot`
- `route`
- `confidence`
- `inhibition`
- `salience`
- `risk`
- `prediction_error`

The head outputs are the auditable motor interface. JSON is only a runtime serialization detail after the heads have selected the bounded action.

## Head Supervision Corpus

The current Phase 2C head-supervision corpus is:

- Corpus: `artifacts/datasets/phase2c_native_head_wide_ood_scenario_holdout_seed313`
- Manifest: `artifacts/reports/phase2c_native_nervous/phase2c_head_dataset_manifest.json`
- Source split: `artifacts/datasets/phase1b_wide_ood_scenario_holdout_seed313_debug_v3`
- Prompt style: `phase2c_head_state_v1`
- Train/validation/test records: `6010 / 1341 / 1343`
- Aggregate records: `8694`
- JSON text target: `false`
- Leakage audit: passed for train, validation, and test

The corpus keeps model-visible text to receptor state, legal action mask, and candidate command/file slots. Hidden `recovery_hint`, `scenario_template`, oracle-action markers, and text JSON targets are not serialized into the prompt. NSI reference signals are stored as sidecar supervision/latent-injection fields, not as the motor output channel.

Label contract:

- `test_failure_reflex` rows use `internal_target=ESCALATE_TO_DEBUG_CORTEX`.
- `external_file_change_reflex` rows with stale/external file evidence use `runtime_overrides=["stale_state_refresh_receptor"]` and `action_type=REFRESH_STATE`.
- `dangerous_action_interception` rows use `internal_target=INHIBIT`, `route=safety_cortex`, and `action_type=BLOCK`.

## Training Entrypoint

The guarded runner is `scripts/run-phase2c-native-heads.ps1`.

- `-Stage prepare`: rebuilds or verifies the native-head corpus.
- `-Stage canary -AllowLongRun`: loads Qwen7B and trains a small limited-record canary.
- `-Stage train -AllowLongRun`: runs the configured native-head adapter training.
- `-Stage evaluate -AllowLongRun`: evaluates `--policy qwen_native_heads`.
- `-Stage gate`: writes the Phase 2C gate report without starting training.

The runner uses `artifacts/control/phase2c_native_heads.lock` to refuse duplicate execution and respects `artifacts/control/phase2c_native_heads.paused` unless `-OverridePause` is explicitly supplied.

Phase 2C training must use `reflexlm.cli.train_phase2c_native_heads`, not the legacy `train_qwen_qlora` SFT path. The training objective is a weighted sum over explicit head losses:

- classification: `action_type`, `internal_target`, `route`, `command_slot`, `file_slot`
- regression: `confidence`, `inhibition`, `salience`, `risk`, `prediction_error`

The saved artifact contract is:

- `backbone_adapter/`: LoRA adapter for the shared backbone
- `native_heads.pt`: explicit head weights
- `head_config.json`: native-head dimensions and latent injection setting
- `training_summary.json`: loss curves, validation head metrics, config hash, and run manifest

Evaluation must use `reflexlm.cli.evaluate --policy qwen_native_heads`. That policy preserves the low-level NSI path for reflex/inhibition states and invokes the Qwen native-head path only for internal Debug/Semantic Cortex escalation. Evaluations that call Qwen for pure low-level reflex tasks do not satisfy the Phase 2C architecture boundary.

The initial 7B matrix should start with the two smallest native-head runs in `configs/phase2c_native_nervous.yaml`; full-matrix or paper-claim evaluation is not allowed until these runs show non-trivial validation accuracy without overfit leakage.

## Rejected Paths

The following paths are no longer accepted as evidence for the paper architecture:

- Direct JSON text generation as the primary motor output.
- Multiple route adapters presented as separate internal experts.
- External agent routing between prompt-only, ReAct, route experts, and reflex policies.
- Free-form shell generation.
- Treating parse failures as model reasoning failures rather than motor-interface design failures.

## Validation Gates

The fixed split is accepted only if the explicit-head path reaches:

- `completion >= 0.95`
- `dangerous_block = 1.0`
- `parse_failure = 0.0`
- no low-level reflex latency regression against `nsi_v20_debug_lexical_tiny`
- no 7B calls for pure low-level reflex tasks
- Debug Cortex gain of at least `30pp` over Qwen7B prompt-only/ReAct and the old JSON motor path

The paper claim cannot be upgraded unless the generalization and overfit audits also pass and a quasi-real trajectory split is evaluated.

## Interpretation

Passing Phase 2C would support the paper claim that an LLM can be reorganized into a native nervous-interface model with receptors, latent synaptic state, inhibition, internal routing, cortex processing, and bounded motor heads. Failing Phase 2C does not invalidate the low-level NSI result; it means the large-model native architecture is still unresolved.
