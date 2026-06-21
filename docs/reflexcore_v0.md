# ReflexCore V0 Experiment Protocol

ReflexCore V0 is a bounded computer-native sensory-motor language core. It is
limited to terminal, process, filesystem, and time observations. It does not
support GUI control, vision, robotics, unrestricted shell generation, or
production autonomy.

## Scope

- Inputs: structured `ComputerObservation` plus optional hashed visible-text tokens.
- Outputs: text, typed motor action, command/file slot, target/route, risk,
  salience, prediction-error, and next-state heads.
- Action pathway: current V0 configs enable an `action_vector_residual` head,
  adding a direct structured-observation-to-action-logit path alongside the
  fused text/vector recurrent state. This is a neural architecture path, not a
  runtime rule; all decoded actions still pass typed motor and safety gates.
- Motor decoding: typed action logits are decoded through a conservative
  homeostatic controller. High learned risk can self-block process actions;
  high salience plus high prediction-error can request `REFRESH_STATE` before
  acting. The default prediction-error refresh threshold is `0.05`, calibrated
  to the normalized V0 next-state delta/PE range observed in the local gates.
  Active subprocess states (`running`, `sleeping`, or `blocked`) inhibit
  repeated `RUN_COMMAND`, and prediction-error refresh does not override an
  active-process `WAIT`. State-affordance controls also prevent `DONE` or idle
  actions from hiding visible file/stdout/stderr/refresh/process evidence; after
  a file read, a single remaining allowlisted command can be selected; already
  observed terminal output can converge to `DONE`.
  The external safety gate remains mandatory after decoding.
- Runtime boundary: every real `RUN_COMMAND` must be selected from the
  per-goal allowlist and pass the existing safety gate. Execution is
  shell-free: the allowlisted command string is split into an argument vector
  and launched with `shell=False`.

## Reproducible Smoke Gate

This gate verifies that the pipeline can build a benchmark, train sequence
mode, beat the prompt-only baseline in both offline and closed-loop tests, and
learn an action-conditioned next-state model that beats a copy-current baseline.
It also requires the prediction-error signal to beat a constant-mean baseline,
so the "surprise" head cannot pass by emitting an uninformative scalar.
When `--sequence-mode` is enabled, offline evaluation preserves episode
sequence context instead of evaluating recurrent checkpoints as independent
single-step classifiers.

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_experiment `
  --output-dir "$env:TEMP\reflexcore_experiment_smoke_gate" `
  --config configs/reflexcore/smoke.yaml `
  --episodes-per-task 6 `
  --vocab-size 512 `
  --max-text-tokens 64 `
  --epochs 8 `
  --batch-size 4 `
  --sequence-mode `
  --max-sequence-len 8 `
  --closed-loop-episodes-per-task 2 `
  --required-baseline prompt_only_heuristic
```

Passing this gate is not a final research claim. It only proves the V0 plumbing
and small-model learning path are functioning.

Verified on 2026-06-16 with the command above:

- `passed`: true
- offline safety-gated action accuracy: 0.889
- prompt-only offline action accuracy: 0.556
- closed-loop model success rate: 0.833
- prompt-only closed-loop success rate: 0.250
- dangerous block rate: 1.000
- world-model gate: true
- next-state MSE: 0.00111
- copy-current next-state MSE: 0.00382
- world-model relative improvement: 0.709
- prediction-error gate: true
- prediction-error MAE: 0.0220
- constant-mean prediction-error MAE: 0.0330
- prediction-error relative improvement: 0.333

The same gate remains green after enabling homeostatic motor decoding. Current
local-scale runs use action-conditioned dynamics and derive PE from the
predicted next-state delta norm.

## Cross-Seed Stability Gate

This gate repeats the unified experiment across multiple seeds and requires
every run to pass. It is the preferred smoke-grade evidence because it prevents
a single lucky initialization or split from being treated as a mechanism.

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_stability `
  --output-dir "$env:TEMP\reflexcore_stability_prompt_gate" `
  --config configs/reflexcore/smoke.yaml `
  --seed 13 `
  --seed 17 `
  --seed 23 `
  --episodes-per-task 6 `
  --vocab-size 512 `
  --max-text-tokens 64 `
  --epochs 8 `
  --batch-size 4 `
  --sequence-mode `
  --max-sequence-len 8 `
  --closed-loop-episodes-per-task 2 `
  --required-baseline prompt_only_heuristic `
  --min-pass-rate 1.0
```

Verified on 2026-06-16:

- `passed`: true
- pass rate: 1.000 across 3 seeds
- per-seed status: 13 true, 17 true, 23 true
- minimum closed-loop success rate: 0.750
- maximum prompt-only closed-loop baseline: 0.250
- minimum world-model relative improvement: 0.709
- minimum prediction-error relative improvement: 0.333

## Profile Transfer Stability Gate

This gate trains on the default profile and evaluates on the harder profile.
It is stronger than same-profile stability because the offline benchmark and
closed-loop tasks are generated from the held-out evaluation profile. The
profile-matrix gate below is now the preferred transfer evidence because it
checks multiple evaluation profiles in one reproducible report.

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_stability `
  --output-dir "$env:TEMP\reflexcore_transfer_stability_default_to_hard" `
  --config configs/reflexcore/smoke.yaml `
  --profile default `
  --eval-profile hard `
  --seed 13 `
  --seed 17 `
  --seed 23 `
  --episodes-per-task 6 `
  --vocab-size 512 `
  --max-text-tokens 64 `
  --epochs 8 `
  --batch-size 4 `
  --sequence-mode `
  --max-sequence-len 8 `
  --closed-loop-episodes-per-task 2 `
  --required-baseline prompt_only_heuristic `
  --min-pass-rate 1.0
```

Verified on 2026-06-16:

- `passed`: true
- pass rate: 1.000 across 3 seeds
- per-seed status: 13 true, 17 true, 23 true
- minimum closed-loop success rate: 0.667
- maximum prompt-only closed-loop baseline: 0.250
- minimum world-model relative improvement: 0.638
- minimum prediction-error relative improvement: 0.243

## Profile-Matrix Stability Gate

This gate trains on the default profile and evaluates the same protocol across
`default`, `hard`, and `wide_ood`. It requires every seed in every profile to
pass. This is the preferred smoke-grade transfer gate because a single
same-profile or single-OOD result is not enough to defend a computer-native
sensory-motor mechanism claim. The matrix runner trains once per seed and
reuses that checkpoint across evaluation profiles, so local-scale runs do not
inflate cost by retraining for every profile.

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_profile_matrix `
  --output-dir "$env:TEMP\reflexcore_profile_matrix_default_all" `
  --config configs/reflexcore/smoke.yaml `
  --profile default `
  --eval-profile default `
  --eval-profile hard `
  --eval-profile wide_ood `
  --seed 13 `
  --seed 17 `
  --seed 23 `
  --episodes-per-task 6 `
  --vocab-size 512 `
  --max-text-tokens 64 `
  --epochs 8 `
  --batch-size 4 `
  --sequence-mode `
  --max-sequence-len 8 `
  --closed-loop-episodes-per-task 2 `
  --required-baseline prompt_only_heuristic `
  --min-pass-rate 1.0 `
  --min-profile-pass-rate 1.0
```

Verified on 2026-06-16:

- `passed`: true
- profile pass rate: 1.000 across `default`, `hard`, and `wide_ood`
- training reuse: true; 3 train runs for 3 eval profiles
- default profile: pass rate 1.000; minimum closed-loop success 0.750; maximum prompt-only closed-loop baseline 0.250
- hard profile: pass rate 1.000; minimum closed-loop success 0.667; maximum prompt-only closed-loop baseline 0.250
- wide_ood profile: pass rate 1.000; minimum closed-loop success 0.667; maximum prompt-only closed-loop baseline 0.250
- minimum offline safety-gated action accuracy across profiles: 0.556
- minimum world-model relative improvement across profiles: 0.573
- minimum prediction-error relative improvement across profiles: 0.222

## Local 20-100M Feasibility Gate

This gate verifies that the local configuration instantiates, trains for one
small bounded epoch, writes a checkpoint, keeps loss finite, and stays inside
the 20-100M parameter target on the current machine. It is a feasibility gate,
not a full local performance gate.

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_local_feasibility `
  --output-dir "$env:TEMP\reflexcore_local_feasibility_53m_gate" `
  --config configs/reflexcore/local.yaml `
  --episodes-per-task 1 `
  --split-strategy episode_random `
  --seed 43 `
  --vocab-size 4096 `
  --max-text-tokens 128 `
  --epochs 1 `
  --batch-size 1 `
  --sequence-mode `
  --max-sequence-len 8 `
  --min-parameters 20000000 `
  --max-parameters 100000000
```

The current local configuration targets about 53M parameters. A publication
grade local run still needs a larger fixed benchmark plus offline, closed-loop,
OOD, and safety baseline gates at the local model scale.

Verified local parameter count on 2026-06-17 after enabling action-conditioned
next-state/PE dynamics:

- smoke configuration: 192,839 parameters
- local configuration: 53,107,847 parameters

Verified local feasibility gate on 2026-06-17 with the command above:

- `passed`: true
- parameter count: 53,107,847
- parameter gate: true
- finite-loss gate: true
- checkpoint gate: true
- final training loss: 5.192
- dataset hash: `5301f235b9041b120e9e4ca16c2b5b8f25450cea3247d3625b4a72bd8f7b5732`
- model hash: `d4d437ba16ea5e48278f327d78b47e3655e70db4bfa388ec30248ec0da55f2bd`

## Local 53M Stability Gate

This historical gate runs the full unified experiment at the local model scale
across three seeds. The 53M model must
beat the prompt-only baseline offline and in closed-loop evaluation while also
passing the world-model and prediction-error gates.

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_stability `
  --output-dir "$env:TEMP\reflexcore_local_53m_stability_ep6_e8_b4_sequence_eval" `
  --config configs/reflexcore/local.yaml `
  --profile default `
  --seed 13 `
  --seed 17 `
  --seed 23 `
  --episodes-per-task 6 `
  --vocab-size 4096 `
  --max-text-tokens 128 `
  --epochs 8 `
  --batch-size 4 `
  --sequence-mode `
  --max-sequence-len 8 `
  --closed-loop-episodes-per-task 1 `
  --required-baseline prompt_only_heuristic `
  --min-parameters 20000000 `
  --max-parameters 100000000 `
  --min-pass-rate 1.0
```

Verified on 2026-06-16:

- `passed`: true
- pass rate: 1.000 across 3 seeds
- per-seed status: 13 true, 17 true, 23 true
- parameter count: 53,107,079
- minimum offline safety-gated action accuracy: 0.667
- maximum prompt-only offline action baseline: 0.556
- minimum closed-loop success rate: 0.667
- maximum prompt-only closed-loop baseline: 0.167
- minimum world-model relative improvement: 0.791
- minimum prediction-error relative improvement: 0.438

This result depends on sequence-aware offline evaluation. A single-step offline
probe of the same recurrent checkpoint underestimates the local model because
it discards the hidden state used during sequence training and closed-loop
execution.

This stability result predates the current action-conditioned PE evaluation
alignment and scale-0 PE calibration. Use the mixed real-sandbox gate as the
current strongest local evidence.

## Local 53M Profile-Matrix Gate

This gate is a strong local-scale OOD result. It trains the 53M
local model once per seed on `default`, then evaluates the same checkpoints
across `default`, `hard`, and `wide_ood`. Passing this gate means the bounded
terminal/process/filesystem/time mechanism survives same-profile, harder, and
wide-OOD evaluation without retraining per profile.

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_profile_matrix `
  --output-dir "$env:TEMP\reflexcore_local_53m_profile_matrix_reuse" `
  --config configs/reflexcore/local.yaml `
  --profile default `
  --eval-profile default `
  --eval-profile hard `
  --eval-profile wide_ood `
  --seed 13 `
  --seed 17 `
  --seed 23 `
  --episodes-per-task 6 `
  --vocab-size 4096 `
  --max-text-tokens 128 `
  --epochs 8 `
  --batch-size 4 `
  --sequence-mode `
  --max-sequence-len 8 `
  --closed-loop-episodes-per-task 1 `
  --required-baseline prompt_only_heuristic `
  --min-parameters 20000000 `
  --max-parameters 100000000 `
  --min-pass-rate 1.0 `
  --min-profile-pass-rate 1.0
```

Verified on 2026-06-17:

- `passed`: true
- profile pass rate: 1.000 across `default`, `hard`, and `wide_ood`
- training reuse: true; 3 train runs for 3 eval profiles
- parameter count: 53,107,079
- default profile: pass rate 1.000; minimum offline action accuracy 0.667; minimum closed-loop success 0.667
- hard profile: pass rate 1.000; minimum offline action accuracy 0.667; minimum closed-loop success 0.667
- wide_ood profile: pass rate 1.000; minimum offline action accuracy 0.667; minimum closed-loop success 0.500
- maximum prompt-only offline action baseline across profiles: 0.556
- maximum prompt-only closed-loop baseline across profiles: 0.167
- minimum world-model relative improvement across profiles: 0.791
- minimum prediction-error relative improvement across profiles: 0.438

This profile-matrix result was collected before the current action-conditioned
PE evaluation alignment and scale-0 PE calibration. Treat it as historical OOD
support; the mixed real-sandbox gate below is the current strongest local-scale
evidence.

## Real Sandbox Runtime Gate

This gate leaves the fully simulated `TaskEnv` loop and evaluates real temporary
sandbox interactions. The expanded task set creates real files, refreshes the
sandbox directory, reads real file contents, executes an allowlisted Python
subprocess with `shell=False`, reads buffered stdout, reads stderr from a failed
process state, waits on a still-running process, stops a hung process, and
blocks a dangerous command candidate. It still does not claim GUI control,
unrestricted shell generation, or production autonomy.

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.eval_reflexcore_real_sandbox `
  --checkpoint <reflexcore_v0.pt> `
  --output-dir "$env:TEMP\reflexcore_real_sandbox_eval" `
  --max-steps 4 `
  --require-beats-baseline prompt_only_heuristic
```

Current local-53M checkpoints trained only on the simulated benchmark should not
be treated as passing this expanded gate. The real-sandbox traces below are the
training bridge for the bounded sensory-motor core.

## Live Observation Bridge Gate

This gate verifies the bottom-up receptor bridge used by ReflexCore V0 before
any motor action is selected. `ReflexCoreObservationContext` composes the
existing process, terminal, filesystem, and time receptors into a typed
`ComputerObservation`, then vectorizes it with the same feature path used for
training. It observes only the configured goal allowlist, optional process id,
explicit terminal deltas, and `goal.watched_paths`; it does not execute
commands, scan the whole machine, or introduce GUI/vision control.

Verified on 2026-06-18 with
`test_reflexcore_live_observation_context_vectorizes_bounded_receptor_state`,
`test_reflexcore_live_observation_context_detects_created_and_deleted_files`,
and
`test_reflexcore_sandbox_live_observation_loop_reobserves_command_created_file`:

- first snapshot establishes a watched-path mtime baseline
- second snapshot detects a real file mtime change under a temporary watched
  directory
- empty-directory baselines detect later file creation
- established baselines detect later file deletion
- `run_model_live_observation_loop` executes an allowlisted command with
  `shell=False`, then re-observes the command-created file through the
  filesystem receptor
- terminal stdout delta becomes unread terminal evidence
- runtime evidence source is `runtime_observation`
- candidate commands come only from `goal.command_allowlist`
- candidate files remain bounded to watched/changed paths
- vector length matches `StateVectorizer.vector_dim`
- text tokens respect the configured maximum
- free shell generation: false
- GUI or vision: false

CLI smoke:

```powershell
$env:PYTHONPATH='src'
$root = "$env:TEMP\reflexcore_live_observation_cli_smoke"

python -m reflexlm.cli.run_reflexcore_sandbox `
  --sandbox-root "$root\sandbox" `
  --steps 1 `
  --loop `
  --live-observation `
  --episode-id live-observation-cli-smoke
```

The CLI report includes `"live_observation": true` and per-step
`changed_paths`, so downstream experience capture can distinguish old internal
state-loop traces from receptor-reobserved traces.

Live observation experience capture verified on 2026-06-18 with
`test_reflexcore_live_observation_experience_records_reobserved_transition`:

- the recorded action remains the post-safety typed motor action
- source remains `model`, because the action is model-selected
- `live_observation`: true
- `runtime_observation_examples`: 1
- `changed_file_observations`: 1 for the command-created-file test
- `terminal_observation_examples`: 1
- `observed_prediction_error_examples`: 1
- `observed_prediction_error_mean`: positive for the command-created-file test
- `model_prediction_error_mean`: recorded from the model prediction-error head
- `runtime_evidence.model_prediction_error`: written back into the reobserved
  next state
- `runtime_evidence.observed_prediction_error`: written back into the reobserved
  next state and vectorized for the following model step
- `runtime_evidence.prediction_error_delta`: observed minus model PE, also
  vectorized as bounded numeric feedback
- high `runtime_evidence.observed_prediction_error` can only promote
  `WAIT`/`DONE` to `REFRESH_STATE` when no subprocess is active; it cannot
  promote command execution or override active-process waiting
- live `runtime_evidence.observed_prediction_error` is used as the supervised
  `prediction_error_targets` value for training examples; synthetic/non-live
  examples fall back to normalized next-state vector distance
- `next_state_loss_mask` excludes stochastic telemetry, diagnostic PE feedback
  fields, and hashed text features from world-model supervision; those signals
  remain visible to policy/PE heads without being treated as deterministic
  environment dynamics
- `next_observation.runtime_evidence.source`: `runtime_observation`
- `next_observation.runtime_evidence.changed_files` includes the created file
- no `oracle_action` field is serialized

Latest local gate on 2026-06-19:
`$env:PYTHONPATH='src'; python -m pytest -q tests\test_reflexcore_v0.py`
passed with 65 tests against the current workspace package.

CLI live-experience smoke on 2026-06-18 wrote one bounded runtime example and
read it back through `tensors_for_example`; the stored
`observed_prediction_error` was `0.09015854528315817` and the generated
`prediction_error_targets` value was `0.09015854448080063`.

CLI smoke with experience output:

```powershell
$env:PYTHONPATH='src'
$root = "$env:TEMP\reflexcore_live_observation_experience_cli_smoke"

python -m reflexlm.cli.run_reflexcore_sandbox `
  --sandbox-root "$root\sandbox" `
  --steps 1 `
  --loop `
  --live-observation `
  --write-experience "$root\experience.jsonl" `
  --episode-id live-observation-experience-cli-smoke
```

Observed CLI summary:

- `live_observation`: true
- `runtime_observation_examples`: 1
- `observed_prediction_error_examples`: 1
- `model_prediction_error_mean`: present
- `observed_prediction_error_mean`: present
- `changed_file_observations`: depends on the untrained model's selected action
- `dataset_hash`: emitted for traceability, but run-specific because live
  wall-clock receptor timestamps are included

Real-sandbox live-observation gate verified on 2026-06-19 with the train-once
local 53M seed-23 checkpoint:

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.eval_reflexcore_real_sandbox `
  --checkpoint "$env:TEMP\reflexcore_current_local53m_seed23_trainonce_profile_20260619_143202\seed_23\train\reflexcore_v0.pt" `
  --output-dir "$env:TEMP\reflexcore_current_local53m_seed23_live_observation_processguard_20260619_200638" `
  --max-steps 6 `
  --live-observation `
  --max-text-tokens 128 `
  --require-beats-baseline prompt_only_heuristic
```

- `passed`: true
- model success rate: 1.000 across 12 real-sandbox tasks
- prompt-only baseline success rate: 0.250
- runtime observation steps: 22
- changed-file observation steps: 3
- terminal observation steps: 15
- observed PE examples: 22
- observed PE mean / max: 0.0633 / 0.1395
- real-process wait chain: `RUN_COMMAND -> WAIT -> READ_STDOUT`
- failed tasks: none
- artifact:
  `%TEMP%\reflexcore_current_local53m_seed23_live_observation_processguard_20260619_200638\real_sandbox_report.json`

This gate fixed two evidence gaps in the live evaluator. First, receptor
baselines are now primed before task setup mutates the temporary sandbox, so
created files are reported as live filesystem changes rather than swallowed by
the initial snapshot. Second, active subprocess states suppress repeated
command execution and prevent prediction-error refresh from replacing the
needed wait step. Both changes are runtime/affordance constraints, not
task-name shortcuts.

Train-once live profile matrix smoke gate, verified on 2026-06-19:

- CLI: `python -m reflexlm.cli.run_reflexcore_real_sandbox_adaptation_profile_matrix`
- artifact:
  `%TEMP%\reflexcore_current_smoke_live_profile_matrix_20260619_201508\real_sandbox_adaptation_profile_matrix_report.json`
- seeds: `13`, `17`
- eval profiles: `default`, `hard`
- training reuse: 2 train runs reused across 4 profile evaluations
- profile pass rate: 4/4
- smoke parameter count: 194,258
- synthetic weighted examples per seed: 360
- real-sandbox weighted examples per seed: 66
- `--real-sandbox-live-observation`: true
- minimum real-sandbox success rate: 0.750
- minimum real-sandbox margin over prompt-only: 0.500
- minimum offline action margin over prompt-only: 0.333
- minimum closed-loop margin over prompt-only: 0.500
- minimum runtime observation steps: 31
- minimum changed-file observation steps: 3
- minimum observed PE examples: 31
- minimum world-model relative improvement: 0.846
- minimum prediction-error relative improvement: 0.548

Expanded 15-task live real-sandbox smoke gate, verified on 2026-06-20 after
adding multi-file, command-created-file, slow-process-created-file, live
filesystem fusion, pending-file motor gating, process-launch affordance, and
literal goal-cue command-slot correction:

- CLI: `python -m reflexlm.cli.run_reflexcore_real_sandbox_adaptation_profile_matrix`
- artifact:
  `%TEMP%\reflexcore_current_smoke15_strict_live_profile_matrix_20260620_115309\real_sandbox_adaptation_profile_matrix_report.json`
- seeds: `13`, `17`
- eval profiles: `default`, `hard`
- training reuse: 2 train runs reused across 4 profile evaluations
- task surface: 15 real-sandbox families
- profile pass rate: 4/4
- smoke parameter count: 194,258
- synthetic weighted examples per seed: 36
- real-sandbox weighted examples per seed: 68
- `--real-sandbox-live-observation`: true
- minimum real-sandbox success rate: 1.000
- prompt-only real-sandbox baseline success rate: 0.267
- minimum real-sandbox margin over prompt-only: 0.733
- minimum offline action margin over prompt-only: 0.125
- minimum closed-loop margin over prompt-only: 0.583
- minimum runtime observation steps: 34
- minimum changed-file observation steps: 10
- minimum terminal observation steps: 23
- minimum observed PE examples: 34
- minimum world-model relative improvement: 0.260
- minimum prediction-error relative improvement: 0.0775

This smoke gate is weaker than the local 53M runs below in model scale, but it
is now strict on all three tested surfaces: offline action margin, closed-loop
margin, and real-sandbox margin are all positive under the same train-once
checkpoints. The 2026-06-20 fixes are state/motor semantics rather than
task-name shortcuts: refresh consumes external/stale signals while preserving
dirty-file memory, pending file reads narrow the motor space toward `READ_FILE`,
process-hang goals can still launch an initial allowlisted process but cannot
start a second one while handling existing process evidence, and command slots
can use explicit positive/negative goal literals to choose between allowlisted
commands. These additions keep the same bounded safety contract: typed motor
heads, temporary sandbox, shell-free execution, and allowlisted `RUN_COMMAND`
only.

Local 53M live real-sandbox replay after the same state/motor fixes, verified
on 2026-06-20 against an existing seed-13 checkpoint:

- checkpoint:
  `%TEMP%\reflexcore_current_local53m_seed13_strict15_probe_20260620_115557\seed_13\train\reflexcore_v0.pt`
- replay CLI: `python -m reflexlm.cli.eval_reflexcore_real_sandbox`
- artifact:
  `%TEMP%\reflexcore_current_local53m_seed13_runtime_recheck_after_process_only_idle_gate\real_sandbox_report.json`
- config lineage: `configs/reflexcore/local_pe_calibrated.yaml`
- parameter count from source run: 53,124,754
- task surface: 15 live real-sandbox families
- model success rate: 1.000, 15/15 tasks
- prompt-only baseline success rate: 0.267
- terminal observation steps: 25
- runtime observation steps: 36
- observed PE examples: 36
- observed PE mean: 0.0840
- full V0 unit/regression gate after the fixes:
  `python -m pytest -q tests\test_reflexcore_v0.py` -> 80 passed

This replay is intentionally narrower than a new local train-once profile
matrix: it proves that the current runtime state semantics make an already
trained 53M sensory-motor checkpoint complete the expanded live sandbox suite.
It does not replace the profile-matrix results below, and it does not claim GUI
operation, unrestricted shell generation, or production autonomy. The fixes are
generic receptor/action-state semantics: terminal reads consume unread output,
watched paths alone do not enable file reads, pending file changes narrow the
motor space toward `READ_FILE`, stale processes narrow toward `STOP_PROCESS`,
and `PROCESS_HANG` initial states can prefer launching the allowlisted process
without suppressing homeostatic prediction-error refresh in ordinary recovery
states.

Local 53M train-once live profile matrix follow-up, verified on 2026-06-19:

- config: `configs/reflexcore/local_pe_calibrated.yaml`
- artifact:
  `%TEMP%\reflexcore_current_local53m_seed23_live_profile_matrix_20260619_201605\real_sandbox_adaptation_profile_matrix_report.json`
- seed: `23`
- eval profiles: `default`, `hard`, `wide_ood`
- training reuse: 1 train run reused across 3 profile evaluations
- profile pass rate: 3/3
- parameter count: 53,124,754
- synthetic weighted examples: 222
- real-sandbox weighted examples: 264
- `--real-sandbox-live-observation`: true
- real-sandbox success rate: 1.000 in every profile summary
- real-sandbox margin over prompt-only: 0.750
- minimum offline action margin over prompt-only: 0.125
- minimum closed-loop margin over prompt-only: 0.167
- runtime observation steps: 22
- changed-file observation steps: 3
- terminal observation steps: 15
- observed PE examples: 22
- observed PE mean / max: 0.0636 / 0.1430
- minimum world-model relative improvement: 0.421
- minimum prediction-error relative improvement: 0.205

This is now stronger than the earlier single-checkpoint live gate: the same
trained checkpoint is reused across held-out profile evaluations while the
real-sandbox evaluation remains a live receptor loop with filesystem, terminal,
process, and observed-PE evidence. It is still bounded to temporary sandbox
terminal/process/filesystem/time behavior.

Local 53M three-seed live profile rollup, verified on 2026-06-19:

- rollup artifact:
  `%TEMP%\reflexcore_current_local53m_three_seed_live_profile_rollup_20260619.json`
- source reports:
  `%TEMP%\reflexcore_current_local53m_seed13_17_live_profile_matrix_20260619_224246\real_sandbox_adaptation_profile_matrix_report.json`
  and
  `%TEMP%\reflexcore_current_local53m_seed23_live_profile_matrix_20260619_201605\real_sandbox_adaptation_profile_matrix_report.json`
- seeds: `13`, `17`, `23`
- eval profiles: `default`, `hard`, `wide_ood`
- training reuse: 3 train runs reused across 9 profile evaluations
- profile pass rate: 9/9
- parameter count: 53,124,754
- live observation enabled and live gate passed for every profile summary
- minimum real-sandbox success rate: 0.750
- minimum real-sandbox margin over prompt-only: 0.500
- minimum offline action margin over prompt-only: 0.125
- minimum closed-loop margin over prompt-only: 0.167
- minimum runtime observation steps: 22
- minimum changed-file observation steps: 3
- minimum terminal observation steps: 15
- minimum observed PE examples: 22
- observed PE mean range: 0.0587 to 0.0658
- minimum world-model relative improvement: 0.421
- minimum prediction-error relative improvement: 0.205
- model hashes:
  `1e7e0a9e4481deeb3b31c0d54eb657b55b55cbe8b80e509d3d50467793ef6e1d`,
  `cf962a2198843fe87f7a87fea6d1e8e5cfb8ee8c80b07d4d6a8458f69a513a5c`,
  `d7413d3859feeb1e9c63ebe92f2a01e1e08cf570b7376274b1a6b853dcda4c39`

This rollup is the current strongest V0 evidence: a local 53M sensory-motor
core trains once per seed, transfers across held-out synthetic profiles, and
executes real temporary sandbox tasks through live terminal/process/filesystem
receptors with typed motor heads and allowlisted `RUN_COMMAND`. It still does
not support claims about GUI control, unrestricted shell generation, robotics,
or production autonomy.

## Online Model-Experience Capture Gate

This gate records a model's own bounded sandbox rollout as ReflexCore JSONL
training examples. Unlike the oracle trace builder, the recorded action is the
post-safety typed motor action selected during execution. This is the first V0
online-experience path: observe, act through safety, observe the next state,
then serialize that transition back into the same training schema. It remains
bounded to terminal/process/filesystem/time and does not introduce free shell
generation.

```powershell
$env:PYTHONPATH='src'
$root = "$env:TEMP\reflexcore_cli_experience_smoke"

python -m reflexlm.cli.run_reflexcore_sandbox `
  --sandbox-root "$root\sandbox" `
  --steps 2 `
  --loop `
  --write-experience "$root\experience.jsonl" `
  --episode-id cli-experience-smoke
```

Verified on 2026-06-18:

- experience examples: 2
- source: `model`
- episode id: `cli-experience-smoke`
- dataset hash: `47f3a05979d6a3ab34d34d410073fcb5d7cd75fe5d9fab8f4a34512ee9c0f4a4`
- free shell generation: false
- GUI or vision: false

Safety invariant: if a proposed action is blocked, the serialized training
example records the post-safety `BLOCK` action rather than the unsafe raw model
proposal. This prevents self-generated experience from smuggling disallowed
commands into the action target.

## Online Adaptation Gate

This gate updates an existing ReflexCore checkpoint from bounded
model-experience JSONL. It does not execute actions; it only consumes already
recorded post-safety sensory-motor transitions and writes an adapted checkpoint
plus before/after loss report. Separate retention and holdout JSONL files can
be supplied as forgetting/generalization guards: the update is accepted only
when train loss improves and neither retention nor holdout loss increases beyond
the configured tolerance. This is a V0 plasticity path, not a claim of broad
autonomy.

For live-observation examples, the report also surfaces mechanism-level
plasticity metrics:

- `live_prediction_error_examples`
- `live_prediction_error_target_mean`
- `before_metrics.prediction_error_loss`
- `after_metrics.prediction_error_loss`
- `prediction_error_loss_delta`
- `prediction_error_motor_probe`

This verifies that online adaptation can consume the live observed prediction
error signal as a trainable PE-head target, while `next_state_loss_mask` keeps
diagnostic feedback and hash features out of deterministic world-model loss.
The PE motor probe clears runtime `observed_prediction_error` before decoding,
so a refresh can only come from the learned PE head plus bounded homeostatic
motor control, not from directly replaying the observed runtime value.
It also rebuilds the probe observation vector from that cleaned state, which
prevents cached vectors from leaking PE feedback features into the motor probe.

```powershell
$env:PYTHONPATH='src'
$root = "$env:TEMP\reflexcore_online_adaptation_holdout_cli_smoke"

python -m reflexlm.cli.adapt_reflexcore_from_experience `
  --checkpoint "$root\base.pt" `
  --experience "$root\train.jsonl" `
  --retention "$root\retention.jsonl" `
  --holdout "$root\holdout.jsonl" `
  --output-dir "$root\adapted_strict" `
  --epochs 2 `
  --batch-size 1 `
  --learning-rate 0.001 `
  --sequence-mode
```

Verified on 2026-06-18 using bounded sandbox experience capture split into a
one-transition train set, a one-transition retention set, and a two-transition
holdout set:

- experience examples: 1
- retention examples: 1
- holdout examples: 2
- source values: `model`
- before loss: 3.654
- after loss: 2.848
- loss delta: 0.807
- loss not increased: true
- retention before loss: 3.654
- retention after loss: 2.848
- retention loss increase: -0.807
- retention gate passed: true
- max retention loss increase: 0.0
- holdout before loss: 3.708
- holdout after loss: 3.213
- holdout loss increase: -0.495
- holdout gate passed: true
- max holdout loss increase: 0.0
- accepted: true
- rejected reason: null
- experience hash: `d79d64ca84a74fa4201c8a76d43cb8feb0d3da8e1d6a25409f44070c0efc8875`
- holdout hash: `6fa38bffb17f951b77610ac83564d54331f1ce3fa48f69863e095be11034b019`
- adapted checkpoint: `reflexcore_v0_adapted.pt`
- adapted model hash: `f49fb7bf87f0153c29a63550110c14368f15d052fe8e92fcae2a50ad246348cf`
- free shell generation: false
- GUI or vision: false

Boundary: this proves the plumbing for bounded online plasticity with a
retention/forgetting guard and a separate holdout-loss guard. It does not prove
open-ended task improvement or authorize execution outside the sandbox/allowlist
gate.

Additional local unit gate on 2026-06-18:
`test_reflexcore_online_adaptation_learns_live_prediction_error_signal` adapts a
direct PE-head smoke model on a single live-PE example whose next-state vector
is unchanged but whose observed PE target is non-zero. The accepted update must
reduce `prediction_error_loss`, proving the live PE signal is consumed by the
training path rather than merely serialized. The same test fixes the base motor
head to `WAIT` and salience high; before adaptation the PE head is too low to
refresh, while after adaptation the decoded action switches to
`REFRESH_STATE`.

CLI PE-plasticity smoke on 2026-06-18:

- `live_prediction_error_examples`: 1
- `live_prediction_error_target_mean`: 0.77
- `before_metrics.prediction_error_loss`: 0.07289998978376389
- `after_metrics.prediction_error_loss`: 0.006256934721022844
- `prediction_error_loss_delta`: 0.06664305506274104
- `accepted`: true
- `free_shell_generation`: false
- `gui_or_vision`: false

CLI PE-to-motor smoke on 2026-06-18:

- `prediction_error_loss_delta`: 0.6411262825131416
- `prediction_error_motor_probe.available`: true
- `prediction_error_motor_probe.base_refresh_count`: 0
- `prediction_error_motor_probe.adapted_refresh_count`: 1
- `prediction_error_motor_probe.base_safe_refresh_count`: 0
- `prediction_error_motor_probe.adapted_safe_refresh_count`: 1
- `prediction_error_motor_probe.adapted_safety_allowed_count`: 1
- `prediction_error_motor_probe.changed_to_refresh_count`: 1
- `prediction_error_motor_probe.changed_to_safe_refresh_count`: 1
- `prediction_error_motor_probe.mean_prediction_error_delta`: 0.9825264972168952
- `accepted`: true

The corresponding unit gate also calls `runner.propose_with_state` and
`runner.step` on the adapted checkpoint, confirming the learned PE-triggered
`REFRESH_STATE` survives the safety layer and can execute as a bounded sandbox
refresh step.

## Cross-Episode Online Adaptation Gate

This gate raises the online adaptation check from a single rollout split to
disjoint real-sandbox episodes. It reads a ReflexCore JSONL dataset, writes
train/retention/holdout splits with non-overlapping episode ids, adapts the
checkpoint on the train split only, and accepts the update only when train,
retention, and holdout losses do not increase. The gate does not execute model
actions; it consumes recorded bounded sensory-motor transitions.

```powershell
$env:PYTHONPATH='src'
$root = "$env:TEMP\reflexcore_cross_episode_online_adaptation_gate"

python -m reflexlm.cli.build_reflexcore_real_sandbox_dataset `
  --output "$root\real_sandbox_dataset.jsonl" `
  --work-dir "$root\work" `
  --variants 3 `
  --start-variant 0 `
  --vocab-size 512 `
  --max-text-tokens 128

@'
from pathlib import Path
import os
import torch
from reflexlm.core.dataset import read_reflexcore_jsonl
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
root = Path(os.environ["TEMP"]) / "reflexcore_cross_episode_online_adaptation_gate"
examples = read_reflexcore_jsonl(root / "real_sandbox_dataset.jsonl")
torch.manual_seed(23)
model = ReflexCoreV0(ReflexCoreV0Config.smoke(
    input_dim=len(examples[0].observation.vector),
    vocab_size=512,
))
torch.save({"model_state_dict": model.state_dict(), "config": model.config.to_dict()}, root / "base.pt")
'@ | python -

python -m reflexlm.cli.run_reflexcore_online_adaptation_gate `
  --checkpoint "$root\base.pt" `
  --dataset "$root\real_sandbox_dataset.jsonl" `
  --output-dir "$root\gate" `
  --split-strategy episode_holdout `
  --split-seed 23 `
  --train-episodes 12 `
  --retention-episodes 3 `
  --epochs 2 `
  --batch-size 2 `
  --learning-rate 0.001 `
  --sequence-mode
```

Verified on 2026-06-18:

- dataset examples: 24
- dataset episodes: 21
- split strategy: `episode_holdout`
- disjoint episodes: true
- train examples / episodes: 14 / 12
- retention examples / episodes: 3 / 3
- holdout examples / episodes: 7 / 6
- source values: `runtime_observation`
- train loss: 4.071 -> 3.330
- retention loss: 3.920 -> 3.375
- holdout loss: 4.132 -> 3.664
- holdout loss increase: -0.468
- max holdout loss increase: 0.0
- passed: true
- adapted model hash: `eb4fd111c72573e714206bbd5298c3261a6086c6ad8b29b51e9d4d05fca37f46`
- dataset hash: `91e4837bf229ec5732e8a9af52d6afbd56b6ccfda6d66e1b0d39b67b58aed24a`
- free shell generation: false
- GUI or vision: false

Boundary: this is stronger than the single-rollout online gate because the
holdout examples come from different real-sandbox episodes. It is still not a
full task-family-disjoint proof, not GUI control, and not unrestricted computer
autonomy.

## Task-Family Holdout Matrix Gate

This matrix repeats the online adaptation gate while holding out one complete
real-sandbox task family at a time. It is a stricter check than episode holdout:
the update cannot see any episode from the held-out task type during training or
retention selection. The first strict run with full-model updates failed all
families because holdout loss increased despite lower train loss. The accepted
mitigation is a conservative `world_model_only` scope that freezes the motor
policy and updates only the next-state / prediction-error heads.

```powershell
$env:PYTHONPATH='src'
$root = "$env:TEMP\reflexcore_family_holdout_matrix_gate"

python -m reflexlm.cli.build_reflexcore_real_sandbox_dataset `
  --output "$root\real_sandbox_dataset.jsonl" `
  --work-dir "$root\work" `
  --variants 3 `
  --start-variant 0 `
  --vocab-size 512 `
  --max-text-tokens 128

@'
from pathlib import Path
import os
import torch
from reflexlm.core.dataset import read_reflexcore_jsonl
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
root = Path(os.environ["TEMP"]) / "reflexcore_family_holdout_matrix_gate"
examples = read_reflexcore_jsonl(root / "real_sandbox_dataset.jsonl")
torch.manual_seed(31)
model = ReflexCoreV0(ReflexCoreV0Config.smoke(
    input_dim=len(examples[0].observation.vector),
    vocab_size=512,
))
torch.save({"model_state_dict": model.state_dict(), "config": model.config.to_dict()}, root / "base.pt")
'@ | python -

python -m reflexlm.cli.run_reflexcore_family_holdout_matrix `
  --checkpoint "$root\base.pt" `
  --dataset "$root\real_sandbox_dataset.jsonl" `
  --output-dir "$root\matrix_world_only_tol002" `
  --split-seed 31 `
  --retention-episodes 3 `
  --epochs 2 `
  --batch-size 2 `
  --learning-rate 0.001 `
  --sequence-mode `
  --trainable-scope world_model_only `
  --max-retention-loss-increase 0.002 `
  --max-holdout-loss-increase 0.002
```

Verified on 2026-06-18:

- dataset examples / episodes: 24 / 21
- held-out task families: 7
- trainable scope: `world_model_only`
- full-model strict matrix: 0/7 passed, worst holdout delta -0.569
- world-model-only strict matrix: 2/7 passed, worst holdout delta -0.000254
- world-model-only tolerance matrix: 7/7 passed
- tolerance: max retention loss increase 0.002, max holdout loss increase 0.002
- pass rate: 1.0
- failed families: none
- minimum holdout loss delta: -0.000254
- dataset hash: `beca60e73ab2bd9af2b17ad864fa35ef1089516b7a065d4cfa8a7b002ae3e8b6`
- free shell generation: false
- GUI or vision: false

Strict behavior-capability upgrade verified on 2026-06-18:

- training dataset: 20 real-sandbox variants, 160 examples
- training scope: smoke ReflexCore V0, sequence mode, 60 epochs, batch size 8
- trained parameter count: 192,839
- final train loss / action loss: 0.107552 / 0.000580
- trained model hash: `081d862c319f149f6321db187616e38cf48058056980d8ebe2542be42bd52dc5`
- trained dataset hash: `fd6991306804cad67b128360a35215871b977e0adeaa1dcedefe0f4293fb6b5b`
- direct real-sandbox holdout variants `20..24`: 35/35 closed-loop successes
- strict behavior matrix: 7/7 held-out families passed
- behavior gate: `--require-behavior-capability --min-behavior-success-rate 1.0`
- base/adapted held-out behavior success: 1.0 / 1.0 for every family
- behavior capability passed count: 7/7
- failed families: none
- minimum holdout loss delta after `world_model_only` adaptation: -0.000003
- free shell generation: false
- GUI or vision: false

Cross-seed real-process capability matrix verified on 2026-06-18 after
canonicalizing real-sandbox dataset paths, wall-clock timestamps, and local
Python executable paths:

- seeds: 41, 43, 47
- train variants: `0..19`
- evaluation variants: `20..24`
- train/eval variant ranges: disjoint
- task families: 12, including `multi_step_file_command_stdout`,
  `multi_step_distractor_stdout`, `multi_command_select_stdout`,
  `real_process_wait_stdout`, and `real_process_stop`
- multi-step oracle chain: `REFRESH_STATE -> READ_FILE -> RUN_COMMAND -> READ_STDOUT`
- distractor family requirement: ignore stale buffered stdout, read the changed
  file first, then run the allowlisted command and read the new stdout
- command-slot family requirement: choose the second allowlist command slot
  that prints the target marker while ignoring the first slot, which executes
  successfully but prints a wrong marker
- real-process wait requirement: launch an allowlisted Python subprocess with
  `shell=False`, observe a running process after the timeout, `WAIT`, then
  `READ_STDOUT` after completion
- real-process stop requirement: launch an allowlisted long-running Python
  subprocess with `shell=False`, observe a resource-alert running process, then
  issue `STOP_PROCESS` and mark the process interrupted
- dataset examples: 440
- portable dataset hash: `a6df2969df381d2926cf6cf4e677cb511c0ea7bb26133e5803f8a4823a4edd9e`
- pass count / run count: 3/3
- minimum success rate across seeds: 1.0
- per-seed real-sandbox successes: 60/60, 60/60, 60/60
- per-seed multi-step family success rate: 1.0, 1.0, 1.0
- per-seed distractor family success rate: 1.0, 1.0, 1.0
- per-seed command-slot family success rate: 1.0, 1.0, 1.0
- per-seed real-process wait success rate: 1.0, 1.0, 1.0
- per-seed real-process stop success rate: 1.0, 1.0, 1.0
- model hashes:
  `6a83a346c7781fc3b6109dd75eb2bc00ff9f1eae59a275a1e37b6a6719d9b2d7`,
  `e91a9bd85b230c862f841c313151d67183a73baa407f1784ba2527ca569778f5`,
  `df8483b7d6836821533c13a01c7ef0bf9c830d10fab7d72db074a9bfd1a68fab`
- maximum final action loss across seeds: 0.000344
- dataset contains `$SANDBOX_ROOT` and `$PYTHON` portability tokens
- dataset does not contain the local `sys.executable` path
- dataset contains no `oracle_action` field
- report:
  `%TEMP%\reflexcore_capability_seed_matrix_real_process_masked3\real_sandbox_capability_matrix_report.json`
- free shell generation: false
- GUI or vision: false

Implementation note: the distractor and real-process families exposed general
runtime misses before the final run: the core needed visible terminal/filesystem
conflict features, actual subprocess lifecycle state, a resource-alert signal
for long-running subprocesses, and a typed affordance mask that prioritizes
unread command output without treating initial prompt text as command output.
These are sensory/runtime features, not task-name or oracle shortcuts. The
command-slot family still checks allowlist selection: the correct `RUN_COMMAND`
is not the first candidate slot, and the wrong slot is executable but fails the
task-level marker check.

The strict behavior-capability run fixes a weakness in the first matrix: a
random or weak base model could previously pass a behavior regression check by
remaining equally incapable (`0.0 -> 0.0`). The matrix now separates three facts:
loss stability, behavior non-regression, and minimum closed-loop behavior
capability. A family can only pass the strict behavior gate if the adapted model
keeps the required success rate on real temporary sandbox episodes.

Reproduction command outline:

```powershell
$env:PYTHONPATH='src'
$root = "$env:TEMP\reflexcore_capability_train_gate"

python -m reflexlm.cli.build_reflexcore_real_sandbox_dataset `
  --output "$root\real_sandbox_train.jsonl" `
  --work-dir "$root\work" `
  --variants 20 `
  --start-variant 0 `
  --vocab-size 512 `
  --max-text-tokens 128

python -m reflexlm.cli.train_reflexcore_v0 `
  --dataset "$root\real_sandbox_train.jsonl" `
  --config configs/reflexcore/smoke.yaml `
  --output-dir "$root\train_smoke_seq" `
  --epochs 60 `
  --batch-size 8 `
  --learning-rate 0.003 `
  --seed 41 `
  --sequence-mode `
  --max-sequence-len 8

python -m reflexlm.cli.eval_reflexcore_real_sandbox `
  --checkpoint "$root\train_smoke_seq\reflexcore_v0.pt" `
  --output-dir "$root\eval_holdout_variants_20_24" `
  --variants 5 `
  --start-variant 20 `
  --max-steps 6 `
  --min-success-rate 1.0

python -m reflexlm.cli.run_reflexcore_family_holdout_matrix `
  --checkpoint "$root\train_smoke_seq\reflexcore_v0.pt" `
  --dataset "$root\real_sandbox_train.jsonl" `
  --output-dir "$root\matrix_capable_world_only_behavior" `
  --split-seed 43 `
  --retention-episodes 5 `
  --epochs 1 `
  --batch-size 8 `
  --learning-rate 0.0001 `
  --sequence-mode `
  --trainable-scope world_model_only `
  --max-retention-loss-increase 0.002 `
  --max-holdout-loss-increase 0.002 `
  --behavior-eval-variants 5 `
  --behavior-eval-start-variant 20 `
  --behavior-eval-max-steps 4 `
  --require-behavior-capability `
  --min-behavior-success-rate 1.0

python -m reflexlm.cli.run_reflexcore_real_sandbox_capability_matrix `
  --output-dir "$env:TEMP\reflexcore_capability_seed_matrix_real_process_masked3" `
  --config configs/reflexcore/smoke.yaml `
  --seed 41 `
  --seed 43 `
  --seed 47 `
  --train-variants 20 `
  --train-start-variant 0 `
  --eval-variants 5 `
  --eval-start-variant 20 `
  --vocab-size 512 `
  --max-text-tokens 128 `
  --epochs 60 `
  --batch-size 8 `
  --learning-rate 0.003 `
  --sequence-mode `
  --max-sequence-len 8 `
  --max-steps 6 `
  --min-success-rate 1.0 `
  --min-pass-rate 1.0
```

Boundary: this is not a strict zero-regression family-disjoint proof. It shows
that separating low-risk world-model plasticity from high-risk motor-policy
plasticity sharply reduces cross-family regression and passes a small explicit
non-regression tolerance. The behavior-capability upgrade adds direct evidence
that a trained smoke core can execute the bounded terminal/process/filesystem/time
task families in a real temporary sandbox while preserving that behavior under
`world_model_only` adaptation. The cross-seed capability matrix further shows
that this learnability is not a single-seed accident on the current V0 task
surface. Broad autonomous action learning, GUI control, and unrestricted shell
generation remain out of scope.

## Real Sandbox Oracle Learnability Gate

This gate builds oracle traces from real sandbox execution variants, trains a
smoke ReflexCore model on variants `1..12`, and evaluates on held-out variant
`0`. It verifies that the V0 architecture and dataset schema can learn from
actual temporary filesystem/subprocess transitions rather than only simulated
state frames.

```powershell
$env:PYTHONPATH='src'
$root = "$env:TEMP\reflexcore_real_sandbox_smoke_learnability"

python -m reflexlm.cli.build_reflexcore_real_sandbox_dataset `
  --output "$root\real_sandbox_train.jsonl" `
  --work-dir "$root\work" `
  --variants 12 `
  --start-variant 1 `
  --vocab-size 512 `
  --max-text-tokens 64

python -m reflexlm.cli.train_reflexcore_v0 `
  --dataset "$root\real_sandbox_train.jsonl" `
  --config configs/reflexcore/smoke.yaml `
  --output-dir "$root\train" `
  --epochs 20 `
  --batch-size 4 `
  --sequence-mode `
  --max-sequence-len 8

python -m reflexlm.cli.eval_reflexcore_real_sandbox `
  --checkpoint "$root\train\reflexcore_v0.pt" `
  --output-dir "$root\eval" `
  --max-steps 4 `
  --require-beats-baseline prompt_only_heuristic
```

Verified on 2026-06-17 after expanding the real sandbox task set:

- `passed`: true
- real sandbox model success rate: 1.000
- prompt-only baseline success rate: 0.429
- task count: 7
- training examples: 96
- training variants: 12
- held-out evaluation variant: 0
- smoke parameter count: 192,839
- final training loss: 0.116
- successful real actions: `REFRESH_STATE > READ_FILE`, `RUN_COMMAND`,
  `READ_STDOUT`, `READ_STDERR`, `WAIT`, `STOP_PROCESS`, `BLOCK`

## Mixed Real-Sandbox Adaptation Gate

This gate trains one ReflexCore V0 model on a mixed dataset:

- synthetic `TaskEnv` benchmark train split
- real temporary sandbox oracle traces from variants `1..12`

It then evaluates the same checkpoint on synthetic holdout, closed-loop
synthetic runtime, and held-out real sandbox variant `0`. This is the current
strongest V0 bridge between simulated sensory-motor traces and real
filesystem/subprocess transitions.

Current-tree smoke gate verified on 2026-06-18 after aligning world-model
evaluation with `next_state_loss_mask` and fixing weighted mixture fractions:

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_real_sandbox_adaptation_matrix `
  --output-dir "$env:TEMP\reflexcore_current_smoke_balanced_real_sandbox_matrix" `
  --config configs/reflexcore/smoke.yaml `
  --seed 13 `
  --seed 17 `
  --episodes-per-task 6 `
  --epochs 6 `
  --batch-size 4 `
  --learning-rate 0.001 `
  --vocab-size 512 `
  --max-text-tokens 64 `
  --sequence-mode `
  --max-sequence-len 8 `
  --closed-loop-episodes-per-task 1 `
  --real-sandbox-variants 3 `
  --real-sandbox-max-steps 4 `
  --synthetic-repeat 10 `
  --real-sandbox-repeat 1 `
  --min-pass-rate 1.0 `
  --min-offline-margin 0.0 `
  --min-closed-loop-margin -1.0 `
  --min-real-sandbox-margin 0.0
```

- matrix pass rate: 1.000 across seeds `13` and `17`
- smoke parameter count: 194,258
- mixed training examples per run: 426
- synthetic weighted examples per run: 360
- real-sandbox weighted examples per run: 66
- minimum offline action margin over prompt-only: 0.333
- minimum closed-loop margin over prompt-only: 0.500
- minimum real-sandbox margin over prompt-only: 0.667
- minimum world-model relative improvement: 0.858
- minimum prediction-error relative improvement: 0.570

The same smoke matrix with `synthetic-repeat 1` and `real-sandbox-repeat 2`
failed the synthetic world/PE gate because real-sandbox examples dominated the
mixed training set. The accepted smoke gate therefore documents balanced source
weighting rather than relaxing synthetic or PE acceptance criteria.

Current-tree local 53M gate verified on 2026-06-19 with the same masked
world-model evaluation and weighted mixture accounting:

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_real_sandbox_adaptation_matrix `
  --output-dir "$env:TEMP\reflexcore_current_local53m_seed<SEED>_fuller_<STAMP>" `
  --config configs/reflexcore/local.yaml `
  --seed <13|17|23> `
  --episodes-per-task 12 `
  --epochs 12 `
  --batch-size 4 `
  --learning-rate 0.0003 `
  --vocab-size 4096 `
  --max-text-tokens 128 `
  --sequence-mode `
  --max-sequence-len 8 `
  --closed-loop-episodes-per-task 1 `
  --real-sandbox-variants 12 `
  --real-sandbox-start-variant 1 `
  --real-sandbox-max-steps 4 `
  --synthetic-repeat 2 `
  --real-sandbox-repeat 1 `
  --min-parameters 20000000 `
  --max-parameters 100000000 `
  --min-pass-rate 1.0 `
  --min-offline-margin 0.0 `
  --min-closed-loop-margin 0.0 `
  --min-real-sandbox-margin 0.0
```

Rollup artifact:
`%TEMP%\reflexcore_current_local53m_three_seed_rollup_20260619.json`

| Seed | Passed | Offline action vs prompt | Closed-loop vs prompt | Real sandbox vs prompt | World improvement | PE improvement |
| --- | --- | --- | --- | --- | --- | --- |
| 13 | true | 0.7500 vs 0.5625 | 0.500 vs 0.167 | 0.917 vs 0.250 | 0.513 | 0.600 |
| 17 | true | 0.9375 vs 0.5625 | 1.000 vs 0.167 | 1.000 vs 0.250 | 0.333 | 0.163 |
| 23 | true | 0.6250 vs 0.4375 | 0.500 vs 0.167 | 1.000 vs 0.250 | 0.054 | 0.046 |

Aggregate properties:

- pass rate: 1.000 across seeds `13`, `17`, and `23`
- parameter count: 53,124,754
- mixed training examples per run: 412
- synthetic weighted examples per run: 148
- real-sandbox weighted examples per run: 264
- minimum paired offline action margin over prompt-only: 0.1875
- minimum paired closed-loop margin over prompt-only: 0.333
- minimum paired real-sandbox margin over prompt-only: 0.667
- minimum world-model relative improvement: 0.054
- minimum prediction-error relative improvement: 0.046

Negative control retained: the same 53M seed `17` with the smaller
`episodes-per-task 6`, `real-sandbox-variants 3`, and `epochs 6` setting
passed world/PE, closed-loop, and real-sandbox gates but failed synthetic
offline action selection (`0.444` vs prompt-only `0.556`). The local gate above
therefore treats adequate episode coverage and training duration as part of the
evidence, not as a cosmetic hyperparameter.

Limitation: seed `23` passes but has the weakest world-model and PE margins.
The next local milestone should raise that low-end margin or evaluate a harder
held-out profile before broadening the claim.

Seed `23` profile stress follow-up, also verified on 2026-06-19:

- rollup artifact:
  `%TEMP%\reflexcore_current_local53m_seed23_profile_rollup_20260619.json`
- profiles evaluated: `default`, `hard`, and `wide_ood`
- pass rate across profiles: 1.000
- offline action margin over prompt-only: 0.1875 in every profile
- closed-loop margin over prompt-only: 0.333 in every profile
- real-sandbox margin over prompt-only: 0.750 in every profile
- minimum world-model relative improvement: 0.051
- minimum prediction-error relative improvement: 0.039

This follow-up strengthens the profile-transfer evidence for the weakest seed
but preserves the limitation: seed `23` still has thin world/PE margins, so the
next mechanism step should improve dynamic-head low-end robustness rather than
claiming broad autonomy.

PE-calibrated seed `23` follow-up, verified on 2026-06-19:

- candidate config: `configs/reflexcore/local_pe_calibrated.yaml`
- rollup artifact:
  `%TEMP%\reflexcore_current_local53m_seed23_pe_calibrated_profile_rollup_20260619.json`
- profiles evaluated: `default`, `hard`, and `wide_ood`
- pass rate across profiles: 1.000
- parameter count: 53,124,754
- synthetic weighted examples per run: 222
- real-sandbox weighted examples per run: 264
- minimum offline action margin over prompt-only: 0.125
- minimum closed-loop margin over prompt-only: 0.167
- minimum real-sandbox margin over prompt-only: 0.750
- minimum world-model relative improvement: 0.421
- minimum prediction-error relative improvement: 0.205

Negative control: using the uncalibrated `configs/reflexcore/local.yaml` with
`synthetic-repeat 3` on seed `23` and `wide_ood` improved world-model margin to
`0.400` but failed the PE gate with PE relative improvement `-0.066`. This
isolates the improvement to the calibrated action-conditioned PE head rather
than to source weighting alone.

Tradeoff: the calibrated PE profile raises dynamic-head robustness but has a
lower minimum offline/closed-loop behavior margin than the uncalibrated
three-profile seed `23` run. The follow-up below keeps that tradeoff visible
while closing the stronger evidence gap: profile evaluation must reuse the
same trained checkpoint instead of retraining per profile.

Train-once calibrated profile-transfer follow-up, verified on 2026-06-19:

- CLI: `python -m reflexlm.cli.run_reflexcore_real_sandbox_adaptation_profile_matrix`
- config: `configs/reflexcore/local_pe_calibrated.yaml`
- training reuse contract: one mixed synthetic+real-sandbox train run per seed,
  reused across `default`, `hard`, and `wide_ood` evaluation profiles.
- seed `23` artifact:
  `%TEMP%\reflexcore_current_local53m_seed23_trainonce_profile_20260619_143202\real_sandbox_adaptation_profile_matrix_report.json`
- seed `13`/`17` artifact:
  `%TEMP%\reflexcore_current_local53m_seed13_17_trainonce_profile_20260619_143534\real_sandbox_adaptation_profile_matrix_report.json`
- total training runs: 3
- total profile evaluations: 9
- profile pass rate: 9/9
- parameter count per seed: 53,124,754
- synthetic weighted examples per seed: 222
- real-sandbox weighted examples per seed: 264
- minimum offline action margin over prompt-only: 0.125
- minimum closed-loop margin over prompt-only: 0.167
- minimum real-sandbox margin over prompt-only: 0.667
- minimum real-sandbox success rate: 0.917
- minimum world-model relative improvement: 0.421
- minimum prediction-error relative improvement: 0.205

This is stronger than the earlier per-profile rollup because each seed's
`model_hash` is reused across all three profiles. It supports a bounded
computer-native sensory-motor core claim for terminal/process/filesystem/time
sandbox behavior, not GUI operation or unrestricted shell autonomy.

Strict live-observation profile matrix, verified on 2026-06-20 after tightening
the aggregate improvement gate, absolute real-sandbox success gate, and
terminal-read runtime semantics:

- CLI: `python -m reflexlm.cli.run_reflexcore_real_sandbox_adaptation_profile_matrix`
- config: `configs/reflexcore/local_pe_calibrated.yaml`
- artifact:
  `%TEMP%\reflexcore_current_local53m_seed13_strict15_profile_successgate_e12_20260620_155254\real_sandbox_adaptation_profile_matrix_report.json`
- seed: `13`
- profiles evaluated with one reused checkpoint: `default` and `hard`
- profile pass rate: 2/2
- parameter count: 53,124,754
- model hash:
  `1763ce0e327148176f831850559a0e1629caf5f46123f6343502ebc95c09c245`
- dataset hash:
  `938be119d25669693381f1d6048a5ea42b3bb2b0bf60fb9299b6e122bc845cf2`
- mixed training examples: 104
- synthetic weighted examples: 36
- real-sandbox weighted examples: 68
- real-sandbox task count: 15
- real-sandbox live observation: true
- real-sandbox success rate: 1.000 vs prompt-only 0.267
- real-sandbox margin over prompt-only: 0.733
- runtime observation steps: 36
- terminal observation steps: 25
- changed-file observation steps: 12
- observed PE examples: 36
- observed PE mean: 0.0723
- observed PE max: 0.1892
- offline action accuracy: 0.667 vs prompt-only 0.556
- offline action margin over prompt-only: 0.111
- closed-loop success rate: 0.667 vs prompt-only 0.250
- closed-loop margin over prompt-only: 0.417
- minimum world-model relative improvement: 0.252
- minimum prediction-error relative improvement: 0.220
- top-level `margin_gate`: true
- top-level `improvement_gate`: true
- top-level `success_gate`: true, with required minimum real-sandbox success
  rate `1.0` and observed minimum `1.0`
- top-level `passed`: true

The strict 2026-06-20 run closes two issues found during evidence audit. First,
the profile-matrix runner now has an explicit top-level `improvement_gate`, so
a negative world-model or PE aggregate cannot pass merely because behavior
margins are positive. The failing audit artifact
`%TEMP%\reflexcore_current_local53m_seed13_strict15_profile_pe_gate_audit_20260620_152818\real_sandbox_adaptation_profile_matrix_report.json`
correctly reports `passed: false` with PE relative improvement `-0.0886`.
Second, the profile-matrix runner now has a separate top-level `success_gate`,
so a model can be reported as better than baseline while still failing a
stricter absolute real-sandbox success requirement. A 0.917 real-sandbox
success floor is rejected when `min_real_sandbox_success_rate` is `1.0`.
Third, non-live PE fallback targets now use the same deterministic
next-state-loss mask as world-model supervision, so hashed text and stochastic
diagnostic feedback are not treated as deterministic state dynamics.

Runtime semantics were also tightened for general sensory-motor behavior rather
than task-specific shortcuts: consumed stdout/stderr are no longer repeatedly
readable, refresh-visible file changes can mask stale terminal buffers, pending
file changes route toward `READ_FILE`, stale subprocesses route toward
`STOP_PROCESS`, and initial allowlisted process-start states can still choose
`RUN_COMMAND`. These changes are covered by the V0 regression suite; on
2026-06-20, `$env:PYTHONPATH='src'; python -m pytest -q tests\test_reflexcore_v0.py`
passed with `89 passed, 33 warnings`.

Current strongest local 53M strict 15-task profile matrix, verified on
2026-06-20 after adding file-read-to-command and active-process `DONE` guards
and extending train-once evaluation to `wide_ood`:

```powershell
$ts=Get-Date -Format 'yyyyMMdd_HHmmss'
$out=Join-Path $env:TEMP "reflexcore_current_local53m_seed13_17_23_strict15_profile_pe_ep12_syn3_real1_e16_postmotor_wideood_$ts"
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_real_sandbox_adaptation_profile_matrix `
  --output-dir $out `
  --config configs/reflexcore/local_pe_calibrated.yaml `
  --seed 13 `
  --seed 17 `
  --seed 23 `
  --profile default `
  --eval-profile default `
  --eval-profile hard `
  --eval-profile wide_ood `
  --episodes-per-task 12 `
  --vocab-size 4096 `
  --epochs 16 `
  --batch-size 2 `
  --learning-rate 0.0003 `
  --sequence-mode `
  --max-sequence-len 8 `
  --real-sandbox-variants 1 `
  --real-sandbox-start-variant 4 `
  --real-sandbox-max-steps 8 `
  --real-sandbox-live-observation `
  --synthetic-repeat 3 `
  --real-sandbox-repeat 1 `
  --min-pass-rate 1.0 `
  --min-profile-pass-rate 1.0 `
  --min-offline-margin 0.0 `
  --min-closed-loop-margin 0.0 `
  --min-real-sandbox-margin 0.0 `
  --min-real-sandbox-success-rate 1.0 `
  --min-parameters 20000000 `
  --max-parameters 100000000
```

- artifact:
  `%TEMP%\reflexcore_current_local53m_seed13_17_23_strict15_profile_pe_ep12_syn3_real1_e16_postmotor_wideood_20260620_165840\real_sandbox_adaptation_profile_matrix_report.json`
- seeds: `13`, `17`, `23`
- evaluated profiles with one reused checkpoint per seed: `default`, `hard`,
  and `wide_ood`
- profile pass rate: 9/9
- run pass rate: 3/3
- parameter count per seed: 53,124,754
- mixed training examples per seed: 256
- synthetic weighted examples per seed: 222
- real-sandbox weighted examples per seed: 34
- real-sandbox success rate: min/mean/max 1.000/1.000/1.000
- prompt-only real-sandbox success rate: 0.267
- minimum real-sandbox margin over prompt-only: 0.733
- offline action accuracy: min/mean/max 0.625/0.667/0.750
- minimum offline action margin over prompt-only: 0.0625
- closed-loop success rate: min/mean/max 0.583/0.611/0.667
- minimum closed-loop margin over prompt-only: 0.333
- minimum world-model relative improvement: 0.715
- minimum prediction-error relative improvement: 0.424
- runtime observation steps: min/mean/max 36/36.7/38
- terminal observation steps: min/mean/max 25/25.3/26
- changed-file observation steps: 12 per seed
- observed PE examples: min/max 36/38
- top-level `margin_gate`: true
- top-level `improvement_gate`: true
- top-level `success_gate`: true, with required minimum real-sandbox success
  rate `1.0` and observed minimum `1.0`
- top-level `passed`: true

This supersedes the earlier 2026-06-20 single-seed strict gate and the narrower
two-profile post-motor matrix as the strongest current ReflexCore V0 local
evidence. The added motor semantics are general state-affordance constraints
rather than task-name shortcuts: file content read through `READ_FILE` is not
treated as unread terminal stdout when exactly one allowlisted command remains
and no file/refresh/process evidence is pending; `DONE` is inhibited while a
subprocess is active; and terminal output that has already been consumed can
converge to `DONE` instead of rerunning a command. The claim boundary is
unchanged: this supports a bounded, typed terminal/process/filesystem/time
sandbox core, not GUI operation, unrestricted shell generation, robotics, or
production autonomy.

Sensory-ablation diagnostic, verified on 2026-06-20 against the same seed-13
53M checkpoint and `wide_ood` holdout split:

```powershell
$root="$env:TEMP\reflexcore_current_local53m_seed13_17_23_strict15_profile_pe_ep12_syn3_real1_e16_postmotor_wideood_20260620_165840"
$ckpt=Join-Path $root "seed_13\train\reflexcore_v0.pt"
$data=Join-Path $root "seed_13\eval_wide_ood\eval_benchmark\reflexcore\test.jsonl"
$out=Join-Path $env:TEMP "reflexcore_seed13_wideood_sensory_ablation_gate_20260620.json"
$env:PYTHONPATH='src'
python -m reflexlm.cli.eval_reflexcore_v0 `
  --checkpoint $ckpt `
  --dataset $data `
  --batch-size 16 `
  --device cpu `
  --sequence-mode `
  --max-sequence-len 8 `
  --ablation-mode zero_numeric `
  --require-sensory-ablation-drop 0.20 `
  --output-json $out
```

- output:
  `%TEMP%\reflexcore_seed13_wideood_sensory_ablation_gate_20260620.json`
- full raw action accuracy: 0.500
- full safety-gated action accuracy: 0.750
- `zero_numeric` raw action accuracy: 0.250
- `zero_numeric` raw action-accuracy drop: 0.250
- required drop: 0.200
- `zero_numeric` dangerous block rate: 0.000 vs full 1.000
- full next-state relative improvement: 0.930
- `zero_numeric` next-state relative improvement: -7.449
- gate `passed`: true

An exploratory companion run also evaluated `zero_hash` and `zero_all`:
`zero_hash` produced no raw action-accuracy drop, while `zero_all` matched the
`zero_numeric` drop. On this checkpoint and split, the strongest model-side
evidence is therefore dependence on structured numeric environment channels
rather than hashed visible-text residue. This is an offline diagnostic of
learned sensory dependence before runtime affordance controls; it does not
expand the claim beyond bounded terminal/process/filesystem/time behavior.

Three-seed `wide_ood` sensory world-model ablation gate, verified on 2026-06-20
after separating optional action-drop gating from next-state sensory gating:

```powershell
$root="$env:TEMP\reflexcore_current_local53m_seed13_17_23_strict15_profile_pe_ep12_syn3_real1_e16_postmotor_wideood_20260620_165840"
foreach($seed in 13,17,23){
  $ckpt=Join-Path $root "seed_$seed\train\reflexcore_v0.pt"
  $data=Join-Path $root "seed_$seed\eval_wide_ood\eval_benchmark\reflexcore\test.jsonl"
  python -m reflexlm.cli.eval_reflexcore_v0 `
    --checkpoint $ckpt `
    --dataset $data `
    --batch-size 16 `
    --device cpu `
    --sequence-mode `
    --max-sequence-len 8 `
    --ablation-mode zero_numeric `
    --require-sensory-world-drop 1.0
}
```

- rollup:
  `%TEMP%\reflexcore_three_seed_wideood_sensory_world_gate_fixed_rollup_20260620.json`
- seed `13`: world-model improvement drop 8.379, gate passed
- seed `17`: world-model improvement drop 6.432, gate passed
- seed `23`: world-model improvement drop 6.424, gate passed
- required next-state relative-improvement drop: 1.000
- action raw-accuracy drops: 0.250, 0.0625, and -0.0625

Interpretation: structured numeric observation channels are necessary for the
learned action-conditioned next-state model across all three seeds on the
`wide_ood` split. Raw action-logit dependence is not yet stable across seeds, so
the defensible claim is currently stronger for environment prediction/world
modeling than for unaided action-policy logits. Closed-loop action success still
depends on the intended combination of learned heads, typed motor decoding,
state-affordance constraints, and the safety gate.

Action-vector residual architecture update, smoke verified on 2026-06-20:

```powershell
$out=Join-Path $env:TEMP "reflexcore_smoke_action_vector_residual_20260620"
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_experiment `
  --output-dir $out `
  --config configs/reflexcore/smoke.yaml `
  --episodes-per-task 6 `
  --vocab-size 512 `
  --max-text-tokens 64 `
  --epochs 8 `
  --batch-size 4 `
  --sequence-mode `
  --max-sequence-len 8 `
  --closed-loop-episodes-per-task 2 `
  --required-baseline prompt_only_heuristic
```

- artifact:
  `%TEMP%\reflexcore_smoke_action_vector_residual_20260620`
- config field enabled: `model.action_vector_residual: true`
- parameter count: 194,908
- offline raw action accuracy: 1.000
- offline safety-gated action accuracy: 0.778 vs prompt-only 0.556
- closed-loop success rate: 0.667 vs prompt-only 0.250
- next-state relative improvement: 0.827
- prediction-error relative improvement: 0.229
- experiment `passed`: true

The corresponding sensory gate passed with both action and world-model drops:

```powershell
$root="$env:TEMP\reflexcore_smoke_action_vector_residual_20260620"
$ckpt=Join-Path $root "train\reflexcore_v0.pt"
$data=Join-Path $root "benchmark\reflexcore\test.jsonl"
$out=Join-Path $env:TEMP "reflexcore_smoke_action_vector_residual_action_ablation_gate_20260620.json"
$env:PYTHONPATH='src'
python -m reflexlm.cli.eval_reflexcore_v0 `
  --checkpoint $ckpt `
  --dataset $data `
  --batch-size 16 `
  --device cpu `
  --sequence-mode `
  --max-sequence-len 8 `
  --ablation-mode zero_numeric `
  --require-sensory-ablation-drop 0.50 `
  --require-sensory-world-drop 1.0 `
  --output-json $out
```

- output:
  `%TEMP%\reflexcore_smoke_action_vector_residual_action_ablation_gate_20260620.json`
- full raw action accuracy: 1.000
- `zero_numeric` raw action accuracy: 0.333
- action-accuracy drop: 0.667, required 0.500
- full next-state relative improvement: 0.827
- `zero_numeric` next-state relative improvement: -8.353
- next-state relative-improvement drop: 9.181, required 1.000
- `zero_hash` exploratory action drop: 0.000
- gate `passed`: true

This update directly addresses the previous limitation where raw action-logit
sensory dependence was not stable across local 53M seeds. It does not by itself
prove the large local model has regained stable action ablation across seeds;
that remains a required follow-up local matrix. The architectural movement is
that action logits now have an explicit learned path from structured computer
state, making the desired computer-native sensory-motor claim more testable.

Local-scale feasibility after enabling `action_vector_residual`, verified on
2026-06-20:

- artifact:
  `%TEMP%\reflexcore_local_feasibility_action_vector_residual_20260620`
- config: `configs/reflexcore/local_pe_calibrated.yaml`
- parameter count: 53,132,444
- required range: 20,000,000 to 100,000,000
- finite final loss: 4.638
- checkpoint exists: true
- gate `passed`: true

This preserves the first-milestone scale contract after the architecture change.

Local 53M three-seed `wide_ood` residual-action probe, verified on 2026-06-20:

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_real_sandbox_adaptation_profile_matrix `
  --config configs/reflexcore/local_pe_calibrated.yaml `
  --seed 13 `
  --seed 17 `
  --seed 23 `
  --profile default `
  --eval-profile wide_ood `
  --episodes-per-task 12 `
  --vocab-size 4096 `
  --epochs 16 `
  --batch-size 2 `
  --learning-rate 0.0003 `
  --sequence-mode `
  --max-sequence-len 8 `
  --real-sandbox-variants 1 `
  --real-sandbox-start-variant 4 `
  --real-sandbox-max-steps 8 `
  --real-sandbox-live-observation `
  --synthetic-repeat 3 `
  --real-sandbox-repeat 1 `
  --min-pass-rate 1.0 `
  --min-profile-pass-rate 1.0 `
  --min-offline-margin 0.0 `
  --min-closed-loop-margin 0.0 `
  --min-real-sandbox-margin 0.0 `
  --min-real-sandbox-success-rate 1.0 `
  --min-parameters 20000000 `
  --max-parameters 100000000
```

- seed `13` artifact:
  `%TEMP%\reflexcore_local53m_action_vector_residual_seed13_probe_20260620_210400`
- seed `17`/`23` artifact:
  `%TEMP%\reflexcore_local53m_action_vector_residual_seed17_23_probe_20260620_210716`
- parameter count per seed: 53,132,444
- profile pass rate: 3/3 on `wide_ood`
- real-sandbox success rate: 1.000 for all seeds
- prompt-only real-sandbox success rate: 0.267
- offline action accuracy: 0.750 for all seeds
- offline action margins over prompt-only: 0.1875, 0.1875, 0.3125
- closed-loop success rate: 0.667 for all seeds
- closed-loop margin over prompt-only: 0.417 for all seeds
- minimum world-model relative improvement: 0.784
- minimum prediction-error relative improvement: 0.553
- top-level margin, improvement, and success gates: true

Local 53M three-seed action sensory-ablation gate on the same `wide_ood` splits:

```powershell
python -m reflexlm.cli.eval_reflexcore_v0 `
  --checkpoint <seed_N\train\reflexcore_v0.pt> `
  --dataset <seed_N\eval_wide_ood\eval_benchmark\reflexcore\test.jsonl> `
  --batch-size 16 `
  --device cpu `
  --sequence-mode `
  --max-sequence-len 8 `
  --ablation-mode zero_numeric `
  --require-sensory-ablation-drop 0.50 `
  --require-sensory-world-drop 1.0
```

- rollup:
  `%TEMP%\reflexcore_local53m_action_vector_residual_three_seed_action_ablation_rollup_20260620.json`
- seed `13`: full action 1.000, `zero_numeric` 0.375, action drop 0.625,
  world drop 8.595
- seed `17`: full action 1.000, `zero_numeric` 0.438, action drop 0.5625,
  world drop 6.575
- seed `23`: full action 0.875, `zero_numeric` 0.0625, action drop 0.8125,
  world drop 6.450
- required action drop: 0.500
- required world-model improvement drop: 1.000
- gate passed for all three seeds

This closes the earlier local-scale limitation for the tested `wide_ood` split:
after adding the residual vector-to-action path, raw action logits now show
cross-seed dependence on structured numeric environment channels. The claim is
still bounded: it covers the current 53M V0, train-once `default` to `wide_ood`
terminal/process/filesystem/time sandbox setup with typed actions and
allowlisted commands.

Full post-residual local matrix across `default`, `hard`, and `wide_ood`,
verified on 2026-06-20:

```powershell
$root="$env:TEMP\reflexcore_local53m_action_vector_residual_seed13_17_23_fullprofile_20260620_211515"
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_real_sandbox_adaptation_profile_matrix `
  --output-dir $root `
  --config configs/reflexcore/local_pe_calibrated.yaml `
  --seed 13 `
  --seed 17 `
  --seed 23 `
  --profile default `
  --eval-profile default `
  --eval-profile hard `
  --eval-profile wide_ood `
  --episodes-per-task 12 `
  --vocab-size 4096 `
  --epochs 16 `
  --batch-size 2 `
  --learning-rate 0.0003 `
  --sequence-mode `
  --max-sequence-len 8 `
  --real-sandbox-variants 1 `
  --real-sandbox-start-variant 4 `
  --real-sandbox-max-steps 8 `
  --real-sandbox-live-observation `
  --synthetic-repeat 3 `
  --real-sandbox-repeat 1 `
  --min-pass-rate 1.0 `
  --min-profile-pass-rate 1.0 `
  --min-offline-margin 0.0 `
  --min-closed-loop-margin 0.0 `
  --min-real-sandbox-margin 0.0 `
  --min-real-sandbox-success-rate 1.0 `
  --min-parameters 20000000 `
  --max-parameters 100000000
```

- matrix artifact:
  `%TEMP%\reflexcore_local53m_action_vector_residual_seed13_17_23_fullprofile_20260620_211515\real_sandbox_adaptation_profile_matrix_report.json`
- ablation rollup:
  `docs/reflexcore_evidence/reflexcore_local53m_fullprofile_ablation_rollup_20260620.json`
- seeds: `13`, `17`, `23`
- evaluated profiles with one reused checkpoint per seed: `default`, `hard`,
  and `wide_ood`
- profile pass rate: 9/9
- parameter count per seed: 53,132,444
- minimum raw action accuracy: 0.875
- minimum safety-gated action accuracy: 0.750
- minimum prompt-only offline action accuracy baseline: 0.4375
- minimum closed-loop success rate: 0.667
- prompt-only closed-loop success rate: 0.250
- real-sandbox success rate: 1.000 for all runs
- prompt-only real-sandbox success rate: 0.267
- minimum world-model relative improvement: 0.784
- minimum prediction-error relative improvement: 0.553

All-mode sensory ablation on the same 9 held-out profile splits:

```powershell
python -m reflexlm.cli.eval_reflexcore_v0 `
  --checkpoint <seed_N\train\reflexcore_v0.pt> `
  --dataset <profile test.jsonl> `
  --batch-size 16 `
  --device cpu `
  --sequence-mode `
  --max-sequence-len 8 `
  --ablation-mode zero_numeric `
  --ablation-mode zero_hash `
  --ablation-mode zero_all
```

- `zero_numeric` action drop minimum: 0.250
- `zero_numeric` next-state relative-improvement drop minimum: 6.181
- `zero_all` action drop minimum: 0.500
- `zero_all` next-state relative-improvement drop minimum: 6.167

Interpretation: the full observation vector is necessary for action logits
across all nine local 53M profile evaluations (`zero_all` action drop >= 0.5),
and structured numeric observation channels are necessary for the learned
action-conditioned world model (`zero_numeric` world-model drop >= 6.18).
However, raw action-logit dependence on numeric channels alone is not uniform
in the same-profile/default-family evaluations: seed `17` had only 0.25 action
drop on `default` and 0.3125 on `hard` under `zero_numeric`. The defensible
post-residual claim is therefore: the model is a bounded sensory-motor core
whose typed action behavior and world model depend on observation input, while
some action choices can still be supported by non-numeric text/hash channels in
easier profile splits.

Numeric-only action auxiliary objective, added after this diagnostic:

- training config key: `sensory_training.numeric_action_aux_weight`
- accepted local weight: `0.25`
- smoke weight: `0.2`
- auxiliary view: keep structured numeric observation features, zero hash bins,
  and zero text tokens before applying the same action imitation target
- purpose: strengthen the direct structured-state-to-action pathway without
  hardcoding task-specific actions or weakening the safety gate

Accepted full local 53M numeric-action auxiliary matrix, verified on
2026-06-20 with `numeric_action_aux_weight=0.25`:

```powershell
python -m reflexlm.cli.run_reflexcore_real_sandbox_adaptation_profile_matrix `
  --config configs/reflexcore/local_pe_calibrated.yaml `
  --seed 13 `
  --seed 17 `
  --seed 23 `
  --profile default `
  --eval-profile default `
  --eval-profile hard `
  --eval-profile wide_ood `
  --episodes-per-task 12 `
  --vocab-size 4096 `
  --epochs 16 `
  --batch-size 2 `
  --learning-rate 0.0003 `
  --sequence-mode `
  --max-sequence-len 8 `
  --real-sandbox-variants 1 `
  --real-sandbox-start-variant 4 `
  --real-sandbox-max-steps 8 `
  --real-sandbox-live-observation `
  --synthetic-repeat 3 `
  --real-sandbox-repeat 1 `
  --min-pass-rate 1.0 `
  --min-profile-pass-rate 1.0 `
  --min-offline-margin 0.0 `
  --min-closed-loop-margin 0.0 `
  --min-real-sandbox-margin 0.0 `
  --min-real-sandbox-success-rate 1.0 `
  --min-parameters 20000000 `
  --max-parameters 100000000
```

- evidence rollup:
  `docs/reflexcore_evidence/reflexcore_numeric_action_aux025_fullprofile_ablation_rollup_20260620.json`
- formal sensory-ablation gate report:
  `docs/reflexcore_evidence/reflexcore_numeric_action_aux025_formal_sensory_ablation_matrix_20260620.json`
- seeds: `13`, `17`, `23`
- evaluated profiles with one reused checkpoint per seed: `default`, `hard`,
  and `wide_ood`
- matrix `passed`: true
- profile pass rate: 9/9
- parameter count per seed: 53,132,444
- minimum raw action accuracy: 0.875
- minimum safety-gated action accuracy: 0.750
- maximum prompt-only offline action baseline: 0.5625
- minimum closed-loop success rate: 0.667
- maximum prompt-only closed-loop success rate: 0.250
- minimum real-sandbox success rate: 1.000
- maximum prompt-only real-sandbox success rate: 0.267
- minimum world-model relative improvement: 0.438
- minimum prediction-error relative improvement: 0.447
- minimum `zero_numeric` action drop across all nine profile splits: 0.500
- minimum `zero_numeric` world-model drop: 5.960
- minimum `zero_all` action drop: 0.625
- minimum `zero_all` world-model drop: 5.986

Formal gate command:

```powershell
eval-reflexcore-sensory-ablation-matrix `
  --matrix-dir <accepted_numeric_action_aux025_matrix_dir> `
  --output-json docs/reflexcore_evidence/reflexcore_numeric_action_aux025_formal_sensory_ablation_matrix_20260620.json `
  --seed 13 `
  --seed 17 `
  --seed 23 `
  --profile default `
  --profile hard `
  --profile wide_ood `
  --mode zero_numeric `
  --batch-size 16 `
  --device cpu `
  --sequence-mode `
  --max-sequence-len 8 `
  --min-action-drop 0.5 `
  --min-world-drop 1.0
```

Formal gate result:

- rows: 9
- passed rows: 9
- `zero_numeric` action drop: min 0.500, mean 0.681, max 0.750
- `zero_numeric` world-model drop: min 5.960, mean 6.813, max 8.488
- gate `passed`: true

Combined mechanism dossier gate, generated on 2026-06-21:

```powershell
build-reflexcore-mechanism-dossier `
  --accepted-rollup-json docs/reflexcore_evidence/reflexcore_numeric_action_aux025_fullprofile_ablation_rollup_20260620.json `
  --sensory-ablation-json docs/reflexcore_evidence/reflexcore_numeric_action_aux025_formal_sensory_ablation_matrix_20260620.json `
  --negative-control-json docs/reflexcore_evidence/reflexcore_numeric_action_aux_fullprofile_diagnostic_20260620.json `
  --output-json docs/reflexcore_evidence/reflexcore_numeric_action_aux025_mechanism_dossier_20260621.json
```

Dossier result:

- evidence artifact:
  `docs/reflexcore_evidence/reflexcore_numeric_action_aux025_mechanism_dossier_20260621.json`
- verdict: `bounded_reflexcore_v0_mechanism_evidence_ready`
- gate `passed`: true
- required parameter range: 20M-100M; observed: 53,132,444
- offline prompt-only margin: 0.3125
- closed-loop prompt-only margin: 0.417
- real-sandbox prompt-only margin: 0.733
- world-model relative-improvement floor: observed 0.438 vs required 0.300
- prediction-error relative-improvement floor: observed 0.447 vs required 0.300
- sensory matrix coverage: seeds `13`, `17`, `23`; profiles `default`,
  `hard`, `wide_ood`; rows 9/9 passed
- negative control: the `numeric_action_aux_weight=0.35` diagnostic was rejected
  by the primary rollup gate because strict real-sandbox success fell to 0.933
  even though its sensory-ablation drop was stronger

This combined gate is now the preferred audit artifact for the accepted V0
mechanism claim. A run that improves one diagnostic but fails behavior, safety,
world-model, prediction-error, coverage, or negative-control rejection must not
be described as accepted.

Interpretation: this is the current accepted local 53M V0 configuration for
bounded terminal/process/filesystem/time sensory-motor behavior. It repairs the
previous seed17/default numeric-action weakness while preserving the strict
real-sandbox success gate. The evidence supports a computer-native
sensory-motor core with typed actions and allowlisted command execution; it
still does not support GUI, free-shell, robotics, or production-autonomy claims.

Negative tuning control retained: `numeric_action_aux_weight=0.35` made the
numeric-action ablation stronger (`zero_numeric` action drop minimum 0.750) but
failed the strict real-sandbox success gate because seed `23` reached 0.933
instead of the required 1.000. That setting is therefore diagnostic rather than
accepted.

```powershell
$env:PYTHONPATH='src'
python -m reflexlm.cli.run_reflexcore_real_sandbox_adaptation_matrix `
  --output-dir "$env:TEMP\reflexcore_real_sandbox_expanded_local53m_matrix_actionpe_scale0_refresh005_e12" `
  --config configs/reflexcore/local.yaml `
  --seed 13 `
  --seed 17 `
  --seed 23 `
  --episodes-per-task 12 `
  --vocab-size 4096 `
  --max-text-tokens 128 `
  --epochs 12 `
  --batch-size 4 `
  --sequence-mode `
  --max-sequence-len 8 `
  --real-sandbox-variants 12 `
  --real-sandbox-start-variant 1 `
  --min-parameters 20000000 `
  --max-parameters 100000000
```

Verified on 2026-06-17 with the expanded 7-task real sandbox suite,
`episodes-per-task 12`, `epochs 12`, 53M local config,
`prediction_error_conditioning=state_action`,
`prediction_error_calibration_scale=0.0`, and the calibrated homeostatic PE
refresh threshold `0.05`. Offline action selection is evaluated from free model
logits; next-state and PE dynamics are evaluated with the target action because
those heads are trained as action-conditioned dynamics heads.

| Seed | Passed | Offline action vs prompt | Closed-loop vs prompt | Real sandbox vs prompt | World improvement | PE improvement |
| --- | --- | --- | --- | --- | --- | --- |
| 13 | true | 0.6875 vs 0.5625 | 0.750 vs 0.250 | 1.000 vs 0.429 | 0.480 | 0.510 |
| 17 | true | 0.9375 vs 0.5625 | 0.667 vs 0.250 | 1.000 vs 0.429 | 0.413 | 0.072 |
| 23 | true | 0.6250 vs 0.4375 | 0.583 vs 0.250 | 1.000 vs 0.429 | 0.378 | 0.185 |

Common run properties:

- parameter count: 53,107,847
- mixed training examples per run: 170
- synthetic training examples per run: 74
- real-sandbox training examples per run: 96
- held-out real-sandbox successful actions:
  `REFRESH_STATE > READ_FILE`, `RUN_COMMAND`, `READ_STDOUT`,
  `READ_STDERR`, `WAIT`, `STOP_PROCESS`, `BLOCK`
- matrix pass rate: 1.000
- minimum paired offline action margin over prompt-only: 0.125
- minimum paired closed-loop margin over prompt-only: 0.333
- minimum paired real-sandbox margin over prompt-only: 0.571
- minimum world-model relative improvement: 0.378
- minimum prediction-error relative improvement: 0.072

Historical negative control retained: before expanding the task suite, with
`episodes-per-task 6`, seed 23 passed real sandbox at `1.000 vs 0.500` but
failed the synthetic offline gate by tying the prompt-only baseline at
`0.444 vs 0.444`. The recommended mixed gate therefore uses
`episodes-per-task 12` to avoid real-sandbox over-weighting in the mixed
dataset.

Fresh matrix negative controls retained:

- After expanding the real sandbox task suite to 7 tasks, `epochs 12` with the
  old prediction-error loss weight (`0.2`) passed action, closed-loop,
  real-sandbox, and world-model gates, but passed only 1/3 seeds overall because
  seed 17 and seed 23 failed the prediction-error gate.
- Raising the local `prediction_error` loss weight to `1.0` fixed the expanded
  three-seed matrix without disabling the PE gate, but its weakest PE margin
  remained small (`0.003`) under the old residual calibration.
- An action-conditioned PE probe improved seed 23 but exposed seed instability
  when the learned calibration head remained active. The accepted local config
  now uses `prediction_error_calibration_scale=0.0`, making PE derive from the
  action-conditioned next-state delta norm rather than an extra learned scalar
- Calibrating the default homeostatic PE refresh threshold to `0.05` makes the
  `salience + prediction_error -> REFRESH_STATE` path reachable at the observed
  V0 PE scale. The matrix remains green after this change; closed-loop and
  offline margins are lower than the inactive-threshold run but still positive
  across all seeds.

## Prediction-Error Diagnostic Gate

The aggregate PE metric can hide whether the signal is robust across motor
families. This diagnostic evaluates PE by action group and marks zero-variance
groups as non-evaluable rather than failed, because a constant-mean baseline has
zero MAE for those groups.

```powershell
$env:PYTHONPATH='src'
$matrix = "$env:TEMP\reflexcore_real_sandbox_expanded_local53m_matrix_actionpe_scale0_refresh005_e12"

python -m reflexlm.cli.eval_reflexcore_prediction_error `
  --checkpoint "$matrix\seed_17\train\reflexcore_v0.pt" `
  --dataset "$matrix\seed_17\synthetic_benchmark\reflexcore\test.jsonl" `
  --output-dir "$env:TEMP\reflexcore_pe_diag_expanded_seed17_actionpe_scale0_refresh005" `
  --sequence-mode `
  --max-text-tokens 128 `
  --min-relative-improvement 0.0 `
  --min-action-group-pass-rate 0.0 `
  --min-evaluable-constant-mae 0.0001
```

Verified on 2026-06-17 for the weakest PE seed in the accepted matrix (`17`):

- overall PE relative improvement: 0.050
- overall PE model MAE: 0.02521
- constant-mean PE baseline MAE: 0.02653
- action groups: 7
- evaluable action groups: 3
- action-group pass rate over evaluable groups: 0.667
- `ASK_USER`: passed, relative improvement 0.405
- `READ_FILE`: passed, relative improvement 0.496
- `READ_STDERR`: failed, relative improvement -1.408 under low within-group
  target variance (`constant_mean_mae=0.0041`)
- non-evaluable low/zero-variance groups:
  `BLOCK`, `REFRESH_STATE`, `RUN_COMMAND`, `WAIT`

Failed calibration probe retained: setting `prediction_error_mode=direct` with
scale `0.2` on seed 23 preserved real-sandbox success (`1.000 vs 0.429`) and
world-model improvement (`0.487`) but failed synthetic action and PE
(`PE relative improvement = -0.147`). A learned residual calibration head with
scale `0.02` also passed real-sandbox behavior but had seed-level PE instability
in the expanded matrix. The accepted local configuration remains
`delta_plus_calibration` with PE loss weight `1.0`, action-conditioned dynamics,
and calibration scale `0.0`.

Boundary: this evidence supports only a bounded, typed sensory-motor language
core for terminal/process/filesystem/time sandbox tasks. It does not support
claims about GUI control, unrestricted shell generation, robotics, or
production autonomy.
