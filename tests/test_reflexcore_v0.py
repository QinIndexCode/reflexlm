from __future__ import annotations

from pathlib import Path
import sys
import time

import pytest
import torch

import reflexlm.runtime.receptors as receptor_module
from reflexlm.core.dataset import (
    ReflexCoreEpisodeDataset,
    ReflexCoreTorchDataset,
    build_reflexcore_examples,
    collate_reflexcore_batch,
    collate_reflexcore_sequence_batch,
    read_reflexcore_jsonl,
    slot_bounds_ok,
    split_examples_by_episode,
    split_hashes,
    tensors_for_example,
    write_reflexcore_jsonl,
)
from reflexlm.core.losses import compute_reflexcore_losses
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
from reflexlm.core.motor import ReflexCoreMotorConfig, decode_reflexcore_motor
from reflexlm.core.observation import ReflexCoreObservationContext
from reflexlm.core.runner import ReflexCoreSandboxConfig, ReflexCoreSandboxRunner
from reflexlm.core.evaluation import (
    acceptance_against_baselines,
    evaluate_baseline_policies,
    evaluate_reflexcore_model,
    evaluate_reflexcore_sensory_ablation,
    prediction_error_acceptance,
    world_model_acceptance,
)
from reflexlm.core.benchmark import (
    ReflexCoreBenchmarkConfig,
    build_reflexcore_benchmark,
)
from reflexlm.core.closed_loop import (
    evaluate_closed_loop_baselines,
    evaluate_reflexcore_closed_loop,
)
from reflexlm.core.experiment import (
    ReflexCoreExperimentConfig,
    run_reflexcore_experiment,
)
from reflexlm.core.stability import (
    ReflexCoreStabilityConfig,
    run_reflexcore_stability,
)
from reflexlm.core.training import numeric_only_observation_vectors, train_reflexcore_v0
from reflexlm.core.profile_matrix import (
    ReflexCoreProfileMatrixConfig,
    run_reflexcore_profile_matrix,
)
from reflexlm.core.local_feasibility import (
    ReflexCoreLocalFeasibilityConfig,
    run_reflexcore_local_feasibility,
)
from reflexlm.core.sandbox_benchmark import (
    RealSandboxEvalConfig,
    build_real_sandbox_oracle_dataset,
    evaluate_reflexcore_real_sandbox,
)
from reflexlm.core.real_sandbox_adaptation import (
    ReflexCoreRealSandboxAdaptationConfig,
    run_reflexcore_real_sandbox_adaptation,
)
from reflexlm.core.real_sandbox_adaptation_matrix import (
    ReflexCoreRealSandboxAdaptationMatrixConfig,
    run_reflexcore_real_sandbox_adaptation_matrix,
)
from reflexlm.core.real_sandbox_adaptation_profile_matrix import (
    ReflexCoreRealSandboxAdaptationProfileMatrixConfig,
    _improvement_gate,
    _success_gate,
    run_reflexcore_real_sandbox_adaptation_profile_matrix,
)
from reflexlm.core.real_sandbox_capability_matrix import (
    ReflexCoreRealSandboxCapabilityMatrixConfig,
    run_reflexcore_real_sandbox_capability_matrix,
)
from reflexlm.core.sensory_ablation_matrix import (
    ReflexCoreSensoryAblationMatrixConfig,
    run_reflexcore_sensory_ablation_matrix,
)
from reflexlm.core.prediction_error_report import (
    ReflexCorePredictionErrorReportConfig,
    build_reflexcore_prediction_error_report,
)
from reflexlm.core.experience import examples_from_step_trace, write_experience_jsonl
from reflexlm.core.online_adaptation import (
    ReflexCoreOnlineAdaptationConfig,
    _prediction_error_motor_probe,
    _rejected_reason,
    _retention_gate,
    adapt_reflexcore_from_experience,
)
from reflexlm.core.online_adaptation_gate import (
    ReflexCoreFamilyHoldoutMatrixConfig,
    ReflexCoreOnlineAdaptationGateConfig,
    run_family_holdout_matrix,
    run_online_adaptation_gate,
    split_online_adaptation_examples,
)
from reflexlm.core.schema import (
    ComputerObservation,
    MotorAction,
    ReflexCoreTrainingExample,
    action_from_index,
    action_to_index,
    dataset_hash,
)
from reflexlm.cli.eval_reflexcore_real_sandbox import _apply_min_success_gate
from reflexlm.data.tasks import build_env, rollout_env
from reflexlm.models.features import (
    MAX_CANDIDATE_SLOTS,
    PREDICTION_FEEDBACK_FEATURES,
    PREDICTION_FEEDBACK_START_INDEX,
    StateVectorizer,
    resolve_structured_action,
    valid_action_mask,
)
from reflexlm.runtime.oracle import RuleOracle
from reflexlm.runtime.safety import SafetyLayer
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    GoalSpec,
    InternalTarget,
    ProcessStatus,
    RouteName,
    SourceType,
    TaskType,
)


def _records() -> list:
    oracle = RuleOracle()
    records = []
    for task_type in (TaskType.TEST_FAILURE, TaskType.FILE_CHANGE, TaskType.DANGEROUS_ACTION):
        env = build_env(task_type, 0)
        records.extend(rollout_env(env, policy=oracle))
    return records


def _without_terminal_output(state):
    return state.model_copy(
        update={
            "terminal": state.terminal.model_copy(
                update={
                    "stdout_delta": "",
                    "stderr_delta": "",
                    "stdout_unread": False,
                    "stderr_unread": False,
                    "stdout_lines": 0,
                    "stderr_lines": 0,
                    "last_output_channel": None,
                }
            )
        }
    )


class _FixedMotorModel(torch.nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        action_type: ActionType,
        command_slot: int = 0,
        file_slot: int = 0,
        vocab_size: int = 512,
    ) -> None:
        super().__init__()
        self.config = ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=vocab_size)
        self.action_type = action_type
        self.command_slot = command_slot
        self.file_slot = file_slot

    def forward(self, observation_vectors, text_tokens=None, *, hidden=None, action_indices=None):
        batch_size, seq_len = observation_vectors.shape[:2]
        action_logits = torch.full(
            (batch_size, seq_len, len(ActionType)),
            -10.0,
            dtype=torch.float32,
        )
        action_logits[..., action_to_index(self.action_type)] = 10.0
        command_logits = torch.full(
            (batch_size, seq_len, self.config.max_command_slots),
            -10.0,
            dtype=torch.float32,
        )
        command_logits[..., self.command_slot] = 10.0
        file_logits = torch.full(
            (batch_size, seq_len, self.config.max_file_slots),
            -10.0,
            dtype=torch.float32,
        )
        file_logits[..., self.file_slot] = 10.0
        return {
            "hidden": hidden,
            "action_logits": action_logits,
            "command_slot_logits": command_logits,
            "file_slot_logits": file_logits,
            "route_logits": torch.zeros(batch_size, seq_len, len(RouteName)),
            "target_logits": torch.zeros(batch_size, seq_len, len(InternalTarget)),
            "risk": torch.zeros(batch_size, seq_len, 1),
            "salience": torch.ones(batch_size, seq_len, 1),
            "prediction_error": torch.zeros(batch_size, seq_len, 1),
            "next_state": observation_vectors,
            "text_logits": torch.zeros(batch_size, seq_len, self.config.vocab_size),
        }


def test_reflexcore_live_observation_context_vectorizes_bounded_receptor_state(
    tmp_path: Path,
) -> None:
    goal = GoalSpec(
        task_type=TaskType.FILE_CHANGE,
        description="observe bounded sandbox filesystem and terminal deltas",
        command_allowlist=["echo safe", "python -m pytest"],
        watched_paths=[str(tmp_path)],
    )
    target = tmp_path / "note.txt"
    target.write_text("initial", encoding="utf-8")
    context = ReflexCoreObservationContext(
        goal=goal,
        vocab_size=512,
        max_text_tokens=32,
        vectorizer=StateVectorizer(hash_bins=16),
    )

    first = context.observe(prompt_visible=True)
    assert first.goal == goal
    assert first.process.status.value == "exited"
    assert first.filesystem.changed_paths == []
    assert first.candidate_commands == ["echo safe", "python -m pytest"]
    assert first.candidate_files == [str(tmp_path)]
    assert first.runtime_evidence.source == SourceType.RUNTIME_OBSERVATION.value
    assert len(first.vector) == context.vectorizer.vector_dim
    assert len(first.text_tokens) <= 32

    time.sleep(0.02)
    target.write_text("changed again", encoding="utf-8")
    second = context.observe(stdout_delta="command finished", last_command="echo safe")

    assert str(target) in second.filesystem.changed_paths
    assert str(target) in second.filesystem.dirty_files
    assert str(target) in second.candidate_files
    assert second.filesystem.external_change_detected is True
    assert second.terminal.stdout_unread is True
    assert second.terminal.last_command == "echo safe"
    assert second.runtime_evidence.changed_files == [str(target)]
    assert second.runtime_evidence.terminal_observations == ["command finished"]
    assert second.source_frame_hash != first.source_frame_hash


def test_reflexcore_live_observation_context_detects_created_and_deleted_files(
    tmp_path: Path,
) -> None:
    goal = GoalSpec(
        task_type=TaskType.FILE_CHANGE,
        description="detect created and deleted sandbox files",
        watched_paths=[str(tmp_path)],
    )
    context = ReflexCoreObservationContext(goal=goal, vectorizer=StateVectorizer(hash_bins=0))
    baseline = context.observe_state()
    assert baseline.filesystem.changed_paths == []

    created = tmp_path / "created.txt"
    created.write_text("created", encoding="utf-8")
    create_state = context.observe_state()
    assert create_state.filesystem.changed_paths == [str(created)]
    assert create_state.filesystem.external_change_detected is True

    created.unlink()
    delete_state = context.observe_state()
    assert delete_state.filesystem.changed_paths == [str(created)]
    assert delete_state.filesystem.dirty_files == [str(created)]


def test_process_receptor_tolerates_pid_that_exits_during_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StaleProcess:
        pid = 12345

        def oneshot(self) -> StaleProcess:
            return self

        def __enter__(self) -> StaleProcess:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def status(self) -> str:
            raise receptor_module.psutil.NoSuchProcess(self.pid)

    monkeypatch.setattr(
        receptor_module.psutil,
        "Process",
        lambda _pid: StaleProcess(),
    )

    state = receptor_module.ProcessReceptor().snapshot(12345)

    assert state.pid == 12345
    assert state.status == ProcessStatus.EXITED


def test_reflexcore_sandbox_live_observation_loop_reobserves_command_created_file(
    tmp_path: Path,
) -> None:
    command = (
        f'"{sys.executable}" -c "from pathlib import Path; '
        "Path('created.txt').write_text('ok', encoding='utf-8'); "
        "print('made-file')\""
    )
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(
            sandbox_root=tmp_path,
            allowed_commands=(command,),
            allow_process_execution=True,
            max_steps=1,
        )
    )
    goal = GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description="run an allowlisted command and reobserve created file",
        command_allowlist=[command],
        watched_paths=[str(tmp_path)],
    )
    model = _FixedMotorModel(
        input_dim=StateVectorizer().vector_dim,
        action_type=ActionType.RUN_COMMAND,
    )

    result = runner.run_model_live_observation_loop(model, goal)

    assert result.initial_state.filesystem.changed_paths == []
    assert len(result.trace) == 1
    step = result.trace[0]
    assert step.safety_decision.allowed is True
    assert step.safety_decision.action is not None
    assert step.safety_decision.action.type == ActionType.RUN_COMMAND
    assert "made-file" in step.stdout
    assert "created.txt" in step.state.filesystem.changed_paths
    assert str(tmp_path / "created.txt") not in step.state.filesystem.changed_paths
    assert "created.txt" in step.state.runtime_evidence.changed_files
    assert str(tmp_path / "created.txt") not in step.state.runtime_evidence.changed_files
    assert step.state.runtime_evidence.source == SourceType.RUNTIME_OBSERVATION.value
    assert step.state.terminal.stdout_unread is True
    assert step.model_prediction_error == 0.0
    assert step.observed_prediction_error is not None
    assert step.observed_prediction_error > 0.0
    assert step.state.runtime_evidence.model_prediction_error == step.model_prediction_error
    assert step.state.runtime_evidence.observed_prediction_error == (
        step.observed_prediction_error
    )
    assert step.state.runtime_evidence.prediction_error_delta == (
        step.observed_prediction_error - step.model_prediction_error
    )


def test_reflexcore_live_observation_experience_records_reobserved_transition(
    tmp_path: Path,
) -> None:
    command = (
        f'"{sys.executable}" -c "from pathlib import Path; '
        "Path('created.txt').write_text('ok', encoding='utf-8'); "
        "print('made-file')\""
    )
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(
            sandbox_root=tmp_path / "sandbox",
            allowed_commands=(command,),
            allow_process_execution=True,
            max_steps=1,
        )
    )
    goal = GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description="capture live reobserved transition as training data",
        command_allowlist=[command],
        watched_paths=[str(tmp_path / "sandbox")],
    )
    model = _FixedMotorModel(
        input_dim=StateVectorizer().vector_dim,
        action_type=ActionType.RUN_COMMAND,
    )
    live_loop = runner.run_model_live_observation_loop(model, goal)
    path = tmp_path / "experience.jsonl"

    summary = write_experience_jsonl(
        path,
        initial_state=live_loop.initial_state,
        trace=live_loop.trace,
        episode_id="live-reobserve-experience",
        vocab_size=model.config.vocab_size,
        max_text_tokens=64,
    )
    examples = read_reflexcore_jsonl(path)

    assert summary.source == SourceType.MODEL.value
    assert summary.live_observation is True
    assert summary.runtime_observation_examples == 1
    assert summary.changed_file_observations == 1
    assert summary.terminal_observation_examples == 1
    assert summary.post_safety_actions is True
    assert summary.observed_prediction_error_examples == 1
    assert summary.observed_prediction_error_mean is not None
    assert summary.observed_prediction_error_mean > 0.0
    assert summary.observed_prediction_error_max == summary.observed_prediction_error_mean
    assert summary.model_prediction_error_mean == 0.0
    assert len(examples) == 1
    assert examples[0].action.type == ActionType.RUN_COMMAND
    assert examples[0].action.command == command
    assert examples[0].next_observation.runtime_evidence.source == (
        SourceType.RUNTIME_OBSERVATION.value
    )
    assert examples[0].next_observation.runtime_evidence.changed_files == [
        "created.txt"
    ]
    assert str(tmp_path / "sandbox" / "created.txt") not in (
        examples[0].next_observation.runtime_evidence.changed_files
    )
    assert "made-file" in examples[0].next_observation.terminal.stdout_delta
    assert "oracle_action" not in examples[0].model_dump_json()
    target_tensors = tensors_for_example(
        examples[0],
        max_text_tokens=64,
        input_dim=len(examples[0].observation.vector),
    )
    assert live_loop.trace[0].observed_prediction_error is not None
    assert target_tensors["prediction_error_targets"].item() == pytest.approx(
        live_loop.trace[0].observed_prediction_error
    )


def test_reflexcore_schema_roundtrip_and_action_mapping() -> None:
    example = build_reflexcore_examples(_records()[:1], vocab_size=512)[0]
    loaded = ReflexCoreTrainingExample.model_validate_json(example.model_dump_json())
    assert loaded.canonical_hash() == example.canonical_hash()
    observation = ComputerObservation.model_validate_json(example.observation.model_dump_json())
    assert observation.candidate_commands == example.observation.candidate_commands
    for action in ActionType:
        assert action_from_index(action_to_index(action)) == action


def test_reflexcore_runner_consumes_terminal_stdout(tmp_path: Path) -> None:
    runner = ReflexCoreSandboxRunner(ReflexCoreSandboxConfig(sandbox_root=tmp_path))
    base_state = runner.initial_state(
        GoalSpec(
            task_type=TaskType.ROUTINE_RECOVERY,
            description="read buffered stdout exactly once",
        )
    )
    state = base_state.model_copy(
        update={
            "terminal": base_state.terminal.model_copy(
                update={
                    "stdout_delta": "buffered output",
                    "stdout_unread": True,
                    "stdout_lines": 1,
                    "prompt_visible": True,
                    "last_output_channel": "stdout",
                }
            )
        }
    )

    result = runner.step(state, ActionDecision(type=ActionType.READ_STDOUT))

    assert result.stdout == "buffered output"
    assert result.state.terminal.stdout_unread is False


def test_reflexcore_motor_reads_visible_file_before_idle(tmp_path: Path) -> None:
    runner = ReflexCoreSandboxRunner(ReflexCoreSandboxConfig(sandbox_root=tmp_path))
    base_state = runner.initial_state(
        GoalSpec(
            task_type=TaskType.FILE_CHANGE,
            description="read pending changed file before idling",
            watched_paths=[str(tmp_path)],
        )
    )
    state = base_state.model_copy(
        update={
            "filesystem": base_state.filesystem.model_copy(
                update={
                    "changed_paths": ["note.txt"],
                    "dirty_files": ["note.txt"],
                    "watched_paths": [str(tmp_path)],
                }
            )
        }
    )
    model = _FixedMotorModel(
        input_dim=StateVectorizer().vector_dim,
        action_type=ActionType.WAIT,
    )

    proposal = runner.propose_with_state(model, state)

    assert proposal.safety_decision.action is not None
    assert proposal.safety_decision.action.type == ActionType.READ_FILE
    assert proposal.safety_decision.action.file_target == "note.txt"


def test_reflexcore_motor_reads_visible_stdout_before_done(tmp_path: Path) -> None:
    runner = ReflexCoreSandboxRunner(ReflexCoreSandboxConfig(sandbox_root=tmp_path))
    base_state = runner.initial_state(
        GoalSpec(
            task_type=TaskType.ROUTINE_RECOVERY,
            description="read terminal output before claiming completion",
        )
    )
    state = base_state.model_copy(
        update={
            "terminal": base_state.terminal.model_copy(
                update={
                    "stdout_delta": "ready",
                    "stdout_unread": True,
                    "stdout_lines": 1,
                    "last_output_channel": "stdout",
                }
            )
        }
    )
    model = _FixedMotorModel(
        input_dim=StateVectorizer().vector_dim,
        action_type=ActionType.DONE,
    )

    proposal = runner.propose_with_state(model, state)

    assert proposal.safety_decision.action is not None
    assert proposal.safety_decision.action.type == ActionType.READ_STDOUT


def test_reflexcore_motor_finishes_after_observed_terminal_output(tmp_path: Path) -> None:
    runner = ReflexCoreSandboxRunner(ReflexCoreSandboxConfig(sandbox_root=tmp_path))
    base_state = runner.initial_state(
        GoalSpec(
            task_type=TaskType.ROUTINE_RECOVERY,
            description="finish after command stdout has already been read",
            command_allowlist=["python -m pytest -q"],
        )
    )
    state = base_state.model_copy(
        update={
            "terminal": base_state.terminal.model_copy(
                update={
                    "stdout_delta": "sandbox-command-ok",
                    "stdout_unread": False,
                    "stdout_lines": 1,
                    "prompt_visible": True,
                    "last_output_channel": "stdout",
                    "last_command": ActionType.READ_STDOUT.value,
                }
            )
        }
    )
    action_logits = torch.full((1, 1, len(ActionType)), -10.0)
    action_logits[0, 0, action_to_index(ActionType.RUN_COMMAND)] = 10.0
    outputs = {
        "action_logits": action_logits,
        "command_slot_logits": torch.tensor([[[10.0, -10.0, -10.0, -10.0]]]),
        "file_slot_logits": torch.zeros(1, 1, 4),
        "route_logits": torch.zeros(1, 1, len(RouteName)),
        "target_logits": torch.zeros(1, 1, len(InternalTarget)),
        "risk": torch.zeros(1, 1, 1),
        "salience": torch.zeros(1, 1, 1),
        "prediction_error": torch.zeros(1, 1, 1),
    }

    decoded = decode_reflexcore_motor(outputs, state)

    assert decoded.action.type == ActionType.DONE
    assert decoded.action.reason == "reflexcore_terminal_output_already_observed"


def test_reflexcore_motor_runs_single_command_after_file_read(tmp_path: Path) -> None:
    command = "python -m pytest -q"
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(sandbox_root=tmp_path, allowed_commands=(command,))
    )
    base_state = runner.initial_state(
        GoalSpec(
            task_type=TaskType.ROUTINE_RECOVERY,
            description="run the remaining command after changed file content is read",
            command_allowlist=[command],
        )
    )
    state = base_state.model_copy(
        update={
            "terminal": base_state.terminal.model_copy(
                update={
                    "stdout_delta": "sandbox-note:0",
                    "stdout_unread": True,
                    "stdout_lines": 1,
                    "prompt_visible": True,
                    "last_output_channel": "stdout",
                    "last_command": ActionType.READ_FILE.value,
                }
            )
        }
    )
    action_logits = torch.full((1, 1, len(ActionType)), -10.0)
    action_logits[0, 0, action_to_index(ActionType.READ_STDOUT)] = 10.0
    outputs = {
        "action_logits": action_logits,
        "command_slot_logits": torch.tensor([[[10.0, -10.0, -10.0, -10.0]]]),
        "file_slot_logits": torch.zeros(1, 1, 4),
        "route_logits": torch.zeros(1, 1, len(RouteName)),
        "target_logits": torch.zeros(1, 1, len(InternalTarget)),
        "risk": torch.zeros(1, 1, 1),
        "salience": torch.zeros(1, 1, 1),
        "prediction_error": torch.zeros(1, 1, 1),
    }

    decoded = decode_reflexcore_motor(outputs, state)

    assert decoded.action.type == ActionType.RUN_COMMAND
    assert decoded.action.command == command
    assert decoded.action.reason == "reflexcore_file_read_complete_command_affordance"


def test_reflexcore_motor_waits_before_done_while_process_active(tmp_path: Path) -> None:
    runner = ReflexCoreSandboxRunner(ReflexCoreSandboxConfig(sandbox_root=tmp_path))
    base_state = runner.initial_state(
        GoalSpec(
            task_type=TaskType.PROCESS_HANG,
            description="wait for active process before declaring done",
        )
    )
    state = base_state.model_copy(
        update={
            "process": base_state.process.model_copy(update={"status": ProcessStatus.RUNNING}),
            "terminal": base_state.terminal.model_copy(update={"prompt_visible": False}),
        }
    )
    action_logits = torch.full((1, 1, len(ActionType)), -10.0)
    action_logits[0, 0, action_to_index(ActionType.DONE)] = 10.0
    outputs = {
        "action_logits": action_logits,
        "command_slot_logits": torch.zeros(1, 1, 4),
        "file_slot_logits": torch.zeros(1, 1, 4),
        "route_logits": torch.zeros(1, 1, len(RouteName)),
        "target_logits": torch.zeros(1, 1, len(InternalTarget)),
        "risk": torch.zeros(1, 1, 1),
        "salience": torch.zeros(1, 1, 1),
        "prediction_error": torch.zeros(1, 1, 1),
    }

    decoded = decode_reflexcore_motor(outputs, state)

    assert decoded.action.type == ActionType.WAIT
    assert decoded.action.reason == "reflexcore_done_blocked_by_active_process"


def test_reflexcore_state_vectorizer_exposes_stdout_file_conflict(tmp_path: Path) -> None:
    runner = ReflexCoreSandboxRunner(ReflexCoreSandboxConfig(sandbox_root=tmp_path))
    state = runner.initial_state(
        GoalSpec(
            task_type=TaskType.ROUTINE_RECOVERY,
            description="observe changed file before trusting stale stdout",
            watched_paths=[str(tmp_path)],
        )
    )
    file_change_state = state.model_copy(
        update={
            "terminal": state.terminal.model_copy(
                update={
                    "stdout_delta": "old buffered output",
                    "stdout_unread": False,
                    "last_output_channel": "stdout",
                }
            ),
            "filesystem": state.filesystem.model_copy(
                update={
                    "changed_paths": ["note.txt"],
                    "dirty_files": ["note.txt"],
                    "external_change_detected": True,
                    "stale_cache_detected": True,
                }
            ),
        }
    )
    stale_stdout_conflict = file_change_state.model_copy(
        update={
            "terminal": file_change_state.terminal.model_copy(update={"stdout_unread": True})
        }
    )
    vectorizer = StateVectorizer(hash_bins=0)
    file_change_vector = vectorizer.vectorize_state(file_change_state)
    conflict_vector = vectorizer.vectorize_state(stale_stdout_conflict)

    assert len(conflict_vector) == vectorizer.numeric_dim
    assert (conflict_vector != file_change_vector).any()


def test_reflexcore_state_vectorizer_exposes_prediction_error_feedback(
    tmp_path: Path,
) -> None:
    runner = ReflexCoreSandboxRunner(ReflexCoreSandboxConfig(sandbox_root=tmp_path))
    state = runner.initial_state(
        GoalSpec(
            task_type=TaskType.ROUTINE_RECOVERY,
            description="observe prediction error feedback before next action",
        )
    )
    feedback_state = state.model_copy(
        update={
            "runtime_evidence": state.runtime_evidence.model_copy(
                update={
                    "model_prediction_error": 0.1,
                    "observed_prediction_error": 1.6,
                    "prediction_error_delta": 1.5,
                }
            )
        }
    )
    vectorizer = StateVectorizer(hash_bins=0)
    base_vector = vectorizer.vectorize_state(state)
    feedback_vector = vectorizer.vectorize_state(feedback_state)

    assert len(feedback_vector) == vectorizer.numeric_dim
    assert (feedback_vector != base_vector).any()
    serialized = feedback_state.model_dump_json()
    assert "observed_prediction_error" in serialized


def test_reflexcore_training_target_prefers_live_observed_prediction_error() -> None:
    example = build_reflexcore_examples(_records()[:1], vocab_size=512)[0]
    observed_error = 0.73
    next_observation = example.observation.model_copy(
        update={
            "runtime_evidence": example.observation.runtime_evidence.model_copy(
                update={"observed_prediction_error": observed_error}
            )
        }
    )
    live_example = example.model_copy(update={"next_observation": next_observation})

    tensors = tensors_for_example(
        live_example,
        max_text_tokens=64,
        input_dim=len(live_example.observation.vector),
    )

    assert tensors["prediction_error_targets"].item() == pytest.approx(observed_error)
    assert tensors["next_state"].tolist() == live_example.observation.vector


def test_reflexcore_prediction_error_fallback_ignores_hash_noise() -> None:
    vectorizer = StateVectorizer(hash_bins=8)
    example = build_reflexcore_examples(
        _records()[:1],
        vectorizer=vectorizer,
        vocab_size=512,
    )[0]
    clean_next = example.next_observation.model_copy(
        update={
            "runtime_evidence": example.next_observation.runtime_evidence.model_copy(
                update={"observed_prediction_error": None}
            )
        }
    )
    clean_example = example.model_copy(update={"next_observation": clean_next})
    noisy_vector = list(clean_next.vector)
    for index in range(len(noisy_vector) - vectorizer.hash_bins, len(noisy_vector)):
        noisy_vector[index] = noisy_vector[index] + 1000.0
    noisy_next = clean_next.model_copy(update={"vector": noisy_vector})
    noisy_example = clean_example.model_copy(update={"next_observation": noisy_next})

    clean_tensors = tensors_for_example(
        clean_example,
        max_text_tokens=64,
        input_dim=len(clean_example.observation.vector),
    )
    noisy_tensors = tensors_for_example(
        noisy_example,
        max_text_tokens=64,
        input_dim=len(noisy_example.observation.vector),
    )

    assert noisy_tensors["prediction_error_targets"].item() == pytest.approx(
        clean_tensors["prediction_error_targets"].item()
    )


def test_reflexcore_next_state_loss_mask_excludes_hashes_and_pe_feedback() -> None:
    vectorizer = StateVectorizer(hash_bins=8)
    example = build_reflexcore_examples(
        _records()[:1],
        vectorizer=vectorizer,
        vocab_size=512,
    )[0]

    tensors = tensors_for_example(
        example,
        max_text_tokens=64,
        input_dim=len(example.observation.vector),
    )
    mask = tensors["next_state_loss_mask"]
    feedback_end = PREDICTION_FEEDBACK_START_INDEX + PREDICTION_FEEDBACK_FEATURES

    assert torch.equal(mask, torch.tensor(vectorizer.world_model_target_mask()))
    assert mask[PREDICTION_FEEDBACK_START_INDEX:feedback_end].sum().item() == 0.0
    assert mask[-vectorizer.hash_bins :].sum().item() == 0.0


def test_reflexcore_dataset_split_hash_and_slot_bounds_are_stable() -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)
    first = split_examples_by_episode(examples, seed=17)
    second = split_examples_by_episode(examples, seed=17)
    assert split_hashes(first) == split_hashes(second)
    assert slot_bounds_ok(examples)
    assert all("oracle_action" not in item.observation.model_dump(mode="json") for item in examples)


def test_reflexcore_benchmark_package_has_reproducible_splits(tmp_path: Path) -> None:
    manifest = build_reflexcore_benchmark(
        ReflexCoreBenchmarkConfig(
            output_dir=tmp_path,
            episodes_per_task=3,
            split_strategy="episode_random",
            seed=23,
            vocab_size=512,
        )
    )
    assert manifest["scope"] == "terminal_process_filesystem_time_only"
    assert manifest["free_shell_generation"] is False
    assert "closed_loop_eval_cli" in manifest["recommended_gate"]
    assert "stability_eval_cli" in manifest["recommended_gate"]
    assert "profile_transfer_stability_cli" in manifest["recommended_gate"]
    assert "profile_matrix_stability_cli" in manifest["recommended_gate"]
    assert "local_feasibility_cli" in manifest["recommended_gate"]
    assert "local_stability_cli" in manifest["recommended_gate"]
    assert "local_profile_matrix_cli" in manifest["recommended_gate"]
    assert "real_sandbox_eval_cli" in manifest["recommended_gate"]
    assert "online_experience_cli" in manifest["recommended_gate"]
    assert "online_adaptation_cli" in manifest["recommended_gate"]
    assert "prediction_error_diagnostic_cli" in manifest["recommended_gate"]
    assert "real_sandbox_dataset_cli" in manifest["recommended_gate"]
    assert "real_sandbox_adaptation_cli" in manifest["recommended_gate"]
    assert "real_sandbox_adaptation_matrix_cli" in manifest["recommended_gate"]
    assert "real_sandbox_adaptation_profile_matrix_cli" in manifest["recommended_gate"]
    assert (tmp_path / "trajectories" / "train.jsonl").exists()
    train_examples = read_reflexcore_jsonl(tmp_path / "reflexcore" / "train.jsonl")
    assert train_examples
    assert manifest["reflexcore"]["split_hashes"]["train"] == split_hashes(
        {"train": train_examples}
    )["train"]


def test_reflexcore_model_loss_and_checkpoint_smoke(tmp_path: Path) -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)[:8]
    dataset = ReflexCoreTorchDataset(examples)
    items = [dataset[index] for index in range(min(4, len(dataset)))]
    batch = collate_reflexcore_batch(items)
    config = ReflexCoreV0Config.smoke(input_dim=dataset.input_dim, vocab_size=512)
    model = ReflexCoreV0(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    outputs = model(
        batch["observation_vectors"],
        batch["text_tokens"],
        action_indices=batch["action_indices"],
    )
    losses = compute_reflexcore_losses(outputs, batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    optimizer.step()
    assert model.parameter_count() > 0
    checkpoint_path = tmp_path / "reflexcore_v0.pt"
    torch.save({"config": config.to_dict(), "model_state_dict": model.state_dict()}, checkpoint_path)
    loaded = ReflexCoreV0(ReflexCoreV0Config(**torch.load(checkpoint_path)["config"]))
    loaded.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])


def test_reflexcore_numeric_only_view_removes_hash_bins() -> None:
    vectorizer = StateVectorizer(hash_bins=4)
    numeric_dim = StateVectorizer(hash_bins=0).numeric_dim
    input_dim = vectorizer.numeric_dim + 4
    vectors = torch.arange(float(input_dim)).reshape(1, 1, input_dim)

    numeric_only = numeric_only_observation_vectors(vectors, zero_hash=True)

    assert torch.equal(numeric_only[..., :numeric_dim], vectors[..., :numeric_dim])
    assert torch.equal(
        numeric_only[..., numeric_dim:],
        torch.zeros_like(vectors[..., numeric_dim:]),
    )


def test_reflexcore_training_records_numeric_action_aux_loss(tmp_path: Path) -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)[:8]
    dataset_path = tmp_path / "train.jsonl"
    write_reflexcore_jsonl(dataset_path, examples)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  vocab_size: 512",
                "  text_embedding_dim: 32",
                "  hidden_dim: 64",
                "  transformer_layers: 1",
                "  transformer_heads: 2",
                "  gru_layers: 1",
                "  dropout: 0.0",
                "  action_vector_residual: true",
                "training:",
                "  seed: 13",
                "  epochs: 1",
                "  batch_size: 4",
                "  learning_rate: 0.001",
                "  device: cpu",
                "sensory_training:",
                "  numeric_action_aux_weight: 0.2",
                "  numeric_action_aux_zero_text: true",
                "  numeric_action_aux_zero_hash: true",
                "dataset:",
                "  max_text_tokens: 64",
                "  sequence_mode: true",
                "  max_sequence_len: 4",
            ]
        ),
        encoding="utf-8",
    )

    summary = train_reflexcore_v0(
        dataset_path=dataset_path,
        config_path=config_path,
        output_dir=tmp_path / "train",
    )

    assert summary["sensory_training"]["numeric_action_aux_weight"] == pytest.approx(0.2)
    assert "numeric_action_aux_loss" in summary["history"][-1]
    assert torch.isfinite(
        torch.tensor(float(summary["history"][-1]["numeric_action_aux_loss"]))
    )


def test_reflexcore_next_state_loss_ignores_masked_diagnostic_dimensions() -> None:
    example = build_reflexcore_examples(_records()[:1], vocab_size=512)[0]
    dataset = ReflexCoreTorchDataset([example])
    batch = collate_reflexcore_batch([dataset[0]])
    config = ReflexCoreV0Config.smoke(input_dim=dataset.input_dim, vocab_size=512)
    model = ReflexCoreV0(config)
    outputs = model(
        batch["observation_vectors"],
        batch["text_tokens"],
        action_indices=batch["action_indices"],
    )
    exact_next_state = batch["next_state"].clone()
    masked_next_state = exact_next_state.clone()
    masked_next_state[batch["next_state_loss_mask"] <= 0.0] = 999.0
    exact_outputs = dict(outputs)
    masked_outputs = dict(outputs)
    exact_outputs["next_state"] = exact_next_state
    masked_outputs["next_state"] = masked_next_state

    exact_loss = compute_reflexcore_losses(exact_outputs, batch)["next_state_loss"]
    masked_loss = compute_reflexcore_losses(masked_outputs, batch)["next_state_loss"]

    assert exact_loss.item() == pytest.approx(0.0)
    assert masked_loss.item() == pytest.approx(0.0)


def test_reflexcore_next_state_head_initializes_as_copy_current_baseline() -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)[:2]
    dataset = ReflexCoreTorchDataset(examples)
    items = [dataset[index] for index in range(2)]
    batch = collate_reflexcore_batch(items)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=dataset.input_dim, vocab_size=512))
    outputs = model(
        batch["observation_vectors"],
        batch["text_tokens"],
        action_indices=batch["action_indices"],
    )
    assert torch.allclose(outputs["next_state"], batch["observation_vectors"], atol=1e-6)


def test_reflexcore_action_vector_residual_uses_structured_observation() -> None:
    input_dim = 4
    config = ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512)
    config.action_vector_residual = True
    model = ReflexCoreV0(config)
    assert model.action_vector_head is not None
    with torch.no_grad():
        for parameter in model.action_head.parameters():
            parameter.zero_()
        for module in model.vector_encoder:
            if isinstance(module, torch.nn.Linear):
                module.weight.zero_()
                module.bias.zero_()
                module.weight[0, 0] = 1.0
            elif isinstance(module, torch.nn.LayerNorm):
                module.weight.fill_(1.0)
                module.bias.zero_()
        model.action_vector_head.weight.zero_()
        model.action_vector_head.bias.zero_()
        model.action_vector_head.weight[action_to_index(ActionType.RUN_COMMAND), 0] = 1.0

    positive = torch.tensor([[[2.0, 0.0, 0.0, 0.0]]])
    negative = torch.tensor([[[-2.0, 0.0, 0.0, 0.0]]])
    tokens = torch.zeros(1, 1, 4, dtype=torch.long)
    pos_logits = model(positive, tokens)["action_logits"]
    neg_logits = model(negative, tokens)["action_logits"]

    action_index = action_to_index(ActionType.RUN_COMMAND)
    assert pos_logits[0, 0, action_index] > neg_logits[0, 0, action_index]


def test_reflexcore_direct_prediction_error_mode_is_bounded() -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)[:2]
    dataset = ReflexCoreTorchDataset(examples)
    batch = collate_reflexcore_batch([dataset[0], dataset[1]])
    config = ReflexCoreV0Config.smoke(input_dim=dataset.input_dim, vocab_size=512)
    config.prediction_error_mode = "direct"
    config.prediction_error_calibration_scale = 0.2
    model = ReflexCoreV0(config)
    outputs = model(batch["observation_vectors"], batch["text_tokens"])
    assert torch.all(outputs["prediction_error"] >= 0.0)
    assert torch.all(outputs["prediction_error"] <= 0.2)


def test_reflexcore_prediction_error_mode_rejects_unknown_value() -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)[:1]
    dataset = ReflexCoreTorchDataset(examples)
    batch = collate_reflexcore_batch([dataset[0]])
    config = ReflexCoreV0Config.smoke(input_dim=dataset.input_dim, vocab_size=512)
    config.prediction_error_mode = "unknown"
    model = ReflexCoreV0(config)
    try:
        model(batch["observation_vectors"], batch["text_tokens"])
    except ValueError as exc:
        assert "prediction_error_mode" in str(exc)
    else:
        raise AssertionError("unknown prediction_error_mode should fail")


def test_reflexcore_prediction_error_can_condition_on_action() -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)[:1]
    dataset = ReflexCoreTorchDataset(examples)
    batch = collate_reflexcore_batch([dataset[0]])
    config = ReflexCoreV0Config.smoke(input_dim=dataset.input_dim, vocab_size=512)
    config.prediction_error_conditioning = "state_action"
    config.prediction_error_mode = "direct"
    config.prediction_error_calibration_scale = 1.0
    model = ReflexCoreV0(config)
    with torch.no_grad():
        model.action_embedding.weight.zero_()
        model.action_embedding.weight[action_to_index(ActionType.BLOCK), 0] = 4.0
        model.prediction_error_head.weight.zero_()
        model.prediction_error_head.bias.zero_()
        action_start = model.config.hidden_dim
        model.prediction_error_head.weight[0, action_start] = 1.0
    wait_action = torch.full_like(batch["action_indices"], action_to_index(ActionType.WAIT))
    block_action = torch.full_like(batch["action_indices"], action_to_index(ActionType.BLOCK))
    wait_pe = model(
        batch["observation_vectors"],
        batch["text_tokens"],
        action_indices=wait_action,
    )["prediction_error"]
    block_pe = model(
        batch["observation_vectors"],
        batch["text_tokens"],
        action_indices=block_action,
    )["prediction_error"]
    assert not torch.allclose(wait_pe, block_pe)


def test_reflexcore_prediction_error_conditioning_rejects_unknown_value() -> None:
    config = ReflexCoreV0Config.smoke(input_dim=4, vocab_size=512)
    config.prediction_error_conditioning = "unknown"
    try:
        ReflexCoreV0(config)
    except ValueError as exc:
        assert "prediction_error_conditioning" in str(exc)
    else:
        raise AssertionError("unknown prediction_error_conditioning should fail")


def test_reflexcore_homeostatic_motor_control_uses_risk_head() -> None:
    state = _without_terminal_output(_records()[0].state).model_copy(
        update={
            "goal": _records()[0].state.goal.model_copy(
                update={"command_allowlist": ["echo ok"]}
            )
        }
    )
    action_logits = torch.full((1, 1, len(ActionType)), -10.0)
    action_logits[0, 0, action_to_index(ActionType.RUN_COMMAND)] = 10.0
    outputs = {
        "action_logits": action_logits,
        "command_slot_logits": torch.tensor([[[10.0, -10.0, -10.0, -10.0]]]),
        "file_slot_logits": torch.zeros(1, 1, 4),
        "route_logits": torch.zeros(1, 1, 4),
        "target_logits": torch.zeros(1, 1, 4),
        "risk": torch.tensor([[[0.95]]]),
        "salience": torch.tensor([[[0.2]]]),
        "prediction_error": torch.tensor([[[0.1]]]),
    }
    decoded = decode_reflexcore_motor(
        outputs,
        state,
        config=ReflexCoreMotorConfig(risk_block_threshold=0.9),
    )
    assert decoded.action.type == ActionType.BLOCK
    assert decoded.action.reason == "reflexcore_risk_threshold"


def test_reflexcore_homeostatic_motor_control_refreshes_on_prediction_error() -> None:
    state = _without_terminal_output(_records()[0].state)
    action_logits = torch.full((1, 1, len(ActionType)), -10.0)
    action_logits[0, 0, action_to_index(ActionType.WAIT)] = 10.0
    outputs = {
        "action_logits": action_logits,
        "command_slot_logits": torch.zeros(1, 1, 4),
        "file_slot_logits": torch.zeros(1, 1, 4),
        "route_logits": torch.zeros(1, 1, 4),
        "target_logits": torch.zeros(1, 1, 4),
        "risk": torch.tensor([[[0.1]]]),
        "salience": torch.tensor([[[0.9]]]),
        "prediction_error": torch.tensor([[[0.08]]]),
    }
    decoded = decode_reflexcore_motor(outputs, state)
    assert decoded.action.type == ActionType.REFRESH_STATE
    assert decoded.action.reason == "reflexcore_prediction_error_refresh"


def test_reflexcore_homeostatic_motor_control_refreshes_on_observed_pe_feedback() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "runtime_evidence": base_state.runtime_evidence.model_copy(
                update={"observed_prediction_error": 0.6}
            )
        }
    )
    action_logits = torch.full((1, 1, len(ActionType)), -10.0)
    action_logits[0, 0, action_to_index(ActionType.WAIT)] = 10.0
    outputs = {
        "action_logits": action_logits,
        "command_slot_logits": torch.zeros(1, 1, 4),
        "file_slot_logits": torch.zeros(1, 1, 4),
        "route_logits": torch.zeros(1, 1, 4),
        "target_logits": torch.zeros(1, 1, 4),
        "risk": torch.tensor([[[0.1]]]),
        "salience": torch.tensor([[[0.1]]]),
        "prediction_error": torch.tensor([[[0.0]]]),
    }
    decoded = decode_reflexcore_motor(
        outputs,
        state,
        config=ReflexCoreMotorConfig(observed_prediction_error_refresh_threshold=0.5),
    )
    assert decoded.action.type == ActionType.REFRESH_STATE
    assert decoded.action.reason == "reflexcore_observed_prediction_error_refresh"


def test_reflexcore_active_process_masks_run_command_affordance() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "goal": base_state.goal.model_copy(update={"command_allowlist": ["echo ok"]}),
            "process": base_state.process.model_copy(
                update={"status": ProcessStatus.SLEEPING}
            ),
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.RUN_COMMAND)] == 0.0
    assert mask[action_to_index(ActionType.REFRESH_STATE)] == 0.0
    assert mask[action_to_index(ActionType.WAIT)] == 1.0


def test_reflexcore_pending_file_read_masks_new_command_until_clean() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    pending_state = base_state.model_copy(
        update={
            "goal": base_state.goal.model_copy(update={"command_allowlist": ["echo ok"]}),
            "filesystem": base_state.filesystem.model_copy(
                update={
                    "changed_paths": ["first.txt", "second.txt"],
                    "dirty_files": ["first.txt", "second.txt"],
                    "external_change_detected": False,
                    "stale_cache_detected": False,
                }
            ),
        }
    )

    pending_mask = valid_action_mask(pending_state)

    assert pending_mask[action_to_index(ActionType.RUN_COMMAND)] == 0.0
    assert pending_mask[action_to_index(ActionType.READ_FILE)] == 1.0
    assert pending_mask[action_to_index(ActionType.WAIT)] == 0.0
    assert pending_mask[action_to_index(ActionType.DONE)] == 0.0
    assert pending_mask[action_to_index(ActionType.REFRESH_STATE)] == 0.0

    clean_state = pending_state.model_copy(
        update={
            "filesystem": pending_state.filesystem.model_copy(
                update={"changed_paths": [], "dirty_files": []}
            )
        }
    )
    clean_mask = valid_action_mask(clean_state)

    assert clean_mask[action_to_index(ActionType.RUN_COMMAND)] == 1.0


def test_reflexcore_manual_input_masks_redundant_ask_user() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "process": base_state.process.model_copy(
                update={"waiting_for_input": True}
            ),
            "terminal": base_state.terminal.model_copy(
                update={"input_requested": True}
            ),
            "user": base_state.user.model_copy(update={"manual_input_active": True}),
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.ASK_USER)] == 0.0
    assert mask[action_to_index(ActionType.WAIT)] == 1.0


def test_reflexcore_refresh_signal_masks_stale_file_read() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "filesystem": base_state.filesystem.model_copy(
                update={
                    "changed_paths": ["src/cache/state.py"],
                    "dirty_files": ["src/cache/state.py"],
                    "external_change_detected": True,
                }
            )
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.REFRESH_STATE)] == 1.0
    assert mask[action_to_index(ActionType.READ_FILE)] == 0.0


def test_reflexcore_refresh_signal_masks_stale_terminal_buffer() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "goal": base_state.goal.model_copy(update={"command_allowlist": ["echo ok"]}),
            "filesystem": base_state.filesystem.model_copy(
                update={
                    "changed_paths": ["note.txt"],
                    "dirty_files": ["note.txt"],
                    "external_change_detected": True,
                    "stale_cache_detected": True,
                }
            ),
            "terminal": base_state.terminal.model_copy(
                update={
                    "stdout_delta": "stale buffered stdout",
                    "stdout_unread": True,
                    "stdout_lines": 1,
                    "last_command": None,
                    "last_output_channel": "stdout",
                }
            ),
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.REFRESH_STATE)] == 1.0
    assert mask[action_to_index(ActionType.READ_STDOUT)] == 0.0


def test_reflexcore_process_hang_goal_masks_new_command() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "goal": base_state.goal.model_copy(
                update={
                    "task_type": TaskType.PROCESS_HANG,
                    "command_allowlist": ["python tools/inspect_process.py"],
                }
            ),
            "process": base_state.process.model_copy(
                update={"status": ProcessStatus.EXITED}
            ),
            "terminal": base_state.terminal.model_copy(
                update={"prompt_visible": True, "last_command": "python worker.py"}
            ),
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.RUN_COMMAND)] == 0.0
    assert mask[action_to_index(ActionType.DONE)] == 1.0


def test_reflexcore_process_hang_initial_goal_allows_process_launch() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "goal": base_state.goal.model_copy(
                update={
                    "task_type": TaskType.PROCESS_HANG,
                    "command_allowlist": ["python worker.py"],
                }
            ),
            "process": base_state.process.model_copy(
                update={"status": ProcessStatus.EXITED}
            ),
            "terminal": base_state.terminal.model_copy(
                update={"prompt_visible": True, "last_command": None}
            ),
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.RUN_COMMAND)] == 1.0


def test_reflexcore_command_slot_uses_goal_literal_semantic_cue() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    wrong = "python -c \"print('sandbox-command-wrong-0')\""
    target = "python -c \"print('sandbox-command-ok-0')\""
    state = base_state.model_copy(
        update={
            "goal": base_state.goal.model_copy(
                update={
                    "description": (
                        "Select the command that prints sandbox-command-ok-0; "
                        "ignore sandbox-command-wrong-0."
                    ),
                    "command_allowlist": [wrong, target],
                }
            )
        }
    )

    action = resolve_structured_action(
        action_index=action_to_index(ActionType.RUN_COMMAND),
        command_index=0,
        file_index=0,
        state=state,
        confidence=0.8,
    )

    assert action.type == ActionType.RUN_COMMAND
    assert action.command == target


def test_reflexcore_unread_command_output_masks_non_terminal_actions() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "goal": base_state.goal.model_copy(update={"command_allowlist": ["echo ok"]}),
            "terminal": base_state.terminal.model_copy(
                update={
                    "stdout_delta": "command output",
                    "stdout_unread": True,
                    "stdout_lines": 1,
                    "last_command": "echo ok",
                    "last_output_channel": "stdout",
                }
            ),
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.READ_STDOUT)] == 1.0
    assert mask[action_to_index(ActionType.RUN_COMMAND)] == 0.0
    assert mask[action_to_index(ActionType.READ_FILE)] == 0.0
    assert mask[action_to_index(ActionType.REFRESH_STATE)] == 0.0
    assert mask[action_to_index(ActionType.WAIT)] == 0.0


def test_reflexcore_consumed_stdout_is_not_readable_again() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "terminal": base_state.terminal.model_copy(
                update={
                    "stdout_delta": "already consumed",
                    "stdout_unread": False,
                    "stdout_lines": 1,
                    "last_command": ActionType.READ_STDOUT.value,
                    "last_output_channel": "stdout",
                }
            ),
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.READ_STDOUT)] == 0.0


def test_reflexcore_buffered_output_without_commands_forces_terminal_read() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "goal": base_state.goal.model_copy(update={"command_allowlist": []}),
            "terminal": base_state.terminal.model_copy(
                update={
                    "stderr_delta": "buffered failure",
                    "stderr_unread": True,
                    "stderr_lines": 1,
                    "prompt_visible": True,
                    "last_output_channel": "stderr",
                }
            ),
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.READ_STDERR)] == 1.0
    assert mask[action_to_index(ActionType.REFRESH_STATE)] == 0.0
    assert mask[action_to_index(ActionType.WAIT)] == 0.0


def test_reflexcore_ready_command_masks_idle_actions() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "goal": base_state.goal.model_copy(
                update={
                    "command_allowlist": ["echo ok"],
                    "task_type": TaskType.PROCESS_HANG,
                }
            ),
            "process": base_state.process.model_copy(
                update={"status": ProcessStatus.EXITED}
            ),
            "terminal": base_state.terminal.model_copy(
                update={"prompt_visible": True, "last_command": None}
            ),
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.RUN_COMMAND)] == 1.0
    assert mask[action_to_index(ActionType.WAIT)] == 0.0
    assert mask[action_to_index(ActionType.DONE)] == 0.0
    assert mask[action_to_index(ActionType.REFRESH_STATE)] == 0.0


def test_reflexcore_stale_process_forces_stop_instead_of_wait() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "time": base_state.time.model_copy(update={"since_last_output_ms": 30_000}),
            "process": base_state.process.model_copy(
                update={
                    "status": ProcessStatus.RUNNING,
                    "runtime_ms": 60_000,
                    "last_output_ms": 30_000,
                    "cpu_percent": 99.0,
                    "resource_alert": True,
                }
            ),
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.STOP_PROCESS)] == 1.0
    assert mask[action_to_index(ActionType.WAIT)] == 0.0
    assert mask[action_to_index(ActionType.RUN_COMMAND)] == 0.0


def test_reflexcore_watched_paths_alone_do_not_enable_read_file() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "filesystem": base_state.filesystem.model_copy(
                update={
                    "watched_paths": ["workspace"],
                    "changed_paths": [],
                    "dirty_files": [],
                }
            )
        }
    )

    mask = valid_action_mask(state)

    assert mask[action_to_index(ActionType.READ_FILE)] == 0.0


def test_reflexcore_active_process_keeps_wait_despite_prediction_error() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "process": base_state.process.model_copy(
                update={"status": ProcessStatus.SLEEPING}
            ),
            "runtime_evidence": base_state.runtime_evidence.model_copy(
                update={"observed_prediction_error": 0.6}
            ),
        }
    )
    action_logits = torch.full((1, 1, len(ActionType)), -10.0)
    action_logits[0, 0, action_to_index(ActionType.WAIT)] = 10.0
    outputs = {
        "action_logits": action_logits,
        "command_slot_logits": torch.zeros(1, 1, 4),
        "file_slot_logits": torch.zeros(1, 1, 4),
        "route_logits": torch.zeros(1, 1, 4),
        "target_logits": torch.zeros(1, 1, 4),
        "risk": torch.tensor([[[0.1]]]),
        "salience": torch.tensor([[[0.9]]]),
        "prediction_error": torch.tensor([[[0.08]]]),
    }

    decoded = decode_reflexcore_motor(
        outputs,
        state,
        config=ReflexCoreMotorConfig(observed_prediction_error_refresh_threshold=0.5),
    )

    assert decoded.action.type == ActionType.WAIT


def test_reflexcore_process_hang_done_not_overridden_by_prediction_error() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "goal": base_state.goal.model_copy(
                update={"task_type": TaskType.PROCESS_HANG}
            ),
            "process": base_state.process.model_copy(
                update={"status": ProcessStatus.EXITED}
            ),
            "terminal": base_state.terminal.model_copy(update={"prompt_visible": True}),
            "runtime_evidence": base_state.runtime_evidence.model_copy(
                update={"observed_prediction_error": 0.6}
            ),
        }
    )
    action_logits = torch.full((1, 1, len(ActionType)), -10.0)
    action_logits[0, 0, action_to_index(ActionType.DONE)] = 10.0
    outputs = {
        "action_logits": action_logits,
        "command_slot_logits": torch.zeros(1, 1, 4),
        "file_slot_logits": torch.zeros(1, 1, 4),
        "route_logits": torch.zeros(1, 1, 4),
        "target_logits": torch.zeros(1, 1, 4),
        "risk": torch.tensor([[[0.1]]]),
        "salience": torch.tensor([[[0.9]]]),
        "prediction_error": torch.tensor([[[0.08]]]),
    }

    decoded = decode_reflexcore_motor(
        outputs,
        state,
        config=ReflexCoreMotorConfig(observed_prediction_error_refresh_threshold=0.5),
    )

    assert decoded.action.type == ActionType.DONE


def test_reflexcore_safety_holds_run_command_while_process_active() -> None:
    base_state = _without_terminal_output(_records()[0].state)
    state = base_state.model_copy(
        update={
            "goal": base_state.goal.model_copy(update={"command_allowlist": ["echo ok"]}),
            "process": base_state.process.model_copy(
                update={"status": ProcessStatus.SLEEPING}
            ),
        }
    )

    decision = SafetyLayer().enforce(
        ActionDecision(type=ActionType.RUN_COMMAND, command="echo ok"),
        state.goal,
        state,
    )

    assert decision.allowed is False
    assert decision.reason == "active_process_running"
    assert decision.action.type == ActionType.WAIT


def test_reflexcore_homeostatic_motor_control_waits_on_low_prediction_error() -> None:
    state = _without_terminal_output(_records()[0].state)
    action_logits = torch.full((1, 1, len(ActionType)), -10.0)
    action_logits[0, 0, action_to_index(ActionType.WAIT)] = 10.0
    outputs = {
        "action_logits": action_logits,
        "command_slot_logits": torch.zeros(1, 1, 4),
        "file_slot_logits": torch.zeros(1, 1, 4),
        "route_logits": torch.zeros(1, 1, 4),
        "target_logits": torch.zeros(1, 1, 4),
        "risk": torch.tensor([[[0.1]]]),
        "salience": torch.tensor([[[0.9]]]),
        "prediction_error": torch.tensor([[[0.01]]]),
    }
    decoded = decode_reflexcore_motor(outputs, state)
    assert decoded.action.type == ActionType.WAIT


def test_reflexcore_episode_sequence_training_smoke() -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)
    dataset = ReflexCoreEpisodeDataset(examples, max_sequence_len=3)
    items = [dataset[index] for index in range(min(3, len(dataset)))]
    batch = collate_reflexcore_sequence_batch(items)
    config = ReflexCoreV0Config.smoke(input_dim=dataset.input_dim, vocab_size=512)
    model = ReflexCoreV0(config)
    outputs = model(
        batch["observation_vectors"],
        batch["text_tokens"],
        action_indices=batch["action_indices"],
    )
    losses = compute_reflexcore_losses(outputs, batch)
    assert batch["observation_vectors"].ndim == 3
    assert batch["loss_mask"].sum().item() >= len(items)
    assert torch.isfinite(losses["loss"])


def test_reflexcore_evaluation_reports_model_and_baselines() -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)
    dataset = ReflexCoreTorchDataset(examples)
    config = ReflexCoreV0Config.smoke(input_dim=dataset.input_dim, vocab_size=512)
    model = ReflexCoreV0(config)
    summary = evaluate_reflexcore_model(model, examples, batch_size=2)
    sequence_summary = evaluate_reflexcore_model(
        model,
        examples,
        batch_size=2,
        sequence_mode=True,
        max_sequence_len=3,
    )
    baselines = evaluate_baseline_policies(examples)
    acceptance = acceptance_against_baselines(
        summary,
        baselines,
        required_baselines=["static_wait"],
    )
    assert summary["counts"]["total"] == len(examples)
    assert sequence_summary["counts"]["total"] == len(examples)
    assert sequence_summary["safety_gated"]["counts"]["total"] == len(examples)
    assert summary["safety_gated"]["dangerous_block_rate"] is not None
    assert summary["copy_current_next_state_mse"] >= 0.0
    assert "next_state_relative_improvement" in summary
    assert "prediction_error_mae" in summary
    assert "prediction_error_constant_mean_mae" in summary
    assert "relative_improvement" in prediction_error_acceptance(summary)
    assert "model_next_state_mse" in world_model_acceptance(summary)
    assert "prompt_only_heuristic" in baselines
    assert "static_wait" in acceptance["details"]


def test_reflexcore_sensory_ablation_reports_vector_dependence() -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)[:2]
    input_dim = len(examples[0].observation.vector)
    wait_example = examples[0].model_copy(
        update={
            "observation": examples[0].observation.model_copy(
                update={"vector": [1.0] + [0.0] * (input_dim - 1)}
            ),
            "action": MotorAction(type=ActionType.WAIT),
        }
    )
    done_example = examples[1].model_copy(
        update={
            "observation": examples[1].observation.model_copy(
                update={"vector": [-1.0] + [0.0] * (input_dim - 1)}
            ),
            "action": MotorAction(type=ActionType.DONE),
        }
    )

    class VectorSignActionModel(torch.nn.Module):
        config = ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512)

        def forward(
            self,
            observation_vectors: torch.Tensor,
            text_tokens: torch.Tensor,
            *,
            action_indices: torch.Tensor | None = None,
            hidden: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor | None]:
            batch_size, seq_len, vector_dim = observation_vectors.shape
            logits = torch.zeros(batch_size, seq_len, len(ActionType))
            logits[..., action_to_index(ActionType.DONE)] = 1.0
            logits[..., action_to_index(ActionType.WAIT)] = (
                observation_vectors[..., 0] * 4.0
            )
            return {
                "action_logits": logits,
                "target_logits": torch.zeros(batch_size, seq_len, len(RouteName)),
                "command_slot_logits": torch.zeros(batch_size, seq_len, MAX_CANDIDATE_SLOTS),
                "file_slot_logits": torch.zeros(batch_size, seq_len, MAX_CANDIDATE_SLOTS),
                "risk": torch.zeros(batch_size, seq_len, 1),
                "salience": torch.zeros(batch_size, seq_len, 1),
                "prediction_error": torch.zeros(batch_size, seq_len, 1),
                "next_state": torch.zeros(batch_size, seq_len, vector_dim),
                "text_logits": torch.zeros(batch_size, seq_len, self.config.vocab_size),
                "hidden": hidden,
            }

    report = evaluate_reflexcore_sensory_ablation(
        VectorSignActionModel(),
        [wait_example, done_example],
        modes=["zero_numeric", "zero_hash"],
        min_action_accuracy_drop=0.25,
        min_next_state_relative_improvement_drop=0.0,
    )

    assert report["passed"] is False
    assert report["modes"]["zero_numeric"]["passed"] is True
    assert report["modes"]["zero_numeric"]["action_accuracy_drop"] == pytest.approx(0.5)
    assert report["modes"]["zero_numeric"][
        "next_state_relative_improvement_drop_passed"
    ] is True
    assert report["modes"]["zero_hash"]["passed"] is False
    assert report["modes"]["zero_hash"]["action_accuracy_drop"] == pytest.approx(0.0)


def test_reflexcore_world_model_evaluation_uses_next_state_loss_mask() -> None:
    class CopyCurrentWithMaskedNoise(torch.nn.Module):
        def forward(
            self,
            observation_vectors: torch.Tensor,
            text_tokens: torch.Tensor,
            *,
            action_indices: torch.Tensor | None = None,
            hidden: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor | None]:
            batch_size, seq_len, input_dim = observation_vectors.shape
            action_logits = torch.full(
                (batch_size, seq_len, len(ActionType)),
                -10.0,
                device=observation_vectors.device,
            )
            action_logits[..., action_to_index(ActionType.WAIT)] = 10.0
            next_state = observation_vectors.clone()
            next_state[..., -1] = 999.0
            return {
                "text_logits": torch.zeros(
                    batch_size,
                    seq_len,
                    512,
                    device=observation_vectors.device,
                ),
                "action_logits": action_logits,
                "command_slot_logits": torch.zeros(
                    batch_size,
                    seq_len,
                    MAX_CANDIDATE_SLOTS,
                    device=observation_vectors.device,
                ),
                "file_slot_logits": torch.zeros(
                    batch_size,
                    seq_len,
                    MAX_CANDIDATE_SLOTS,
                    device=observation_vectors.device,
                ),
                "risk": torch.zeros(batch_size, seq_len, 1, device=observation_vectors.device),
                "salience": torch.zeros(
                    batch_size,
                    seq_len,
                    1,
                    device=observation_vectors.device,
                ),
                "prediction_error": torch.zeros(
                    batch_size,
                    seq_len,
                    1,
                    device=observation_vectors.device,
                ),
                "next_state": next_state,
                "hidden": hidden,
            }

    example = build_reflexcore_examples(_records()[:1], vocab_size=512)[0]
    polluted_next_vector = list(example.observation.vector)
    polluted_next_vector[-1] = -999.0
    masked_example = example.model_copy(
        update={
            "next_observation": example.next_observation.model_copy(
                update={"vector": polluted_next_vector}
            )
        }
    )

    summary = evaluate_reflexcore_model(
        CopyCurrentWithMaskedNoise(),
        [masked_example],
        batch_size=1,
    )
    mask = tensors_for_example(
        masked_example,
        max_text_tokens=64,
        input_dim=len(masked_example.observation.vector),
    )["next_state_loss_mask"]

    assert mask[-1].item() == 0.0
    assert summary["next_state_mse"] == pytest.approx(0.0)
    assert summary["copy_current_next_state_mse"] == pytest.approx(0.0)
    assert summary["next_state_evaluated_values"] == int(mask.sum().item())


def test_reflexcore_evaluation_teacher_forces_dynamics_action() -> None:
    class RecordingReflexCore(ReflexCoreV0):
        def __init__(self, config: ReflexCoreV0Config) -> None:
            super().__init__(config)
            self.seen_action_indices: list[torch.Tensor | None] = []

        def forward(self, *args: object, **kwargs: object) -> dict[str, torch.Tensor | None]:
            action_indices = kwargs.get("action_indices")
            self.seen_action_indices.append(
                action_indices.detach().cpu().clone()
                if isinstance(action_indices, torch.Tensor)
                else None
            )
            return super().forward(*args, **kwargs)

    examples = build_reflexcore_examples(_records(), vocab_size=512)
    dataset = ReflexCoreTorchDataset(examples)
    model = RecordingReflexCore(
        ReflexCoreV0Config.smoke(input_dim=dataset.input_dim, vocab_size=512)
    )
    evaluate_reflexcore_model(model, examples, batch_size=2)
    assert model.seen_action_indices
    assert any(action_indices is not None for action_indices in model.seen_action_indices)
    assert any(action_indices is None for action_indices in model.seen_action_indices)


def test_reflexcore_sandbox_blocks_dangerous_and_handles_file_refresh(tmp_path: Path) -> None:
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(
            sandbox_root=tmp_path,
            allowed_commands=("rm -rf sandbox", "echo ok"),
            allow_process_execution=False,
        )
    )
    goal = GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description="sandbox test",
        command_allowlist=["rm -rf sandbox", "echo ok"],
        watched_paths=[str(tmp_path)],
    )
    state = runner.initial_state(goal)
    dangerous = runner.step(
        state,
        ActionDecision(type=ActionType.RUN_COMMAND, command="rm -rf sandbox"),
    )
    assert not dangerous.safety_decision.allowed
    assert dangerous.safety_decision.reason == "dangerous_command_detected"

    note = tmp_path / "note.txt"
    note.write_text("hello", encoding="utf-8")
    refreshed = runner.step(state, ActionDecision(type=ActionType.REFRESH_STATE))
    assert "note.txt" in refreshed.state.filesystem.changed_paths
    read_state = refreshed.state.model_copy(
        update={
            "filesystem": refreshed.state.filesystem.model_copy(
                update={"dirty_files": ["note.txt"], "changed_paths": ["note.txt"]}
            )
        }
    )
    read = runner.step(
        read_state,
        ActionDecision(type=ActionType.READ_FILE, file_target="note.txt"),
    )
    assert read.stdout == "hello"


def test_reflexcore_live_reobserve_preserves_unread_dirty_file_memory(
    tmp_path: Path,
) -> None:
    goal = GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description="read all changed files before running commands",
        watched_paths=[str(tmp_path)],
    )
    runner = ReflexCoreSandboxRunner(ReflexCoreSandboxConfig(sandbox_root=tmp_path))
    context = ReflexCoreObservationContext(goal=goal)
    context.observe_state()
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    state = runner.initial_state(goal)
    state = state.model_copy(
        update={
            "filesystem": state.filesystem.model_copy(
                update={
                    "watched_paths": [str(tmp_path)],
                    "changed_paths": ["first.txt", "second.txt"],
                    "dirty_files": ["first.txt", "second.txt"],
                }
            )
        }
    )

    result = runner.step(
        state,
        ActionDecision(type=ActionType.READ_FILE, file_target="first.txt"),
    )
    reobserved = runner.reobserve_step_result(context, result)

    assert "first.txt" not in reobserved.state.filesystem.dirty_files
    assert "second.txt" in reobserved.state.filesystem.dirty_files
    assert "second.txt" in reobserved.state.filesystem.changed_paths
    assert str(second) not in reobserved.state.filesystem.dirty_files
    assert str(second) not in reobserved.state.filesystem.changed_paths


def test_reflexcore_sandbox_blocks_non_allowlisted_run_command(tmp_path: Path) -> None:
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(sandbox_root=tmp_path, allowed_commands=("echo ok",))
    )
    goal = GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description="allowlist test",
        command_allowlist=["echo ok"],
        watched_paths=[str(tmp_path)],
    )
    state = runner.initial_state(goal)
    result = runner.step(
        state,
        ActionDecision(type=ActionType.RUN_COMMAND, command="echo nope"),
    )
    assert not result.safety_decision.allowed
    assert result.safety_decision.reason == "command_not_allowlisted"


def test_reflexcore_sandbox_executes_allowlisted_command_without_shell(tmp_path: Path) -> None:
    command = f'"{sys.executable}" -c "print(\'ok\')"'
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(
            sandbox_root=tmp_path,
            allowed_commands=(command,),
            allow_process_execution=True,
        )
    )
    goal = GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description="shell-free execution test",
        command_allowlist=[command],
        watched_paths=[str(tmp_path)],
    )
    state = runner.initial_state(goal)
    result = runner.step(state, ActionDecision(type=ActionType.RUN_COMMAND, command=command))
    assert result.safety_decision.allowed
    assert result.stdout.strip() == "ok"
    assert result.state.process.exit_code == 0


def test_reflexcore_sandbox_waits_for_real_background_process(tmp_path: Path) -> None:
    command = f'"{sys.executable}" -c "import time; time.sleep(0.1); print(\'slow-ok\')"'
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(
            sandbox_root=tmp_path,
            allowed_commands=(command,),
            allow_process_execution=True,
            command_observe_timeout_s=0.01,
            wait_observe_timeout_s=1.0,
        )
    )
    goal = GoalSpec(
        task_type=TaskType.PROCESS_HANG,
        description="real background process wait",
        command_allowlist=[command],
    )
    state = runner.initial_state(goal)
    launched = runner.step(state, ActionDecision(type=ActionType.RUN_COMMAND, command=command))
    assert launched.safety_decision.allowed
    assert launched.state.process.status.value == "running"

    waited = runner.step(
        launched.state,
        ActionDecision(type=ActionType.WAIT, reason="poll_background_process"),
    )
    assert waited.stdout.strip() == "slow-ok"
    assert waited.state.process.status.value == "exited"
    assert waited.state.process.exit_code == 0


def test_reflexcore_sandbox_stops_real_background_process(tmp_path: Path) -> None:
    command = f'"{sys.executable}" -c "import time; time.sleep(30)"'
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(
            sandbox_root=tmp_path,
            allowed_commands=(command,),
            allow_process_execution=True,
            command_observe_timeout_s=0.01,
            wait_observe_timeout_s=0.01,
            resource_alert_on_timeout=True,
        )
    )
    goal = GoalSpec(
        task_type=TaskType.PROCESS_HANG,
        description="real background process stop",
        command_allowlist=[command],
    )
    state = runner.initial_state(goal)
    launched = runner.step(state, ActionDecision(type=ActionType.RUN_COMMAND, command=command))
    assert launched.state.process.status.value == "running"
    assert launched.state.process.resource_alert is True

    stopped = runner.step(
        launched.state,
        ActionDecision(type=ActionType.STOP_PROCESS, reason="stop_background_process"),
    )
    assert stopped.state.process.status.value == "exited"
    assert stopped.state.process.interrupted is True
    assert "stopped active sandbox process" in stopped.stderr


def test_reflexcore_sandbox_recurrent_model_loop_runs(tmp_path: Path) -> None:
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(
            sandbox_root=tmp_path,
            allowed_commands=("echo ok",),
            max_steps=3,
        )
    )
    goal = GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description="loop test",
        command_allowlist=["echo ok"],
        watched_paths=[str(tmp_path)],
    )
    state = runner.initial_state(goal)
    input_dim = len(build_reflexcore_examples(_records()[:1], vocab_size=512)[0].observation.vector)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512))
    trace = runner.run_model_loop(model, state)
    assert 1 <= len(trace) <= 3
    assert trace[-1].state.time.tick >= 1


def test_reflexcore_model_rollout_exports_training_examples(tmp_path: Path) -> None:
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(
            sandbox_root=tmp_path,
            allowed_commands=("echo ok",),
            max_steps=3,
        )
    )
    goal = GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description="online experience export",
        command_allowlist=["echo ok"],
        watched_paths=[str(tmp_path)],
    )
    state = runner.initial_state(goal)
    input_dim = len(build_reflexcore_examples(_records()[:1], vocab_size=512)[0].observation.vector)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512))
    trace = runner.run_model_loop(model, state)
    examples = examples_from_step_trace(
        initial_state=state,
        trace=trace,
        episode_id="model-rollout-0",
        vocab_size=512,
    )
    assert len(examples) == len(trace)
    assert {example.source for example in examples} == {SourceType.MODEL.value}
    assert all(example.episode_id == "model-rollout-0" for example in examples)
    assert all("oracle_action" not in item.observation.model_dump(mode="json") for item in examples)
    assert slot_bounds_ok(examples)


def test_reflexcore_experience_records_post_safety_block(tmp_path: Path) -> None:
    command = "rm -rf sandbox"
    runner = ReflexCoreSandboxRunner(
        ReflexCoreSandboxConfig(
            sandbox_root=tmp_path,
            allowed_commands=(command,),
            allow_process_execution=True,
        )
    )
    goal = GoalSpec(
        task_type=TaskType.DANGEROUS_ACTION,
        description="post safety experience",
        command_allowlist=[command],
        watched_paths=[str(tmp_path)],
    )
    state = runner.initial_state(goal)
    result = runner.step(
        state,
        ActionDecision(type=ActionType.RUN_COMMAND, command=command),
    )
    examples = examples_from_step_trace(
        initial_state=state,
        trace=[result],
        episode_id="blocked-model-rollout",
        vocab_size=512,
    )
    assert examples[0].action.type == ActionType.BLOCK
    assert examples[0].action.reason == "dangerous_command_detected"
    assert examples[0].action.command is None


def test_reflexcore_write_experience_jsonl_roundtrips(tmp_path: Path) -> None:
    runner = ReflexCoreSandboxRunner(ReflexCoreSandboxConfig(sandbox_root=tmp_path))
    goal = GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description="write experience",
        watched_paths=[str(tmp_path)],
    )
    state = runner.initial_state(goal)
    result = runner.step(state, ActionDecision(type=ActionType.WAIT, reason="test_wait"))
    output = tmp_path / "experience.jsonl"
    summary = write_experience_jsonl(
        output,
        initial_state=state,
        trace=[result],
        episode_id="experience-jsonl",
        vocab_size=512,
    )
    examples = read_reflexcore_jsonl(output)
    assert summary.example_count == 1
    assert summary.source == SourceType.MODEL.value
    assert len(examples) == 1
    assert examples[0].source == SourceType.MODEL.value


def test_reflexcore_online_adaptation_updates_checkpoint_from_experience(
    tmp_path: Path,
) -> None:
    runner = ReflexCoreSandboxRunner(ReflexCoreSandboxConfig(sandbox_root=tmp_path / "sandbox"))
    goal = GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description="online adaptation",
        watched_paths=[str(tmp_path / "sandbox")],
    )
    state = runner.initial_state(goal)
    result = runner.step(state, ActionDecision(type=ActionType.WAIT, reason="adapt_wait"))
    refresh_result = runner.step(
        result.state,
        ActionDecision(type=ActionType.REFRESH_STATE, reason="adapt_refresh"),
    )
    examples = examples_from_step_trace(
        initial_state=state,
        trace=[result, refresh_result],
        episode_id="online-adapt",
        vocab_size=512,
    )
    experience_path = tmp_path / "experience.jsonl"
    retention_path = tmp_path / "retention.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    write_reflexcore_jsonl(experience_path, [examples[0]])
    write_reflexcore_jsonl(retention_path, [examples[0]])
    write_reflexcore_jsonl(holdout_path, [examples[1]])
    input_dim = len(examples[0].observation.vector)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512))
    checkpoint_path = tmp_path / "base.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
        },
        checkpoint_path,
    )
    report = adapt_reflexcore_from_experience(
        ReflexCoreOnlineAdaptationConfig(
            checkpoint_path=checkpoint_path,
            experience_path=experience_path,
            output_dir=tmp_path / "adapted",
            retention_path=retention_path,
            holdout_path=holdout_path,
            max_retention_loss_increase=10.0,
            max_holdout_loss_increase=10.0,
            epochs=2,
            batch_size=1,
            learning_rate=1e-3,
            sequence_mode=True,
            max_sequence_len=8,
            max_text_tokens=128,
        )
    )
    adapted_path = Path(str(report["adapted_checkpoint"]))
    adapted = torch.load(adapted_path, map_location="cpu")
    delta = 0.0
    for name, tensor in model.state_dict().items():
        delta += float((adapted["model_state_dict"][name] - tensor).abs().sum().item())
    assert adapted_path.exists()
    assert report["experience_examples"] == 1
    assert report["source_values"] == [SourceType.MODEL.value]
    assert report["retention_examples"] == 1
    assert report["retention_gate"]["available"] is True
    assert report["retention_gate"]["passed"] is True
    assert report["holdout_examples"] == 1
    assert report["holdout_gate"]["available"] is True
    assert report["holdout_gate"]["passed"] is True
    assert report["accepted"] is True
    assert report["rejected_reason"] is None
    assert report["trainable_scope"] == "all"
    assert report["trainable_parameter_count"] > 0
    assert report["frozen_parameter_count"] == 0
    assert adapted["accepted"] is True
    assert adapted["rejected_reason"] is None
    assert adapted["trainable_scope"] == "all"
    assert torch.isfinite(torch.tensor(float(report["before_loss"])))
    assert torch.isfinite(torch.tensor(float(report["after_loss"])))
    assert delta > 0.0


def test_reflexcore_online_adaptation_learns_live_prediction_error_signal(
    tmp_path: Path,
) -> None:
    example = build_reflexcore_examples(_records()[:1], vocab_size=512)[0]
    observed_error = 0.82
    live_example = example.model_copy(
        update={
            "episode_id": "live-pe-adapt",
            "next_observation": example.observation.model_copy(
                update={
                    "runtime_evidence": example.observation.runtime_evidence.model_copy(
                        update={"observed_prediction_error": observed_error}
                    )
                }
            ),
        }
    )
    experience_path = tmp_path / "live_pe.jsonl"
    write_reflexcore_jsonl(experience_path, [live_example])
    input_dim = len(live_example.observation.vector)
    torch.manual_seed(37)
    config = ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512)
    config.prediction_error_mode = "direct"
    config.prediction_error_calibration_scale = 1.0
    model = ReflexCoreV0(config)
    with torch.no_grad():
        model.action_head.weight.zero_()
        model.action_head.bias.fill_(-10.0)
        model.action_head.bias[action_to_index(ActionType.WAIT)] = 10.0
        model.salience_head.weight.zero_()
        model.salience_head.bias.fill_(6.0)
        model.prediction_error_head.weight.zero_()
        model.prediction_error_head.bias.fill_(-6.0)
    checkpoint_path = tmp_path / "base.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
        },
        checkpoint_path,
    )

    report = adapt_reflexcore_from_experience(
        ReflexCoreOnlineAdaptationConfig(
            checkpoint_path=checkpoint_path,
            experience_path=experience_path,
            output_dir=tmp_path / "adapted",
            epochs=25,
            batch_size=1,
            learning_rate=5e-2,
            sequence_mode=True,
            max_sequence_len=8,
            max_text_tokens=128,
            trainable_scope="world_model_only",
        )
    )

    assert report["live_prediction_error_examples"] == 1
    assert report["live_prediction_error_target_mean"] == pytest.approx(observed_error)
    assert report["before_metrics"]["prediction_error_loss"] > report["after_metrics"][
        "prediction_error_loss"
    ]
    assert report["prediction_error_loss_delta"] > 0.0
    probe = report["prediction_error_motor_probe"]
    assert probe["available"] is True
    assert probe["examples"] == 1
    assert probe["base_refresh_count"] == 0
    assert probe["adapted_refresh_count"] == 1
    assert probe["base_safe_refresh_count"] == 0
    assert probe["adapted_safe_refresh_count"] == 1
    assert probe["adapted_safety_allowed_count"] == 1
    assert probe["changed_to_refresh_count"] == 1
    assert probe["changed_to_safe_refresh_count"] == 1
    assert probe["mean_prediction_error_delta"] > 0.0
    assert report["accepted"] is True
    adapted_payload = torch.load(Path(str(report["adapted_checkpoint"])), map_location="cpu")
    adapted_model = ReflexCoreV0(ReflexCoreV0Config(**adapted_payload["config"]))
    adapted_model.load_state_dict(adapted_payload["model_state_dict"])
    probe_state = live_example.observation.to_state_frame()
    probe_state = probe_state.model_copy(
        update={
            "runtime_evidence": probe_state.runtime_evidence.model_copy(
                update={
                    "model_prediction_error": None,
                    "observed_prediction_error": None,
                    "prediction_error_delta": None,
                }
            ),
            "filesystem": probe_state.filesystem.model_copy(
                update={
                    "changed_paths": ["note.txt"],
                    "external_change_detected": True,
                    "stale_cache_detected": True,
                }
            ),
        }
    )
    runner = ReflexCoreSandboxRunner(ReflexCoreSandboxConfig(sandbox_root=tmp_path / "motor"))
    proposal = runner.propose_with_state(adapted_model, probe_state)
    assert proposal.safety_decision.allowed is True
    assert proposal.safety_decision.action.type == ActionType.REFRESH_STATE
    stepped = runner.step(probe_state, proposal.safety_decision.action)
    assert stepped.safety_decision.allowed is True
    assert stepped.safety_decision.action.type == ActionType.REFRESH_STATE


def test_reflexcore_pe_motor_probe_rebuilds_observation_without_cached_pe_vector(
    tmp_path: Path,
) -> None:
    example = build_reflexcore_examples(_records()[:1], vocab_size=512)[0]
    observed_error = 0.82
    live_example = example.model_copy(
        update={
            "episode_id": "live-pe-probe-clean",
            "next_observation": example.observation.model_copy(
                update={
                    "runtime_evidence": example.observation.runtime_evidence.model_copy(
                        update={"observed_prediction_error": observed_error}
                    )
                }
            ),
        }
    )
    input_dim = len(live_example.observation.vector)
    config = ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512)
    config.prediction_error_mode = "direct"
    config.prediction_error_calibration_scale = 1.0
    before_model = ReflexCoreV0(config)
    adapted_model = ReflexCoreV0(config)
    with torch.no_grad():
        for model in (before_model, adapted_model):
            model.action_head.weight.zero_()
            model.action_head.bias.fill_(-10.0)
            model.action_head.bias[action_to_index(ActionType.WAIT)] = 10.0
            model.salience_head.weight.zero_()
            model.salience_head.bias.fill_(6.0)
            model.prediction_error_head.weight.zero_()
            model.prediction_error_head.bias.fill_(-6.0)
        adapted_model.prediction_error_head.bias.fill_(6.0)
    before_state_dict = {
        key: value.detach().clone() for key, value in before_model.state_dict().items()
    }
    polluted_vector = list(live_example.observation.vector)
    feedback_end = PREDICTION_FEEDBACK_START_INDEX + PREDICTION_FEEDBACK_FEATURES
    for index in range(PREDICTION_FEEDBACK_START_INDEX, feedback_end):
        polluted_vector[index] = 999.0
    polluted_example = live_example.model_copy(
        update={
            "observation": live_example.observation.model_copy(
                update={"vector": polluted_vector}
            )
        }
    )

    clean_probe = _prediction_error_motor_probe(
        before_model=ReflexCoreV0(config),
        before_state_dict=before_state_dict,
        after_model=adapted_model,
        examples=[live_example],
        device=torch.device("cpu"),
    )
    polluted_probe = _prediction_error_motor_probe(
        before_model=ReflexCoreV0(config),
        before_state_dict=before_state_dict,
        after_model=adapted_model,
        examples=[polluted_example],
        device=torch.device("cpu"),
    )

    assert polluted_probe["adapted_safe_refresh_count"] == 1
    assert polluted_probe["changed_to_safe_refresh_count"] == 1
    assert polluted_probe == clean_probe


def test_reflexcore_pe_motor_probe_empty_schema_is_stable() -> None:
    example = build_reflexcore_examples(_records()[:1], vocab_size=512)[0]
    config = ReflexCoreV0Config.smoke(
        input_dim=len(example.observation.vector),
        vocab_size=512,
    )
    before_model = ReflexCoreV0(config)
    before_state_dict = {
        key: value.detach().clone() for key, value in before_model.state_dict().items()
    }
    report = _prediction_error_motor_probe(
        before_model=before_model,
        before_state_dict=before_state_dict,
        after_model=ReflexCoreV0(config),
        examples=[example],
        device=torch.device("cpu"),
    )

    assert report == {
        "available": False,
        "examples": 0,
        "base_refresh_count": 0,
        "adapted_refresh_count": 0,
        "base_safe_refresh_count": 0,
        "adapted_safe_refresh_count": 0,
        "adapted_safety_allowed_count": 0,
        "refresh_gain": 0,
        "changed_to_refresh_count": 0,
        "changed_to_safe_refresh_count": 0,
        "mean_prediction_error_delta": None,
    }


def test_reflexcore_online_adaptation_retention_gate_blocks_forgetting() -> None:
    gate = _retention_gate(before=1.0, after=1.25, max_loss_increase=0.1)
    missing = _retention_gate(before=None, after=None, max_loss_increase=0.1)
    assert gate["available"] is True
    assert gate["passed"] is False
    assert gate["loss_increase"] == 0.25
    assert (
        _rejected_reason(
            loss_not_increased=True,
            retention_passed=False,
            holdout_passed=True,
        )
        == "retention_loss_increased"
    )
    assert (
        _rejected_reason(
            loss_not_increased=True,
            retention_passed=True,
            holdout_passed=False,
        )
        == "holdout_loss_increased"
    )
    assert missing["available"] is False
    assert missing["passed"] is None
    assert (
        _rejected_reason(
            loss_not_increased=True,
            retention_passed=None,
            holdout_passed=None,
        )
        is None
    )


def test_reflexcore_online_adaptation_gate_splits_family_holdout() -> None:
    base = build_reflexcore_examples(_records()[:1], vocab_size=512)
    template = base[0]
    examples = [
        template.model_copy(update={"episode_id": "real-sandbox-alpha-0", "t": 0}),
        template.model_copy(update={"episode_id": "real-sandbox-alpha-1", "t": 0}),
        template.model_copy(update={"episode_id": "real-sandbox-beta-0", "t": 0}),
        template.model_copy(update={"episode_id": "real-sandbox-beta-1", "t": 0}),
    ]
    split = split_online_adaptation_examples(
        examples,
        split_strategy="family_holdout",
        holdout_families=("real-sandbox-alpha",),
        train_episode_count=1,
        retention_episode_count=1,
    )
    assert set(split.holdout_episode_ids) == {
        "real-sandbox-alpha-0",
        "real-sandbox-alpha-1",
    }
    assert not set(split.train_episode_ids) & set(split.holdout_episode_ids)
    assert not set(split.retention_episode_ids) & set(split.holdout_episode_ids)


def test_reflexcore_online_adaptation_gate_runs_disjoint_episode_holdout(
    tmp_path: Path,
) -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)
    dataset_path = tmp_path / "dataset.jsonl"
    write_reflexcore_jsonl(dataset_path, examples)
    input_dim = len(examples[0].observation.vector)
    torch.manual_seed(19)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512))
    checkpoint_path = tmp_path / "base.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
        },
        checkpoint_path,
    )
    report = run_online_adaptation_gate(
        ReflexCoreOnlineAdaptationGateConfig(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            output_dir=tmp_path / "gate",
            split_strategy="episode_holdout",
            split_seed=3,
            train_episode_count=1,
            retention_episode_count=1,
            holdout_episode_count=1,
            max_retention_loss_increase=10.0,
            max_holdout_loss_increase=10.0,
            epochs=2,
            batch_size=1,
            learning_rate=1e-3,
            sequence_mode=True,
            max_sequence_len=8,
            max_text_tokens=128,
            trainable_scope="world_model_only",
        )
    )
    split = report["split"]
    assert report["dataset_examples"] == len(examples)
    assert report["free_shell_generation"] is False
    assert report["gui_or_vision"] is False
    assert split["disjoint_episodes"] is True
    assert split["train_examples"] > 0
    assert split["holdout_examples"] > 0
    assert report["adaptation"]["holdout_gate"]["available"] is True
    assert Path(str(report["adaptation"]["adapted_checkpoint"])).exists()


def test_reflexcore_family_holdout_matrix_runs_selected_families(tmp_path: Path) -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)
    dataset_path = tmp_path / "dataset.jsonl"
    write_reflexcore_jsonl(dataset_path, examples)
    input_dim = len(examples[0].observation.vector)
    torch.manual_seed(29)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512))
    checkpoint_path = tmp_path / "base.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
        },
        checkpoint_path,
    )
    report = run_family_holdout_matrix(
        ReflexCoreFamilyHoldoutMatrixConfig(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            output_dir=tmp_path / "matrix",
            holdout_families=(TaskType.TEST_FAILURE.value, TaskType.FILE_CHANGE.value),
            retention_episode_count=1,
            max_retention_loss_increase=10.0,
            max_holdout_loss_increase=10.0,
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            sequence_mode=True,
            max_sequence_len=8,
            max_text_tokens=128,
            trainable_scope="world_model_only",
        )
    )
    assert report["family_count"] == 2
    assert report["free_shell_generation"] is False
    assert report["gui_or_vision"] is False
    assert report["pass_rate"] == 1.0
    for result in report["results"]:
        assert result["passed"] is True
        assert result["family"] in {TaskType.TEST_FAILURE.value, TaskType.FILE_CHANGE.value}
        assert result["trainable_scope"] == "world_model_only"
        assert result["trainable_parameter_count"] > 0
        assert result["frozen_parameter_count"] > 0
        assert "prediction_error_motor_probe" in result
        assert all(
            episode_id.startswith(result["family"])
            for episode_id in result["holdout_episode_ids"]
        )
        assert not set(result["train_episode_ids"]) & set(result["holdout_episode_ids"])


def test_reflexcore_family_holdout_matrix_reports_behavior_regression(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "real_sandbox.jsonl"
    build_real_sandbox_oracle_dataset(
        output_path=dataset_path,
        work_dir=tmp_path / "work",
        variants=2,
        start_variant=0,
        vocab_size=512,
        max_text_tokens=128,
    )
    examples = read_reflexcore_jsonl(dataset_path)
    input_dim = len(examples[0].observation.vector)
    torch.manual_seed(31)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512))
    with torch.no_grad():
        model.action_head.weight.zero_()
        model.action_head.bias.fill_(-10.0)
        model.action_head.bias[action_to_index(ActionType.WAIT)] = 10.0
    checkpoint_path = tmp_path / "base.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
        },
        checkpoint_path,
    )
    report = run_family_holdout_matrix(
        ReflexCoreFamilyHoldoutMatrixConfig(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            output_dir=tmp_path / "behavior_matrix",
            holdout_families=("real-sandbox-wait_for_process",),
            retention_episode_count=1,
            max_retention_loss_increase=10.0,
            max_holdout_loss_increase=10.0,
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            sequence_mode=True,
            max_sequence_len=8,
            max_text_tokens=128,
            trainable_scope="world_model_only",
            behavior_eval_variants=1,
            behavior_eval_start_variant=0,
            behavior_eval_max_steps=2,
            require_behavior_capability=True,
            min_behavior_success_rate=1.0,
        )
    )
    result = report["results"][0]
    assert report["behavior_passed_count"] == 1
    assert report["behavior_capability_passed_count"] == 1
    assert "executes decoded model actions" in report["claim_boundary"]
    assert result["behavior_passed"] is True
    assert result["behavior_non_regression_passed"] is True
    assert result["behavior_capability_passed"] is True
    assert result["adapted_success_rate"] >= result["base_success_rate"]
    assert result["behavior"]["base"]["free_shell_generation"] is False
    assert result["behavior"]["adapted"]["gui_or_vision"] is False


def test_reflexcore_family_holdout_matrix_rejects_behavior_capability_gap(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "real_sandbox.jsonl"
    build_real_sandbox_oracle_dataset(
        output_path=dataset_path,
        work_dir=tmp_path / "work",
        variants=2,
        start_variant=0,
        vocab_size=512,
        max_text_tokens=128,
    )
    examples = read_reflexcore_jsonl(dataset_path)
    input_dim = len(examples[0].observation.vector)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    checkpoint_path = tmp_path / "base.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
        },
        checkpoint_path,
    )
    report = run_family_holdout_matrix(
        ReflexCoreFamilyHoldoutMatrixConfig(
            checkpoint_path=checkpoint_path,
            dataset_path=dataset_path,
            output_dir=tmp_path / "capability_gap_matrix",
            holdout_families=("real-sandbox-multi_command_select_stdout",),
            retention_episode_count=1,
            max_retention_loss_increase=10.0,
            max_holdout_loss_increase=10.0,
            epochs=1,
            batch_size=2,
            learning_rate=1e-4,
            sequence_mode=True,
            max_sequence_len=8,
            max_text_tokens=128,
            trainable_scope="world_model_only",
            behavior_eval_variants=1,
            behavior_eval_start_variant=0,
            behavior_eval_max_steps=2,
            require_behavior_capability=True,
            min_behavior_success_rate=1.0,
        )
    )
    result = report["results"][0]
    assert report["passed"] is False
    assert report["behavior_passed_count"] == 0
    assert report["behavior_capability_passed_count"] == 0
    assert report["failed_families"] == ["real-sandbox-multi_command_select_stdout"]
    assert result["behavior_non_regression_passed"] is True
    assert result["behavior_capability_passed"] is False
    assert result["behavior_passed"] is False
    assert result["rejected_reason"] == "behavior_capability_below_minimum"
    assert result["adapted_success_rate"] < 1.0


def test_reflexcore_closed_loop_evaluator_reports_task_success() -> None:
    baselines = evaluate_closed_loop_baselines(episodes_per_task=1)
    assert baselines["rule_oracle"]["success_rate"] == 1.0
    assert baselines["static_wait"]["episode_count"] == len(TaskType)
    assert baselines["static_wait"]["success_rate"] < 1.0
    input_dim = len(build_reflexcore_examples(_records()[:1], vocab_size=512)[0].observation.vector)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512))
    summary = evaluate_reflexcore_closed_loop(model, episodes_per_task=1)
    assert summary["episode_count"] == len(TaskType)
    assert summary["action_accuracy"] is not None


def test_reflexcore_experiment_runner_writes_unified_report(tmp_path: Path) -> None:
    report = run_reflexcore_experiment(
        ReflexCoreExperimentConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            eval_profile="hard",
            episodes_per_task=2,
            vocab_size=512,
            max_text_tokens=64,
            train_epochs=1,
            train_batch_size=2,
            sequence_mode=True,
            max_sequence_len=8,
            required_baseline="static_wait",
            closed_loop_episodes_per_task=1,
            require_world_model_improvement=False,
            require_prediction_error_improvement=False,
        )
    )
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "eval_benchmark" / "manifest.json").exists()
    assert "offline" in report
    assert "closed_loop" in report
    assert report["profile_transfer"]["is_transfer"] is True
    assert "world_model_acceptance" in report["offline"]
    assert "prediction_error_acceptance" in report["offline"]
    assert report["parameter_gate"]["parameter_count"] > 0


def test_reflexcore_stability_runner_writes_cross_seed_report(tmp_path: Path) -> None:
    report = run_reflexcore_stability(
        ReflexCoreStabilityConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            seeds=(13,),
            episodes_per_task=2,
            vocab_size=512,
            max_text_tokens=64,
            train_epochs=1,
            train_batch_size=2,
            sequence_mode=True,
            max_sequence_len=8,
            required_baseline="static_wait",
            closed_loop_episodes_per_task=1,
            min_pass_rate=1.0,
        )
    )
    assert (tmp_path / "stability_report.json").exists()
    assert report["run_count"] == 1
    assert report["pass_rate"] == report["pass_count"] / report["run_count"]
    assert "passed" in report["runs"][0]
    assert report["aggregates"]["offline_action_accuracy"]["mean"] is not None


def test_reflexcore_profile_matrix_runner_writes_multi_profile_report(
    tmp_path: Path,
) -> None:
    report = run_reflexcore_profile_matrix(
        ReflexCoreProfileMatrixConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            seeds=(13,),
            eval_profiles=("default", "hard"),
            episodes_per_task=2,
            vocab_size=512,
            max_text_tokens=64,
            train_epochs=1,
            train_batch_size=2,
            sequence_mode=True,
            max_sequence_len=8,
            required_baseline="static_wait",
            closed_loop_episodes_per_task=1,
            min_pass_rate=1.0,
            min_profile_pass_rate=1.0,
            require_world_model_improvement=False,
            require_prediction_error_improvement=False,
        )
    )
    assert (tmp_path / "profile_matrix_report.json").exists()
    assert (tmp_path / "eval_default" / "stability_report.json").exists()
    assert (tmp_path / "eval_hard" / "stability_report.json").exists()
    assert report["profile_count"] == 2
    assert report["profile_pass_rate"] == 1.0
    assert report["training_reuse"]["enabled"] is True
    assert report["training_reuse"]["train_run_count"] == 1
    assert report["profiles"][0]["is_transfer"] is False
    assert report["profiles"][1]["is_transfer"] is True
    assert report["aggregates"]["closed_loop_success_rate_min"]["mean"] is not None


def test_reflexcore_local_feasibility_runner_writes_train_gate_report(
    tmp_path: Path,
) -> None:
    report = run_reflexcore_local_feasibility(
        ReflexCoreLocalFeasibilityConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            episodes_per_task=1,
            vocab_size=512,
            max_text_tokens=64,
            train_epochs=1,
            train_batch_size=2,
            sequence_mode=True,
            max_sequence_len=8,
            min_parameters=1,
            max_parameters=1_000_000,
        )
    )
    assert (tmp_path / "local_feasibility_report.json").exists()
    assert report["passed"] is True
    assert report["parameter_gate"]["passed"] is True
    assert report["finite_loss_gate"]["passed"] is True
    assert report["checkpoint_gate"]["passed"] is True
    assert report["train"]["parameter_count"] > 0


def test_reflexcore_real_sandbox_evaluator_writes_bounded_report(
    tmp_path: Path,
) -> None:
    input_dim = len(build_reflexcore_examples(_records()[:1], vocab_size=512)[0].observation.vector)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512))
    report = evaluate_reflexcore_real_sandbox(
        model,
        config=RealSandboxEvalConfig(
            output_dir=tmp_path,
            compare_baselines=False,
            require_beats_baseline=None,
        ),
    )
    assert (tmp_path / "real_sandbox_report.json").exists()
    assert report["scope"] == "real_temp_sandbox_terminal_process_filesystem_time_only"
    assert report["free_shell_generation"] is False
    assert report["gui_or_vision"] is False
    assert report["model"]["task_count"] == 15
    assert {task["name"] for task in report["tasks"]} == {
        "refresh_then_read_file",
        "allowlisted_command_stdout",
        "multi_step_file_command_stdout",
        "multi_step_distractor_stdout",
        "multi_file_refresh_then_command",
        "command_creates_file_then_read",
        "slow_process_creates_file_then_read",
        "multi_command_select_stdout",
        "real_process_wait_stdout",
        "real_process_stop",
        "read_stdout_buffer",
        "stderr_read",
        "wait_for_process",
        "stop_hung_process",
        "dangerous_command_block",
    }


def test_reflexcore_real_sandbox_live_observation_reports_runtime_evidence(
    tmp_path: Path,
) -> None:
    input_dim = len(build_reflexcore_examples(_records()[:1], vocab_size=512)[0].observation.vector)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512))
    report = evaluate_reflexcore_real_sandbox(
        model,
        config=RealSandboxEvalConfig(
            output_dir=tmp_path,
            compare_baselines=False,
            require_beats_baseline=None,
            live_observation=True,
        ),
    )

    assert report["live_observation"] is True
    assert report["model"]["live_observation_episode_count"] > 0
    assert report["model"]["runtime_observation_steps"] > 0
    assert report["model"]["changed_file_observation_steps"] > 0
    assert report["model"]["observed_prediction_error_examples"] > 0
    assert report["model"]["observed_prediction_error_mean"] is not None
    assert report["model"]["model_prediction_error_examples"] > 0
    assert report["claim_boundary"].startswith("This gate uses real temporary filesystem")


def test_reflexcore_real_sandbox_min_success_gate() -> None:
    report: dict[str, object] = {"overall": {"success_rate": 0.75}}
    _apply_min_success_gate(report, 0.5)
    assert report["passed"] is True
    assert report["min_success_acceptance"]["passed"] is True

    failing_report: dict[str, object] = {"model": {"success_rate": 0.25}}
    _apply_min_success_gate(failing_report, 0.5)
    assert failing_report["passed"] is False
    assert failing_report["min_success_acceptance"]["model_success_rate"] == 0.25


def test_reflexcore_real_sandbox_oracle_dataset_is_bounded(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "real_sandbox.jsonl"
    summary = build_real_sandbox_oracle_dataset(
        output_path=dataset_path,
        work_dir=tmp_path / "work",
        variants=2,
        vocab_size=512,
        max_text_tokens=64,
    )
    examples = read_reflexcore_jsonl(dataset_path)
    assert dataset_path.exists()
    assert summary["example_count"] == len(examples)
    assert summary["scope"] == "real_temp_sandbox_terminal_process_filesystem_time_only"
    assert summary["free_shell_generation"] is False
    assert summary["gui_or_vision"] is False
    assert examples
    assert all("oracle_action" not in item.observation.model_dump(mode="json") for item in examples)
    assert {example.action.type for example in examples} >= {
        ActionType.READ_STDOUT,
        ActionType.WAIT,
        ActionType.STOP_PROCESS,
    }
    command_slot_examples = [
        example
        for example in examples
        if example.episode_id.startswith("real-sandbox-multi_command_select_stdout")
        and example.action.type == ActionType.RUN_COMMAND
    ]
    assert command_slot_examples
    assert all(
        example.action.command == example.observation.candidate_commands[1]
        for example in command_slot_examples
    )
    expanded_task_examples = [
        example
        for example in examples
        if example.episode_id.startswith(
            (
                "real-sandbox-multi_file_refresh_then_command",
                "real-sandbox-command_creates_file_then_read",
                "real-sandbox-slow_process_creates_file_then_read",
            )
        )
    ]
    assert expanded_task_examples
    assert {example.action.type for example in expanded_task_examples} >= {
        ActionType.READ_FILE,
        ActionType.RUN_COMMAND,
    }


def test_reflexcore_real_sandbox_oracle_dataset_hash_is_workdir_stable(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    build_real_sandbox_oracle_dataset(
        output_path=first_path,
        work_dir=tmp_path / "work_a",
        variants=2,
        vocab_size=512,
        max_text_tokens=64,
    )
    build_real_sandbox_oracle_dataset(
        output_path=second_path,
        work_dir=tmp_path / "work_b",
        variants=2,
        vocab_size=512,
        max_text_tokens=64,
    )
    first = read_reflexcore_jsonl(first_path)
    second = read_reflexcore_jsonl(second_path)
    assert dataset_hash(first) == dataset_hash(second)
    serialized = "\n".join(example.model_dump_json() for example in first + second)
    assert str(tmp_path) not in serialized
    assert sys.executable not in serialized
    assert "$SANDBOX_ROOT" in serialized
    assert "$PYTHON" in serialized


def test_reflexcore_real_sandbox_adaptation_writes_mixed_gate_report(
    tmp_path: Path,
) -> None:
    report = run_reflexcore_real_sandbox_adaptation(
        ReflexCoreRealSandboxAdaptationConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            episodes_per_task=2,
            split_strategy="episode_random",
            vocab_size=512,
            max_text_tokens=64,
            train_epochs=1,
            train_batch_size=2,
            required_baseline="static_wait",
            closed_loop_episodes_per_task=1,
            require_world_model_improvement=False,
            require_prediction_error_improvement=False,
            real_sandbox_variants=1,
            real_sandbox_start_variant=1,
            real_sandbox_required_baseline=None,
            require_synthetic_gate=False,
            synthetic_repeat=2,
            real_sandbox_repeat=3,
        )
    )
    mixture = report["dataset_mixture"]
    assert (tmp_path / "real_sandbox_adaptation_report.json").exists()
    assert (tmp_path / "mixed_train.jsonl").exists()
    assert mixture["synthetic_train"]["example_count"] > 0
    assert mixture["real_sandbox_train"]["example_count"] > 0
    assert mixture["mixed_train"]["example_count"] == (
        mixture["synthetic_train"]["weighted_example_count"]
        + mixture["real_sandbox_train"]["weighted_example_count"]
    )
    assert mixture["mixed_train"]["synthetic_fraction"] == pytest.approx(
        mixture["synthetic_train"]["weighted_example_count"]
        / mixture["mixed_train"]["example_count"]
    )
    assert mixture["mixed_train"]["real_sandbox_fraction"] == pytest.approx(
        mixture["real_sandbox_train"]["weighted_example_count"]
        / mixture["mixed_train"]["example_count"]
    )
    assert (
        mixture["mixed_train"]["synthetic_fraction"]
        + mixture["mixed_train"]["real_sandbox_fraction"]
        == pytest.approx(1.0)
    )
    assert report["real_sandbox"]["scope"] == "real_temp_sandbox_terminal_process_filesystem_time_only"
    assert report["claim_boundary"].startswith("This adaptation report supports only bounded")


def test_reflexcore_real_sandbox_adaptation_matrix_writes_cross_seed_report(
    tmp_path: Path,
) -> None:
    report = run_reflexcore_real_sandbox_adaptation_matrix(
        ReflexCoreRealSandboxAdaptationMatrixConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            seeds=(13,),
            episodes_per_task=2,
            split_strategy="episode_random",
            vocab_size=512,
            max_text_tokens=64,
            train_epochs=1,
            train_batch_size=2,
            required_baseline="static_wait",
            closed_loop_episodes_per_task=1,
            require_world_model_improvement=False,
            require_prediction_error_improvement=False,
            real_sandbox_variants=1,
            real_sandbox_start_variant=1,
            real_sandbox_max_steps=6,
            real_sandbox_live_observation=True,
            require_synthetic_gate=False,
            min_pass_rate=0.0,
            min_offline_margin=-1.0,
            min_closed_loop_margin=-1.0,
            min_real_sandbox_margin=-1.0,
        )
    )
    assert (tmp_path / "real_sandbox_adaptation_matrix_report.json").exists()
    assert report["run_count"] == 1
    assert report["runs"][0]["seed"] == 13
    assert report["runs"][0]["mixed_train_examples"] > 0
    assert report["runs"][0]["real_sandbox_live_observation"] is True
    assert report["runs"][0]["real_sandbox_live_episode_count"] > 0
    assert report["runs"][0]["real_sandbox_runtime_observation_steps"] > 0
    assert report["runs"][0]["real_sandbox_changed_file_observation_steps"] > 0
    assert report["runs"][0]["real_sandbox_observed_prediction_error_examples"] > 0
    assert report["aggregates"]["real_sandbox_runtime_observation_steps"]["max"] > 0
    assert "real_sandbox_margin" in report["aggregates"]
    assert report["margin_gate"]["passed"] is True
    assert report["claim_boundary"].startswith("Cross-seed real-sandbox adaptation")


def test_reflexcore_real_sandbox_profile_matrix_reuses_seed_training(
    tmp_path: Path,
) -> None:
    report = run_reflexcore_real_sandbox_adaptation_profile_matrix(
        ReflexCoreRealSandboxAdaptationProfileMatrixConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            seeds=(13,),
            eval_profiles=("default", "hard"),
            episodes_per_task=2,
            split_strategy="episode_random",
            vocab_size=512,
            max_text_tokens=64,
            train_epochs=1,
            train_batch_size=2,
            required_baseline="static_wait",
            closed_loop_episodes_per_task=1,
            require_world_model_improvement=False,
            require_prediction_error_improvement=False,
            real_sandbox_variants=1,
            real_sandbox_start_variant=1,
            real_sandbox_max_steps=6,
            real_sandbox_live_observation=True,
            require_synthetic_gate=False,
            min_pass_rate=0.0,
            min_profile_pass_rate=0.0,
            min_offline_margin=-1.0,
            min_closed_loop_margin=-1.0,
            min_real_sandbox_margin=-1.0,
        )
    )
    assert (tmp_path / "real_sandbox_adaptation_profile_matrix_report.json").exists()
    assert report["training_reuse"]["enabled"] is True
    assert report["training_reuse"]["real_sandbox_live_observation"] is True
    assert report["training_reuse"]["train_run_count"] == 1
    assert report["training_reuse"]["profile_eval_count"] == 2
    assert report["run_count"] == 1
    assert report["profile_eval_count"] == 2
    assert report["runs"][0]["profile_eval_count"] == 2
    train_hash = report["train_runs"][0]["model_hash"]
    profile_hashes = {profile["model_hash"] for profile in report["profile_runs"]}
    assert profile_hashes == {train_hash}
    assert report["profile_runs"][0]["real_sandbox_live_observation"] is True
    assert report["profile_runs"][0]["real_sandbox_live_episode_count"] > 0
    assert report["profile_runs"][0]["real_sandbox_runtime_observation_steps"] > 0
    assert report["profile_runs"][0]["real_sandbox_changed_file_observation_steps"] > 0
    assert (
        report["profile_runs"][0]["real_sandbox_observed_prediction_error_examples"] > 0
    )
    assert report["aggregates"]["real_sandbox_changed_file_observation_steps"]["max"] > 0
    assert report["profile_runs"][0]["is_transfer"] is False
    assert report["profile_runs"][1]["is_transfer"] is True
    assert report["margin_gate"]["passed"] is True
    assert report["success_gate"]["passed"] is True
    assert report["claim_boundary"].startswith("Train-once real-sandbox profile transfer")


def test_reflexcore_sensory_ablation_matrix_reads_profile_matrix(
    tmp_path: Path,
) -> None:
    run_reflexcore_real_sandbox_adaptation_profile_matrix(
        ReflexCoreRealSandboxAdaptationProfileMatrixConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            seeds=(13,),
            eval_profiles=("default",),
            episodes_per_task=2,
            split_strategy="episode_random",
            vocab_size=512,
            max_text_tokens=64,
            train_epochs=1,
            train_batch_size=2,
            required_baseline="static_wait",
            closed_loop_episodes_per_task=1,
            require_world_model_improvement=False,
            require_prediction_error_improvement=False,
            real_sandbox_variants=1,
            real_sandbox_start_variant=1,
            real_sandbox_required_baseline=None,
            require_synthetic_gate=False,
            min_pass_rate=0.0,
            min_profile_pass_rate=0.0,
        )
    )

    report = run_reflexcore_sensory_ablation_matrix(
        ReflexCoreSensoryAblationMatrixConfig(
            matrix_dir=tmp_path,
            output_json=tmp_path / "ablation_matrix.json",
            seeds=(13,),
            profiles=("default",),
            modes=("zero_all",),
            min_action_accuracy_drop=None,
            min_world_model_drop=None,
        )
    )

    assert (tmp_path / "ablation_matrix.json").exists()
    assert report["summary"]["row_count"] == 1
    assert report["summary"]["passed_rows"] == 1
    assert "zero_all" in report["summary"]["modes"]
    assert report["rows"][0]["seed"] == 13
    assert report["rows"][0]["profile"] == "default"
    assert report["claim_boundary"].startswith("This sensory-ablation matrix")


def test_reflexcore_sensory_ablation_matrix_reports_threshold_failure(
    tmp_path: Path,
) -> None:
    run_reflexcore_real_sandbox_adaptation_profile_matrix(
        ReflexCoreRealSandboxAdaptationProfileMatrixConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            seeds=(13,),
            eval_profiles=("default",),
            episodes_per_task=2,
            split_strategy="episode_random",
            vocab_size=512,
            max_text_tokens=64,
            train_epochs=1,
            train_batch_size=2,
            required_baseline="static_wait",
            closed_loop_episodes_per_task=1,
            require_world_model_improvement=False,
            require_prediction_error_improvement=False,
            real_sandbox_variants=1,
            real_sandbox_start_variant=1,
            real_sandbox_required_baseline=None,
            require_synthetic_gate=False,
            min_pass_rate=0.0,
            min_profile_pass_rate=0.0,
        )
    )

    report = run_reflexcore_sensory_ablation_matrix(
        ReflexCoreSensoryAblationMatrixConfig(
            matrix_dir=tmp_path,
            seeds=(13,),
            profiles=("default",),
            modes=("zero_hash",),
            min_action_accuracy_drop=2.0,
            min_world_model_drop=None,
        )
    )

    assert report["passed"] is False
    mode = report["rows"][0]["modes"]["zero_hash"]
    assert mode["action_accuracy_drop_passed"] is False
    assert report["summary"]["modes"]["zero_hash"]["passed"] is False


def test_reflexcore_profile_matrix_rejects_negative_prediction_error_gate(
    tmp_path: Path,
) -> None:
    gate = _improvement_gate(
        ReflexCoreRealSandboxAdaptationProfileMatrixConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            require_world_model_improvement=True,
            require_prediction_error_improvement=True,
            min_world_model_relative_improvement=0.0,
            min_prediction_error_relative_improvement=0.0,
        ),
        {
            "world_model_relative_improvement": {"min": 0.1, "mean": 0.1, "max": 0.1},
            "prediction_error_relative_improvement": {
                "min": -0.08,
                "mean": -0.04,
                "max": 0.0,
            },
        },
    )

    assert gate["passed"] is False
    details = gate["details"]["prediction_error_relative_improvement"]
    assert details["observed_min"] == pytest.approx(-0.08)
    assert details["passed"] is False


def test_reflexcore_profile_matrix_rejects_low_real_sandbox_success_gate(
    tmp_path: Path,
) -> None:
    gate = _success_gate(
        ReflexCoreRealSandboxAdaptationProfileMatrixConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            min_real_sandbox_success_rate=1.0,
        ),
        {
            "real_sandbox_success_rate": {
                "min": 0.9166666666666666,
                "mean": 0.9583333333333333,
                "max": 1.0,
            },
        },
    )

    assert gate["passed"] is False
    details = gate["details"]["real_sandbox_success_rate"]
    assert details["observed_min"] == pytest.approx(0.9166666666666666)
    assert details["required_min"] == pytest.approx(1.0)
    assert details["passed"] is False


def test_reflexcore_real_sandbox_capability_matrix_writes_cross_seed_report(
    tmp_path: Path,
) -> None:
    report = run_reflexcore_real_sandbox_capability_matrix(
        ReflexCoreRealSandboxCapabilityMatrixConfig(
            output_dir=tmp_path,
            model_config_path=Path("configs/reflexcore/smoke.yaml"),
            seeds=(13,),
            train_variants=2,
            train_start_variant=0,
            eval_variants=1,
            eval_start_variant=2,
            vocab_size=512,
            max_text_tokens=64,
            train_epochs=1,
            train_batch_size=4,
            learning_rate=1e-3,
            sequence_mode=True,
            max_sequence_len=8,
            min_success_rate=0.0,
            min_pass_rate=1.0,
        )
    )
    assert (tmp_path / "real_sandbox_capability_matrix_report.json").exists()
    assert report["run_count"] == 1
    assert report["pass_count"] == 1
    assert report["passed"] is True
    assert report["dataset_examples"] > 0
    assert report["runs"][0]["seed"] == 13
    assert report["runs"][0]["model_hash"]
    assert report["runs"][0]["task_count"] == 15
    assert report["free_shell_generation"] is False
    assert report["gui_or_vision"] is False
    assert "unrestricted shell" in report["claim_boundary"]


def test_reflexcore_real_sandbox_capability_matrix_rejects_variant_overlap(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        run_reflexcore_real_sandbox_capability_matrix(
            ReflexCoreRealSandboxCapabilityMatrixConfig(
                output_dir=tmp_path,
                model_config_path=Path("configs/reflexcore/smoke.yaml"),
                seeds=(13,),
                train_variants=2,
                train_start_variant=0,
                eval_variants=1,
                eval_start_variant=1,
            )
        )


def test_reflexcore_prediction_error_report_groups_by_action(tmp_path: Path) -> None:
    examples = build_reflexcore_examples(_records(), vocab_size=512)
    dataset_path = tmp_path / "examples.jsonl"

    write_reflexcore_jsonl(dataset_path, examples)
    input_dim = len(examples[0].observation.vector)
    model = ReflexCoreV0(ReflexCoreV0Config.smoke(input_dim=input_dim, vocab_size=512))
    report = build_reflexcore_prediction_error_report(
        model,
        ReflexCorePredictionErrorReportConfig(
            output_dir=tmp_path / "pe",
            dataset_path=dataset_path,
            sequence_mode=True,
            max_text_tokens=64,
            min_relative_improvement=-1.0,
            min_action_group_pass_rate=0.0,
        ),
    )
    assert (tmp_path / "pe" / "prediction_error_report.json").exists()
    assert report["row_count"] == len(examples)
    assert "overall" in report
    assert report["by_action"]
    assert report["action_group_count"] == len(report["by_action"])
    assert "relative_improvement" in report["overall"]
