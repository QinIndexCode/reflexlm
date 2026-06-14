import pytest
import torch

from reflexlm.eval import SequenceModelPolicy
from reflexlm.llm.hybrid import HybridPolicyConfig, HybridSynapticPolicy
from reflexlm.llm.native_head_policy import NativeHeadPolicy
from reflexlm.llm.prompts import SynapseSummary
from reflexlm.models.features import ACTION_ORDER, ROUTE_ORDER, StateVectorizer
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    FileSystemState,
    GoalSpec,
    InternalTarget,
    ProcessState,
    ProcessStatus,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


class IdentityNextStateModel(torch.nn.Module):
    def __init__(self, input_dim: int, *, model_error: float = 0.05) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.input_dim = input_dim
        self.model_error = model_error

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, seq_len, _dim = inputs.shape
        action_logits = torch.zeros(batch, seq_len, len(ACTION_ORDER), device=inputs.device)
        action_logits[..., ACTION_ORDER.index(ActionType.WAIT)] = 4.0
        return {
            "memory": torch.zeros(batch, seq_len, self.input_dim, device=inputs.device),
            "hidden": hidden,
            "action_logits": action_logits,
            "target_logits": torch.zeros(batch, seq_len, len(INTERNAL_TARGET_ORDER), device=inputs.device),
            "route_logits": torch.zeros(batch, seq_len, len(ROUTE_ORDER), device=inputs.device),
            "command_slot_logits": torch.zeros(batch, seq_len, 4, device=inputs.device),
            "file_slot_logits": torch.zeros(batch, seq_len, 4, device=inputs.device),
            "salience": torch.zeros(batch, seq_len, device=inputs.device),
            "risk": torch.zeros(batch, seq_len, device=inputs.device),
            "prediction_error": torch.full(
                (batch, seq_len),
                self.model_error,
                device=inputs.device,
            ),
            "next_state": inputs.clone(),
        }


class RecordingNextStateModel(IdentityNextStateModel):
    def __init__(self, input_dim: int) -> None:
        super().__init__(input_dim)
        self.recorded_action_indices: torch.Tensor | None = None

    def predict_next_state(
        self,
        memory: torch.Tensor,
        inputs: torch.Tensor,
        *,
        action_indices: torch.Tensor | None = None,
        action_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.recorded_action_indices = (
            action_indices.detach().cpu() if action_indices is not None else None
        )
        return inputs.clone()


def _state(*, changed: bool = False) -> SystemStateFrame:
    return SystemStateFrame(
        time=TimeState(
            tick=1 if changed else 0,
            runtime_ms=60000 if changed else 0,
            since_last_output_ms=60000 if changed else 0,
            since_last_state_change_ms=60000 if changed else 0,
        ),
        goal=GoalSpec(
            task_type=TaskType.PROCESS_HANG,
            description="keep a bounded process healthy",
            command_allowlist=["pytest -q", "python -m pip install -r requirements.txt"],
        ),
        process=ProcessState(
            status=ProcessStatus.EXITED if changed else ProcessStatus.RUNNING,
            exit_code=255 if changed else None,
            cpu_percent=100.0 if changed else 0.0,
            memory_mb=4096.0 if changed else 0.0,
            runtime_ms=60000 if changed else 0,
            resource_alert=changed,
        ),
        terminal=TerminalState(
            stdout_delta="stable" if not changed else "",
            stderr_delta="AssertionError: still failing" if changed else "",
            stdout_lines=0 if changed else 1,
            stderr_lines=50 if changed else 0,
            prompt_visible=changed,
            last_command="pytest -q" if changed else None,
        ),
        filesystem=FileSystemState(
            watched_paths=["tests/test_app.py"],
            changed_paths=["src/app.py"] if changed else [],
            dirty_files=["src/app.py"] if changed else [],
            external_change_detected=changed,
            stale_cache_detected=changed,
        ),
    )


def _policy(model_error: float = 0.05) -> SequenceModelPolicy:
    vectorizer = StateVectorizer(hash_bins=0)
    model = IdentityNextStateModel(vectorizer.vector_dim, model_error=model_error)
    return SequenceModelPolicy(model, vectorizer, policy_label="temporal_prediction_error_test")


def test_sequence_policy_uses_observed_temporal_next_state_error() -> None:
    policy = _policy(model_error=0.05)

    policy.act(_state(changed=False))
    first = policy.last_call
    assert first["temporal_observation_available"] is False
    assert first["prediction_error"] == pytest.approx(0.05)
    assert first["prediction_error_source"] == "model_head"

    policy.act(_state(changed=False))
    stable = policy.last_call
    assert stable["temporal_observation_available"] is True
    assert stable["observed_temporal_prediction_error"] == 0.0
    assert stable["prediction_error"] == pytest.approx(0.05)
    assert stable["prediction_error_source"] == "model_head"

    policy.act(_state(changed=True))
    changed = policy.last_call
    assert changed["temporal_observation_available"] is True
    assert changed["observed_temporal_prediction_error"] > 0.05
    assert changed["observed_full_state_novelty"] > 0.05
    assert changed["prediction_error"] == changed["observed_temporal_prediction_error"]
    assert changed["prediction_error_source"] == "observed_temporal_next_state"


def test_sequence_policy_reset_clears_temporal_prediction() -> None:
    policy = _policy()
    policy.act(_state(changed=False))
    assert policy.last_call["next_state_prediction_available"] is True

    policy.reset()
    policy.act(_state(changed=True))
    assert policy.last_call["temporal_observation_available"] is False
    assert policy.last_call["prediction_error_source"] == "model_head"


def test_temporal_prediction_error_can_block_direct_hybrid_reflex() -> None:
    policy = HybridSynapticPolicy.__new__(HybridSynapticPolicy)
    policy.hybrid_config = HybridPolicyConfig(
        base_model_name="unused",
        confidence_threshold=0.7,
        prediction_error_threshold=0.45,
    )
    action = ActionDecision(type=ActionType.WAIT, confidence=0.95)
    low_error = SynapseSummary(
        route_name="terminal_cortex",
        salience=0.1,
        risk=0.1,
        prediction_error=0.2,
        confidence=0.95,
        reflex_action=ActionType.WAIT.value,
    )
    high_error = SynapseSummary(
        route_name="terminal_cortex",
        salience=0.1,
        risk=0.1,
        prediction_error=0.8,
        confidence=0.95,
        reflex_action=ActionType.WAIT.value,
    )

    assert policy._use_direct_reflex(action, low_error) is True
    assert policy._use_direct_reflex(action, high_error) is False


def test_native_head_policy_escalates_reflex_state_on_observed_prediction_error() -> None:
    policy = NativeHeadPolicy.__new__(NativeHeadPolicy)
    policy.prediction_error_escalation_threshold = 0.45
    policy.nsi_policy = type(
        "Nsi",
        (),
        {
            "last_call": {
                "prediction_error": 0.8,
                "temporal_observation_available": True,
            }
        },
    )()

    target, source = policy._internal_target_with_prediction_error(_state(changed=False))

    assert target == InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX
    assert source == "observed_temporal_prediction_error"


def test_native_head_policy_does_not_escalate_first_frame_model_head_error() -> None:
    policy = NativeHeadPolicy.__new__(NativeHeadPolicy)
    policy.prediction_error_escalation_threshold = 0.45
    policy.nsi_policy = type(
        "Nsi",
        (),
        {
            "last_call": {
                "prediction_error": 0.8,
                "temporal_observation_available": False,
            }
        },
    )()

    target, source = policy._internal_target_with_prediction_error(_state(changed=False))

    assert target == InternalTarget.REFLEX_MOTOR
    assert source == "state_receptor"


def test_native_head_policy_propagates_low_level_homeostatic_wake() -> None:
    policy = NativeHeadPolicy.__new__(NativeHeadPolicy)
    policy.prediction_error_escalation_threshold = 0.45
    policy.nsi_policy = type(
        "Nsi",
        (),
        {
            "last_call": {
                "homeostatic_decision": {
                    "target": InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX.value,
                    "reason": "homeostatic_persistent_failure_wake",
                },
                "prediction_error": 0.1,
                "temporal_observation_available": True,
            }
        },
    )()

    target, source = policy._internal_target_with_prediction_error(
        _state(changed=False)
    )

    assert target == InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX
    assert source == "homeostatic_persistent_failure_wake"


def test_sequence_policy_conditions_next_state_on_executed_motor_action() -> None:
    vectorizer = StateVectorizer(hash_bins=0)
    model = RecordingNextStateModel(vectorizer.vector_dim)
    policy = SequenceModelPolicy(
        model,
        vectorizer,
        policy_label="executed_action_prediction_test",
    )

    policy.act(_state(changed=True))

    assert model.recorded_action_indices is not None
    assert int(model.recorded_action_indices[0, 0]) == ACTION_ORDER.index(ActionType.REFRESH_STATE)
    assert policy.last_call["next_state_action_source"] == "executed_motor_action"
