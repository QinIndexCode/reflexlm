import hashlib
import pytest
import torch
import json

from reflexlm.eval import SequenceModelPolicy
from reflexlm.models.features import ACTION_ORDER, ROUTE_ORDER, StateVectorizer
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER
from reflexlm.runtime.homeostasis import (
    HomeostaticControlConfig,
    HomeostaticSynapticController,
)
from reflexlm.schema import (
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


class _ControlledSequenceModel(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action: ActionType,
        salience_logit: float = -10.0,
        risk: float = 0.0,
        prediction_error: float = 0.01,
        command_slot: int = 0,
    ) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.input_dim = input_dim
        self.action = action
        self.salience_logit = salience_logit
        self.risk = risk
        self.prediction_error = prediction_error
        self.command_slot = command_slot

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, seq_len, _ = inputs.shape
        action_logits = torch.zeros(batch, seq_len, len(ACTION_ORDER))
        action_logits[..., ACTION_ORDER.index(self.action)] = 8.0
        target_logits = torch.zeros(batch, seq_len, len(INTERNAL_TARGET_ORDER))
        target_logits[
            ..., INTERNAL_TARGET_ORDER.index(InternalTarget.REFLEX_MOTOR)
        ] = 8.0
        command_slot_logits = torch.zeros(batch, seq_len, 4)
        command_slot_logits[..., self.command_slot] = 8.0
        return {
            "hidden": hidden,
            "action_logits": action_logits,
            "target_logits": target_logits,
            "route_logits": torch.zeros(batch, seq_len, len(ROUTE_ORDER)),
            "command_slot_logits": command_slot_logits,
            "file_slot_logits": torch.zeros(batch, seq_len, 4),
            "salience": torch.full((batch, seq_len), self.salience_logit),
            "risk": torch.full((batch, seq_len), self.risk),
            "prediction_error": torch.full(
                (batch, seq_len),
                self.prediction_error,
            ),
            "next_state": inputs.clone(),
        }


def _runtime_state(
    *,
    changed: bool = False,
    failed: bool | None = None,
) -> SystemStateFrame:
    failure_visible = changed if failed is None else failed
    return SystemStateFrame(
        time=TimeState(tick=int(changed), runtime_ms=1000 * int(changed)),
        goal=GoalSpec(
            task_type=TaskType.ROUTINE_RECOVERY,
            description="bounded runtime recovery",
            command_allowlist=["python -m pytest -q"],
        ),
        process=ProcessState(
            status=ProcessStatus.EXITED if failure_visible else ProcessStatus.RUNNING,
            exit_code=1 if failure_visible else None,
            resource_alert=failure_visible,
        ),
        terminal=TerminalState(
            stderr_delta="changed runtime state" if changed else "",
        ),
        filesystem=FileSystemState(
            changed_paths=["src/runtime.py"] if changed else [],
        ),
    )


def _sequence_policy(
    *,
    action: ActionType,
    salience_logit: float = -10.0,
    risk: float = 0.0,
    prediction_error: float = 0.01,
    enable_homeostatic_control: bool = True,
    homeostatic_config: HomeostaticControlConfig | None = None,
    command_slot: int = 0,
) -> SequenceModelPolicy:
    vectorizer = StateVectorizer(hash_bins=0)
    return SequenceModelPolicy(
        _ControlledSequenceModel(
            vectorizer.vector_dim,
            action=action,
            salience_logit=salience_logit,
            risk=risk,
            prediction_error=prediction_error,
            command_slot=command_slot,
        ),
        vectorizer,
        policy_label="homeostatic_integration_test",
        enable_homeostatic_control=enable_homeostatic_control,
        homeostatic_config=homeostatic_config,
    )


def _stable_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _unsigned_artifact(artifact: dict) -> dict:
    return {
        key: value
        for key, value in artifact.items()
        if key not in {"integrity_sha256", "authenticator"}
    }


def test_surprising_cross_frame_state_wakes_semantic_cortex() -> None:
    controller = HomeostaticSynapticController()

    decision = controller.observe(
        proposed_action=ActionType.WAIT,
        salience=0.30,
        risk=0.10,
        prediction_error=0.90,
        temporal_observation_available=True,
    )

    assert decision.target == InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX
    assert decision.reason == "homeostatic_surprise_wake"
    assert controller.snapshot()["wake_events"] == 1


def test_high_learned_risk_inhibits_side_effect_but_not_observation() -> None:
    side_effect_controller = HomeostaticSynapticController()
    observation_controller = HomeostaticSynapticController()

    side_effect = side_effect_controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.50,
        risk=0.95,
        prediction_error=0.20,
        temporal_observation_available=False,
    )
    observation = observation_controller.observe(
        proposed_action=ActionType.READ_STDERR,
        salience=0.50,
        risk=0.95,
        prediction_error=0.20,
        temporal_observation_available=False,
    )

    assert side_effect.inhibited is True
    assert side_effect.target == InternalTarget.INHIBIT
    assert observation.inhibited is False


def test_repeated_low_value_side_effect_habituates_without_task_label() -> None:
    controller = HomeostaticSynapticController(
        HomeostaticControlConfig(habituation_repetitions=2)
    )

    first = controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.05,
        risk=0.05,
        prediction_error=0.05,
        temporal_observation_available=True,
    )
    second = controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.05,
        risk=0.05,
        prediction_error=0.05,
        temporal_observation_available=True,
    )

    assert first.habituated is False
    assert second.habituated is True
    assert second.reason == "homeostatic_low_value_habituation"


def test_visible_failure_adapts_surprise_sensitivity_without_task_label() -> None:
    adaptive = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.80,
            failure_sensitivity_rate=1.0,
            minimum_surprise_wake_threshold=0.05,
        )
    )
    fixed = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.80,
            online_failure_sensitivity_enabled=False,
        )
    )

    adaptive_decision = adaptive.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.20,
        risk=0.10,
        prediction_error=0.40,
        temporal_observation_available=True,
        failure_visible=True,
    )
    fixed_decision = fixed.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.20,
        risk=0.10,
        prediction_error=0.40,
        temporal_observation_available=True,
        failure_visible=True,
    )

    assert adaptive_decision.reason == "homeostatic_surprise_wake"
    assert fixed_decision.reason == "homeostatic_reflex_preserved"
    assert adaptive.snapshot()["active_surprise_wake_threshold"] == pytest.approx(0.36)
    assert adaptive.snapshot()["failure_sensitivity_adaptations"] == 1
    assert fixed.snapshot()["active_surprise_wake_threshold"] == 0.80


def test_stable_outcomes_restore_calibrated_surprise_set_point() -> None:
    controller = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.80,
            failure_sensitivity_rate=1.0,
            set_point_recovery_rate=0.50,
        )
    )
    controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.20,
        risk=0.10,
        prediction_error=0.40,
        temporal_observation_available=True,
        failure_visible=True,
    )
    adapted = float(controller.snapshot()["active_surprise_wake_threshold"])

    controller.observe(
        proposed_action=ActionType.WAIT,
        salience=0.10,
        risk=0.05,
        prediction_error=0.05,
        temporal_observation_available=True,
        failure_visible=False,
    )
    recovered = controller.snapshot()

    assert adapted == pytest.approx(0.36)
    assert adapted < recovered["active_surprise_wake_threshold"] < 0.80
    assert recovered["set_point_recovery_adaptations"] == 1
    assert (
        recovered["last_threshold_adaptation"]["reason"]
        == "stable_outcome_restored_set_point"
    )


def test_failure_sensitivity_respects_minimum_threshold_bound() -> None:
    controller = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.80,
            failure_sensitivity_rate=1.0,
            minimum_surprise_wake_threshold=0.20,
        )
    )

    controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.10,
        risk=0.05,
        prediction_error=0.0,
        temporal_observation_available=True,
        failure_visible=True,
    )

    assert controller.snapshot()["active_surprise_wake_threshold"] == 0.20


def test_failure_sensitivity_hysteresis_rejects_near_threshold_jitter() -> None:
    controller = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.50,
            failure_sensitivity_rate=1.0,
            failure_sensitivity_hysteresis=0.01,
        )
    )

    near_threshold_decision = controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.10,
        risk=0.05,
        prediction_error=0.495,
        temporal_observation_available=True,
        failure_visible=True,
    )
    near_threshold = controller.snapshot()
    controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.10,
        risk=0.05,
        prediction_error=0.40,
        temporal_observation_available=True,
        failure_visible=True,
    )

    assert near_threshold["active_surprise_wake_threshold"] == 0.50
    assert near_threshold["failure_sensitivity_adaptations"] == 0
    assert near_threshold_decision.reason == "homeostatic_surprise_wake"
    assert controller.snapshot()["failure_sensitivity_adaptations"] == 1


def test_decision_signal_resolution_stabilizes_cross_runtime_control_edges() -> None:
    failure_decisions = []
    failure_states = []
    for prediction_error in (0.14863471686840057, 0.15021659433841705):
        controller = HomeostaticSynapticController(
            HomeostaticControlConfig(
                surprise_wake_threshold=0.15372,
                failure_sensitivity_hysteresis=0.005,
                decision_signal_resolution=0.005,
            )
        )
        failure_decisions.append(
            controller.observe(
                proposed_action=ActionType.READ_STDERR,
                salience=0.20,
                risk=0.05,
                prediction_error=prediction_error,
                temporal_observation_available=True,
                failure_visible=True,
            ).reason
        )
        failure_states.append(controller.snapshot())

    wake_reasons = []
    for wake_pressure in (0.1407648189599819, 0.14131860730118656):
        controller = HomeostaticSynapticController(
            HomeostaticControlConfig(
                surprise_wake_threshold=0.14115,
                decision_signal_resolution=0.005,
            )
        )
        wake_reasons.append(
            controller.observe(
                proposed_action=ActionType.DONE,
                salience=0.20,
                risk=0.05,
                prediction_error=wake_pressure,
                temporal_observation_available=True,
            ).reason
        )

    assert failure_decisions == ["homeostatic_surprise_wake"] * 2
    assert [state["failure_sensitivity_adaptations"] for state in failure_states] == [
        0,
        0,
    ]
    assert wake_reasons == ["homeostatic_reflex_preserved"] * 2


def test_quantized_surprise_wake_requires_strict_threshold_crossing() -> None:
    boundary = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.135,
            decision_signal_resolution=0.005,
        )
    )
    above = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.135,
            decision_signal_resolution=0.005,
        )
    )

    boundary_decision = boundary.observe(
        proposed_action=ActionType.WAIT,
        salience=0.10,
        risk=0.05,
        prediction_error=0.135,
        temporal_observation_available=True,
    )
    above_decision = above.observe(
        proposed_action=ActionType.WAIT,
        salience=0.10,
        risk=0.05,
        prediction_error=0.140,
        temporal_observation_available=True,
    )

    assert boundary_decision.reason == "homeostatic_reflex_preserved"
    assert above_decision.reason == "homeostatic_surprise_wake"


def test_failure_hysteresis_requires_strict_quantized_threshold_crossing() -> None:
    boundary = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.135,
            failure_sensitivity_hysteresis=0.005,
            decision_signal_resolution=0.005,
            online_failure_sensitivity_enabled=False,
        )
    )
    above = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.135,
            failure_sensitivity_hysteresis=0.005,
            decision_signal_resolution=0.005,
            online_failure_sensitivity_enabled=False,
        )
    )

    boundary_decision = boundary.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.10,
        risk=0.05,
        prediction_error=0.130,
        temporal_observation_available=True,
        failure_visible=True,
    )
    above_decision = above.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.10,
        risk=0.05,
        prediction_error=0.131,
        temporal_observation_available=True,
        failure_visible=True,
    )

    assert boundary_decision.reason == "homeostatic_reflex_preserved"
    assert above_decision.reason == "homeostatic_surprise_wake"


def test_cross_episode_memory_preserves_only_adaptive_threshold_and_lifetime_counts() -> None:
    controller = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.80,
            failure_sensitivity_rate=1.0,
            preserve_adaptive_threshold_across_reset=True,
            cross_episode_threshold_retention=1.0,
        )
    )
    controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.20,
        risk=0.10,
        prediction_error=0.40,
        temporal_observation_available=True,
        failure_visible=True,
    )

    controller.reset()
    state = controller.snapshot()

    assert state["active_surprise_wake_threshold"] == pytest.approx(0.36)
    assert state["failure_sensitivity_adaptations"] == 0
    assert state["lifetime_failure_sensitivity_adaptations"] == 1
    assert state["adaptive_threshold_preserved_resets"] == 1
    assert state["adaptive_threshold_reset_decay_events"] == 0
    assert state["observations"] == 0
    assert state["ema_failure"] == 0.0


def test_default_reset_erases_adaptive_threshold_memory() -> None:
    controller = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.80,
            failure_sensitivity_rate=1.0,
        )
    )
    controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.20,
        risk=0.10,
        prediction_error=0.40,
        temporal_observation_available=True,
        failure_visible=True,
    )

    controller.reset()
    state = controller.snapshot()

    assert state["active_surprise_wake_threshold"] == 0.80
    assert state["lifetime_failure_sensitivity_adaptations"] == 0
    assert state["adaptive_threshold_preserved_resets"] == 0


def test_cross_episode_memory_decays_toward_calibrated_set_point_on_reset() -> None:
    controller = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.80,
            failure_sensitivity_rate=1.0,
            preserve_adaptive_threshold_across_reset=True,
            cross_episode_threshold_retention=0.50,
        )
    )
    controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.20,
        risk=0.10,
        prediction_error=0.40,
        temporal_observation_available=True,
        failure_visible=True,
    )

    controller.reset()
    state = controller.snapshot()

    assert state["active_surprise_wake_threshold"] == pytest.approx(0.58)
    assert state["adaptive_threshold_preserved_resets"] == 1
    assert state["adaptive_threshold_reset_decay_events"] == 1


def test_persistent_state_round_trip_preserves_only_bounded_memory(
    tmp_path,
) -> None:
    config = HomeostaticControlConfig(
        surprise_wake_threshold=0.80,
        failure_sensitivity_rate=1.0,
        preserve_adaptive_threshold_across_reset=True,
    )
    source = HomeostaticSynapticController(config)
    source.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.20,
        risk=0.10,
        prediction_error=0.40,
        temporal_observation_available=True,
        failure_visible=True,
    )
    artifact_path = tmp_path / "homeostatic-state.json"
    artifact = source.save_persistent_state(artifact_path)
    restored = HomeostaticSynapticController(config)

    restored.load_persistent_state_file(artifact_path)
    state = restored.snapshot()

    assert artifact["integrity_sha256"]
    assert artifact["authenticator"]["algorithm"] == "sha256"
    assert artifact["authenticator"]["key_fingerprint_sha256"] is None
    assert state["active_surprise_wake_threshold"] == pytest.approx(0.36)
    assert state["lifetime_failure_sensitivity_adaptations"] == 1
    assert state["observations"] == 0
    assert state["ema_failure"] == 0.0
    assert state["last_action"] is None
    assert state["failure_sensitivity_adaptations"] == 0


def test_persistent_state_rejects_tampering_and_config_drift(tmp_path) -> None:
    config = HomeostaticControlConfig(
        surprise_wake_threshold=0.80,
        preserve_adaptive_threshold_across_reset=True,
    )
    source = HomeostaticSynapticController(config)
    artifact_path = tmp_path / "homeostatic-state.json"
    source.save_persistent_state(artifact_path)
    tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    tampered["state"]["active_surprise_wake_threshold"] = 0.20
    artifact_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="integrity mismatch"):
        HomeostaticSynapticController(config).load_persistent_state_file(artifact_path)

    clean = source.export_persistent_state()
    incompatible = HomeostaticSynapticController(
        HomeostaticControlConfig(
            surprise_wake_threshold=0.70,
            preserve_adaptive_threshold_across_reset=True,
        )
    )
    with pytest.raises(ValueError, match="incompatible with config"):
        incompatible.load_persistent_state(clean)

    algorithm_drift = json.loads(json.dumps(clean))
    algorithm_drift["controller_schema_version"] = (
        "reflexlm.homeostatic_synaptic_control.v1"
    )
    with pytest.raises(ValueError, match="controller schema mismatch"):
        HomeostaticSynapticController(config).load_persistent_state(algorithm_drift)

    scoped = source.export_persistent_state(persistence_scope="package-a")
    with pytest.raises(ValueError, match="scope mismatch"):
        HomeostaticSynapticController(config).load_persistent_state(
            scoped,
            persistence_scope="package-b",
        )


def test_persistent_state_hmac_authenticator_requires_matching_key(tmp_path) -> None:
    config = HomeostaticControlConfig(
        surprise_wake_threshold=0.80,
        failure_sensitivity_rate=1.0,
        preserve_adaptive_threshold_across_reset=True,
    )
    source = HomeostaticSynapticController(config)
    source.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.20,
        risk=0.10,
        prediction_error=0.40,
        temporal_observation_available=True,
        failure_visible=True,
    )
    artifact_path = tmp_path / "homeostatic-hmac-state.json"
    artifact = source.save_persistent_state(
        artifact_path,
        authenticity_key="bounded-test-key",
    )

    restored = HomeostaticSynapticController(config)
    restored.load_persistent_state_file(
        artifact_path,
        authenticity_key="bounded-test-key",
    )

    assert artifact["authenticator"]["algorithm"] == "hmac-sha256"
    assert artifact["authenticator"]["key_fingerprint_sha256"] == hashlib.sha256(
        b"bounded-test-key"
    ).hexdigest()
    assert "bounded-test-key" not in artifact_path.read_text(encoding="utf-8")
    assert restored.snapshot()["active_surprise_wake_threshold"] == pytest.approx(0.36)
    with pytest.raises(ValueError, match="requires an authenticity key"):
        HomeostaticSynapticController(config).load_persistent_state_file(artifact_path)
    with pytest.raises(ValueError, match="authenticator mismatch"):
        HomeostaticSynapticController(config).load_persistent_state_file(
            artifact_path,
            authenticity_key="wrong-key",
        )


def test_persistent_state_hmac_rejects_recomputed_hash_tampering(tmp_path) -> None:
    config = HomeostaticControlConfig(
        surprise_wake_threshold=0.80,
        failure_sensitivity_rate=1.0,
        preserve_adaptive_threshold_across_reset=True,
    )
    source = HomeostaticSynapticController(config)
    artifact_path = tmp_path / "homeostatic-hmac-state.json"
    source.save_persistent_state(artifact_path, authenticity_key="bounded-test-key")
    tampered = json.loads(artifact_path.read_text(encoding="utf-8"))
    tampered["state"]["active_surprise_wake_threshold"] = 0.20
    tampered["integrity_sha256"] = _stable_hash(_unsigned_artifact(tampered))
    artifact_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="authenticator mismatch"):
        HomeostaticSynapticController(config).load_persistent_state_file(
            artifact_path,
            authenticity_key="bounded-test-key",
        )


def test_persistent_state_requires_explicit_cross_episode_memory() -> None:
    controller = HomeostaticSynapticController()

    with pytest.raises(ValueError, match="requires online adaptation"):
        controller.export_persistent_state()


def test_priority_receptor_is_not_suppressed_by_surprise_or_habituation() -> None:
    controller = HomeostaticSynapticController()

    decision = controller.observe(
        proposed_action=ActionType.RUN_COMMAND,
        salience=0.01,
        risk=0.01,
        prediction_error=0.99,
        temporal_observation_available=True,
        receptor_priority=True,
    )

    assert decision.target == InternalTarget.REFLEX_MOTOR
    assert decision.habituated is False


def test_visible_persistent_failure_wakes_recovery_after_observations() -> None:
    controller = HomeostaticSynapticController()

    decision = controller.observe(
        proposed_action=ActionType.WAIT,
        salience=0.05,
        risk=0.05,
        prediction_error=0.05,
        temporal_observation_available=True,
        failure_visible=True,
    )

    assert decision.target == InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX
    assert decision.reason == "homeostatic_persistent_failure_wake"
    assert decision.failure_pressure == 1.0


def test_priority_receptor_defers_persistent_failure_wake() -> None:
    controller = HomeostaticSynapticController()

    decision = controller.observe(
        proposed_action=ActionType.READ_STDERR,
        salience=0.05,
        risk=0.05,
        prediction_error=0.05,
        temporal_observation_available=True,
        receptor_priority=True,
        failure_visible=True,
    )

    assert decision.target == InternalTarget.REFLEX_MOTOR
    assert decision.reason == "homeostatic_reflex_preserved"


def test_sequence_runtime_inhibits_high_risk_side_effect() -> None:
    policy = _sequence_policy(action=ActionType.RUN_COMMAND, risk=0.99)

    action = policy.act(_runtime_state())

    assert action.type == ActionType.BLOCK
    assert action.reason == "learned_risk_inhibition"
    assert policy.last_call["homeostatic_decision"]["inhibited"] is True


def test_sequence_runtime_wakes_on_observed_cross_frame_surprise() -> None:
    policy = _sequence_policy(
        action=ActionType.WAIT,
        homeostatic_config=HomeostaticControlConfig(
            surprise_wake_threshold=0.10
        ),
    )
    policy.act(_runtime_state())

    action = policy.act(_runtime_state(changed=True, failed=False))

    assert action.type == ActionType.WAIT
    assert action.reason == "homeostatic_surprise_wake"
    assert (
        policy.last_call["synaptic_plan"]["internal_target"]
        == InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX.value
    )


def test_sequence_runtime_uses_checkpoint_calibrated_wake_threshold() -> None:
    vectorizer = StateVectorizer(hash_bins=0)
    policy = SequenceModelPolicy(
        _ControlledSequenceModel(
            vectorizer.vector_dim,
            action=ActionType.WAIT,
        ),
        vectorizer,
        policy_label="homeostatic_calibration_test",
        training_summary={
                "prediction_error_calibration": {
                    "threshold": 0.10,
            }
        },
    )
    policy.act(_runtime_state())

    policy.act(_runtime_state(changed=True, failed=False))

    assert policy.last_call["homeostatic_decision"]["reason"] == "homeostatic_surprise_wake"
    assert (
        policy.metadata()["homeostatic_control"]["config"]["surprise_wake_threshold"]
        == 0.10
    )


def test_sequence_runtime_can_disable_online_adaptation_without_losing_calibration() -> None:
    vectorizer = StateVectorizer(hash_bins=0)
    policy = SequenceModelPolicy(
        _ControlledSequenceModel(
            vectorizer.vector_dim,
            action=ActionType.WAIT,
        ),
        vectorizer,
        policy_label="homeostatic_online_adaptation_ablation",
        training_summary={
            "prediction_error_calibration": {
                "threshold": 0.10,
            }
        },
        enable_online_homeostatic_adaptation=False,
    )

    metadata = policy.metadata()["homeostatic_control"]

    assert metadata["config"]["surprise_wake_threshold"] == 0.10
    assert metadata["active_surprise_wake_threshold"] == 0.10
    assert metadata["config"]["online_failure_sensitivity_enabled"] is False


def test_sequence_runtime_habituates_repeated_low_value_side_effect() -> None:
    policy = _sequence_policy(action=ActionType.RUN_COMMAND)
    first = policy.act(_runtime_state())

    second = policy.act(_runtime_state())

    assert first.type == ActionType.RUN_COMMAND
    assert second.type == ActionType.WAIT
    assert second.reason == "homeostatic_low_value_habituation"


def test_sequence_runtime_control_off_preserves_repeated_side_effect_for_ablation() -> None:
    policy = _sequence_policy(
        action=ActionType.RUN_COMMAND,
        enable_homeostatic_control=False,
    )
    first = policy.act(_runtime_state())

    second = policy.act(_runtime_state())

    assert first.type == ActionType.RUN_COMMAND
    assert second.type == ActionType.RUN_COMMAND
    assert policy.last_call["homeostatic_control_enabled"] is False


def test_persistent_failure_recovery_does_not_override_observation_action() -> None:
    policy = _sequence_policy(action=ActionType.READ_STDERR)
    policy.authorize_bounded_debug_cortex_recovery = True
    state = _runtime_state(changed=True)
    state = state.model_copy(
        update={
            "goal": state.goal.model_copy(
                update={"command_allowlist": ["failed-command", "recovery-command"]}
            ),
            "terminal": state.terminal.model_copy(
                update={"last_command": "failed-command"}
            ),
        }
    )

    action = policy.act(state)

    assert action.type == ActionType.READ_STDERR
    assert policy.last_call["persistent_recovery_constraint_applied"] is False


def test_persistent_failure_recovery_wakes_from_wait_after_observations() -> None:
    policy = _sequence_policy(action=ActionType.WAIT, command_slot=1)
    policy.authorize_bounded_debug_cortex_recovery = True
    state = _runtime_state(changed=True)
    state = state.model_copy(
        update={
            "goal": state.goal.model_copy(
                update={"command_allowlist": ["failed-command", "recovery-command"]}
            ),
            "terminal": state.terminal.model_copy(
                update={"last_command": "failed-command"}
            ),
        }
    )

    action = policy.act(state)

    assert action.type == ActionType.RUN_COMMAND
    assert action.command == "recovery-command"
    assert policy.last_call["persistent_recovery_constraint_applied"] is True
