from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from reflexlm.core.motor import ReflexCoreMotorConfig, decode_reflexcore_motor
from reflexlm.core.schema import action_to_index
from reflexlm.schema import (
    ActionType,
    FileSystemState,
    GoalSpec,
    ProcessState,
    ProcessStatus,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


@dataclass(slots=True)
class ReflexCoreHomeostaticMotorAuditConfig:
    output_json: Path | None = None
    risk_block_threshold: float = 0.9
    prediction_error_refresh_threshold: float = 0.05
    observed_prediction_error_refresh_threshold: float = 0.5
    salience_refresh_threshold: float = 0.75


def audit_reflexcore_homeostatic_motor(
    config: ReflexCoreHomeostaticMotorAuditConfig,
) -> dict[str, object]:
    motor_config = ReflexCoreMotorConfig(
        risk_block_threshold=config.risk_block_threshold,
        prediction_error_refresh_threshold=config.prediction_error_refresh_threshold,
        observed_prediction_error_refresh_threshold=(
            config.observed_prediction_error_refresh_threshold
        ),
        salience_refresh_threshold=config.salience_refresh_threshold,
    )
    cases = [
        _case_high_risk_blocks_run_command(motor_config),
        _case_prediction_error_refreshes_idle(motor_config),
        _case_observed_prediction_error_refreshes_idle(motor_config),
        _case_low_error_idle_waits(motor_config),
        _case_active_process_masks_prediction_error_refresh(motor_config),
    ]
    passed = all(case["passed"] is True for case in cases)
    report: dict[str, object] = {
        "artifact_family": "reflexcore_v0_homeostatic_motor_audit",
        "passed": passed,
        "verdict": (
            "bounded_reflexcore_v0_homeostatic_motor_ready"
            if passed
            else "repair_reflexcore_v0_homeostatic_motor"
        ),
        "config": _json_config(config),
        "checks": {str(case["name"]): case for case in cases},
        "claim_boundary": (
            "Audits typed motor-head modulation by bounded risk, salience, "
            "model prediction-error, and observed prediction-error signals. It "
            "does not bypass the safety layer and does not support GUI, "
            "free-shell, robotics, or production autonomy claims."
        ),
    }
    if config.output_json is not None:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def _case_high_risk_blocks_run_command(
    config: ReflexCoreMotorConfig,
) -> dict[str, object]:
    state = _base_state()
    decoded = decode_reflexcore_motor(
        _outputs(
            action=ActionType.RUN_COMMAND,
            risk=config.risk_block_threshold + 0.05,
            salience=0.2,
            prediction_error=0.0,
        ),
        state,
        config=config,
    )
    return _case_result(
        name="high_risk_blocks_run_command",
        decoded=decoded.action.type,
        expected=ActionType.BLOCK,
        reason=decoded.action.reason,
        expected_reason="reflexcore_risk_threshold",
        signal={"risk": decoded.risk},
    )


def _case_prediction_error_refreshes_idle(
    config: ReflexCoreMotorConfig,
) -> dict[str, object]:
    decoded = decode_reflexcore_motor(
        _outputs(
            action=ActionType.WAIT,
            risk=0.1,
            salience=config.salience_refresh_threshold + 0.05,
            prediction_error=config.prediction_error_refresh_threshold + 0.02,
        ),
        _base_state(),
        config=config,
    )
    return _case_result(
        name="prediction_error_refreshes_idle",
        decoded=decoded.action.type,
        expected=ActionType.REFRESH_STATE,
        reason=decoded.action.reason,
        expected_reason="reflexcore_prediction_error_refresh",
        signal={
            "salience": decoded.salience,
            "prediction_error": decoded.prediction_error,
        },
    )


def _case_observed_prediction_error_refreshes_idle(
    config: ReflexCoreMotorConfig,
) -> dict[str, object]:
    base = _base_state()
    state = base.model_copy(
        update={
            "runtime_evidence": base.runtime_evidence.model_copy(
                update={
                    "observed_prediction_error": (
                        config.observed_prediction_error_refresh_threshold + 0.1
                    )
                }
            )
        }
    )
    decoded = decode_reflexcore_motor(
        _outputs(
            action=ActionType.WAIT,
            risk=0.1,
            salience=0.1,
            prediction_error=0.0,
        ),
        state,
        config=config,
    )
    return _case_result(
        name="observed_prediction_error_refreshes_idle",
        decoded=decoded.action.type,
        expected=ActionType.REFRESH_STATE,
        reason=decoded.action.reason,
        expected_reason="reflexcore_observed_prediction_error_refresh",
        signal={
            "observed_prediction_error": (
                state.runtime_evidence.observed_prediction_error
            )
        },
    )


def _case_low_error_idle_waits(config: ReflexCoreMotorConfig) -> dict[str, object]:
    decoded = decode_reflexcore_motor(
        _outputs(
            action=ActionType.WAIT,
            risk=0.1,
            salience=config.salience_refresh_threshold + 0.05,
            prediction_error=max(0.0, config.prediction_error_refresh_threshold - 0.02),
        ),
        _base_state(),
        config=config,
    )
    return _case_result(
        name="low_error_idle_waits",
        decoded=decoded.action.type,
        expected=ActionType.WAIT,
        reason=decoded.action.reason,
        expected_reason=None,
        signal={
            "salience": decoded.salience,
            "prediction_error": decoded.prediction_error,
        },
    )


def _case_active_process_masks_prediction_error_refresh(
    config: ReflexCoreMotorConfig,
) -> dict[str, object]:
    base = _base_state()
    state = base.model_copy(
        update={
            "process": base.process.model_copy(
                update={"status": ProcessStatus.RUNNING}
            )
        }
    )
    decoded = decode_reflexcore_motor(
        _outputs(
            action=ActionType.WAIT,
            risk=0.1,
            salience=config.salience_refresh_threshold + 0.05,
            prediction_error=config.prediction_error_refresh_threshold + 0.02,
        ),
        state,
        config=config,
    )
    return _case_result(
        name="active_process_masks_prediction_error_refresh",
        decoded=decoded.action.type,
        expected=ActionType.WAIT,
        reason=decoded.action.reason,
        expected_reason=None,
        signal={
            "process_status": state.process.status.value,
            "salience": decoded.salience,
            "prediction_error": decoded.prediction_error,
        },
    )


def _case_result(
    *,
    name: str,
    decoded: ActionType,
    expected: ActionType,
    reason: str | None,
    expected_reason: str | None,
    signal: dict[str, object],
) -> dict[str, object]:
    return {
        "name": name,
        "passed": decoded == expected and reason == expected_reason,
        "observed": {"action": decoded.value, "reason": reason, "signal": signal},
        "required": {"action": expected.value, "reason": expected_reason},
    }


def _outputs(
    *,
    action: ActionType,
    risk: float,
    salience: float,
    prediction_error: float,
) -> dict[str, torch.Tensor]:
    action_logits = torch.full((1, 1, len(ActionType)), -10.0)
    action_logits[0, 0, action_to_index(action)] = 10.0
    command_logits = torch.full((1, 1, 4), -10.0)
    command_logits[0, 0, 0] = 10.0
    return {
        "action_logits": action_logits,
        "command_slot_logits": command_logits,
        "file_slot_logits": torch.zeros(1, 1, 4),
        "route_logits": torch.zeros(1, 1, 4),
        "target_logits": torch.zeros(1, 1, 4),
        "risk": torch.tensor([[[risk]]]),
        "salience": torch.tensor([[[salience]]]),
        "prediction_error": torch.tensor([[[prediction_error]]]),
    }


def _base_state() -> SystemStateFrame:
    return SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="homeostatic motor audit",
            command_allowlist=["echo ok"],
        ),
        process=ProcessState(status=ProcessStatus.EXITED),
        terminal=TerminalState(prompt_visible=True),
        filesystem=FileSystemState(),
    )


def _json_config(config: ReflexCoreHomeostaticMotorAuditConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_json"] = str(config.output_json) if config.output_json else None
    return payload
