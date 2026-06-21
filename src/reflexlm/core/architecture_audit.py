from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
from reflexlm.core.runner import ReflexCoreSandboxRunner
from reflexlm.core.schema import ComputerObservation, MotorAction
from reflexlm.core.training import load_reflexcore_yaml
from reflexlm.models.features import MAX_CANDIDATE_SLOTS, ROUTE_ORDER
from reflexlm.runtime.safety import SafetyLayer
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    FileSystemState,
    GoalSpec,
    InternalTarget,
    ProcessState,
    ProcessStatus,
    SafetyState,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


@dataclass(slots=True)
class ReflexCoreArchitectureAuditConfig:
    config_path: Path
    output_json: Path | None = None
    input_dim: int = 64
    batch_size: int = 2
    sequence_len: int = 3
    min_parameter_count: int | None = None
    max_parameter_count: int | None = None
    require_action_vector_residual: bool = True
    min_numeric_action_aux_weight: float = 0.0
    require_numeric_aux_zero_text: bool = True
    require_numeric_aux_zero_hash: bool = True
    require_acceptance_scope: str = "terminal_process_filesystem_time_sandbox_only"
    require_free_shell_disabled: bool = True
    require_gui_vision_disabled: bool = True


def audit_reflexcore_architecture(
    config: ReflexCoreArchitectureAuditConfig,
) -> dict[str, object]:
    raw_config = load_reflexcore_yaml(config.config_path)
    model_config = dict(_object(raw_config.get("model")))
    model_config["input_dim"] = config.input_dim
    resolved_model_config = ReflexCoreV0Config(**model_config)
    model = ReflexCoreV0(resolved_model_config)
    outputs = model(
        torch.zeros(config.batch_size, config.sequence_len, config.input_dim),
        torch.zeros(config.batch_size, config.sequence_len, 4, dtype=torch.long),
    )
    parameter_count = model.parameter_count()
    checks = {
        "acceptance_scope": _equality_check(
            _object(raw_config.get("acceptance")).get("scope"),
            config.require_acceptance_scope,
        ),
        "free_shell_generation_disabled": _equality_check(
            _object(raw_config.get("acceptance")).get("free_shell_generation"),
            False if config.require_free_shell_disabled else _object(raw_config.get("acceptance")).get("free_shell_generation"),
        ),
        "gui_or_vision_disabled": _equality_check(
            _object(raw_config.get("acceptance")).get("gui_or_vision"),
            False if config.require_gui_vision_disabled else _object(raw_config.get("acceptance")).get("gui_or_vision"),
        ),
        "action_vector_residual_enabled": _equality_check(
            resolved_model_config.action_vector_residual,
            True if config.require_action_vector_residual else resolved_model_config.action_vector_residual,
        ),
        "parameter_count_range": _parameter_count_check(parameter_count, config),
        "required_output_heads_present": _output_head_check(outputs),
        "typed_action_head_shape": _shape_check(
            outputs.get("action_logits"),
            expected_last_dim=len(ActionType),
        ),
        "typed_target_head_shape": _shape_check(
            outputs.get("target_logits"),
            expected_last_dim=len(InternalTarget),
        ),
        "route_head_shape": _shape_check(
            outputs.get("route_logits"),
            expected_last_dim=len(ROUTE_ORDER),
        ),
        "command_slot_head_shape": _shape_check(
            outputs.get("command_slot_logits"),
            expected_last_dim=resolved_model_config.max_command_slots,
            max_last_dim=MAX_CANDIDATE_SLOTS,
        ),
        "file_slot_head_shape": _shape_check(
            outputs.get("file_slot_logits"),
            expected_last_dim=resolved_model_config.max_file_slots,
            max_last_dim=MAX_CANDIDATE_SLOTS,
        ),
        "risk_salience_are_bounded": _bounded_sigmoid_check(outputs),
        "prediction_error_nonnegative": _nonnegative_check(outputs.get("prediction_error")),
        "next_state_shape_matches_observation": _next_state_shape_check(
            outputs.get("next_state"),
            config,
        ),
        "numeric_action_auxiliary_configured": _numeric_aux_check(raw_config, config),
        "loss_heads_supervised": _loss_weight_check(raw_config),
        "observation_schema_blocks_non_allowlisted_candidates": _observation_schema_check(),
        "motor_schema_requires_command_payload": _motor_schema_check(),
        "safety_layer_blocks_non_allowlisted_and_dangerous_commands": _safety_check(),
        "runner_uses_shell_false": _runner_shell_false_check(),
    }
    passed = all(
        isinstance(check, dict) and check.get("passed") is True
        for check in checks.values()
    )
    report: dict[str, object] = {
        "artifact_family": "reflexcore_v0_architecture_audit",
        "passed": passed,
        "verdict": (
            "bounded_reflexcore_v0_architecture_ready"
            if passed
            else "repair_reflexcore_v0_architecture"
        ),
        "config": _json_config(config),
        "model_config": resolved_model_config.to_dict(),
        "parameter_count": parameter_count,
        "checks": checks,
        "claim_boundary": (
            "Audits ReflexCore V0 structural compliance only. It does not prove "
            "task performance; pair it with profile, sensory-ablation, and "
            "mechanism-dossier gates."
        ),
    }
    if config.output_json is not None:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def _output_head_check(outputs: dict[str, torch.Tensor | None]) -> dict[str, object]:
    required = {
        "text_logits",
        "action_logits",
        "target_logits",
        "route_logits",
        "command_slot_logits",
        "file_slot_logits",
        "risk",
        "salience",
        "prediction_error",
        "next_state",
    }
    observed = {key for key, value in outputs.items() if isinstance(value, torch.Tensor)}
    missing = sorted(required - observed)
    return {
        "passed": not missing,
        "observed": sorted(observed),
        "required": sorted(required),
        "missing": missing,
    }


def _shape_check(
    tensor: object,
    *,
    expected_last_dim: int,
    max_last_dim: int | None = None,
) -> dict[str, object]:
    shape = list(tensor.shape) if isinstance(tensor, torch.Tensor) else None
    observed_last_dim = shape[-1] if shape else None
    passed = observed_last_dim == expected_last_dim
    if max_last_dim is not None:
        passed = passed and observed_last_dim is not None and observed_last_dim <= max_last_dim
    return {
        "passed": passed,
        "observed_shape": shape,
        "expected_last_dim": expected_last_dim,
        "max_last_dim": max_last_dim,
    }


def _bounded_sigmoid_check(outputs: dict[str, torch.Tensor | None]) -> dict[str, object]:
    observed: dict[str, object] = {}
    passed = True
    for key in ("risk", "salience"):
        tensor = outputs.get(key)
        if not isinstance(tensor, torch.Tensor):
            observed[key] = None
            passed = False
            continue
        minimum = float(tensor.min().item())
        maximum = float(tensor.max().item())
        observed[key] = {"min": minimum, "max": maximum}
        passed = passed and 0.0 <= minimum <= maximum <= 1.0
    return {"passed": passed, "observed": observed, "required_range": [0.0, 1.0]}


def _nonnegative_check(tensor: object) -> dict[str, object]:
    if not isinstance(tensor, torch.Tensor):
        return {"passed": False, "observed_min": None, "required_min": 0.0}
    observed_min = float(tensor.min().item())
    return {
        "passed": observed_min >= 0.0,
        "observed_min": observed_min,
        "required_min": 0.0,
    }


def _next_state_shape_check(
    tensor: object,
    config: ReflexCoreArchitectureAuditConfig,
) -> dict[str, object]:
    expected = [config.batch_size, config.sequence_len, config.input_dim]
    observed = list(tensor.shape) if isinstance(tensor, torch.Tensor) else None
    return {"passed": observed == expected, "observed_shape": observed, "expected_shape": expected}


def _numeric_aux_check(
    raw_config: dict[str, object],
    config: ReflexCoreArchitectureAuditConfig,
) -> dict[str, object]:
    sensory = _object(raw_config.get("sensory_training"))
    weight = _number(sensory.get("numeric_action_aux_weight"))
    zero_text = bool(sensory.get("numeric_action_aux_zero_text"))
    zero_hash = bool(sensory.get("numeric_action_aux_zero_hash"))
    passed = weight is not None and weight >= config.min_numeric_action_aux_weight
    if config.require_numeric_aux_zero_text:
        passed = passed and zero_text
    if config.require_numeric_aux_zero_hash:
        passed = passed and zero_hash
    return {
        "passed": passed,
        "observed": {
            "numeric_action_aux_weight": weight,
            "numeric_action_aux_zero_text": zero_text,
            "numeric_action_aux_zero_hash": zero_hash,
        },
        "required": {
            "min_numeric_action_aux_weight": config.min_numeric_action_aux_weight,
            "zero_text": config.require_numeric_aux_zero_text,
            "zero_hash": config.require_numeric_aux_zero_hash,
        },
    }


def _loss_weight_check(raw_config: dict[str, object]) -> dict[str, object]:
    required = ("action", "command_slot", "file_slot", "risk", "salience", "prediction_error", "next_state")
    losses = _object(raw_config.get("loss_weights"))
    observed = {name: _number(losses.get(name)) for name in required}
    missing_or_zero = [
        name for name, value in observed.items() if value is None or value <= 0.0
    ]
    return {
        "passed": not missing_or_zero,
        "observed": observed,
        "required_positive": list(required),
        "missing_or_zero": missing_or_zero,
    }


def _observation_schema_check() -> dict[str, object]:
    goal = _audit_goal()
    try:
        ComputerObservation(
            time=TimeState(),
            goal=goal,
            process=ProcessState(status=ProcessStatus.EXITED),
            terminal=TerminalState(),
            filesystem=FileSystemState(),
            candidate_commands=["python -m pytest", "not allowlisted"],
        )
    except ValueError as exc:
        return {
            "passed": "candidate command is not allowlisted" in str(exc),
            "observed": str(exc),
            "required": "reject non-allowlisted candidate commands",
        }
    return {
        "passed": False,
        "observed": "accepted invalid candidate command",
        "required": "reject non-allowlisted candidate commands",
    }


def _motor_schema_check() -> dict[str, object]:
    try:
        MotorAction(type=ActionType.RUN_COMMAND)
    except ValueError as exc:
        return {
            "passed": "RUN_COMMAND requires a command payload" in str(exc),
            "observed": str(exc),
            "required": "RUN_COMMAND requires command payload",
        }
    return {
        "passed": False,
        "observed": "accepted RUN_COMMAND without command",
        "required": "RUN_COMMAND requires command payload",
    }


def _safety_check() -> dict[str, object]:
    goal = _audit_goal()
    state = _audit_state(goal)
    safety = SafetyLayer()
    non_allowlisted = safety.enforce(
        ActionDecision(type=ActionType.RUN_COMMAND, command="python setup.py"),
        goal,
        state,
    )
    dangerous = safety.enforce(
        ActionDecision(type=ActionType.RUN_COMMAND, command="rm -rf /"),
        goal.model_copy(update={"command_allowlist": ["rm -rf /"]}),
        state,
    )
    return {
        "passed": (not non_allowlisted.allowed) and (not dangerous.allowed),
        "observed": {
            "non_allowlisted_reason": non_allowlisted.reason,
            "dangerous_reason": dangerous.reason,
        },
        "required": [
            "block command_not_allowlisted",
            "block dangerous_command_detected",
        ],
    }


def _runner_shell_false_check() -> dict[str, object]:
    source = inspect.getsource(ReflexCoreSandboxRunner._run_command)
    observed = "shell=False" in source
    return {
        "passed": observed,
        "observed": "shell=False" if observed else "shell flag not found",
        "required": "ReflexCoreSandboxRunner._run_command uses shell=False",
    }


def _parameter_count_check(
    parameter_count: int,
    config: ReflexCoreArchitectureAuditConfig,
) -> dict[str, object]:
    passed = True
    if config.min_parameter_count is not None:
        passed = passed and parameter_count >= config.min_parameter_count
    if config.max_parameter_count is not None:
        passed = passed and parameter_count <= config.max_parameter_count
    return {
        "passed": passed,
        "observed": parameter_count,
        "required": {
            "min": config.min_parameter_count,
            "max": config.max_parameter_count,
        },
    }


def _audit_goal() -> GoalSpec:
    return GoalSpec(
        task_type=TaskType.ROUTINE_RECOVERY,
        description="audit bounded command selection",
        command_allowlist=["python -m pytest"],
        success_criteria=["typed_motor_action"],
        safety_notes=["allowlist_only", "shell_false"],
    )


def _audit_state(goal: GoalSpec) -> SystemStateFrame:
    return SystemStateFrame(
        time=TimeState(),
        goal=goal,
        process=ProcessState(status=ProcessStatus.EXITED),
        terminal=TerminalState(),
        filesystem=FileSystemState(),
        safety=SafetyState(),
    )


def _equality_check(observed: object, required: object) -> dict[str, object]:
    return {"passed": observed == required, "observed": observed, "required": required}


def _object(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _json_config(config: ReflexCoreArchitectureAuditConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["config_path"] = _path_label(config.config_path)
    payload["output_json"] = _path_label(config.output_json) if config.output_json else None
    return payload


def _path_label(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name
