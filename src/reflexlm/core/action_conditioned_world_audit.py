from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
from reflexlm.core.schema import action_to_index
from reflexlm.schema import ActionType


@dataclass(slots=True)
class ReflexCoreActionConditionedWorldAuditConfig:
    output_json: Path | None = None
    input_dim: int = 8
    vocab_size: int = 512
    action_signal: float = 4.0
    min_next_state_delta: float = 1.0
    min_prediction_error_delta: float = 0.5


def audit_reflexcore_action_conditioned_world(
    config: ReflexCoreActionConditionedWorldAuditConfig,
) -> dict[str, object]:
    model_config = ReflexCoreV0Config.smoke(
        input_dim=config.input_dim,
        vocab_size=config.vocab_size,
    )
    model_config.prediction_error_conditioning = "state_action"
    model_config.prediction_error_mode = "direct"
    model_config.prediction_error_calibration_scale = 1.0
    model = ReflexCoreV0(model_config)
    observation = torch.zeros(1, 1, config.input_dim)
    text_tokens = torch.zeros(1, 1, 4, dtype=torch.long)

    baseline_outputs = model(
        observation,
        text_tokens,
        action_indices=_action_tensor(ActionType.WAIT),
    )
    _configure_action_probe(model, config)
    wait_outputs = model(
        observation,
        text_tokens,
        action_indices=_action_tensor(ActionType.WAIT),
    )
    block_outputs = model(
        observation,
        text_tokens,
        action_indices=_action_tensor(ActionType.BLOCK),
    )

    checks = {
        "default_next_state_copies_current_observation": _copy_baseline_check(
            baseline_outputs["next_state"],
            observation,
        ),
        "explicit_action_changes_next_state": _delta_check(
            _tensor(block_outputs["next_state"]),
            _tensor(wait_outputs["next_state"]),
            required_min=config.min_next_state_delta,
            source="next_state",
        ),
        "explicit_action_changes_prediction_error": _delta_check(
            _tensor(block_outputs["prediction_error"]),
            _tensor(wait_outputs["prediction_error"]),
            required_min=config.min_prediction_error_delta,
            source="prediction_error",
        ),
        "prediction_error_conditioning_is_state_action": {
            "passed": model.config.prediction_error_conditioning == "state_action",
            "observed": model.config.prediction_error_conditioning,
            "required": "state_action",
            "source": "model_config",
        },
    }
    passed = all(
        isinstance(check, dict) and check.get("passed") is True
        for check in checks.values()
    )
    report: dict[str, object] = {
        "artifact_family": "reflexcore_v0_action_conditioned_world_audit",
        "passed": passed,
        "verdict": (
            "bounded_reflexcore_v0_action_conditioned_world_ready"
            if passed
            else "repair_reflexcore_v0_action_conditioned_world"
        ),
        "config": _json_config(config),
        "observed_summary": {
            "wait_next_state": _last_vector(wait_outputs["next_state"]),
            "block_next_state": _last_vector(block_outputs["next_state"]),
            "wait_prediction_error": _last_scalar(wait_outputs["prediction_error"]),
            "block_prediction_error": _last_scalar(block_outputs["prediction_error"]),
        },
        "checks": checks,
        "claim_boundary": (
            "Audits the ReflexCore V0 action-conditioned next-state and "
            "prediction-error paths under a controlled probe. It verifies that "
            "typed motor action input can alter world-model outputs for the same "
            "observation; it does not claim GUI, free-shell, robotics, or "
            "production autonomy."
        ),
    }
    if config.output_json is not None:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def _configure_action_probe(
    model: ReflexCoreV0,
    config: ReflexCoreActionConditionedWorldAuditConfig,
) -> None:
    with torch.no_grad():
        model.action_embedding.weight.zero_()
        model.action_embedding.weight[action_to_index(ActionType.BLOCK), 0] = (
            config.action_signal
        )
        first_next_state = model.next_state_head[0]
        final_next_state = model.next_state_head[-1]
        first_next_state.weight.zero_()
        first_next_state.bias.zero_()
        first_next_state.weight[0, model.config.hidden_dim] = 1.0
        final_next_state.weight.zero_()
        final_next_state.bias.zero_()
        final_next_state.weight[0, 0] = 1.0
        model.prediction_error_head.weight.zero_()
        model.prediction_error_head.bias.fill_(-config.action_signal)
        model.prediction_error_head.weight[0, model.config.hidden_dim] = (
            config.action_signal
        )


def _action_tensor(action: ActionType) -> torch.Tensor:
    return torch.tensor([[action_to_index(action)]], dtype=torch.long)


def _copy_baseline_check(
    next_state: object,
    observation: torch.Tensor,
) -> dict[str, object]:
    tensor = _tensor(next_state)
    max_abs_error = float((tensor - observation).abs().max().item())
    return {
        "passed": max_abs_error <= 1e-6,
        "observed": {"max_abs_error": max_abs_error},
        "required": "default next_state equals current observation before learned delta",
        "source": "next_state_head_initialization",
    }


def _delta_check(
    high: torch.Tensor,
    low: torch.Tensor,
    *,
    required_min: float,
    source: str,
) -> dict[str, object]:
    observed_delta = float((high - low).abs().max().item())
    return {
        "passed": observed_delta >= required_min,
        "observed": {"max_abs_delta": observed_delta},
        "required_min": required_min,
        "source": source,
    }


def _tensor(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("expected tensor output")
    return value


def _last_vector(value: object) -> list[float]:
    tensor = _tensor(value)
    return [float(item) for item in tensor[0, -1].detach().cpu().tolist()]


def _last_scalar(value: object) -> float:
    tensor = _tensor(value)
    return float(tensor[0, -1].detach().cpu().reshape(-1)[0].item())


def _json_config(
    config: ReflexCoreActionConditionedWorldAuditConfig,
) -> dict[str, object]:
    payload = asdict(config)
    payload["output_json"] = str(config.output_json) if config.output_json else None
    return payload
