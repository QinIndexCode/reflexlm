from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from reflexlm.core.dataset import read_reflexcore_jsonl, tensors_for_example
from reflexlm.core.model import ReflexCoreV0
from reflexlm.core.schema import ReflexCoreTrainingExample


@dataclass(slots=True)
class ReflexCorePredictionErrorReportConfig:
    output_dir: Path
    dataset_path: Path
    device: str = "cpu"
    sequence_mode: bool = True
    max_text_tokens: int = 128
    min_relative_improvement: float = 0.0
    min_action_group_pass_rate: float = 0.0
    min_evaluable_constant_mae: float = 1e-4


def build_reflexcore_prediction_error_report(
    model: ReflexCoreV0,
    config: ReflexCorePredictionErrorReportConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    examples = read_reflexcore_jsonl(config.dataset_path)
    if not examples:
        raise ValueError("prediction-error report requires at least one example")
    rows = _prediction_error_rows(
        model,
        examples,
        device=config.device,
        sequence_mode=config.sequence_mode,
        max_text_tokens=config.max_text_tokens,
    )
    overall = _summarize_rows(rows, min_relative_improvement=config.min_relative_improvement)
    by_action = {
        action: _summarize_rows(
            action_rows,
            min_relative_improvement=config.min_relative_improvement,
            min_evaluable_constant_mae=config.min_evaluable_constant_mae,
        )
        for action, action_rows in sorted(_group_by(rows, "action").items())
    }
    evaluable_action_groups = [
        summary for summary in by_action.values() if summary["evaluable"] is True
    ]
    passed_action_groups = sum(
        1 for summary in evaluable_action_groups if summary["passed"] is True
    )
    action_group_pass_rate = passed_action_groups / max(len(evaluable_action_groups), 1)
    report: dict[str, object] = {
        "config": _json_config(config),
        "row_count": len(rows),
        "overall": overall,
        "by_action": by_action,
        "action_group_count": len(by_action),
        "action_group_evaluable_count": len(evaluable_action_groups),
        "action_group_pass_count": passed_action_groups,
        "action_group_pass_rate": action_group_pass_rate,
        "min_action_group_pass_rate": config.min_action_group_pass_rate,
        "passed": bool(
            overall["passed"] is True
            and action_group_pass_rate >= config.min_action_group_pass_rate
        ),
        "claim_boundary": (
            "Prediction-error diagnostics only evaluate bounded ReflexCore V0 "
            "terminal/process/filesystem/time datasets."
        ),
    }
    (config.output_dir / "prediction_error_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _prediction_error_rows(
    model: ReflexCoreV0,
    examples: list[ReflexCoreTrainingExample],
    *,
    device: str,
    sequence_mode: bool,
    max_text_tokens: int,
) -> list[dict[str, object]]:
    input_dim = len(examples[0].observation.vector)
    model.to(device)
    model.eval()
    rows: list[dict[str, object]] = []
    grouped: dict[str, list[ReflexCoreTrainingExample]] = defaultdict(list)
    for example in examples:
        grouped[example.episode_id].append(example)
    with torch.no_grad():
        for episode_id, episode_examples in sorted(grouped.items()):
            hidden: torch.Tensor | None = None
            for example in sorted(episode_examples, key=lambda item: item.t):
                tensors = tensors_for_example(
                    example,
                    max_text_tokens=max_text_tokens,
                    input_dim=input_dim,
                )
                vector = tensors["observation_vectors"].to(device).unsqueeze(0).unsqueeze(0)
                text = tensors["text_tokens"].to(device).unsqueeze(0).unsqueeze(0)
                action_indices = tensors["action_indices"].to(device).view(1, 1)
                outputs = model(
                    vector,
                    text,
                    action_indices=action_indices,
                    hidden=hidden if sequence_mode else None,
                )
                hidden_value = outputs.get("hidden")
                hidden = (
                    hidden_value
                    if sequence_mode and isinstance(hidden_value, torch.Tensor)
                    else None
                )
                prediction = outputs.get("prediction_error")
                if not isinstance(prediction, torch.Tensor):
                    raise RuntimeError("missing prediction_error tensor")
                predicted = float(prediction.reshape(-1)[0].detach().cpu().item())
                target = float(tensors["prediction_error_targets"].item())
                rows.append(
                    {
                        "episode_id": episode_id,
                        "t": example.t,
                        "action": example.action.type.value,
                        "prediction": predicted,
                        "target": target,
                        "absolute_error": abs(predicted - target),
                    }
                )
    return rows


def _summarize_rows(
    rows: list[dict[str, object]],
    *,
    min_relative_improvement: float,
    min_evaluable_constant_mae: float = 0.0,
) -> dict[str, object]:
    targets = [_float(row["target"]) for row in rows]
    predictions = [_float(row["prediction"]) for row in rows]
    target_mean = mean(targets)
    model_mae = mean(abs(prediction - target) for prediction, target in zip(predictions, targets, strict=True))
    constant_mean_mae = mean(abs(target - target_mean) for target in targets)
    evaluable = constant_mean_mae > min_evaluable_constant_mae
    relative_improvement = (
        (constant_mean_mae - model_mae) / constant_mean_mae
        if evaluable
        else None
    )
    passed = (
        isinstance(relative_improvement, float)
        and model_mae < constant_mean_mae
        and relative_improvement >= min_relative_improvement
        if evaluable
        else None
    )
    return {
        "count": len(rows),
        "target_mean": target_mean,
        "target_min": min(targets),
        "target_max": max(targets),
        "prediction_mean": mean(predictions),
        "model_mae": model_mae,
        "constant_mean_mae": constant_mean_mae,
        "evaluable": evaluable,
        "min_evaluable_constant_mae": min_evaluable_constant_mae,
        "relative_improvement": relative_improvement,
        "min_relative_improvement": min_relative_improvement,
        "passed": passed,
    }


def _group_by(rows: list[dict[str, object]], key: str) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return dict(grouped)


def _float(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"expected numeric value, got {type(value).__name__}")
    return float(value)


def _json_config(config: ReflexCorePredictionErrorReportConfig) -> dict[str, object]:
    payload: dict[str, Any] = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["dataset_path"] = str(config.dataset_path)
    return payload
