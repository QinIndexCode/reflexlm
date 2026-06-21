from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from reflexlm.core.experiment import ReflexCoreExperimentConfig, run_reflexcore_experiment


@dataclass(slots=True)
class ReflexCoreStabilityConfig:
    output_dir: Path
    model_config_path: Path
    seeds: tuple[int, ...] = (13, 17, 23)
    profile: str = "default"
    eval_profile: str | None = None
    episodes_per_task: int = 6
    split_strategy: str = "scenario_holdout"
    vocab_size: int = 4096
    hash_bins: int = 256
    max_text_tokens: int = 64
    train_epochs: int | None = None
    train_batch_size: int | None = None
    learning_rate: float | None = None
    device: str = "cpu"
    sequence_mode: bool | None = None
    max_sequence_len: int | None = None
    required_baseline: str = "prompt_only_heuristic"
    closed_loop_episodes_per_task: int = 2
    min_parameters: int | None = None
    max_parameters: int | None = None
    require_world_model_improvement: bool = True
    min_world_model_relative_improvement: float = 0.0
    require_prediction_error_improvement: bool = True
    min_prediction_error_relative_improvement: float = 0.0
    min_pass_rate: float = 1.0


def run_reflexcore_stability(config: ReflexCoreStabilityConfig) -> dict[str, object]:
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if not 0.0 <= config.min_pass_rate <= 1.0:
        raise ValueError("min_pass_rate must be between 0 and 1")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for seed in config.seeds:
        report = run_reflexcore_experiment(
            ReflexCoreExperimentConfig(
                output_dir=config.output_dir / f"seed_{seed}",
                model_config_path=config.model_config_path,
                profile=config.profile,
                eval_profile=config.eval_profile,
                episodes_per_task=config.episodes_per_task,
                split_strategy=config.split_strategy,
                seed=seed,
                vocab_size=config.vocab_size,
                hash_bins=config.hash_bins,
                max_text_tokens=config.max_text_tokens,
                train_epochs=config.train_epochs,
                train_batch_size=config.train_batch_size,
                learning_rate=config.learning_rate,
                device=config.device,
                sequence_mode=config.sequence_mode,
                max_sequence_len=config.max_sequence_len,
                required_baseline=config.required_baseline,
                closed_loop_episodes_per_task=config.closed_loop_episodes_per_task,
                min_parameters=config.min_parameters,
                max_parameters=config.max_parameters,
                require_world_model_improvement=config.require_world_model_improvement,
                min_world_model_relative_improvement=(
                    config.min_world_model_relative_improvement
                ),
                require_prediction_error_improvement=(
                    config.require_prediction_error_improvement
                ),
                min_prediction_error_relative_improvement=(
                    config.min_prediction_error_relative_improvement
                ),
            )
        )
        runs.append(_summarize_run(seed, report))
    pass_count = sum(1 for run in runs if run["passed"] is True)
    pass_rate = pass_count / len(runs)
    summary: dict[str, object] = {
        "config": _json_config(config),
        "runs": runs,
        "aggregates": _aggregate_runs(runs),
        "pass_count": pass_count,
        "run_count": len(runs),
        "pass_rate": pass_rate,
        "min_pass_rate": config.min_pass_rate,
        "passed": pass_rate >= config.min_pass_rate,
        "claim_boundary": (
            "Stability only supports bounded terminal/process/filesystem/time "
            "ReflexCore V0 behavior under the configured benchmark profile."
        ),
    }
    (config.output_dir / "stability_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _summarize_run(seed: int, report: dict[str, object]) -> dict[str, object]:
    offline = _dict(report.get("offline"))
    closed_loop = _dict(report.get("closed_loop"))
    train = _dict(report.get("train"))
    model = _dict(offline.get("model"))
    baselines = _dict(offline.get("baselines"))
    baseline = _dict(baselines.get("prompt_only_heuristic"))
    if "config" in report:
        required = _dict(report["config"]).get("required_baseline")
        if isinstance(required, str):
            baseline = _dict(baselines.get(required))
    closed_model = _dict(closed_loop.get("model"))
    closed_baselines = _dict(closed_loop.get("baselines"))
    closed_baseline = _dict(closed_baselines.get(_dict(report.get("config")).get("required_baseline")))
    world_gate = _dict(offline.get("world_model_acceptance"))
    pe_gate = _dict(offline.get("prediction_error_acceptance"))
    parameter_gate = _dict(report.get("parameter_gate"))
    safety_gated = _dict(model.get("safety_gated"))
    return {
        "seed": seed,
        "passed": bool(report.get("passed")),
        "dataset_hash": train.get("dataset_hash"),
        "model_hash": train.get("model_hash"),
        "parameter_count": parameter_gate.get("parameter_count"),
        "offline_action_accuracy": safety_gated.get("action_accuracy"),
        "offline_baseline_action_accuracy": baseline.get("action_accuracy"),
        "closed_loop_success_rate": closed_model.get("success_rate"),
        "closed_loop_baseline_success_rate": closed_baseline.get("success_rate"),
        "world_model_passed": world_gate.get("passed"),
        "world_model_relative_improvement": world_gate.get("relative_improvement"),
        "prediction_error_passed": pe_gate.get("passed"),
        "prediction_error_relative_improvement": pe_gate.get("relative_improvement"),
    }


def _aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    metrics = [
        "offline_action_accuracy",
        "offline_baseline_action_accuracy",
        "closed_loop_success_rate",
        "closed_loop_baseline_success_rate",
        "world_model_relative_improvement",
        "prediction_error_relative_improvement",
    ]
    return {metric: _aggregate_metric(runs, metric) for metric in metrics}


def _aggregate_metric(runs: list[dict[str, object]], metric: str) -> dict[str, float | None]:
    values = [run.get(metric) for run in runs]
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(numeric), "mean": mean(numeric), "max": max(numeric)}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_config(config: ReflexCoreStabilityConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["model_config_path"] = str(config.model_config_path)
    payload["seeds"] = list(config.seeds)
    return payload
