from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from reflexlm.core.real_sandbox_adaptation import (
    ReflexCoreRealSandboxAdaptationConfig,
    run_reflexcore_real_sandbox_adaptation,
)


@dataclass(slots=True)
class ReflexCoreRealSandboxAdaptationMatrixConfig:
    output_dir: Path
    model_config_path: Path
    seeds: tuple[int, ...] = (13, 17, 23)
    profile: str = "default"
    eval_profile: str | None = None
    episodes_per_task: int = 12
    split_strategy: str = "scenario_holdout"
    vocab_size: int = 4096
    hash_bins: int = 256
    max_text_tokens: int = 128
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
    real_sandbox_variants: int = 12
    real_sandbox_start_variant: int = 1
    real_sandbox_max_steps: int = 4
    real_sandbox_required_baseline: str | None = "prompt_only_heuristic"
    real_sandbox_live_observation: bool = False
    require_synthetic_gate: bool = True
    synthetic_repeat: int = 1
    real_sandbox_repeat: int = 1
    min_pass_rate: float = 1.0
    min_offline_margin: float = 0.0
    min_closed_loop_margin: float = 0.0
    min_real_sandbox_margin: float = 0.0


def run_reflexcore_real_sandbox_adaptation_matrix(
    config: ReflexCoreRealSandboxAdaptationMatrixConfig,
) -> dict[str, object]:
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if not 0.0 <= config.min_pass_rate <= 1.0:
        raise ValueError("min_pass_rate must be between 0 and 1")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for seed in config.seeds:
        report = run_reflexcore_real_sandbox_adaptation(
            ReflexCoreRealSandboxAdaptationConfig(
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
                real_sandbox_variants=config.real_sandbox_variants,
                real_sandbox_start_variant=config.real_sandbox_start_variant,
                real_sandbox_max_steps=config.real_sandbox_max_steps,
                real_sandbox_required_baseline=config.real_sandbox_required_baseline,
                real_sandbox_live_observation=config.real_sandbox_live_observation,
                require_synthetic_gate=config.require_synthetic_gate,
                synthetic_repeat=config.synthetic_repeat,
                real_sandbox_repeat=config.real_sandbox_repeat,
            )
        )
        runs.append(_summarize_run(seed, report))
    pass_count = sum(1 for run in runs if run["passed"] is True)
    pass_rate = pass_count / len(runs)
    aggregates = _aggregate_runs(runs)
    margin_gate = _margin_gate(config, aggregates)
    report: dict[str, object] = {
        "config": _json_config(config),
        "runs": runs,
        "aggregates": aggregates,
        "margin_gate": margin_gate,
        "pass_count": pass_count,
        "run_count": len(runs),
        "pass_rate": pass_rate,
        "min_pass_rate": config.min_pass_rate,
        "passed": bool(pass_rate >= config.min_pass_rate and margin_gate["passed"]),
        "claim_boundary": (
            "Cross-seed real-sandbox adaptation only supports bounded "
            "terminal/process/filesystem/time ReflexCore V0 behavior with "
            "typed motor heads and allowlisted RUN_COMMAND execution."
        ),
    }
    (config.output_dir / "real_sandbox_adaptation_matrix_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _summarize_run(seed: int, report: dict[str, object]) -> dict[str, object]:
    offline = _dict(report.get("offline"))
    closed_loop = _dict(report.get("closed_loop"))
    real_sandbox = _dict(report.get("real_sandbox"))
    train = _dict(report.get("train"))
    mixture = _dict(report.get("dataset_mixture"))
    parameter_gate = _dict(report.get("parameter_gate"))
    offline_model = _dict(_dict(offline.get("model")).get("safety_gated"))
    offline_baselines = _dict(offline.get("baselines"))
    offline_baseline = _dict(offline_baselines.get(_required_baseline(report)))
    closed_model = _dict(closed_loop.get("model"))
    closed_baselines = _dict(closed_loop.get("baselines"))
    closed_baseline = _dict(closed_baselines.get(_required_baseline(report)))
    real_model = _dict(real_sandbox.get("model"))
    real_acceptance = _dict(real_sandbox.get("acceptance"))
    real_live_gate = _dict(report.get("real_sandbox_live_gate"))
    offline_margin = _margin(
        offline_model.get("action_accuracy"),
        offline_baseline.get("action_accuracy"),
    )
    closed_margin = _margin(
        closed_model.get("success_rate"),
        closed_baseline.get("success_rate"),
    )
    real_margin = _margin(
        real_model.get("success_rate"),
        real_acceptance.get("baseline_success_rate"),
    )
    return {
        "seed": seed,
        "passed": bool(report.get("passed")),
        "dataset_hash": train.get("dataset_hash"),
        "model_hash": train.get("model_hash"),
        "parameter_count": parameter_gate.get("parameter_count"),
        "mixed_train_hash": _dict(mixture.get("mixed_train")).get("hash"),
        "mixed_train_examples": _dict(mixture.get("mixed_train")).get("example_count"),
        "synthetic_train_examples": _dict(mixture.get("synthetic_train")).get("example_count"),
        "real_sandbox_train_examples": _dict(mixture.get("real_sandbox_train")).get("example_count"),
        "synthetic_weighted_examples": _dict(
            mixture.get("synthetic_train")
        ).get("weighted_example_count"),
        "real_sandbox_weighted_examples": _dict(
            mixture.get("real_sandbox_train")
        ).get("weighted_example_count"),
        "offline_action_accuracy": offline_model.get("action_accuracy"),
        "offline_baseline_action_accuracy": offline_baseline.get("action_accuracy"),
        "offline_action_margin": offline_margin,
        "closed_loop_success_rate": closed_model.get("success_rate"),
        "closed_loop_baseline_success_rate": closed_baseline.get("success_rate"),
        "closed_loop_margin": closed_margin,
        "real_sandbox_success_rate": real_model.get("success_rate"),
        "real_sandbox_baseline_success_rate": real_acceptance.get("baseline_success_rate"),
        "real_sandbox_margin": real_margin,
        "synthetic_gate_passed": _dict(report.get("synthetic_gate")).get("passed"),
        "real_sandbox_gate_passed": real_sandbox.get("passed"),
        "real_sandbox_live_gate_passed": real_live_gate.get("passed"),
        "real_sandbox_live_observation": real_sandbox.get("live_observation"),
        "real_sandbox_live_episode_count": real_model.get(
            "live_observation_episode_count"
        ),
        "real_sandbox_runtime_observation_steps": real_model.get(
            "runtime_observation_steps"
        ),
        "real_sandbox_changed_file_observation_steps": real_model.get(
            "changed_file_observation_steps"
        ),
        "real_sandbox_terminal_observation_steps": real_model.get(
            "terminal_observation_steps"
        ),
        "real_sandbox_observed_prediction_error_examples": real_model.get(
            "observed_prediction_error_examples"
        ),
        "real_sandbox_observed_prediction_error_mean": real_model.get(
            "observed_prediction_error_mean"
        ),
        "real_sandbox_observed_prediction_error_max": real_model.get(
            "observed_prediction_error_max"
        ),
        "world_model_relative_improvement": _dict(
            offline.get("world_model_acceptance")
        ).get("relative_improvement"),
        "prediction_error_relative_improvement": _dict(
            offline.get("prediction_error_acceptance")
        ).get("relative_improvement"),
    }


def _aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    metrics = [
        "parameter_count",
        "mixed_train_examples",
        "synthetic_train_examples",
        "real_sandbox_train_examples",
        "synthetic_weighted_examples",
        "real_sandbox_weighted_examples",
        "offline_action_accuracy",
        "offline_baseline_action_accuracy",
        "offline_action_margin",
        "closed_loop_success_rate",
        "closed_loop_baseline_success_rate",
        "closed_loop_margin",
        "real_sandbox_success_rate",
        "real_sandbox_baseline_success_rate",
        "real_sandbox_margin",
        "real_sandbox_live_episode_count",
        "real_sandbox_runtime_observation_steps",
        "real_sandbox_changed_file_observation_steps",
        "real_sandbox_terminal_observation_steps",
        "real_sandbox_observed_prediction_error_examples",
        "real_sandbox_observed_prediction_error_mean",
        "real_sandbox_observed_prediction_error_max",
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


def _margin_gate(
    config: ReflexCoreRealSandboxAdaptationMatrixConfig,
    aggregates: dict[str, object],
) -> dict[str, object]:
    checks = {
        "offline_action_margin": config.min_offline_margin,
        "closed_loop_margin": config.min_closed_loop_margin,
        "real_sandbox_margin": config.min_real_sandbox_margin,
    }
    details: dict[str, object] = {}
    passed = True
    for metric, threshold in checks.items():
        observed = _dict(aggregates.get(metric)).get("min")
        metric_passed = isinstance(observed, int | float) and float(observed) >= threshold
        details[metric] = {
            "observed_min": observed,
            "required_min": threshold,
            "passed": metric_passed,
        }
        passed = passed and metric_passed
    return {"passed": passed, "details": details}


def _margin(value: object, baseline: object) -> float | None:
    if isinstance(value, int | float) and isinstance(baseline, int | float):
        return float(value) - float(baseline)
    return None


def _required_baseline(report: dict[str, object]) -> str:
    config = _dict(report.get("config"))
    baseline = config.get("required_baseline")
    return baseline if isinstance(baseline, str) else "prompt_only_heuristic"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_config(config: ReflexCoreRealSandboxAdaptationMatrixConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["model_config_path"] = str(config.model_config_path)
    payload["seeds"] = list(config.seeds)
    return payload
