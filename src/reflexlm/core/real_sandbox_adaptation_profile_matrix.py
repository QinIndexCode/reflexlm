from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from reflexlm.core.benchmark import (
    ReflexCoreBenchmarkConfig,
    build_reflexcore_benchmark,
)
from reflexlm.core.closed_loop import (
    closed_loop_acceptance_against_baselines,
    evaluate_closed_loop_baselines,
    evaluate_reflexcore_closed_loop,
)
from reflexlm.core.dataset import read_reflexcore_jsonl, write_reflexcore_jsonl
from reflexlm.core.evaluation import (
    acceptance_against_baselines,
    evaluate_baseline_policies,
    evaluate_reflexcore_model,
    prediction_error_acceptance,
    world_model_acceptance,
)
from reflexlm.core.experiment import _load_model
from reflexlm.core.real_sandbox_adaptation import (
    _dataset_mixture_summary,
    _parameter_gate,
    _real_sandbox_live_gate,
)
from reflexlm.core.sandbox_benchmark import (
    RealSandboxEvalConfig,
    build_real_sandbox_oracle_dataset,
    evaluate_reflexcore_real_sandbox,
)
from reflexlm.core.training import train_reflexcore_v0


@dataclass(slots=True)
class ReflexCoreRealSandboxAdaptationProfileMatrixConfig:
    output_dir: Path
    model_config_path: Path
    seeds: tuple[int, ...] = (13, 17, 23)
    profile: str = "default"
    eval_profiles: tuple[str, ...] = ("default", "hard", "wide_ood")
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
    min_profile_pass_rate: float = 1.0
    min_offline_margin: float = 0.0
    min_closed_loop_margin: float = 0.0
    min_real_sandbox_margin: float = 0.0
    min_real_sandbox_success_rate: float = 0.0


def run_reflexcore_real_sandbox_adaptation_profile_matrix(
    config: ReflexCoreRealSandboxAdaptationProfileMatrixConfig,
) -> dict[str, object]:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    seed_runs: list[dict[str, object]] = []
    train_runs: list[dict[str, object]] = []
    profile_runs: list[dict[str, object]] = []

    for seed in config.seeds:
        seed_root = config.output_dir / f"seed_{seed}"
        train_context = _train_seed_model(config, seed=seed, seed_root=seed_root)
        train_summary = _summarize_train_run(seed, train_context["train"])
        train_runs.append(train_summary)
        seed_profile_runs: list[dict[str, object]] = []
        for eval_profile in config.eval_profiles:
            profile_report = _evaluate_seed_profile(
                config,
                seed=seed,
                eval_profile=eval_profile,
                train_context=train_context,
                output_dir=seed_root / _profile_dir_name(eval_profile),
            )
            profile_summary = _summarize_profile_run(seed, eval_profile, profile_report)
            seed_profile_runs.append(profile_summary)
            profile_runs.append(profile_summary)
        seed_runs.append(
            _summarize_seed_run(
                config,
                seed=seed,
                train_summary=train_summary,
                train_context=train_context,
                profile_runs=seed_profile_runs,
            )
        )

    pass_count = sum(1 for run in seed_runs if run["passed"] is True)
    pass_rate = pass_count / len(seed_runs)
    profile_pass_count = sum(1 for run in profile_runs if run["passed"] is True)
    profile_pass_rate = profile_pass_count / len(profile_runs)
    aggregates = _aggregate_profile_runs(profile_runs)
    margin_gate = _margin_gate(config, aggregates)
    improvement_gate = _improvement_gate(config, aggregates)
    success_gate = _success_gate(config, aggregates)
    report: dict[str, object] = {
        "config": _json_config(config),
        "runs": seed_runs,
        "train_runs": train_runs,
        "profile_runs": profile_runs,
        "aggregates": aggregates,
        "margin_gate": margin_gate,
        "improvement_gate": improvement_gate,
        "success_gate": success_gate,
        "training_reuse": {
            "enabled": True,
            "scope": "one_train_run_per_seed_reused_across_eval_profiles",
            "train_run_count": len(train_runs),
            "profile_eval_count": len(profile_runs),
            "eval_profiles_per_train": len(config.eval_profiles),
            "real_sandbox_live_observation": config.real_sandbox_live_observation,
        },
        "pass_count": pass_count,
        "run_count": len(seed_runs),
        "pass_rate": pass_rate,
        "min_pass_rate": config.min_pass_rate,
        "profile_pass_count": profile_pass_count,
        "profile_eval_count": len(profile_runs),
        "profile_pass_rate": profile_pass_rate,
        "min_profile_pass_rate": config.min_profile_pass_rate,
        "passed": bool(
            pass_rate >= config.min_pass_rate
            and profile_pass_rate >= config.min_profile_pass_rate
            and margin_gate["passed"]
            and improvement_gate["passed"]
            and success_gate["passed"]
        ),
        "claim_boundary": (
            "Train-once real-sandbox profile transfer only supports bounded "
            "terminal/process/filesystem/time ReflexCore V0 behavior with "
            "typed motor heads and allowlisted RUN_COMMAND execution."
        ),
    }
    (config.output_dir / "real_sandbox_adaptation_profile_matrix_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _validate_config(
    config: ReflexCoreRealSandboxAdaptationProfileMatrixConfig,
) -> None:
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if not config.eval_profiles:
        raise ValueError("at least one eval profile is required")
    if len(set(config.eval_profiles)) != len(config.eval_profiles):
        raise ValueError("eval profiles must be unique")
    if config.real_sandbox_variants <= 0:
        raise ValueError("real_sandbox_variants must be positive")
    if config.synthetic_repeat <= 0:
        raise ValueError("synthetic_repeat must be positive")
    if config.real_sandbox_repeat <= 0:
        raise ValueError("real_sandbox_repeat must be positive")
    if not 0.0 <= config.min_pass_rate <= 1.0:
        raise ValueError("min_pass_rate must be between 0 and 1")
    if not 0.0 <= config.min_profile_pass_rate <= 1.0:
        raise ValueError("min_profile_pass_rate must be between 0 and 1")
    if not 0.0 <= config.min_real_sandbox_success_rate <= 1.0:
        raise ValueError("min_real_sandbox_success_rate must be between 0 and 1")


def _train_seed_model(
    config: ReflexCoreRealSandboxAdaptationProfileMatrixConfig,
    *,
    seed: int,
    seed_root: Path,
) -> dict[str, object]:
    synthetic_dir = seed_root / "synthetic_benchmark"
    real_dataset_dir = seed_root / "real_sandbox_train"
    train_dir = seed_root / "train"
    mixed_train_path = seed_root / "mixed_train.jsonl"
    benchmark_manifest = build_reflexcore_benchmark(
        ReflexCoreBenchmarkConfig(
            output_dir=synthetic_dir,
            profile=config.profile,
            episodes_per_task=config.episodes_per_task,
            split_strategy=config.split_strategy,
            seed=seed,
            hash_bins=config.hash_bins,
            vocab_size=config.vocab_size,
            max_text_tokens=config.max_text_tokens,
        )
    )
    real_dataset_path = real_dataset_dir / "train.jsonl"
    real_dataset_summary = build_real_sandbox_oracle_dataset(
        output_path=real_dataset_path,
        work_dir=real_dataset_dir / "work",
        variants=config.real_sandbox_variants,
        start_variant=config.real_sandbox_start_variant,
        vocab_size=config.vocab_size,
        max_text_tokens=config.max_text_tokens,
    )
    synthetic_examples = read_reflexcore_jsonl(synthetic_dir / "reflexcore" / "train.jsonl")
    real_examples = read_reflexcore_jsonl(real_dataset_path)
    mixed_examples = (
        synthetic_examples * config.synthetic_repeat
        + real_examples * config.real_sandbox_repeat
    )
    write_reflexcore_jsonl(mixed_train_path, mixed_examples)
    train_summary = train_reflexcore_v0(
        dataset_path=mixed_train_path,
        config_path=config.model_config_path,
        output_dir=train_dir,
        epochs=config.train_epochs,
        batch_size=config.train_batch_size,
        learning_rate=config.learning_rate,
        device=config.device,
        seed=seed,
        sequence_mode=config.sequence_mode,
        max_sequence_len=config.max_sequence_len,
    )
    model = _load_model(Path(str(_dict(train_summary)["checkpoint"])), device=config.device)
    real_sandbox_report = evaluate_reflexcore_real_sandbox(
        model,
        config=RealSandboxEvalConfig(
            output_dir=seed_root / "real_sandbox_eval",
            max_steps=config.real_sandbox_max_steps,
            compare_baselines=True,
            require_beats_baseline=config.real_sandbox_required_baseline,
            live_observation=config.real_sandbox_live_observation,
            max_text_tokens=config.max_text_tokens,
        ),
    )
    real_sandbox_live_gate = _real_sandbox_live_gate(real_sandbox_report)
    parameter_gate = _parameter_gate(
        int(_dict(train_summary)["parameter_count"]),
        min_parameters=config.min_parameters,
        max_parameters=config.max_parameters,
    )
    dataset_mixture = _dataset_mixture_summary(
        synthetic_path=synthetic_dir / "reflexcore" / "train.jsonl",
        synthetic_examples=synthetic_examples,
        real_path=real_dataset_path,
        real_examples=real_examples,
        real_summary=real_dataset_summary,
        mixed_path=mixed_train_path,
        mixed_examples=mixed_examples,
        synthetic_repeat=config.synthetic_repeat,
        real_sandbox_repeat=config.real_sandbox_repeat,
    )
    return {
        "benchmark": benchmark_manifest,
        "synthetic_dir": synthetic_dir,
        "train": train_summary,
        "model": model,
        "real_sandbox": real_sandbox_report,
        "real_sandbox_live_gate": real_sandbox_live_gate,
        "parameter_gate": parameter_gate,
        "dataset_mixture": dataset_mixture,
    }


def _evaluate_seed_profile(
    config: ReflexCoreRealSandboxAdaptationProfileMatrixConfig,
    *,
    seed: int,
    eval_profile: str,
    train_context: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_benchmark_manifest: dict[str, object] | None = None
    eval_benchmark_dir = Path(str(train_context["synthetic_dir"]))
    if eval_profile != config.profile:
        eval_benchmark_dir = output_dir / "eval_benchmark"
        eval_benchmark_manifest = build_reflexcore_benchmark(
            ReflexCoreBenchmarkConfig(
                output_dir=eval_benchmark_dir,
                profile=eval_profile,
                episodes_per_task=config.episodes_per_task,
                split_strategy=config.split_strategy,
                seed=seed,
                hash_bins=config.hash_bins,
                vocab_size=config.vocab_size,
                max_text_tokens=config.max_text_tokens,
            )
        )
    test_examples = read_reflexcore_jsonl(eval_benchmark_dir / "reflexcore" / "test.jsonl")
    offline_model = evaluate_reflexcore_model(
        train_context["model"],
        test_examples,
        batch_size=config.train_batch_size or 16,
        device=config.device,
        sequence_mode=bool(config.sequence_mode),
        max_sequence_len=config.max_sequence_len,
    )
    offline_baselines = evaluate_baseline_policies(test_examples)
    offline_acceptance = acceptance_against_baselines(
        offline_model,
        offline_baselines,
        required_baselines=[config.required_baseline],
    )
    world_acceptance = world_model_acceptance(
        offline_model,
        min_relative_improvement=config.min_world_model_relative_improvement,
    )
    prediction_error_gate = prediction_error_acceptance(
        offline_model,
        min_relative_improvement=config.min_prediction_error_relative_improvement,
    )
    closed_loop_model = evaluate_reflexcore_closed_loop(
        train_context["model"],
        profile=eval_profile,
        episodes_per_task=config.closed_loop_episodes_per_task,
        device=config.device,
    )
    closed_loop_baselines = evaluate_closed_loop_baselines(
        profile=eval_profile,
        episodes_per_task=config.closed_loop_episodes_per_task,
    )
    closed_loop_acceptance = closed_loop_acceptance_against_baselines(
        closed_loop_model,
        closed_loop_baselines,
        required_baselines=[config.required_baseline],
    )
    parameter_gate = _dict(train_context.get("parameter_gate"))
    synthetic_gate_passed = bool(
        offline_acceptance["passed"]
        and closed_loop_acceptance["passed"]
        and parameter_gate["passed"]
        and (world_acceptance["passed"] or not config.require_world_model_improvement)
        and (
            prediction_error_gate["passed"]
            or not config.require_prediction_error_improvement
        )
    )
    real_sandbox_report = _dict(train_context.get("real_sandbox"))
    real_sandbox_live_gate = _dict(train_context.get("real_sandbox_live_gate"))
    report: dict[str, object] = {
        "config": _json_config(config)
        | {
            "seed": seed,
            "eval_profile": eval_profile,
            "output_dir": str(output_dir),
        },
        "benchmark": train_context["benchmark"],
        "eval_benchmark": eval_benchmark_manifest,
        "dataset_mixture": train_context["dataset_mixture"],
        "train": train_context["train"],
        "offline": {
            "model": offline_model,
            "baselines": offline_baselines,
            "acceptance": offline_acceptance,
            "world_model_acceptance": world_acceptance,
            "prediction_error_acceptance": prediction_error_gate,
        },
        "closed_loop": {
            "model": closed_loop_model,
            "baselines": closed_loop_baselines,
            "acceptance": closed_loop_acceptance,
        },
        "real_sandbox": real_sandbox_report,
        "real_sandbox_live_gate": real_sandbox_live_gate,
        "parameter_gate": parameter_gate,
        "synthetic_gate": {
            "required": config.require_synthetic_gate,
            "passed": synthetic_gate_passed,
        },
        "passed": bool(
            real_sandbox_report.get("passed")
            and real_sandbox_live_gate.get("passed", True)
            and parameter_gate["passed"]
            and (synthetic_gate_passed or not config.require_synthetic_gate)
        ),
        "training_reuse": {
            "enabled": True,
            "seed": seed,
            "model_hash": _dict(train_context["train"]).get("model_hash"),
            "checkpoint": _dict(train_context["train"]).get("checkpoint"),
            "real_sandbox_evaluation_reused": True,
        },
        "claim_boundary": (
            "This profile-transfer report supports only bounded terminal/"
            "process/filesystem/time sensory-motor behavior with allowlisted "
            "RUN_COMMAND actions."
        ),
        "profile_transfer": {
            "train_profile": config.profile,
            "eval_profile": eval_profile,
            "is_transfer": eval_profile != config.profile,
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _summarize_seed_run(
    config: ReflexCoreRealSandboxAdaptationProfileMatrixConfig,
    *,
    seed: int,
    train_summary: dict[str, object],
    train_context: dict[str, object],
    profile_runs: list[dict[str, object]],
) -> dict[str, object]:
    profile_pass_count = sum(1 for run in profile_runs if run["passed"] is True)
    profile_pass_rate = profile_pass_count / len(profile_runs)
    real_sandbox = _dict(train_context.get("real_sandbox"))
    real_sandbox_live_gate = _dict(train_context.get("real_sandbox_live_gate"))
    parameter_gate = _dict(train_context.get("parameter_gate"))
    return {
        "seed": seed,
        "passed": bool(
            profile_pass_rate >= config.min_profile_pass_rate
            and real_sandbox.get("passed")
            and real_sandbox_live_gate.get("passed", True)
            and parameter_gate.get("passed")
        ),
        "model_hash": train_summary.get("model_hash"),
        "dataset_hash": train_summary.get("dataset_hash"),
        "parameter_count": train_summary.get("parameter_count"),
        "mixed_train_hash": _dict(_dict(train_context.get("dataset_mixture")).get("mixed_train")).get("hash"),
        "profile_pass_count": profile_pass_count,
        "profile_eval_count": len(profile_runs),
        "profile_pass_rate": profile_pass_rate,
        "real_sandbox_gate_passed": real_sandbox.get("passed"),
        "real_sandbox_live_gate_passed": real_sandbox_live_gate.get("passed"),
        "real_sandbox_live_observation": real_sandbox.get("live_observation"),
        "parameter_gate_passed": parameter_gate.get("passed"),
        "profiles": profile_runs,
    }


def _summarize_train_run(seed: int, train_summary: object) -> dict[str, object]:
    train = _dict(train_summary)
    return {
        "seed": seed,
        "dataset_hash": train.get("dataset_hash"),
        "model_hash": train.get("model_hash"),
        "parameter_count": train.get("parameter_count"),
        "checkpoint": train.get("checkpoint"),
    }


def _summarize_profile_run(
    seed: int,
    eval_profile: str,
    report: dict[str, object],
) -> dict[str, object]:
    offline = _dict(report.get("offline"))
    closed_loop = _dict(report.get("closed_loop"))
    real_sandbox = _dict(report.get("real_sandbox"))
    train = _dict(report.get("train"))
    mixture = _dict(report.get("dataset_mixture"))
    profile_transfer = _dict(report.get("profile_transfer"))
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
    return {
        "seed": seed,
        "train_profile": profile_transfer.get("train_profile"),
        "eval_profile": eval_profile,
        "is_transfer": profile_transfer.get("is_transfer"),
        "passed": bool(report.get("passed")),
        "dataset_hash": train.get("dataset_hash"),
        "model_hash": train.get("model_hash"),
        "parameter_count": parameter_gate.get("parameter_count"),
        "mixed_train_hash": _dict(mixture.get("mixed_train")).get("hash"),
        "mixed_train_examples": _dict(mixture.get("mixed_train")).get("example_count"),
        "synthetic_weighted_examples": _dict(
            mixture.get("synthetic_train")
        ).get("weighted_example_count"),
        "real_sandbox_weighted_examples": _dict(
            mixture.get("real_sandbox_train")
        ).get("weighted_example_count"),
        "offline_action_accuracy": offline_model.get("action_accuracy"),
        "offline_baseline_action_accuracy": offline_baseline.get("action_accuracy"),
        "offline_action_margin": _margin(
            offline_model.get("action_accuracy"),
            offline_baseline.get("action_accuracy"),
        ),
        "closed_loop_success_rate": closed_model.get("success_rate"),
        "closed_loop_baseline_success_rate": closed_baseline.get("success_rate"),
        "closed_loop_margin": _margin(
            closed_model.get("success_rate"),
            closed_baseline.get("success_rate"),
        ),
        "real_sandbox_success_rate": real_model.get("success_rate"),
        "real_sandbox_baseline_success_rate": real_acceptance.get("baseline_success_rate"),
        "real_sandbox_margin": _margin(
            real_model.get("success_rate"),
            real_acceptance.get("baseline_success_rate"),
        ),
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


def _aggregate_profile_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    metrics = [
        "parameter_count",
        "mixed_train_examples",
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


def _aggregate_metric(
    runs: list[dict[str, object]],
    metric: str,
) -> dict[str, float | None]:
    values = [run.get(metric) for run in runs]
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(numeric), "mean": mean(numeric), "max": max(numeric)}


def _margin_gate(
    config: ReflexCoreRealSandboxAdaptationProfileMatrixConfig,
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


def _improvement_gate(
    config: ReflexCoreRealSandboxAdaptationProfileMatrixConfig,
    aggregates: dict[str, object],
) -> dict[str, object]:
    checks: dict[str, float] = {}
    if config.require_world_model_improvement:
        checks["world_model_relative_improvement"] = (
            config.min_world_model_relative_improvement
        )
    if config.require_prediction_error_improvement:
        checks["prediction_error_relative_improvement"] = (
            config.min_prediction_error_relative_improvement
        )
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


def _success_gate(
    config: ReflexCoreRealSandboxAdaptationProfileMatrixConfig,
    aggregates: dict[str, object],
) -> dict[str, object]:
    observed = _dict(aggregates.get("real_sandbox_success_rate")).get("min")
    metric_passed = (
        isinstance(observed, int | float)
        and float(observed) >= config.min_real_sandbox_success_rate
    )
    return {
        "passed": metric_passed,
        "details": {
            "real_sandbox_success_rate": {
                "observed_min": observed,
                "required_min": config.min_real_sandbox_success_rate,
                "passed": metric_passed,
            }
        },
    }


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


def _profile_dir_name(profile: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile).strip("._")
    return f"eval_{safe or 'profile'}"


def _json_config(
    config: ReflexCoreRealSandboxAdaptationProfileMatrixConfig,
) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["model_config_path"] = str(config.model_config_path)
    payload["seeds"] = list(config.seeds)
    payload["eval_profiles"] = list(config.eval_profiles)
    return payload
