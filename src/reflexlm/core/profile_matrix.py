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
from reflexlm.core.dataset import read_reflexcore_jsonl
from reflexlm.core.evaluation import (
    acceptance_against_baselines,
    evaluate_baseline_policies,
    evaluate_reflexcore_model,
    prediction_error_acceptance,
    world_model_acceptance,
)
from reflexlm.core.experiment import _load_model
from reflexlm.core.stability import _aggregate_runs, _summarize_run
from reflexlm.core.training import train_reflexcore_v0


@dataclass(slots=True)
class ReflexCoreProfileMatrixConfig:
    output_dir: Path
    model_config_path: Path
    seeds: tuple[int, ...] = (13, 17, 23)
    profile: str = "default"
    eval_profiles: tuple[str, ...] = ("default", "hard", "wide_ood")
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
    min_profile_pass_rate: float = 1.0


def run_reflexcore_profile_matrix(
    config: ReflexCoreProfileMatrixConfig,
) -> dict[str, object]:
    if not config.eval_profiles:
        raise ValueError("at least one eval profile is required")
    if not 0.0 <= config.min_profile_pass_rate <= 1.0:
        raise ValueError("min_profile_pass_rate must be between 0 and 1")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    profile_runs: dict[str, list[dict[str, object]]] = {
        eval_profile: [] for eval_profile in config.eval_profiles
    }
    train_runs: list[dict[str, object]] = []
    for seed in config.seeds:
        train_context = _train_seed_model(
            config,
            seed=seed,
            seed_root=config.output_dir / f"seed_{seed}",
        )
        train_runs.append(_summarize_train_run(seed, train_context["train"]))
        for eval_profile in config.eval_profiles:
            report = _evaluate_seed_profile(
                config,
                seed=seed,
                eval_profile=eval_profile,
                train_context=train_context,
                output_dir=(
                    config.output_dir
                    / _profile_dir_name(eval_profile)
                    / f"seed_{seed}"
                ),
            )
            profile_runs[eval_profile].append(_summarize_run(seed, report))
    profile_reports = [
        _write_profile_stability_report(config, eval_profile, profile_runs[eval_profile])
        for eval_profile in config.eval_profiles
    ]
    profiles = [
        _summarize_profile(eval_profile, stability)
        for eval_profile, stability in zip(config.eval_profiles, profile_reports, strict=True)
    ]
    passed_profile_count = sum(1 for profile in profiles if profile["passed"] is True)
    profile_pass_rate = passed_profile_count / len(profiles)
    report: dict[str, object] = {
        "config": _json_config(config),
        "train_runs": train_runs,
        "training_reuse": {
            "enabled": True,
            "train_run_count": len(train_runs),
            "profile_eval_count": len(config.eval_profiles),
        },
        "profiles": profiles,
        "aggregates": _aggregate_profiles(profiles),
        "profile_count": len(profiles),
        "passed_profile_count": passed_profile_count,
        "profile_pass_rate": profile_pass_rate,
        "min_profile_pass_rate": config.min_profile_pass_rate,
        "passed": bool(profile_pass_rate >= config.min_profile_pass_rate),
        "claim_boundary": (
            "Profile-matrix stability only supports bounded terminal/process/"
            "filesystem/time ReflexCore V0 behavior under the configured "
            "training and evaluation profiles."
        ),
    }
    (config.output_dir / "profile_matrix_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _train_seed_model(
    config: ReflexCoreProfileMatrixConfig,
    *,
    seed: int,
    seed_root: Path,
) -> dict[str, object]:
    benchmark_dir = seed_root / "benchmark"
    train_dir = seed_root / "train"
    benchmark_manifest = build_reflexcore_benchmark(
        ReflexCoreBenchmarkConfig(
            output_dir=benchmark_dir,
            profile=config.profile,
            episodes_per_task=config.episodes_per_task,
            split_strategy=config.split_strategy,
            seed=seed,
            hash_bins=config.hash_bins,
            vocab_size=config.vocab_size,
            max_text_tokens=config.max_text_tokens,
        )
    )
    train_summary = train_reflexcore_v0(
        dataset_path=benchmark_dir / "reflexcore" / "train.jsonl",
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
    model = _load_model(Path(str(train_summary["checkpoint"])), device=config.device)
    return {
        "benchmark": benchmark_manifest,
        "benchmark_dir": benchmark_dir,
        "train": train_summary,
        "model": model,
    }


def _evaluate_seed_profile(
    config: ReflexCoreProfileMatrixConfig,
    *,
    seed: int,
    eval_profile: str,
    train_context: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_benchmark_dir = Path(str(train_context["benchmark_dir"]))
    eval_benchmark_manifest: dict[str, object] | None = None
    eval_benchmark_dir = train_benchmark_dir
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
    parameter_gate = _parameter_gate(
        int(_dict(train_context["train"])["parameter_count"]),
        min_parameters=config.min_parameters,
        max_parameters=config.max_parameters,
    )
    report: dict[str, object] = {
        "config": _json_config(config)
        | {
            "seed": seed,
            "eval_profile": eval_profile,
            "output_dir": str(output_dir),
        },
        "benchmark": train_context["benchmark"],
        "eval_benchmark": eval_benchmark_manifest,
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
        "parameter_gate": parameter_gate,
        "passed": bool(
            offline_acceptance["passed"]
            and closed_loop_acceptance["passed"]
            and parameter_gate["passed"]
            and (world_acceptance["passed"] or not config.require_world_model_improvement)
            and (
                prediction_error_gate["passed"]
                or not config.require_prediction_error_improvement
            )
        ),
        "claim_boundary": (
            "This report supports only bounded terminal/process/filesystem/time "
            "sensory-motor behavior with allowlisted RUN_COMMAND actions."
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


def _write_profile_stability_report(
    config: ReflexCoreProfileMatrixConfig,
    eval_profile: str,
    runs: list[dict[str, object]],
) -> dict[str, object]:
    profile_dir = config.output_dir / _profile_dir_name(eval_profile)
    pass_count = sum(1 for run in runs if run["passed"] is True)
    pass_rate = pass_count / len(runs)
    summary: dict[str, object] = {
        "config": _json_config(config)
        | {
            "output_dir": str(profile_dir),
            "eval_profile": eval_profile,
        },
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
        "training_reuse": {
            "enabled": True,
            "trained_profile": config.profile,
        },
    }
    (profile_dir / "stability_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _parameter_gate(
    parameter_count: int,
    *,
    min_parameters: int | None,
    max_parameters: int | None,
) -> dict[str, object]:
    passed = True
    if min_parameters is not None:
        passed = passed and parameter_count >= min_parameters
    if max_parameters is not None:
        passed = passed and parameter_count <= max_parameters
    return {
        "parameter_count": parameter_count,
        "min_parameters": min_parameters,
        "max_parameters": max_parameters,
        "passed": passed,
    }


def _summarize_train_run(seed: int, train_summary: object) -> dict[str, object]:
    train = _dict(train_summary)
    return {
        "seed": seed,
        "dataset_hash": train.get("dataset_hash"),
        "model_hash": train.get("model_hash"),
        "parameter_count": train.get("parameter_count"),
    }


def _summarize_profile(
    eval_profile: str,
    stability: dict[str, object],
) -> dict[str, object]:
    config = _dict(stability.get("config"))
    return {
        "profile": eval_profile,
        "train_profile": config.get("profile"),
        "eval_profile": eval_profile,
        "is_transfer": config.get("profile") != eval_profile,
        "passed": bool(stability.get("passed")),
        "pass_rate": stability.get("pass_rate"),
        "run_count": stability.get("run_count"),
        "pass_count": stability.get("pass_count"),
        "aggregates": stability.get("aggregates"),
    }


def _aggregate_profiles(profiles: list[dict[str, object]]) -> dict[str, object]:
    return {
        "pass_rate": _aggregate_metric(profiles, "pass_rate"),
        "offline_action_accuracy_min": _aggregate_nested_metric(
            profiles,
            ("aggregates", "offline_action_accuracy", "min"),
        ),
        "closed_loop_success_rate_min": _aggregate_nested_metric(
            profiles,
            ("aggregates", "closed_loop_success_rate", "min"),
        ),
        "world_model_relative_improvement_min": _aggregate_nested_metric(
            profiles,
            ("aggregates", "world_model_relative_improvement", "min"),
        ),
        "prediction_error_relative_improvement_min": _aggregate_nested_metric(
            profiles,
            ("aggregates", "prediction_error_relative_improvement", "min"),
        ),
    }


def _aggregate_metric(
    rows: list[dict[str, object]],
    metric: str,
) -> dict[str, float | None]:
    values = [row.get(metric) for row in rows]
    return _numeric_summary(values)


def _aggregate_nested_metric(
    rows: list[dict[str, object]],
    path: tuple[str, ...],
) -> dict[str, float | None]:
    values: list[object] = []
    for row in rows:
        value: object = row
        for key in path:
            value = _dict(value).get(key)
        values.append(value)
    return _numeric_summary(values)


def _numeric_summary(values: list[object]) -> dict[str, float | None]:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(numeric), "mean": mean(numeric), "max": max(numeric)}


def _profile_dir_name(profile: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile).strip("._")
    return f"eval_{safe or 'profile'}"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _json_config(config: ReflexCoreProfileMatrixConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["model_config_path"] = str(config.model_config_path)
    payload["seeds"] = list(config.seeds)
    payload["eval_profiles"] = list(config.eval_profiles)
    return payload
