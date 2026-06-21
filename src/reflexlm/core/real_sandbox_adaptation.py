from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

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
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
from reflexlm.core.sandbox_benchmark import (
    RealSandboxEvalConfig,
    build_real_sandbox_oracle_dataset,
    evaluate_reflexcore_real_sandbox,
)
from reflexlm.core.schema import ReflexCoreTrainingExample, dataset_hash
from reflexlm.core.training import train_reflexcore_v0


@dataclass(slots=True)
class ReflexCoreRealSandboxAdaptationConfig:
    output_dir: Path
    model_config_path: Path
    profile: str = "default"
    eval_profile: str | None = None
    episodes_per_task: int = 6
    split_strategy: str = "scenario_holdout"
    seed: int = 13
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
    real_sandbox_variants: int = 12
    real_sandbox_start_variant: int = 1
    real_sandbox_max_steps: int = 4
    real_sandbox_required_baseline: str | None = "prompt_only_heuristic"
    real_sandbox_live_observation: bool = False
    require_synthetic_gate: bool = True
    synthetic_repeat: int = 1
    real_sandbox_repeat: int = 1


def run_reflexcore_real_sandbox_adaptation(
    config: ReflexCoreRealSandboxAdaptationConfig,
) -> dict[str, object]:
    if config.real_sandbox_variants <= 0:
        raise ValueError("real_sandbox_variants must be positive")
    if config.synthetic_repeat <= 0:
        raise ValueError("synthetic_repeat must be positive")
    if config.real_sandbox_repeat <= 0:
        raise ValueError("real_sandbox_repeat must be positive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir = config.output_dir / "synthetic_benchmark"
    real_dataset_dir = config.output_dir / "real_sandbox_train"
    train_dir = config.output_dir / "train"
    mixed_train_path = config.output_dir / "mixed_train.jsonl"

    benchmark_manifest = build_reflexcore_benchmark(
        ReflexCoreBenchmarkConfig(
            output_dir=synthetic_dir,
            profile=config.profile,
            episodes_per_task=config.episodes_per_task,
            split_strategy=config.split_strategy,
            seed=config.seed,
            hash_bins=config.hash_bins,
            vocab_size=config.vocab_size,
            max_text_tokens=config.max_text_tokens,
        )
    )
    eval_profile = config.eval_profile or config.profile
    eval_benchmark_manifest: dict[str, object] | None = None
    eval_benchmark_dir = synthetic_dir
    if eval_profile != config.profile:
        eval_benchmark_dir = config.output_dir / "synthetic_eval_benchmark"
        eval_benchmark_manifest = build_reflexcore_benchmark(
            ReflexCoreBenchmarkConfig(
                output_dir=eval_benchmark_dir,
                profile=eval_profile,
                episodes_per_task=config.episodes_per_task,
                split_strategy=config.split_strategy,
                seed=config.seed,
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
        seed=config.seed,
        sequence_mode=config.sequence_mode,
        max_sequence_len=config.max_sequence_len,
    )
    model = _load_model(Path(str(train_summary["checkpoint"])), device=config.device)

    test_examples = read_reflexcore_jsonl(eval_benchmark_dir / "reflexcore" / "test.jsonl")
    offline_model = evaluate_reflexcore_model(
        model,
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
        model,
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
        int(train_summary["parameter_count"]),
        min_parameters=config.min_parameters,
        max_parameters=config.max_parameters,
    )
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
    real_sandbox_report = evaluate_reflexcore_real_sandbox(
        model,
        config=RealSandboxEvalConfig(
            output_dir=config.output_dir / "real_sandbox_eval",
            max_steps=config.real_sandbox_max_steps,
            compare_baselines=True,
            require_beats_baseline=config.real_sandbox_required_baseline,
            live_observation=config.real_sandbox_live_observation,
            max_text_tokens=config.max_text_tokens,
        ),
    )
    real_gate_passed = bool(real_sandbox_report.get("passed"))
    live_gate = _real_sandbox_live_gate(real_sandbox_report)
    if config.real_sandbox_live_observation:
        real_gate_passed = bool(real_gate_passed and live_gate["passed"])
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
    report: dict[str, object] = {
        "config": _json_config(config),
        "benchmark": benchmark_manifest,
        "eval_benchmark": eval_benchmark_manifest,
        "dataset_mixture": dataset_mixture,
        "train": train_summary,
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
        "real_sandbox_live_gate": live_gate,
        "parameter_gate": parameter_gate,
        "synthetic_gate": {
            "required": config.require_synthetic_gate,
            "passed": synthetic_gate_passed,
        },
        "passed": bool(
            real_gate_passed
            and parameter_gate["passed"]
            and (synthetic_gate_passed or not config.require_synthetic_gate)
        ),
        "claim_boundary": (
            "This adaptation report supports only bounded terminal/process/"
            "filesystem/time sandbox behavior with typed motor heads and "
            "allowlisted RUN_COMMAND execution."
        ),
        "profile_transfer": {
            "train_profile": config.profile,
            "eval_profile": eval_profile,
            "is_transfer": eval_profile != config.profile,
        },
    }
    (config.output_dir / "real_sandbox_adaptation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _load_model(checkpoint_path: Path, *, device: str) -> ReflexCoreV0:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ReflexCoreV0(ReflexCoreV0Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _dataset_mixture_summary(
    *,
    synthetic_path: Path,
    synthetic_examples: list[ReflexCoreTrainingExample],
    real_path: Path,
    real_examples: list[ReflexCoreTrainingExample],
    real_summary: dict[str, object],
    mixed_path: Path,
    mixed_examples: list[ReflexCoreTrainingExample],
    synthetic_repeat: int,
    real_sandbox_repeat: int,
) -> dict[str, object]:
    return {
        "synthetic_train": {
            "path": str(synthetic_path),
            "example_count": len(synthetic_examples),
            "hash": dataset_hash(synthetic_examples),
            "repeat": synthetic_repeat,
            "weighted_example_count": len(synthetic_examples) * synthetic_repeat,
        },
        "real_sandbox_train": {
            "path": str(real_path),
            "example_count": len(real_examples),
            "hash": dataset_hash(real_examples),
            "summary": real_summary,
            "repeat": real_sandbox_repeat,
            "weighted_example_count": len(real_examples) * real_sandbox_repeat,
        },
        "mixed_train": {
            "path": str(mixed_path),
            "example_count": len(mixed_examples),
            "hash": dataset_hash(mixed_examples),
            "synthetic_fraction": (
                (len(synthetic_examples) * synthetic_repeat) / max(len(mixed_examples), 1)
            ),
            "real_sandbox_fraction": (
                (len(real_examples) * real_sandbox_repeat) / max(len(mixed_examples), 1)
            ),
        },
    }


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


def _real_sandbox_live_gate(report: dict[str, object]) -> dict[str, object]:
    if not report.get("live_observation"):
        return {
            "required": False,
            "passed": True,
            "reason": "live_observation_not_requested",
        }
    model = report.get("model")
    if not isinstance(model, dict):
        model = {}
    checks = {
        "live_observation_episode_count": model.get("live_observation_episode_count"),
        "runtime_observation_steps": model.get("runtime_observation_steps"),
        "changed_file_observation_steps": model.get("changed_file_observation_steps"),
        "observed_prediction_error_examples": model.get(
            "observed_prediction_error_examples"
        ),
    }
    details = {
        key: {
            "observed": value,
            "required_min": 1,
            "passed": isinstance(value, int | float) and float(value) >= 1.0,
        }
        for key, value in checks.items()
    }
    return {
        "required": True,
        "passed": all(item["passed"] for item in details.values()),
        "details": details,
    }


def _json_config(config: ReflexCoreRealSandboxAdaptationConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["model_config_path"] = str(config.model_config_path)
    return payload
