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
from reflexlm.core.dataset import read_reflexcore_jsonl
from reflexlm.core.evaluation import (
    acceptance_against_baselines,
    evaluate_baseline_policies,
    evaluate_reflexcore_model,
    prediction_error_acceptance,
    world_model_acceptance,
)
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
from reflexlm.core.training import train_reflexcore_v0


@dataclass(slots=True)
class ReflexCoreExperimentConfig:
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


def run_reflexcore_experiment(config: ReflexCoreExperimentConfig) -> dict[str, object]:
    output_dir = config.output_dir
    benchmark_dir = output_dir / "benchmark"
    train_dir = output_dir / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_manifest = build_reflexcore_benchmark(
        ReflexCoreBenchmarkConfig(
            output_dir=benchmark_dir,
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
    eval_benchmark_dir = benchmark_dir
    if eval_profile != config.profile:
        eval_benchmark_dir = output_dir / "eval_benchmark"
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
    train_summary = train_reflexcore_v0(
        dataset_path=benchmark_dir / "reflexcore" / "train.jsonl",
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
    report: dict[str, object] = {
        "config": _json_config(config),
        "benchmark": benchmark_manifest,
        "eval_benchmark": eval_benchmark_manifest,
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


def _load_model(checkpoint_path: Path, *, device: str) -> ReflexCoreV0:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ReflexCoreV0(ReflexCoreV0Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


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


def _json_config(config: ReflexCoreExperimentConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["model_config_path"] = str(config.model_config_path)
    return payload
