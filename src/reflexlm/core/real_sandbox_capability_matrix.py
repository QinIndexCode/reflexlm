from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from reflexlm.core.sandbox_benchmark import (
    build_real_sandbox_oracle_dataset,
    evaluate_reflexcore_real_sandbox_families,
)
from reflexlm.core.schema import dataset_hash
from reflexlm.core.training import train_reflexcore_v0
from reflexlm.core.experiment import _load_model
from reflexlm.core.dataset import read_reflexcore_jsonl


@dataclass(slots=True)
class ReflexCoreRealSandboxCapabilityMatrixConfig:
    output_dir: Path
    model_config_path: Path
    seeds: tuple[int, ...] = (13, 17, 23)
    train_variants: int = 20
    train_start_variant: int = 0
    eval_variants: int = 5
    eval_start_variant: int = 20
    vocab_size: int = 512
    max_text_tokens: int = 128
    train_epochs: int = 60
    train_batch_size: int = 8
    learning_rate: float = 3e-3
    device: str = "cpu"
    sequence_mode: bool = True
    max_sequence_len: int | None = 8
    max_steps: int = 4
    families: tuple[str, ...] = ()
    min_success_rate: float = 1.0
    min_pass_rate: float = 1.0


def run_reflexcore_real_sandbox_capability_matrix(
    config: ReflexCoreRealSandboxCapabilityMatrixConfig,
) -> dict[str, object]:
    """Train/evaluate ReflexCore V0 across seeds on real sandbox variants."""

    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = config.output_dir / "real_sandbox_train.jsonl"
    dataset_summary = build_real_sandbox_oracle_dataset(
        output_path=dataset_path,
        work_dir=config.output_dir / "dataset_work",
        variants=config.train_variants,
        start_variant=config.train_start_variant,
        vocab_size=config.vocab_size,
        max_text_tokens=config.max_text_tokens,
    )
    examples = read_reflexcore_jsonl(dataset_path)
    runs: list[dict[str, object]] = []
    for seed in config.seeds:
        seed_dir = config.output_dir / f"seed_{seed}"
        train_summary = train_reflexcore_v0(
            dataset_path=dataset_path,
            config_path=config.model_config_path,
            output_dir=seed_dir / "train",
            epochs=config.train_epochs,
            batch_size=config.train_batch_size,
            learning_rate=config.learning_rate,
            device=config.device,
            seed=seed,
            sequence_mode=config.sequence_mode,
            max_sequence_len=config.max_sequence_len,
        )
        model = _load_model(Path(str(train_summary["checkpoint"])), device=config.device)
        eval_report = evaluate_reflexcore_real_sandbox_families(
            model,
            output_dir=seed_dir / "eval",
            families=config.families,
            variants=config.eval_variants,
            start_variant=config.eval_start_variant,
            max_steps=config.max_steps,
        )
        runs.append(_summarize_run(seed, train_summary, eval_report, config))
    pass_count = sum(1 for run in runs if run["passed"] is True)
    pass_rate = pass_count / max(len(runs), 1)
    report: dict[str, object] = {
        "config": _json_config(config),
        "dataset": str(dataset_path),
        "dataset_summary": dataset_summary,
        "dataset_hash": dataset_hash(examples),
        "dataset_examples": len(examples),
        "runs": runs,
        "aggregates": _aggregate_runs(runs),
        "pass_count": pass_count,
        "run_count": len(runs),
        "pass_rate": pass_rate,
        "min_pass_rate": config.min_pass_rate,
        "passed": pass_rate >= config.min_pass_rate,
        "free_shell_generation": False,
        "gui_or_vision": False,
        "claim_boundary": (
            "This cross-seed matrix supports only bounded ReflexCore V0 "
            "terminal/process/filesystem/time behavior in real temporary "
            "sandboxes with typed motor heads and allowlisted RUN_COMMAND. "
            "It does not evaluate GUI, vision, robotics, or unrestricted shell "
            "autonomy."
        ),
    }
    (config.output_dir / "real_sandbox_capability_matrix_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _validate_config(config: ReflexCoreRealSandboxCapabilityMatrixConfig) -> None:
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if config.train_variants <= 0 or config.eval_variants <= 0:
        raise ValueError("train_variants and eval_variants must be positive")
    if config.train_start_variant < 0 or config.eval_start_variant < 0:
        raise ValueError("variant starts must be non-negative")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0.0 <= config.min_success_rate <= 1.0:
        raise ValueError("min_success_rate must be between 0 and 1")
    if not 0.0 <= config.min_pass_rate <= 1.0:
        raise ValueError("min_pass_rate must be between 0 and 1")
    if _ranges_overlap(
        config.train_start_variant,
        config.train_variants,
        config.eval_start_variant,
        config.eval_variants,
    ):
        raise ValueError("train and eval variant ranges must be disjoint")


def _ranges_overlap(start_a: int, count_a: int, start_b: int, count_b: int) -> bool:
    end_a = start_a + count_a
    end_b = start_b + count_b
    return start_a < end_b and start_b < end_a


def _summarize_run(
    seed: int,
    train_summary: dict[str, object],
    eval_report: dict[str, object],
    config: ReflexCoreRealSandboxCapabilityMatrixConfig,
) -> dict[str, object]:
    overall = _dict(eval_report.get("overall"))
    success_rate = overall.get("success_rate")
    passed = isinstance(success_rate, int | float) and float(success_rate) >= config.min_success_rate
    family_rates = {
        family: _dict(summary).get("success_rate")
        for family, summary in _dict(eval_report.get("families_summary")).items()
    }
    return {
        "seed": seed,
        "passed": passed,
        "min_success_rate": config.min_success_rate,
        "success_rate": success_rate,
        "success_count": overall.get("success_count"),
        "task_count": overall.get("task_count"),
        "family_success_rates": family_rates,
        "checkpoint": train_summary.get("checkpoint"),
        "dataset_hash": train_summary.get("dataset_hash"),
        "model_hash": train_summary.get("model_hash"),
        "parameter_count": train_summary.get("parameter_count"),
        "epochs": train_summary.get("epochs"),
        "batch_size": train_summary.get("batch_size"),
        "learning_rate": train_summary.get("learning_rate"),
        "final_loss": _last_history_value(train_summary, "loss"),
        "final_action_loss": _last_history_value(train_summary, "action_loss"),
        "free_shell_generation": eval_report.get("free_shell_generation"),
        "gui_or_vision": eval_report.get("gui_or_vision"),
    }


def _last_history_value(summary: dict[str, object], key: str) -> float | None:
    history = summary.get("history")
    if not isinstance(history, list) or not history:
        return None
    value = _dict(history[-1]).get(key)
    return float(value) if isinstance(value, int | float) else None


def _aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    metrics = [
        "success_rate",
        "success_count",
        "task_count",
        "parameter_count",
        "final_loss",
        "final_action_loss",
    ]
    return {metric: _aggregate_metric(runs, metric) for metric in metrics}


def _aggregate_metric(runs: list[dict[str, object]], metric: str) -> dict[str, float | None]:
    values = [run.get(metric) for run in runs]
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(numeric), "mean": mean(numeric), "max": max(numeric)}


def _json_config(config: ReflexCoreRealSandboxCapabilityMatrixConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["model_config_path"] = str(config.model_config_path)
    payload["seeds"] = list(config.seeds)
    payload["families"] = list(config.families)
    return payload


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}
