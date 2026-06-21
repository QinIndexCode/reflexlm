from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reflexlm.core.benchmark import (
    ReflexCoreBenchmarkConfig,
    build_reflexcore_benchmark,
)
from reflexlm.core.training import train_reflexcore_v0


@dataclass(slots=True)
class ReflexCoreLocalFeasibilityConfig:
    output_dir: Path
    model_config_path: Path = Path("configs/reflexcore/local.yaml")
    profile: str = "default"
    episodes_per_task: int = 1
    split_strategy: str = "episode_random"
    seed: int = 43
    vocab_size: int = 4096
    hash_bins: int = 256
    max_text_tokens: int = 128
    train_epochs: int = 1
    train_batch_size: int = 1
    learning_rate: float | None = None
    device: str | None = None
    sequence_mode: bool = True
    max_sequence_len: int | None = 8
    min_parameters: int | None = 20_000_000
    max_parameters: int | None = 100_000_000


def run_reflexcore_local_feasibility(
    config: ReflexCoreLocalFeasibilityConfig,
) -> dict[str, object]:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir = config.output_dir / "benchmark"
    train_dir = config.output_dir / "train"
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
    parameter_gate = _parameter_gate(
        int(train_summary["parameter_count"]),
        min_parameters=config.min_parameters,
        max_parameters=config.max_parameters,
    )
    finite_loss_gate = _finite_loss_gate(train_summary.get("history"))
    checkpoint_path = Path(str(train_summary["checkpoint"]))
    checkpoint_gate = {
        "checkpoint": str(checkpoint_path),
        "exists": checkpoint_path.exists(),
        "passed": checkpoint_path.exists(),
    }
    report: dict[str, object] = {
        "config": _json_config(config),
        "benchmark": benchmark_manifest,
        "train": train_summary,
        "parameter_gate": parameter_gate,
        "finite_loss_gate": finite_loss_gate,
        "checkpoint_gate": checkpoint_gate,
        "duration_seconds": time.perf_counter() - started,
        "passed": bool(
            parameter_gate["passed"]
            and finite_loss_gate["passed"]
            and checkpoint_gate["passed"]
        ),
        "claim_boundary": (
            "This gate proves only that the configured ReflexCore V0 model can "
            "be instantiated and trained locally on a small bounded benchmark. "
            "It is not a full local performance or autonomy claim."
        ),
    }
    (config.output_dir / "local_feasibility_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


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


def _finite_loss_gate(history: object) -> dict[str, object]:
    if not isinstance(history, list) or not history:
        return {"final_loss": None, "passed": False}
    final = history[-1]
    final_loss = final.get("loss") if isinstance(final, dict) else None
    passed = isinstance(final_loss, int | float) and math.isfinite(float(final_loss))
    return {"final_loss": final_loss, "passed": passed}


def _json_config(config: ReflexCoreLocalFeasibilityConfig) -> dict[str, object]:
    payload: dict[str, Any] = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["model_config_path"] = str(config.model_config_path)
    return payload
