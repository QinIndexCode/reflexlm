from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from reflexlm.core.dataset import read_reflexcore_jsonl
from reflexlm.core.evaluation import evaluate_reflexcore_sensory_ablation
from reflexlm.core.experiment import _load_model


@dataclass(slots=True)
class ReflexCoreSensoryAblationMatrixConfig:
    matrix_dir: Path
    output_json: Path | None = None
    seeds: tuple[int, ...] = (13, 17, 23)
    profiles: tuple[str, ...] = ("default", "hard", "wide_ood")
    modes: tuple[str, ...] = ("zero_numeric",)
    batch_size: int = 16
    device: str = "cpu"
    sequence_mode: bool = True
    max_sequence_len: int | None = 8
    min_action_accuracy_drop: float | None = 0.5
    min_world_model_drop: float | None = 1.0


def run_reflexcore_sensory_ablation_matrix(
    config: ReflexCoreSensoryAblationMatrixConfig,
) -> dict[str, object]:
    _validate_config(config)
    rows: list[dict[str, object]] = []
    for seed in config.seeds:
        seed_dir = config.matrix_dir / f"seed_{seed}"
        checkpoint_path = seed_dir / "train" / "reflexcore_v0.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing checkpoint for seed {seed}: {checkpoint_path}")
        model = _load_model(checkpoint_path, device=config.device)
        for profile in config.profiles:
            dataset_path = _profile_dataset_path(seed_dir, profile)
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"missing test split for seed {seed} profile {profile}: {dataset_path}"
                )
            examples = read_reflexcore_jsonl(dataset_path)
            report = evaluate_reflexcore_sensory_ablation(
                model,
                examples,
                modes=list(config.modes),
                batch_size=config.batch_size,
                device=config.device,
                sequence_mode=config.sequence_mode,
                max_sequence_len=config.max_sequence_len,
                min_action_accuracy_drop=config.min_action_accuracy_drop,
                min_next_state_relative_improvement_drop=config.min_world_model_drop,
            )
            rows.append(
                _summarize_ablation_row(
                    seed=seed,
                    profile=profile,
                    checkpoint_path=checkpoint_path,
                    dataset_path=dataset_path,
                    report=report,
                )
            )

    summary = _summarize_rows(rows)
    result: dict[str, object] = {
        "config": _json_config(config),
        "summary": summary,
        "rows": rows,
        "passed": bool(summary["passed"]),
        "claim_boundary": (
            "This sensory-ablation matrix supports only bounded ReflexCore V0 "
            "dependence on terminal/process/filesystem/time observation vectors. "
            "It does not evaluate GUI, free shell generation, robotics, or "
            "production autonomy."
        ),
    }
    output_json = config.output_json or (
        config.matrix_dir / "sensory_ablation_matrix_report.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _validate_config(config: ReflexCoreSensoryAblationMatrixConfig) -> None:
    if not config.matrix_dir.exists():
        raise FileNotFoundError(f"matrix_dir does not exist: {config.matrix_dir}")
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if not config.profiles:
        raise ValueError("at least one profile is required")
    if not config.modes:
        raise ValueError("at least one ablation mode is required")
    valid_modes = {"zero_numeric", "zero_hash", "zero_all"}
    unknown = sorted(set(config.modes) - valid_modes)
    if unknown:
        raise ValueError(f"unknown ablation mode(s): {', '.join(unknown)}")


def _profile_dataset_path(seed_dir: Path, profile: str) -> Path:
    if profile == "default":
        return seed_dir / "synthetic_benchmark" / "reflexcore" / "test.jsonl"
    return (
        seed_dir
        / _profile_dir_name(profile)
        / "eval_benchmark"
        / "reflexcore"
        / "test.jsonl"
    )


def _profile_dir_name(profile: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile).strip("._")
    return f"eval_{safe or 'profile'}"


def _summarize_ablation_row(
    *,
    seed: int,
    profile: str,
    checkpoint_path: Path,
    dataset_path: Path,
    report: dict[str, object],
) -> dict[str, object]:
    modes = report.get("modes")
    if not isinstance(modes, dict):
        modes = {}
    full = report.get("full")
    if not isinstance(full, dict):
        full = {}
    mode_rows: dict[str, object] = {}
    row_passed = bool(report.get("passed"))
    for mode, payload in modes.items():
        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            summary = {}
        mode_rows[str(mode)] = {
            "passed": bool(payload.get("passed")),
            "action_accuracy": summary.get("action_accuracy"),
            "action_accuracy_drop": payload.get("action_accuracy_drop"),
            "action_accuracy_drop_passed": payload.get("action_accuracy_drop_passed"),
            "next_state_relative_improvement": summary.get(
                "next_state_relative_improvement"
            ),
            "next_state_relative_improvement_drop": payload.get(
                "next_state_relative_improvement_drop"
            ),
            "next_state_relative_improvement_drop_passed": payload.get(
                "next_state_relative_improvement_drop_passed"
            ),
        }
        row_passed = row_passed and bool(payload.get("passed"))
    return {
        "seed": seed,
        "profile": profile,
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "full_action_accuracy": full.get("action_accuracy"),
        "full_next_state_relative_improvement": full.get(
            "next_state_relative_improvement"
        ),
        "modes": mode_rows,
        "passed": row_passed,
    }


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    mode_names = sorted(
        {
            mode
            for row in rows
            for mode in (
                row.get("modes").keys()
                if isinstance(row.get("modes"), dict)
                else []
            )
        }
    )
    mode_summary: dict[str, object] = {}
    for mode in mode_names:
        mode_payloads = [
            row["modes"][mode]
            for row in rows
            if isinstance(row.get("modes"), dict) and mode in row["modes"]
        ]
        mode_summary[mode] = {
            "passed": all(bool(payload.get("passed")) for payload in mode_payloads),
            "action_accuracy_drop": _aggregate_numeric(
                payload.get("action_accuracy_drop") for payload in mode_payloads
            ),
            "next_state_relative_improvement_drop": _aggregate_numeric(
                payload.get("next_state_relative_improvement_drop")
                for payload in mode_payloads
            ),
        }
    return {
        "row_count": len(rows),
        "passed_rows": sum(1 for row in rows if row.get("passed") is True),
        "passed": all(row.get("passed") is True for row in rows),
        "modes": mode_summary,
    }


def _aggregate_numeric(values: object) -> dict[str, float | None]:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    if not numeric:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(numeric),
        "mean": mean(numeric),
        "max": max(numeric),
    }


def _json_config(config: ReflexCoreSensoryAblationMatrixConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["matrix_dir"] = str(config.matrix_dir)
    payload["output_json"] = str(config.output_json) if config.output_json else None
    payload["seeds"] = list(config.seeds)
    payload["profiles"] = list(config.profiles)
    payload["modes"] = list(config.modes)
    return payload
