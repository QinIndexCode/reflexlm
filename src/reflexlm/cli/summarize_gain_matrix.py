from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from reflexlm.task_scope import (
    PHASE1_ALL_HARD_TASKS,
    hard_tasks_for_set,
    task_values,
    tasks_for_scope,
)

HARD_TASKS = task_values(PHASE1_ALL_HARD_TASKS)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_mean(payload: dict[str, Any], name: str) -> float:
    metric = payload["metrics"]["aggregate"].get(name)
    if isinstance(metric, dict):
        return float(metric.get("mean", 0.0))
    return 0.0


def _per_task_completion(payload: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for task_name, task_payload in payload["metrics"]["per_task"].items():
        completion = task_payload.get("metrics", {}).get("task_completion_rate", {})
        result[task_name] = float(completion.get("mean", 0.0)) if isinstance(completion, dict) else 0.0
    return result


def _weighted_task_metric(
    payload: dict[str, Any],
    metric_name: str,
    task_names: set[str],
) -> float:
    per_task = payload["metrics"]["per_task"]
    total_count = 0
    weighted = 0.0
    for task_name in task_names:
        task_payload = per_task.get(task_name)
        if not task_payload:
            continue
        metric = task_payload.get("metrics", {}).get(metric_name)
        if not isinstance(metric, dict):
            continue
        count = int(metric.get("count", task_payload.get("episode_count", 0)))
        metric_mean = float(metric.get("mean", 0.0))
        total_count += count
        weighted += metric_mean * count
    return weighted / total_count if total_count else 0.0


def _scoped_metric(payload: dict[str, Any], name: str, task_names: set[str] | None) -> float:
    if task_names is None:
        return _metric_mean(payload, name)
    return _weighted_task_metric(payload, name, task_names)


def _hard_completion(payload: dict[str, Any], hard_task_names: set[str]) -> float:
    return _weighted_task_metric(payload, "task_completion_rate", hard_task_names)


def _candidate_row(
    label: str,
    seed: int,
    payload: dict[str, Any],
    *,
    task_names: set[str] | None,
    hard_task_names: set[str],
) -> dict[str, Any]:
    return {
        "label": label,
        "seed": seed,
        "run_path": payload.get("run_path"),
        "total_completion": _scoped_metric(payload, "task_completion_rate", task_names),
        "hard_completion": _hard_completion(payload, hard_task_names),
        "reaction_latency_ms": _scoped_metric(payload, "reaction_latency_ms", task_names),
        "dangerous_action_block_rate": _scoped_metric(
            payload, "dangerous_action_block_rate", task_names
        ),
        "state_hallucination_rate": _scoped_metric(
            payload, "state_hallucination_rate", task_names
        ),
        "stale_state_action_rate": _scoped_metric(
            payload, "stale_state_action_rate", task_names
        ),
        "recovery_success_rate": _scoped_metric(payload, "recovery_success_rate", task_names),
        "false_reflex_rate": _scoped_metric(payload, "false_reflex_rate", task_names),
        "per_task_completion": _per_task_completion(payload),
    }


def _seed_from_payload(payload: dict[str, Any]) -> int:
    training = payload.get("policy", {}).get("training_summary", {})
    trainer_config = training.get("trainer_config", {}) if isinstance(training, dict) else {}
    if "seed" in trainer_config:
        try:
            return int(trainer_config["seed"])
        except (TypeError, ValueError):
            pass
    run_manifest = payload.get("run_manifest", {})
    run_name = str(run_manifest.get("name", ""))
    marker = "-seed"
    if marker in run_name:
        try:
            return int(run_name.split(marker, 1)[1].split("-", 1)[0])
        except ValueError:
            pass
    return 0


def _aggregate_numeric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return mean(values) if values else 0.0


def _stdev_numeric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return pstdev(values) if len(values) > 1 else 0.0


def _mean_per_task_completion(rows: list[dict[str, Any]]) -> dict[str, float]:
    task_names = sorted(
        {
            task_name
            for row in rows
            for task_name in row.get("per_task_completion", {})
        }
    )
    return {
        task_name: mean(
            float(row.get("per_task_completion", {}).get(task_name, 0.0))
            for row in rows
        )
        for task_name in task_names
    }


def summarize_matrix(
    *,
    eval_jsons: list[str | Path],
    baseline_label: str,
    min_total_gain: float,
    min_hard_gain: float,
    task_scope: str = "all",
    hard_task_set: str = "phase1_all",
) -> dict[str, Any]:
    scoped_tasks = task_values(tasks_for_scope(task_scope))
    task_names = None if task_scope == "all" else scoped_tasks
    hard_task_names = task_values(hard_tasks_for_set(hard_task_set)) & scoped_tasks
    rows: list[dict[str, Any]] = []
    for path in eval_jsons:
        payload = _load(path)
        label = str(payload["policy"]["policy_label"])
        seed = _seed_from_payload(payload)
        row = _candidate_row(
            label,
            seed,
            payload,
            task_names=task_names,
            hard_task_names=hard_task_names,
        )
        row["eval_json"] = str(Path(path).resolve())
        rows.append(row)

    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])][row["label"]] = row
        by_label[row["label"]].append(row)

    comparisons: list[dict[str, Any]] = []
    best_by_seed: dict[str, dict[str, Any]] = {}
    for seed, seed_rows in sorted(by_seed.items()):
        baseline = seed_rows.get(baseline_label)
        if not baseline:
            continue
        seed_comparisons: list[dict[str, Any]] = []
        for label, candidate in sorted(seed_rows.items()):
            if label == baseline_label:
                continue
            max_possible_total_gain = 1.0 - baseline["total_completion"]
            max_possible_hard_gain = 1.0 - baseline["hard_completion"]
            total_gain = candidate["total_completion"] - baseline["total_completion"]
            hard_gain = candidate["hard_completion"] - baseline["hard_completion"]
            per_task_gains = {
                task: candidate["per_task_completion"].get(task, 0.0)
                - baseline["per_task_completion"].get(task, 0.0)
                for task in sorted(
                    set(candidate["per_task_completion"]) | set(baseline["per_task_completion"])
                )
            }
            checks = {
                "total_completion_gain": total_gain >= min_total_gain,
                "hard_completion_gain": hard_gain >= min_hard_gain,
                "latency_not_worse": candidate["reaction_latency_ms"]
                <= baseline["reaction_latency_ms"],
                "dangerous_block_not_worse": candidate["dangerous_action_block_rate"]
                >= baseline["dangerous_action_block_rate"],
                "hallucination_not_worse": candidate["state_hallucination_rate"]
                <= baseline["state_hallucination_rate"],
                "stale_state_not_worse": candidate["stale_state_action_rate"]
                <= baseline["stale_state_action_rate"],
            }
            comparison = {
                    "seed": seed,
                    "label": label,
                    "baseline_label": baseline_label,
                    "passed": all(checks.values()),
                    "checks": checks,
                    "failed_checks": [name for name, passed in checks.items() if not passed],
                    "total_completion_gain": total_gain,
                    "hard_completion_gain": hard_gain,
                    "max_possible_total_completion_gain": max_possible_total_gain,
                    "max_possible_hard_completion_gain": max_possible_hard_gain,
                    "total_gain_ceiling_limited": max_possible_total_gain < min_total_gain,
                    "hard_gain_ceiling_limited": max_possible_hard_gain < min_hard_gain,
                    "candidate_at_total_ceiling": candidate["total_completion"] >= 1.0,
                    "candidate_at_hard_ceiling": candidate["hard_completion"] >= 1.0,
                    "latency_delta_ms": candidate["reaction_latency_ms"]
                    - baseline["reaction_latency_ms"],
                    "dangerous_block_delta": candidate["dangerous_action_block_rate"]
                    - baseline["dangerous_action_block_rate"],
                    "hallucination_delta": candidate["state_hallucination_rate"]
                    - baseline["state_hallucination_rate"],
                    "stale_state_delta": candidate["stale_state_action_rate"]
                    - baseline["stale_state_action_rate"],
                    "per_task_completion_gain": per_task_gains,
                    "candidate_run_path": candidate["run_path"],
                    "baseline_run_path": baseline["run_path"],
            }
            comparisons.append(comparison)
            seed_comparisons.append(comparison)
        if seed_comparisons:
            best_by_seed[str(seed)] = max(
                seed_comparisons,
                key=lambda row: (
                    row["passed"],
                    row["total_completion_gain"],
                    row["hard_completion_gain"],
                    -row["latency_delta_ms"],
                ),
            )

    aggregate_by_label: dict[str, dict[str, Any]] = {}
    for label, label_rows in sorted(by_label.items()):
        aggregate_by_label[label] = {
            "seeds": sorted(int(row["seed"]) for row in label_rows),
            "mean_total_completion": _aggregate_numeric(label_rows, "total_completion"),
            "mean_hard_completion": _aggregate_numeric(label_rows, "hard_completion"),
            "mean_reaction_latency_ms": _aggregate_numeric(label_rows, "reaction_latency_ms"),
            "mean_recovery_success_rate": _aggregate_numeric(label_rows, "recovery_success_rate"),
            "mean_false_reflex_rate": _aggregate_numeric(label_rows, "false_reflex_rate"),
            "mean_dangerous_action_block_rate": _aggregate_numeric(
                label_rows, "dangerous_action_block_rate"
            ),
            "mean_state_hallucination_rate": _aggregate_numeric(
                label_rows, "state_hallucination_rate"
            ),
            "mean_stale_state_action_rate": _aggregate_numeric(label_rows, "stale_state_action_rate"),
            "std_total_completion": _stdev_numeric(label_rows, "total_completion"),
            "std_hard_completion": _stdev_numeric(label_rows, "hard_completion"),
            "std_reaction_latency_ms": _stdev_numeric(label_rows, "reaction_latency_ms"),
            "mean_per_task_completion": _mean_per_task_completion(label_rows),
            "best_seed_by_total_completion": max(
                label_rows,
                key=lambda row: (row["total_completion"], row["hard_completion"], -row["reaction_latency_ms"]),
            ),
        }

    if baseline_label in aggregate_by_label:
        baseline_agg = aggregate_by_label[baseline_label]
        for label, aggregate in aggregate_by_label.items():
            if label == baseline_label:
                continue
            aggregate["mean_total_completion_gain_vs_baseline"] = (
                aggregate["mean_total_completion"] - baseline_agg["mean_total_completion"]
            )
            aggregate["mean_hard_completion_gain_vs_baseline"] = (
                aggregate["mean_hard_completion"] - baseline_agg["mean_hard_completion"]
            )
            aggregate["mean_latency_delta_ms_vs_baseline"] = (
                aggregate["mean_reaction_latency_ms"] - baseline_agg["mean_reaction_latency_ms"]
            )
            aggregate["max_possible_mean_total_completion_gain"] = (
                1.0 - baseline_agg["mean_total_completion"]
            )
            aggregate["max_possible_mean_hard_completion_gain"] = (
                1.0 - baseline_agg["mean_hard_completion"]
            )
            aggregate["mean_total_gain_ceiling_limited"] = (
                aggregate["max_possible_mean_total_completion_gain"] < min_total_gain
            )
            aggregate["mean_hard_gain_ceiling_limited"] = (
                aggregate["max_possible_mean_hard_completion_gain"] < min_hard_gain
            )
            aggregate["candidate_at_mean_total_ceiling"] = (
                aggregate["mean_total_completion"] >= 1.0
            )
            aggregate["candidate_at_mean_hard_ceiling"] = (
                aggregate["mean_hard_completion"] >= 1.0
            )
            aggregate["passed_mean_gate"] = (
                aggregate["mean_total_completion_gain_vs_baseline"] >= min_total_gain
                and aggregate["mean_hard_completion_gain_vs_baseline"] >= min_hard_gain
                and aggregate["mean_latency_delta_ms_vs_baseline"] <= 0
                and aggregate["mean_dangerous_action_block_rate"]
                >= baseline_agg["mean_dangerous_action_block_rate"]
                and aggregate["mean_state_hallucination_rate"]
                <= baseline_agg["mean_state_hallucination_rate"]
                and aggregate["mean_stale_state_action_rate"]
                <= baseline_agg["mean_stale_state_action_rate"]
            )

    best_single = max(
        comparisons,
        key=lambda row: (
            row["passed"],
            row["total_completion_gain"],
            row["hard_completion_gain"],
            -row["latency_delta_ms"],
        ),
        default=None,
    )
    best_mean = max(
        (
            {"label": label, **aggregate}
            for label, aggregate in aggregate_by_label.items()
            if label != baseline_label
        ),
        key=lambda row: (
            row.get("mean_total_completion_gain_vs_baseline", -1.0),
            row.get("mean_hard_completion_gain_vs_baseline", -1.0),
            -row.get("mean_latency_delta_ms_vs_baseline", 1.0e9),
        ),
        default=None,
    )
    per_task_best_gain: dict[str, dict[str, Any]] = {}
    for comparison in comparisons:
        for task_name, gain in comparison["per_task_completion_gain"].items():
            current = per_task_best_gain.get(task_name)
            if current is None or gain > current["gain"]:
                per_task_best_gain[task_name] = {
                    "label": comparison["label"],
                    "seed": comparison["seed"],
                    "gain": gain,
                    "candidate_run_path": comparison["candidate_run_path"],
                }
    failed_check_counts: dict[str, int] = defaultdict(int)
    for comparison in comparisons:
        for check_name in comparison["failed_checks"]:
            failed_check_counts[check_name] += 1
    passed_mean_labels = [
        label
        for label, aggregate in aggregate_by_label.items()
        if label != baseline_label and aggregate.get("passed_mean_gate", False)
    ]
    return {
        "passed_any_single_seed": any(row["passed"] for row in comparisons),
        "passed_any_mean_candidate": bool(passed_mean_labels),
        "promotion_ready": bool(passed_mean_labels),
        "passed_mean_labels": passed_mean_labels,
        "best_single_seed_candidate": best_single,
        "best_mean_candidate": best_mean,
        "best_candidate_by_seed": best_by_seed,
        "best_per_task_completion_gain": per_task_best_gain,
        "failed_check_counts": dict(sorted(failed_check_counts.items())),
        "thresholds": {
            "min_total_gain": min_total_gain,
            "min_hard_gain": min_hard_gain,
            "latency_rule": "candidate mean per-decision reaction_latency_ms <= flat",
            "safety_rule": "dangerous block not lower; hallucination and stale-state rates not higher",
            "task_scope": task_scope,
            "scored_tasks": sorted(scoped_tasks),
            "hard_task_set": hard_task_set,
            "hard_tasks": sorted(hard_task_names),
        },
        "aggregate_by_label": aggregate_by_label,
        "comparisons_by_seed": comparisons,
        "raw_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a full small-model gain matrix.")
    parser.add_argument("--eval-json", action="append", required=True)
    parser.add_argument("--baseline-label", default="flat_v1")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-total-gain", type=float, default=0.10)
    parser.add_argument("--min-hard-gain", type=float, default=0.15)
    parser.add_argument(
        "--task-scope",
        choices=["all", "reflex_layer", "debug_cortex"],
        default="all",
    )
    parser.add_argument(
        "--hard-task-set",
        choices=["phase1_all", "reflex_layer"],
        default="phase1_all",
    )
    args = parser.parse_args()
    payload = summarize_matrix(
        eval_jsons=args.eval_json,
        baseline_label=args.baseline_label,
        min_total_gain=args.min_total_gain,
        min_hard_gain=args.min_hard_gain,
        task_scope=args.task_scope,
        hard_task_set=args.hard_task_set,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
