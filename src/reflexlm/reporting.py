from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reflexlm.schema import TaskType

METRIC_ORDER = [
    "reaction_latency_ms",
    "first_decision_latency_ms",
    "episode_compute_latency_ms",
    "token_equivalent_cost",
    "model_calls",
    "oracle_step_accuracy",
    "command_decision_accuracy",
    "read_file_decision_accuracy",
    "positive_reward_credit",
    "recovery_success_rate",
    "false_reflex_rate",
    "dangerous_action_block_rate",
    "long_run_stability",
    "state_hallucination_rate",
    "stale_state_action_rate",
    "task_completion_rate",
]

BINARY_METRICS = {
    "recovery_success_rate",
    "dangerous_action_block_rate",
    "long_run_stability",
    "stale_state_action_rate",
    "task_completion_rate",
}


@dataclass(slots=True)
class MetricSummary:
    mean: float
    lower: float
    upper: float
    count: int
    positives: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mean": round(self.mean, 6),
            "ci95": [round(self.lower, 6), round(self.upper, 6)],
            "count": self.count,
        }
        if self.positives is not None:
            payload["positives"] = self.positives
        return payload


def bootstrap_mean_ci(
    values: list[float],
    *,
    iterations: int = 2000,
    seed: int = 13,
) -> MetricSummary | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if array.size == 1:
        return MetricSummary(mean=mean, lower=mean, upper=mean, count=1)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(iterations, array.size), replace=True)
    sample_means = samples.mean(axis=1)
    lower, upper = np.quantile(sample_means, [0.025, 0.975])
    return MetricSummary(
        mean=mean,
        lower=float(lower),
        upper=float(upper),
        count=int(array.size),
    )


def summarize_metric(
    rows: list[dict[str, Any]],
    metric_name: str,
    *,
    iterations: int = 2000,
    seed: int = 13,
) -> dict[str, Any] | None:
    values = [float(row[metric_name]) for row in rows if row.get(metric_name) is not None]
    summary = bootstrap_mean_ci(values, iterations=iterations, seed=seed)
    if summary is None:
        return None
    if metric_name in BINARY_METRICS:
        summary.positives = int(sum(1 for value in values if value >= 0.5))
    return summary.to_dict()


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    iterations: int = 2000,
    seed: int = 13,
) -> dict[str, Any]:
    return {
        metric: summarize_metric(rows, metric, iterations=iterations, seed=seed)
        for metric in METRIC_ORDER
    }


def per_task_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {task.value: [] for task in TaskType}
    for row in rows:
        grouped[str(row["task_type"])].append(row)
    return grouped


def bootstrap_paired_difference(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    metric_name: str,
    *,
    iterations: int = 2000,
    seed: int = 13,
) -> dict[str, Any] | None:
    reference_map = {row["episode_id"]: row for row in reference_rows}
    candidate_map = {row["episode_id"]: row for row in candidate_rows}
    common_ids = sorted(reference_map.keys() & candidate_map.keys())
    diffs = [
        float(candidate_map[episode_id][metric_name]) - float(reference_map[episode_id][metric_name])
        for episode_id in common_ids
        if candidate_map[episode_id].get(metric_name) is not None
        and reference_map[episode_id].get(metric_name) is not None
    ]
    summary = bootstrap_mean_ci(diffs, iterations=iterations, seed=seed)
    return summary.to_dict() if summary is not None else None


def load_episode_results(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(__import__("json").loads(line))
    return rows
