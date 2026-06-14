from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reflexlm.task_scope import (
    PHASE1_ALL_HARD_TASKS,
    REFLEX_LAYER_HARD_TASKS,
    scope_for_task,
    task_values,
)

HARD_TASKS = task_values(PHASE1_ALL_HARD_TASKS)
REFLEX_HARD_TASKS = task_values(REFLEX_LAYER_HARD_TASKS)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _bin(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if number < 0.25:
        return "low"
    if number < 0.5:
        return "mid"
    if number < 0.75:
        return "high"
    return "very_high"


def _group_completion(episodes: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        buckets[str(key_fn(row))].append(row)
    return {
        key: {
            "episodes": len(rows),
            "completion_rate": _rate(
                sum(1 for row in rows if float(row.get("task_completion_rate", 0.0)) > 0.0),
                len(rows),
            ),
            "mean_false_reflex_rate": round(
                sum(float(row.get("false_reflex_rate", 0.0)) for row in rows) / max(len(rows), 1),
                6,
            ),
        }
        for key, rows in sorted(buckets.items())
    }


def analyze_run(run_dir: Path, *, max_failures: int) -> dict[str, Any]:
    episodes = _read_jsonl(run_dir / "episode_results.jsonl")
    traces = _read_jsonl(run_dir / "trace_rows.jsonl")
    failed_episodes = {
        row["episode_id"]
        for row in episodes
        if float(row.get("task_completion_rate", 0.0)) <= 0.0
    }
    first_steps: dict[str, dict[str, Any]] = {}
    for row in traces:
        if int(row.get("step_index", 0)) == 0:
            first_steps[row["episode_id"]] = row

    confusion = Counter()
    incorrect_by_task = Counter()
    hallucinated_by_task = Counter()
    debug_bins: dict[str, Counter[str]] = {
        "salience": Counter(),
        "risk": Counter(),
        "prediction_error": Counter(),
    }
    command_slot = Counter()
    file_slot = Counter()
    for row in traces:
        task_type = row["task_type"]
        action = row.get("action", {})
        oracle = row.get("oracle_action", {})
        action_type = action.get("type")
        oracle_type = oracle.get("type")
        confusion[(oracle_type, action_type)] += 1
        if not row.get("correct", False):
            incorrect_by_task[task_type] += 1
        if row.get("hallucinated", False):
            hallucinated_by_task[task_type] += 1
        debug = row.get("policy_debug", {})
        for key in debug_bins:
            debug_bins[key][_bin(debug.get(key))] += int(not row.get("correct", False))
        command_slot[str(debug.get("command_index", "unknown"))] += int(
            action_type == "RUN_COMMAND" and not row.get("correct", False)
        )
        file_slot[str(debug.get("file_index", "unknown"))] += int(
            action_type == "READ_FILE" and not row.get("correct", False)
        )

    failure_samples = []
    for episode_id in sorted(failed_episodes)[:max_failures]:
        first = first_steps.get(episode_id, {})
        failure_samples.append(
            {
                "episode_id": episode_id,
                "task_type": first.get("task_type"),
                "first_action": first.get("action"),
                "first_oracle_action": first.get("oracle_action"),
                "policy_debug": first.get("policy_debug", {}),
            }
        )

    return {
        "run_dir": str(run_dir.resolve()),
        "episode_count": len(episodes),
        "trace_count": len(traces),
        "failed_episode_count": len(failed_episodes),
        "completion_by_task": _group_completion(episodes, lambda row: row.get("task_type")),
        "completion_by_hard_subset": _group_completion(
            episodes,
            lambda row: "hard" if row.get("task_type") in HARD_TASKS else "easy",
        ),
        "completion_by_reflex_hard_subset": _group_completion(
            episodes,
            lambda row: "reflex_hard"
            if row.get("task_type") in REFLEX_HARD_TASKS
            else "other",
        ),
        "completion_by_cortex_scope": _group_completion(
            episodes,
            lambda row: scope_for_task(str(row.get("task_type"))),
        ),
        "first_step_action_on_failed_episodes": Counter(
            (first_steps.get(episode_id, {}).get("action") or {}).get("type", "missing")
            for episode_id in failed_episodes
        ),
        "incorrect_steps_by_task": dict(sorted(incorrect_by_task.items())),
        "hallucinated_steps_by_task": dict(sorted(hallucinated_by_task.items())),
        "confusion_matrix": [
            {"oracle_action": oracle, "policy_action": policy, "count": count}
            for (oracle, policy), count in sorted(confusion.items())
        ],
        "incorrect_debug_bins": {
            key: dict(sorted(counter.items())) for key, counter in debug_bins.items()
        },
        "incorrect_command_slot_counts": dict(sorted(command_slot.items())),
        "incorrect_file_slot_counts": dict(sorted(file_slot.items())),
        "failure_samples": failure_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Phase 1/2 evaluation errors from a run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-failures", type=int, default=25)
    args = parser.parse_args()

    payload = analyze_run(Path(args.run_dir), max_failures=args.max_failures)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
