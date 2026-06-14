from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reflexlm.data.jsonl import read_jsonl
from reflexlm.schema import ActionDecision, SystemStateFrame, TrajectoryRecord


SPLIT_NAMES = ("train", "val", "test")


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_signature(state: SystemStateFrame) -> str:
    return _stable_hash(state.model_dump(mode="json"))


def _action_signature(action: ActionDecision | None) -> str:
    return _stable_hash(action.model_dump(mode="json") if action else None)


def _split_stats(records: list[TrajectoryRecord]) -> dict[str, Any]:
    state_keys = [_state_signature(record.state) for record in records]
    state_action_keys = [
        f"{_state_signature(record.state)}::{_action_signature(record.action)}"
        for record in records
    ]
    episode_ids = {record.episode_id for record in records}
    per_task_records: dict[str, int] = defaultdict(int)
    per_task_unique_states: dict[str, set[str]] = defaultdict(set)
    for record, state_key in zip(records, state_keys, strict=True):
        task = record.goal.task_type.value
        per_task_records[task] += 1
        per_task_unique_states[task].add(state_key)
    return {
        "record_count": len(records),
        "episode_count": len(episode_ids),
        "unique_state_count": len(set(state_keys)),
        "unique_state_action_count": len(set(state_action_keys)),
        "state_duplicate_rate": 1.0 - (len(set(state_keys)) / max(len(records), 1)),
        "state_action_duplicate_rate": 1.0
        - (len(set(state_action_keys)) / max(len(records), 1)),
        "per_task": {
            task: {
                "record_count": count,
                "unique_state_count": len(per_task_unique_states[task]),
                "state_duplicate_rate": 1.0
                - (len(per_task_unique_states[task]) / max(count, 1)),
            }
            for task, count in sorted(per_task_records.items())
        },
    }


def _overlap_stats(
    records: list[TrajectoryRecord],
    *,
    train_state_keys: set[str],
    train_state_action_keys: set[str],
) -> dict[str, Any]:
    state_hits = 0
    state_action_hits = 0
    per_task: dict[str, dict[str, int]] = defaultdict(
        lambda: {"records": 0, "state_hits": 0, "state_action_hits": 0}
    )
    for record in records:
        state_key = _state_signature(record.state)
        state_action_key = f"{state_key}::{_action_signature(record.action)}"
        task = record.goal.task_type.value
        per_task[task]["records"] += 1
        if state_key in train_state_keys:
            state_hits += 1
            per_task[task]["state_hits"] += 1
        if state_action_key in train_state_action_keys:
            state_action_hits += 1
            per_task[task]["state_action_hits"] += 1
    return {
        "record_count": len(records),
        "state_overlap_with_train_rate": state_hits / max(len(records), 1),
        "state_action_overlap_with_train_rate": state_action_hits / max(len(records), 1),
        "per_task": {
            task: {
                "record_count": stats["records"],
                "state_overlap_with_train_rate": stats["state_hits"]
                / max(stats["records"], 1),
                "state_action_overlap_with_train_rate": stats["state_action_hits"]
                / max(stats["records"], 1),
            }
            for task, stats in sorted(per_task.items())
        },
    }


def _load_episode_metadata(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    metadata_path = dataset_dir / "episode_metadata.json"
    if not metadata_path.exists():
        return {}
    rows = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {str(row["episode_id"]): row for row in rows}


def _scenario_key(metadata: dict[str, Any] | None) -> str | None:
    if not metadata:
        return None
    return f"{metadata['task_type']}::{metadata['scenario_template']}"


def _scenario_split_stats(
    records: list[TrajectoryRecord],
    metadata_by_episode: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scenarios_by_task: dict[str, set[str]] = defaultdict(set)
    scenarios: set[str] = set()
    episodes_without_metadata = 0
    for episode_id in {record.episode_id for record in records}:
        metadata = metadata_by_episode.get(episode_id)
        key = _scenario_key(metadata)
        if key is None:
            episodes_without_metadata += 1
            continue
        scenarios.add(key)
        scenarios_by_task[str(metadata["task_type"])].add(str(metadata["scenario_template"]))
    return {
        "scenario_count": len(scenarios),
        "episodes_without_metadata": episodes_without_metadata,
        "per_task": {
            task: {"scenario_count": len(values)}
            for task, values in sorted(scenarios_by_task.items())
        },
    }


def _scenario_overlap_stats(
    records: list[TrajectoryRecord],
    metadata_by_episode: dict[str, dict[str, Any]],
    *,
    train_scenarios: set[str],
) -> dict[str, Any]:
    episode_ids = {record.episode_id for record in records}
    record_task_by_episode = {
        record.episode_id: record.goal.task_type.value for record in records
    }
    overlap = 0
    missing = 0
    per_task: dict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "overlap": 0, "missing": 0}
    )
    for episode_id in episode_ids:
        metadata = metadata_by_episode.get(episode_id)
        key = _scenario_key(metadata)
        if key is None:
            missing += 1
            task = record_task_by_episode.get(episode_id)
            if task is not None:
                per_task[task]["episodes"] += 1
                per_task[task]["missing"] += 1
            continue
        task = str(metadata["task_type"])
        per_task[task]["episodes"] += 1
        if key in train_scenarios:
            overlap += 1
            per_task[task]["overlap"] += 1
    return {
        "episode_count": len(episode_ids),
        "scenario_overlap_with_train_rate": overlap / max(len(episode_ids), 1),
        "episodes_without_metadata": missing,
        "per_task": {
            task: {
                "episode_count": stats["episodes"],
                "scenario_overlap_with_train_rate": stats["overlap"]
                / max(stats["episodes"], 1),
                "episodes_without_metadata": stats["missing"],
            }
            for task, stats in sorted(per_task.items())
        },
    }


def analyze_dataset_leakage(dataset_dir: str | Path) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    splits = {
        split_name: read_jsonl(dataset_dir / f"{split_name}.jsonl")
        for split_name in SPLIT_NAMES
    }
    metadata_by_episode = _load_episode_metadata(dataset_dir)
    train_records = splits["train"]
    train_state_keys = {_state_signature(record.state) for record in train_records}
    train_state_action_keys = {
        f"{_state_signature(record.state)}::{_action_signature(record.action)}"
        for record in train_records
    }
    train_scenarios = {
        key
        for episode_id in {record.episode_id for record in train_records}
        if (key := _scenario_key(metadata_by_episode.get(episode_id))) is not None
    }
    split_stats = {name: _split_stats(records) for name, records in splits.items()}
    scenario_split_stats = {
        name: _scenario_split_stats(records, metadata_by_episode)
        for name, records in splits.items()
    }
    overlap_with_train = {
        name: _overlap_stats(
            records,
            train_state_keys=train_state_keys,
            train_state_action_keys=train_state_action_keys,
        )
        for name, records in splits.items()
        if name != "train"
    }
    scenario_overlap_with_train = {
        name: _scenario_overlap_stats(
            records,
            metadata_by_episode,
            train_scenarios=train_scenarios,
        )
        for name, records in splits.items()
        if name != "train"
    }
    test_overlap = overlap_with_train.get("test", {})
    test_scenario_overlap = scenario_overlap_with_train.get("test", {})
    ceiling_risk = (
        test_overlap.get("state_action_overlap_with_train_rate", 0.0) >= 0.8
        or split_stats.get("test", {}).get("state_duplicate_rate", 0.0) >= 0.8
        or test_scenario_overlap.get("scenario_overlap_with_train_rate", 0.0) > 0.0
    )
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "split_stats": split_stats,
        "scenario_split_stats": scenario_split_stats,
        "overlap_with_train": overlap_with_train,
        "scenario_overlap_with_train": scenario_overlap_with_train,
        "ceiling_risk": ceiling_risk,
        "interpretation": (
            "High train/test state-action overlap means completion gains can be hidden by "
            "memorization-like baselines; use this report before promoting a small-model result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure train/test state overlap for Phase 1 splits.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    payload = analyze_dataset_leakage(args.dataset_dir)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
