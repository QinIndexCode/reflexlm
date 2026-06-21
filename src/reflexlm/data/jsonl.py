from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from pathlib import Path
from random import Random

from reflexlm.schema import TrajectoryRecord


def write_jsonl(path: Path, records: list[TrajectoryRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def read_jsonl(path: Path) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(TrajectoryRecord.model_validate(payload))
    return records


def split_records_by_episode(
    records: list[TrajectoryRecord],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[TrajectoryRecord]]:
    grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.episode_id].append(record)
    episode_ids = list(grouped)
    rng = Random(seed)
    rng.shuffle(episode_ids)
    total = len(episode_ids)
    train_cut = int(total * train_ratio)
    val_cut = int(total * (train_ratio + val_ratio))
    split_ids = {
        "train": set(episode_ids[:train_cut]),
        "val": set(episode_ids[train_cut:val_cut]),
        "test": set(episode_ids[val_cut:]),
    }
    return {
        split: [record for episode_id in ids for record in grouped[episode_id]]
        for split, ids in split_ids.items()
    }


def _episode_fingerprint(records: list[TrajectoryRecord]) -> str:
    payload = [
        {
            "state": record.state.model_dump(mode="json"),
            "action": record.action.model_dump(mode="json") if record.action else None,
        }
        for record in sorted(records, key=lambda item: item.t)
    ]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_records_by_episode_fingerprint(
    records: list[TrajectoryRecord],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[TrajectoryRecord]]:
    grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.episode_id].append(record)
    fingerprint_groups_by_task: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode_id, episode_records in grouped.items():
        task_type = episode_records[0].goal.task_type.value
        fingerprint_groups_by_task[task_type][_episode_fingerprint(episode_records)].append(
            episode_id
        )

    rng = Random(seed)
    split_ids = {"train": set(), "val": set(), "test": set()}
    for task_type, fingerprint_groups in sorted(fingerprint_groups_by_task.items()):
        fingerprints = list(fingerprint_groups)
        rng.shuffle(fingerprints)
        task_split_ids = {"train": set(), "val": set(), "test": set()}
        fingerprint_count = len(fingerprints)
        if fingerprint_count >= 3:
            val_count = max(1, int(fingerprint_count * val_ratio))
            test_count = max(1, fingerprint_count - int(fingerprint_count * (train_ratio + val_ratio)))
            train_count = max(1, fingerprint_count - val_count - test_count)
        elif fingerprint_count == 2:
            train_count, val_count, test_count = 1, 0, 1
        else:
            train_count, val_count, test_count = fingerprint_count, 0, 0
        split_for_index = (
            ["train"] * train_count
            + ["val"] * val_count
            + ["test"] * test_count
        )
        for index, fingerprint in enumerate(fingerprints):
            episode_ids = fingerprint_groups[fingerprint]
            split = split_for_index[min(index, len(split_for_index) - 1)]
            task_split_ids[split].update(episode_ids)
        for split, episode_ids in task_split_ids.items():
            split_ids[split].update(episode_ids)
    return {
        split: [record for episode_id in ids for record in grouped[episode_id]]
        for split, ids in split_ids.items()
    }


def split_records_by_scenario_holdout(
    records: list[TrajectoryRecord],
    *,
    episode_metadata: dict[str, dict[str, object]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[TrajectoryRecord]]:
    grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        grouped[record.episode_id].append(record)

    scenario_groups_by_task: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode_id, episode_records in grouped.items():
        metadata = episode_metadata.get(episode_id)
        if metadata is None:
            raise ValueError(f"Missing episode metadata for {episode_id}")
        task_type = str(metadata["task_type"])
        scenario_template = str(metadata["scenario_template"])
        scenario_groups_by_task[task_type][scenario_template].append(episode_id)

    rng = Random(seed)
    split_ids = {"train": set(), "val": set(), "test": set()}
    for _task_type, scenario_groups in sorted(scenario_groups_by_task.items()):
        scenarios = list(scenario_groups)
        rng.shuffle(scenarios)
        scenario_count = len(scenarios)
        if scenario_count >= 3:
            val_count = max(1, int(scenario_count * val_ratio))
            test_count = max(1, scenario_count - int(scenario_count * (train_ratio + val_ratio)))
            train_count = max(1, scenario_count - val_count - test_count)
        elif scenario_count == 2:
            train_count, val_count, test_count = 1, 0, 1
        else:
            train_count, val_count, test_count = scenario_count, 0, 0
        split_for_index = (
            ["train"] * train_count
            + ["val"] * val_count
            + ["test"] * test_count
        )
        for index, scenario_template in enumerate(scenarios):
            split = split_for_index[min(index, len(split_for_index) - 1)]
            split_ids[split].update(scenario_groups[scenario_template])

    return {
        split: [record for episode_id in ids for record in grouped[episode_id]]
        for split, ids in split_ids.items()
    }
