from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reflexlm.data.jsonl import read_jsonl, write_jsonl
from reflexlm.data.tasks import build_env_from_episode_id
from reflexlm.eval import _actions_equivalent
from reflexlm.schema import SourceType, TrajectoryRecord


SPLIT_NAMES = ("train", "val", "test")


def _episode_order(records: list[TrajectoryRecord]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for record in records:
        if record.episode_id not in seen:
            seen.add(record.episode_id)
            ordered.append(record.episode_id)
    return ordered


def _canonical_rollout(episode_id: str, *, env_profile: str) -> list[TrajectoryRecord]:
    env = build_env_from_episode_id(episode_id, profile=env_profile)
    state = env.reset()
    records: list[TrajectoryRecord] = []
    done = False
    step_index = 0
    while not done and step_index < env.max_steps:
        action = env.oracle_action(state)
        next_state, reward, done, _ = env.step(action)
        records.append(
            TrajectoryRecord(
                episode_id=episode_id,
                t=step_index,
                goal=state.goal,
                state=state,
                action=action,
                next_state=next_state,
                reward=reward,
                done=done,
                source=SourceType.RULE_ORACLE,
            )
        )
        state = next_state
        step_index += 1
    return records


def _compare_split(
    original_records: list[TrajectoryRecord],
    canonical_records: list[TrajectoryRecord],
) -> dict[str, Any]:
    original_by_key = {
        (record.episode_id, record.t): record for record in original_records
    }
    canonical_by_key = {
        (record.episode_id, record.t): record for record in canonical_records
    }
    action_mismatches: list[dict[str, Any]] = []
    state_mismatches = 0
    missing_in_original = 0
    extra_in_original = 0
    for key, canonical in canonical_by_key.items():
        original = original_by_key.get(key)
        if original is None:
            missing_in_original += 1
            continue
        if original.state.model_dump(mode="json") != canonical.state.model_dump(mode="json"):
            state_mismatches += 1
        if original.action is None or not _actions_equivalent(original.action, canonical.action):
            action_mismatches.append(
                {
                    "episode_id": canonical.episode_id,
                    "t": canonical.t,
                    "task_type": canonical.goal.task_type.value,
                    "original_action": original.action.model_dump(mode="json")
                    if original.action
                    else None,
                    "canonical_action": canonical.action.model_dump(mode="json"),
                }
            )
    for key in original_by_key:
        if key not in canonical_by_key:
            extra_in_original += 1
    by_task: dict[str, int] = defaultdict(int)
    for mismatch in action_mismatches:
        by_task[str(mismatch["task_type"])] += 1
    return {
        "original_record_count": len(original_records),
        "canonical_record_count": len(canonical_records),
        "action_mismatch_count": len(action_mismatches),
        "action_mismatch_count_by_task": dict(sorted(by_task.items())),
        "state_mismatch_count": state_mismatches,
        "missing_in_original": missing_in_original,
        "extra_in_original": extra_in_original,
        "action_mismatch_examples": action_mismatches[:20],
    }


def repair_dataset(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    env_profile: str = "default",
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("output_dir must differ from input_dir to preserve the source dataset")

    split_reports: dict[str, Any] = {}
    total_action_mismatches = 0
    total_state_mismatches = 0
    for split_name in SPLIT_NAMES:
        source_path = input_dir / f"{split_name}.jsonl"
        original_records = read_jsonl(source_path)
        repaired_records: list[TrajectoryRecord] = []
        for episode_id in _episode_order(original_records):
            repaired_records.extend(_canonical_rollout(episode_id, env_profile=env_profile))
        write_jsonl(output_dir / f"{split_name}.jsonl", repaired_records)
        split_report = _compare_split(original_records, repaired_records)
        split_reports[split_name] = split_report
        total_action_mismatches += int(split_report["action_mismatch_count"])
        total_state_mismatches += int(split_report["state_mismatch_count"])

    manifest = {
        "dataset_revision": "env_consistent_oracle_rollout",
        "source_dataset_dir": str(input_dir.resolve()),
        "output_dataset_dir": str(output_dir.resolve()),
        "env_profile": env_profile,
        "split_reports": split_reports,
        "total_action_mismatch_count": total_action_mismatches,
        "total_state_mismatch_count": total_state_mismatches,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an env-consistent Phase 1 dataset without overwriting the source split."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-profile", default="default")
    parser.add_argument("--report-json")
    args = parser.parse_args()
    payload = repair_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        env_profile=args.env_profile,
    )
    if args.report_json:
        output_path = Path(args.report_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
