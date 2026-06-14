from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reflexlm.data.jsonl import read_jsonl
from reflexlm.llm.sft import SynapseSignalExtractor, _build_examples


HIDDEN_LEAKAGE_MARKERS = (
    "recovery_hint",
    "scenario_template",
    "profile_seed",
    "oracle_action",
    "future_state",
)


def _read_jsonl_dicts(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_payload(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _hash_text(text)


def _action_tuple(row: dict[str, Any]) -> tuple[str, str | None, str | None]:
    return (
        str(row.get("action_type") or ""),
        row.get("command"),
        row.get("file_target"),
    )


def _sft_row_signature(row: dict[str, Any]) -> dict[str, str]:
    prompt_hash = _hash_text(str(row.get("user_prompt") or ""))
    target_hash = _hash_text(str(row.get("target_text") or ""))
    return {
        "prompt": prompt_hash,
        "target": target_hash,
        "prompt_target": _hash_payload([prompt_hash, target_hash]),
    }


def _metadata_by_episode(dataset_dir: Path | None) -> dict[str, dict[str, Any]]:
    if dataset_dir is None:
        return {}
    path = dataset_dir / "episode_metadata.json"
    if not path.exists():
        return {}
    return {
        str(row["episode_id"]): row
        for row in json.loads(path.read_text(encoding="utf-8"))
    }


def _scenario_key(
    episode_id: str,
    metadata_by_episode: dict[str, dict[str, Any]],
) -> str | None:
    metadata = metadata_by_episode.get(episode_id)
    if not metadata:
        return None
    return f"{metadata['task_type']}::{metadata['scenario_template']}"


def _build_test_sft_rows(
    *,
    test_jsonl: str | Path,
    prompt_style: str,
    synapse_checkpoint: str | Path,
    synapse_device: str,
) -> list[dict[str, Any]]:
    records = read_jsonl(Path(test_jsonl))
    extractor = SynapseSignalExtractor(synapse_checkpoint, device=synapse_device)
    examples = _build_examples(
        records,
        prompt_style=prompt_style,
        synapse_extractor=extractor,
    )
    return [example.to_dict() for example in examples]


def _split_index(
    rows: list[dict[str, Any]],
    *,
    metadata_by_episode: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    signatures = [_sft_row_signature(row) for row in rows]
    prompts = {signature["prompt"] for signature in signatures}
    targets = {signature["target"] for signature in signatures}
    prompt_targets = {signature["prompt_target"] for signature in signatures}
    episode_ids = {str(row.get("episode_id")) for row in rows}
    scenarios = {
        key
        for episode_id in episode_ids
        if (key := _scenario_key(episode_id, metadata_by_episode)) is not None
    }
    action_tuples = {_action_tuple(row) for row in rows}
    command_slots = {
        row.get("command")
        for row in rows
        if row.get("command") is not None
    }
    file_slots = {
        row.get("file_target")
        for row in rows
        if row.get("file_target") is not None
    }
    per_task = Counter(str(row.get("task_type") or "") for row in rows)
    leakage_hits = [
        {
            "example_id": row.get("example_id"),
            "marker": marker,
        }
        for row in rows
        for marker in HIDDEN_LEAKAGE_MARKERS
        if marker in str(row.get("user_prompt") or "")
    ]
    return {
        "row_count": len(rows),
        "episode_count": len(episode_ids),
        "scenario_count": len(scenarios),
        "prompts": prompts,
        "targets": targets,
        "prompt_targets": prompt_targets,
        "episode_ids": episode_ids,
        "scenarios": scenarios,
        "action_tuples": action_tuples,
        "command_slots": command_slots,
        "file_slots": file_slots,
        "per_task": dict(sorted(per_task.items())),
        "hidden_leakage_hits": leakage_hits,
    }


def _overlap_rate(values: set[Any], reference: set[Any]) -> float:
    return len(values & reference) / max(len(values), 1)


def _per_task_overlap(
    rows: list[dict[str, Any]],
    *,
    train_prompt_hashes: set[str],
    train_prompt_target_hashes: set[str],
    train_episode_ids: set[str],
    train_scenarios: set[str],
    metadata_by_episode: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        task_rows[str(row.get("task_type") or "")].append(row)
    report: dict[str, Any] = {}
    for task_name, task_items in sorted(task_rows.items()):
        signatures = [_sft_row_signature(row) for row in task_items]
        prompts = {signature["prompt"] for signature in signatures}
        prompt_targets = {signature["prompt_target"] for signature in signatures}
        episode_ids = {str(row.get("episode_id")) for row in task_items}
        scenarios = {
            key
            for episode_id in episode_ids
            if (key := _scenario_key(episode_id, metadata_by_episode)) is not None
        }
        report[task_name] = {
            "row_count": len(task_items),
            "prompt_overlap_with_train_rate": _overlap_rate(prompts, train_prompt_hashes),
            "prompt_target_overlap_with_train_rate": _overlap_rate(
                prompt_targets,
                train_prompt_target_hashes,
            ),
            "episode_overlap_with_train_rate": _overlap_rate(episode_ids, train_episode_ids),
            "scenario_overlap_with_train_rate": _overlap_rate(scenarios, train_scenarios),
        }
    return report


def analyze_phase2b_generalization(
    *,
    train_sft_jsonl: str | Path,
    val_sft_jsonl: str | Path,
    test_jsonl: str | Path,
    synapse_checkpoint: str | Path,
    dataset_dir: str | Path | None = None,
    prompt_style: str = "nsi_state_v2",
    synapse_device: str = "cpu",
) -> dict[str, Any]:
    metadata_by_episode = _metadata_by_episode(Path(dataset_dir) if dataset_dir else None)
    train_rows = _read_jsonl_dicts(train_sft_jsonl)
    val_rows = _read_jsonl_dicts(val_sft_jsonl)
    test_rows = _build_test_sft_rows(
        test_jsonl=test_jsonl,
        prompt_style=prompt_style,
        synapse_checkpoint=synapse_checkpoint,
        synapse_device=synapse_device,
    )

    train_index = _split_index(train_rows, metadata_by_episode=metadata_by_episode)
    val_index = _split_index(val_rows, metadata_by_episode=metadata_by_episode)
    test_index = _split_index(test_rows, metadata_by_episode=metadata_by_episode)

    prompt_overlap = _overlap_rate(test_index["prompts"], train_index["prompts"])
    prompt_target_overlap = _overlap_rate(
        test_index["prompt_targets"],
        train_index["prompt_targets"],
    )
    episode_overlap = _overlap_rate(test_index["episode_ids"], train_index["episode_ids"])
    scenario_overlap = _overlap_rate(test_index["scenarios"], train_index["scenarios"])
    hidden_leakage_count = len(train_index["hidden_leakage_hits"]) + len(
        test_index["hidden_leakage_hits"]
    )
    passed = (
        prompt_overlap == 0.0
        and prompt_target_overlap == 0.0
        and episode_overlap == 0.0
        and scenario_overlap == 0.0
        and hidden_leakage_count == 0
    )
    return {
        "passed": passed,
        "scope": "fast train/val SFT versus held-out test SFT prompts",
        "inputs": {
            "train_sft_jsonl": str(Path(train_sft_jsonl).resolve()),
            "val_sft_jsonl": str(Path(val_sft_jsonl).resolve()),
            "test_jsonl": str(Path(test_jsonl).resolve()),
            "dataset_dir": str(Path(dataset_dir).resolve()) if dataset_dir else None,
            "prompt_style": prompt_style,
            "synapse_checkpoint": str(Path(synapse_checkpoint).resolve()),
        },
        "split_counts": {
            "train": {
                "row_count": train_index["row_count"],
                "episode_count": train_index["episode_count"],
                "scenario_count": train_index["scenario_count"],
                "per_task": train_index["per_task"],
            },
            "val": {
                "row_count": val_index["row_count"],
                "episode_count": val_index["episode_count"],
                "scenario_count": val_index["scenario_count"],
                "per_task": val_index["per_task"],
            },
            "test": {
                "row_count": test_index["row_count"],
                "episode_count": test_index["episode_count"],
                "scenario_count": test_index["scenario_count"],
                "per_task": test_index["per_task"],
            },
        },
        "overlap_with_train": {
            "test_prompt_overlap_rate": prompt_overlap,
            "test_prompt_target_overlap_rate": prompt_target_overlap,
            "test_episode_overlap_rate": episode_overlap,
            "test_scenario_overlap_rate": scenario_overlap,
            "test_target_overlap_rate": _overlap_rate(
                test_index["targets"],
                train_index["targets"],
            ),
            "test_action_tuple_overlap_rate": _overlap_rate(
                test_index["action_tuples"],
                train_index["action_tuples"],
            ),
            "test_command_slot_overlap_rate": _overlap_rate(
                test_index["command_slots"],
                train_index["command_slots"],
            ),
            "test_file_slot_overlap_rate": _overlap_rate(
                test_index["file_slots"],
                train_index["file_slots"],
            ),
        },
        "per_task_overlap_with_train": _per_task_overlap(
            test_rows,
            train_prompt_hashes=train_index["prompts"],
            train_prompt_target_hashes=train_index["prompt_targets"],
            train_episode_ids=train_index["episode_ids"],
            train_scenarios=train_index["scenarios"],
            metadata_by_episode=metadata_by_episode,
        ),
        "hidden_leakage": {
            "markers": list(HIDDEN_LEAKAGE_MARKERS),
            "hit_count": hidden_leakage_count,
            "train_hits": train_index["hidden_leakage_hits"][:20],
            "test_hits": test_index["hidden_leakage_hits"][:20],
        },
        "interpretation": (
            "A pass only rules out exact episode/scenario/prompt memorization and hidden "
            "field leakage. It does not prove real-world generalization; final claims still "
            "depend on held-out evaluation and later quasi-real trajectories."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase 2B SFT/test overlap to detect memorization risk."
    )
    parser.add_argument("--train-sft-jsonl", required=True)
    parser.add_argument("--val-sft-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--synapse-checkpoint", required=True)
    parser.add_argument("--dataset-dir")
    parser.add_argument("--prompt-style", default="nsi_state_v2")
    parser.add_argument("--synapse-device", default="cpu")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    payload = analyze_phase2b_generalization(
        train_sft_jsonl=args.train_sft_jsonl,
        val_sft_jsonl=args.val_sft_jsonl,
        test_jsonl=args.test_jsonl,
        synapse_checkpoint=args.synapse_checkpoint,
        dataset_dir=args.dataset_dir,
        prompt_style=args.prompt_style,
        synapse_device=args.synapse_device,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not args.no_fail and not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
