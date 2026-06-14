from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.data.jsonl import read_jsonl
from reflexlm.llm.candidate_features import command_intent_for_text


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./-]+")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text)}


def _jaccard(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _commands_from_head_jsonl(path: str | Path | None) -> list[str]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    commands = []
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("head_scope") == "debug_cortex" and row.get("action_type") == "RUN_COMMAND":
            command = row.get("command")
            if command:
                commands.append(str(command))
    return commands


def _commands_from_trajectory(path: str | Path) -> list[str]:
    commands = []
    for record in read_jsonl(Path(path)):
        if record.action and record.action.command:
            commands.append(record.action.command)
    return commands


def _files_from_trajectory(path: str | Path) -> list[str]:
    files = []
    for record in read_jsonl(Path(path)):
        if record.action and record.action.file_target:
            files.append(record.action.file_target)
    return files


def build_phase2d_dataset_audit(
    *,
    dataset_jsonl: str | Path,
    train_head_jsonl: str | Path | None = None,
    manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    challenge_commands = sorted(set(_commands_from_trajectory(dataset_jsonl)))
    train_commands = sorted(set(_commands_from_head_jsonl(train_head_jsonl)))
    challenge_files = sorted(set(_files_from_trajectory(dataset_jsonl)))
    exact_overlap = sorted(set(challenge_commands) & set(train_commands))
    nearest = []
    for command in challenge_commands:
        best_command = None
        best_score = 0.0
        for train_command in train_commands:
            score = _jaccard(command, train_command)
            if score > best_score:
                best_score = score
                best_command = train_command
        nearest.append(
            {
                "challenge_command": command,
                "nearest_train_command": best_command,
                "semantic_nn_jaccard": round(best_score, 6),
                "intent": command_intent_for_text(command),
                "exact_overlap": command in set(train_commands),
            }
        )
    manifest = (
        json.loads(Path(manifest_json).read_text(encoding="utf-8-sig"))
        if manifest_json
        else None
    )
    return {
        "dataset_jsonl": str(Path(dataset_jsonl)),
        "train_head_jsonl": str(Path(train_head_jsonl)) if train_head_jsonl else None,
        "manifest_json": str(Path(manifest_json)) if manifest_json else None,
        "manifest": manifest,
        "command_overlap": {
            "train_command_count": len(train_commands),
            "challenge_command_count": len(challenge_commands),
            "exact_overlap_count": len(exact_overlap),
            "exact_overlap": exact_overlap,
            "challenge_only_commands": sorted(set(challenge_commands) - set(train_commands)),
        },
        "slot_overlap": {
            "challenge_file_slot_count": len(challenge_files),
            "challenge_file_slots": challenge_files,
        },
        "command_intent_counts": dict(
            Counter(command_intent_for_text(command) for command in challenge_commands)
        ),
        "semantic_nn_audit": nearest,
        "passed": manifest is None
        or (
            manifest.get("model_visible_hidden_hint_leaks") == 0
            and manifest.get("model_visible_scenario_template_leaks") == 0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2D dataset overlap and semantic nearest neighbors.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--train-head-jsonl")
    parser.add_argument("--manifest-json")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    report = build_phase2d_dataset_audit(
        dataset_jsonl=args.dataset_jsonl,
        train_head_jsonl=args.train_head_jsonl,
        manifest_json=args.manifest_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
