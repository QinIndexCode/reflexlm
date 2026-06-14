from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.data.jsonl import read_jsonl
from reflexlm.llm.candidate_features import command_intent_for_text
from reflexlm.llm.native_head_training import _balance_debug_command_intent_rows, _balanced_limited_rows
from reflexlm.models.features import candidate_commands, candidate_files, serialize_state_as_text


DEFAULT_OUTPUT = Path("artifacts/reports/phase2i_data_health_audit.json")
DEFAULT_HEAD_SPLITS = {
    "phase2i_head_train": Path("artifacts/datasets/phase2i_semantic_pairwise_head_canary/train.jsonl"),
    "phase2i_head_val": Path("artifacts/datasets/phase2i_semantic_pairwise_head_canary/val.jsonl"),
}
DEFAULT_CHALLENGE_SPLITS = {
    "phase2i_semantic_train": Path("artifacts/datasets/phase2i_semantic_train/challenge.jsonl"),
    "phase2i_semantic_val": Path("artifacts/datasets/phase2i_semantic_val/challenge.jsonl"),
    "external_trace_v3_semantic_required": Path(
        "artifacts/datasets/phase2i_external_trace_v3_semantic_required/challenge.jsonl"
    ),
}

FORBIDDEN_VISIBLE_MARKERS = (
    "recovery_hint=",
    "scenario_template",
    "oracle_action",
    "oracle_label",
    "oracle=",
    "oracle:",
    "correct_action",
    "correct_command",
    '"target_text"',
    "'target_text'",
    "target_text=",
    "target_text:",
    "return only json",
)


def _tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-zA-Z0-9_./-]+", value.lower()) if token}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def _parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, path = value.split("=", 1)
    return name, Path(path)


def _read_head_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _canonical_rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _command_slot(row: dict[str, Any]) -> int:
    try:
        return int(row.get("command_slot", -100))
    except (TypeError, ValueError):
        return -100


def _file_slot(row: dict[str, Any]) -> int:
    try:
        return int(row.get("file_slot", -100))
    except (TypeError, ValueError):
        return -100


def _max_share(counter: Counter[Any]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    return max(counter.values()) / total


def _head_split_summary(
    path: Path,
    *,
    limit: int | None = None,
    balance_debug_command_intents: bool = False,
) -> dict[str, Any]:
    rows = _read_head_rows(path)
    source_rows = len(rows)
    rows_after_intent_balance = len(rows)
    if balance_debug_command_intents:
        rows = _balance_debug_command_intent_rows(rows)
        rows_after_intent_balance = len(rows)
    if limit is not None and limit > 0:
        rows = _balanced_limited_rows(rows, limit)
    hidden_prompt_hits: Counter[str] = Counter()
    bad_labels: list[dict[str, Any]] = []
    command_slots: Counter[int] = Counter()
    file_slots: Counter[int] = Counter()
    command_intents: Counter[str] = Counter()
    command_candidate_counts: Counter[int] = Counter()
    file_candidate_counts: Counter[int] = Counter()
    actions: Counter[str] = Counter()
    head_scopes: Counter[str] = Counter()
    task_types: Counter[str] = Counter()
    same_intent_sets = 0
    run_rows = 0
    all_run_rows = 0
    all_command_slots: Counter[int] = Counter()
    target_commands: set[str] = set()
    candidate_command_set: set[str] = set()
    target_files: set[str] = set()
    candidate_file_set: set[str] = set()

    for row in rows:
        prompt = str(row.get("state_prompt") or "")
        lowered = prompt.lower()
        for marker in FORBIDDEN_VISIBLE_MARKERS:
            if marker in lowered:
                hidden_prompt_hits[marker] += 1

        action = str(row.get("action_type") or "")
        actions[action] += 1
        head_scopes[str(row.get("head_scope") or "")] += 1
        task_types[str(row.get("task_type") or "")] += 1
        commands = list(row.get("candidate_commands") or [])
        files = list(row.get("candidate_files") or [])
        candidate_command_set.update(commands)
        candidate_file_set.update(files)
        command_candidate_counts[len(commands)] += 1
        file_candidate_counts[len(files)] += 1

        command_slot = _command_slot(row)
        if command_slot != -100:
            all_run_rows += 1
            all_command_slots[command_slot] += 1
            command_intents[str(row.get("command_intent") or command_intent_for_text(row.get("command")))] += 1
            if command_slot < 0 or command_slot >= len(commands):
                bad_labels.append(
                    {
                        "example_id": row.get("example_id"),
                        "label": "command_slot_oob",
                        "slot": command_slot,
                        "candidate_count": len(commands),
                    }
                )
            else:
                target_commands.add(commands[command_slot])
            if str(row.get("head_scope") or "") == "debug_cortex":
                run_rows += 1
                command_slots[command_slot] += 1
                intents = [command_intent_for_text(command) for command in commands]
                if len(intents) != len(set(intents)):
                    same_intent_sets += 1

        file_slot = _file_slot(row)
        if file_slot != -100:
            file_slots[file_slot] += 1
            if file_slot < 0 or file_slot >= len(files):
                bad_labels.append(
                    {
                        "example_id": row.get("example_id"),
                        "label": "file_slot_oob",
                        "slot": file_slot,
                        "candidate_count": len(files),
                    }
                )
            else:
                target_files.add(files[file_slot])

    return {
        "path": str(path),
        "exists": path.exists(),
        "source_rows": source_rows,
        "rows_after_intent_balance": rows_after_intent_balance,
        "effective_limit": limit,
        "effective_split_sha256": _canonical_rows_sha256(rows),
        "effective_split_hash_rows": len(rows),
        "rows": len(rows),
        "episodes": len({row.get("episode_id") for row in rows}),
        "task_types": dict(sorted(task_types.items())),
        "actions": dict(sorted(actions.items())),
        "head_scopes": dict(sorted(head_scopes.items())),
        "run_rows": run_rows,
        "all_run_rows": all_run_rows,
        "command_slots": dict(sorted(command_slots.items())),
        "command_slot_max_share": round(_max_share(command_slots), 6),
        "all_command_slots": dict(sorted(all_command_slots.items())),
        "all_command_slot_max_share": round(_max_share(all_command_slots), 6),
        "file_slots": dict(sorted(file_slots.items())),
        "file_slot_max_share": round(_max_share(file_slots), 6),
        "command_intents": dict(sorted(command_intents.items())),
        "command_candidate_counts": dict(sorted(command_candidate_counts.items())),
        "file_candidate_counts": dict(sorted(file_candidate_counts.items())),
        "same_intent_candidate_sets": same_intent_sets,
        "same_intent_candidate_rate": round(same_intent_sets / run_rows, 6) if run_rows else 0.0,
        "target_commands": sorted(target_commands),
        "candidate_commands": sorted(candidate_command_set),
        "target_files": sorted(target_files),
        "candidate_files": sorted(candidate_file_set),
        "unique_target_command_count": len(target_commands),
        "unique_candidate_command_count": len(candidate_command_set),
        "unique_target_file_count": len(target_files),
        "hidden_prompt_hits": dict(sorted(hidden_prompt_hits.items())),
        "bad_labels": bad_labels[:25],
        "bad_label_count": len(bad_labels),
        "passed": path.exists()
        and not hidden_prompt_hits
        and not bad_labels,
    }


def _challenge_split_summary(path: Path) -> dict[str, Any]:
    records = read_jsonl(path) if path.exists() else []
    visible_hidden_hits: Counter[str] = Counter()
    bad_allowlist_targets: list[dict[str, Any]] = []
    task_types: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    command_slots: Counter[int] = Counter()
    command_intents: Counter[str] = Counter()
    target_commands: set[str] = set()
    candidate_command_set: set[str] = set()
    target_files: set[str] = set()
    candidate_file_set: set[str] = set()
    state_texts: list[str] = []
    by_episode: dict[str, list[Any]] = {}

    for record in records:
        by_episode.setdefault(record.episode_id, []).append(record)
        task_types[record.goal.task_type.value] += 1
        actions[record.action.type.value if record.action else "<none>"] += 1
        state_text = serialize_state_as_text(record.state)
        lowered = state_text.lower()
        for marker in FORBIDDEN_VISIBLE_MARKERS:
            if marker in lowered:
                visible_hidden_hits[marker] += 1

        commands = candidate_commands(record.state)
        files = candidate_files(record.state)
        candidate_command_set.update(commands)
        candidate_file_set.update(files)
        if "source inspected:" in record.state.terminal.stdout_delta.lower():
            state_texts.append(
                "\n".join(
                    [
                        record.state.terminal.stdout_delta,
                        record.state.terminal.stderr_delta,
                        " ".join(candidate_files(record.state)),
                        " ".join(record.state.filesystem.watched_paths),
                    ]
                )
            )
        if record.action and record.action.command:
            target_commands.add(record.action.command)
            command_intents[command_intent_for_text(record.action.command)] += 1
            if commands and record.action.command not in commands:
                bad_allowlist_targets.append(
                    {
                        "episode_id": record.episode_id,
                        "t": record.t,
                        "command": record.action.command,
                    }
                )
            if record.action.command in commands:
                command_slots[commands.index(record.action.command)] += 1
        if record.action and record.action.file_target:
            target_files.add(record.action.file_target)

    ambiguous_episodes = 0
    semantic_source_episodes = 0
    non_last_command_episodes = 0
    for episode_records in by_episode.values():
        ordered = sorted(episode_records, key=lambda item: item.t)
        first_state = ordered[0].state
        commands = candidate_commands(first_state)
        intent_counts = Counter(command_intent_for_text(command) for command in commands)
        if any(count >= 2 for count in intent_counts.values()):
            ambiguous_episodes += 1
        if any(
            "semantic disambiguation required"
            in f"{record.state.terminal.stdout_delta} {record.state.terminal.stderr_delta}".lower()
            for record in ordered
        ):
            semantic_source_episodes += 1
        run_records = [record for record in ordered if record.action and record.action.command]
        correct_command = run_records[-1].action.command if run_records else None
        if correct_command and correct_command != first_state.terminal.last_command:
            non_last_command_episodes += 1

    episode_count = len(by_episode)
    return {
        "path": str(path),
        "exists": path.exists(),
        "rows": len(records),
        "episodes": episode_count,
        "task_types": dict(sorted(task_types.items())),
        "actions": dict(sorted(actions.items())),
        "command_slots": dict(sorted(command_slots.items())),
        "command_slot_max_share": round(_max_share(command_slots), 6),
        "command_intents": dict(sorted(command_intents.items())),
        "target_commands": sorted(target_commands),
        "candidate_commands": sorted(candidate_command_set),
        "target_files": sorted(target_files),
        "candidate_files": sorted(candidate_file_set),
        "unique_target_command_count": len(target_commands),
        "unique_candidate_command_count": len(candidate_command_set),
        "unique_target_file_count": len(target_files),
        "model_visible_hidden_hits": dict(sorted(visible_hidden_hits.items())),
        "bad_allowlist_targets": bad_allowlist_targets[:25],
        "bad_allowlist_target_count": len(bad_allowlist_targets),
        "ambiguous_same_intent_episode_count": ambiguous_episodes,
        "semantic_source_episode_count": semantic_source_episodes,
        "correct_command_differs_from_last_command_episode_count": non_last_command_episodes,
        "same_intent_episode_rate": round(ambiguous_episodes / episode_count, 6) if episode_count else 0.0,
        "semantic_source_episode_rate": round(semantic_source_episodes / episode_count, 6) if episode_count else 0.0,
        "non_last_command_episode_rate": round(non_last_command_episodes / episode_count, 6) if episode_count else 0.0,
        "_state_texts": state_texts,
        "passed": path.exists()
        and not visible_hidden_hits
        and not bad_allowlist_targets
        and episode_count > 0
        and ambiguous_episodes == episode_count
        and semantic_source_episodes == episode_count
        and non_last_command_episodes == episode_count,
    }


def _overlap(left: set[str], right: set[str]) -> dict[str, Any]:
    items = sorted(left & right)
    return {
        "count": len(items),
        "rate_vs_left": round(len(items) / max(len(left), 1), 6),
        "rate_vs_right": round(len(items) / max(len(right), 1), 6),
        "items": items[:50],
    }


def _semantic_nn(left_texts: list[str], right_texts: list[str]) -> dict[str, Any]:
    left_tokens = [_tokens(text) for text in left_texts]
    right_tokens = [_tokens(text) for text in right_texts]
    best = {"score": 0.0, "left_index": None, "right_index": None}
    for left_index, left in enumerate(left_tokens):
        for right_index, right in enumerate(right_tokens):
            score = _jaccard(left, right)
            if score > float(best["score"]):
                best = {
                    "score": round(score, 6),
                    "left_index": left_index,
                    "right_index": right_index,
                }
    return best


def build_phase2i_data_health_audit(
    *,
    head_splits: dict[str, Path],
    challenge_splits: dict[str, Path],
    reference_splits: dict[str, Path] | None = None,
    head_limits: dict[str, int] | None = None,
    balance_debug_command_intents: bool = False,
    max_command_slot_share: float = 0.45,
    min_val_target_commands: int = 6,
    max_train_val_target_overlap: float = 0.25,
    max_semantic_nn: float = 0.80,
) -> dict[str, Any]:
    effective_head_limits = head_limits or {}
    head = {
        name: _head_split_summary(
            path,
            limit=effective_head_limits.get(name),
            balance_debug_command_intents=balance_debug_command_intents,
        )
        for name, path in head_splits.items()
    }
    challenge = {
        name: _challenge_split_summary(path)
        for name, path in challenge_splits.items()
    }
    references = {
        name: _challenge_split_summary(path)
        for name, path in (reference_splits or {}).items()
    }

    overlaps: dict[str, Any] = {}
    for left_name, left in {**head, **challenge}.items():
        left_targets = set(left.get("target_commands", []))
        left_candidates = set(left.get("candidate_commands", []))
        for right_name, right in {**head, **challenge, **references}.items():
            if left_name == right_name:
                continue
            right_targets = set(right.get("target_commands", []))
            right_candidates = set(right.get("candidate_commands", []))
            overlaps[f"{left_name}__vs__{right_name}"] = {
                "target_command_overlap": _overlap(left_targets, right_targets),
                "candidate_command_overlap": _overlap(left_candidates, right_candidates),
            }

    semantic_nn: dict[str, Any] = {}
    challenge_names = list(challenge)
    for index, left_name in enumerate(challenge_names):
        for right_name in challenge_names[index + 1 :]:
            best = _semantic_nn(
                list(challenge[left_name].get("_state_texts", [])),
                list(challenge[right_name].get("_state_texts", [])),
            )
            semantic_nn[f"{left_name}__vs__{right_name}"] = best
    for summary in challenge.values():
        summary.pop("_state_texts", None)
    for summary in references.values():
        summary.pop("_state_texts", None)

    head_checks: dict[str, bool] = {}
    for name, summary in head.items():
        head_checks[f"{name}_basic_health"] = bool(summary["passed"])
        if summary["run_rows"]:
            head_checks[f"{name}_command_slot_max_share"] = (
                float(summary["command_slot_max_share"]) <= max_command_slot_share
            )
        if "val" in name:
            head_checks[f"{name}_target_command_coverage"] = (
                int(summary["unique_target_command_count"]) >= min_val_target_commands
            )

    challenge_checks = {
        f"{name}_basic_health": bool(summary["passed"])
        for name, summary in challenge.items()
    }

    train_val_overlap_ok = True
    train_val_intent_gap: dict[str, Any] = {
        "missing_train_command_intents": [],
        "train_command_intents": [],
        "val_command_intents": [],
    }
    train_val_intent_coverage_ok = True
    if "phase2i_head_train" in head and "phase2i_head_val" in head:
        overlap = _overlap(
            set(head["phase2i_head_train"]["target_commands"]),
            set(head["phase2i_head_val"]["target_commands"]),
        )
        train_val_overlap_ok = float(overlap["rate_vs_right"]) <= max_train_val_target_overlap
        overlaps["phase2i_head_train__vs__phase2i_head_val"]["target_command_overlap"] = overlap
        train_intents = set(head["phase2i_head_train"].get("command_intents", {}))
        val_intents = set(head["phase2i_head_val"].get("command_intents", {}))
        missing_train_intents = sorted(val_intents - train_intents)
        train_val_intent_gap = {
            "missing_train_command_intents": missing_train_intents,
            "train_command_intents": sorted(train_intents),
            "val_command_intents": sorted(val_intents),
        }
        train_val_intent_coverage_ok = not missing_train_intents

    external_v3_overlap_ok = True
    if "external_trace_v3_semantic_required" in challenge:
        v3_targets = set(challenge["external_trace_v3_semantic_required"]["target_commands"])
        v3_candidates = set(challenge["external_trace_v3_semantic_required"]["candidate_commands"])
        phase2i_sources = {
            **{name: summary for name, summary in head.items() if name.startswith("phase2i")},
            **{
                name: summary
                for name, summary in challenge.items()
                if name.startswith("phase2i") and name != "external_trace_v3_semantic_required"
            },
        }
        for name, summary in phase2i_sources.items():
            target_overlap = _overlap(v3_targets, set(summary["target_commands"]))
            candidate_overlap = _overlap(v3_candidates, set(summary["candidate_commands"]))
            overlaps[
                f"external_trace_v3_semantic_required__phase2i_guard__{name}"
            ] = {
                "target_command_overlap": target_overlap,
                "candidate_command_overlap": candidate_overlap,
            }
            if target_overlap["count"]:
                external_v3_overlap_ok = False

    semantic_nn_ok = all(
        float(best["score"]) < max_semantic_nn for best in semantic_nn.values()
    )
    checks = {
        **head_checks,
        **challenge_checks,
        "phase2i_effective_split_hashes_present": all(
            bool(summary.get("effective_split_sha256")) for summary in head.values()
        ),
        "phase2i_train_val_target_overlap": train_val_overlap_ok,
        "phase2i_train_val_command_intent_coverage": train_val_intent_coverage_ok,
        "external_v3_has_no_phase2i_command_overlap": external_v3_overlap_ok,
        "semantic_nearest_neighbor_below_threshold": semantic_nn_ok,
    }
    return {
        "audit_family": "phase2i_data_health",
        "thresholds": {
            "max_command_slot_share": max_command_slot_share,
            "min_val_target_commands": min_val_target_commands,
            "max_train_val_target_overlap": max_train_val_target_overlap,
            "max_semantic_nn": max_semantic_nn,
        },
        "passed": all(checks.values()),
        "checks": checks,
        "effective_split_hashes": {
            name: summary["effective_split_sha256"] for name, summary in head.items()
        },
        "head_splits": head,
        "challenge_splits": challenge,
        "reference_splits": references,
        "train_val_command_intent_gap": train_val_intent_gap,
        "overlaps": overlaps,
        "semantic_nearest_neighbors": semantic_nn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2I data health and overlap.")
    parser.add_argument("--head-split", action="append", default=[])
    parser.add_argument("--challenge-split", action="append", default=[])
    parser.add_argument("--reference-split", action="append", default=[])
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-command-slot-share", type=float, default=0.45)
    parser.add_argument("--min-val-target-commands", type=int, default=6)
    parser.add_argument("--max-train-val-target-overlap", type=float, default=0.25)
    parser.add_argument("--max-semantic-nn", type=float, default=0.80)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=0)
    parser.add_argument("--balance-debug-command-intents", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    head_splits = (
        dict(_parse_named_path(value) for value in args.head_split)
        if args.head_split
        else DEFAULT_HEAD_SPLITS
    )
    challenge_splits = (
        dict(_parse_named_path(value) for value in args.challenge_split)
        if args.challenge_split
        else DEFAULT_CHALLENGE_SPLITS
    )
    reference_splits = dict(_parse_named_path(value) for value in args.reference_split)
    head_limits: dict[str, int] = {}
    if args.max_train_records > 0:
        for name in head_splits:
            if "train" in name:
                head_limits[name] = args.max_train_records
    if args.max_val_records > 0:
        for name in head_splits:
            if "val" in name:
                head_limits[name] = args.max_val_records
    report = build_phase2i_data_health_audit(
        head_splits=head_splits,
        challenge_splits=challenge_splits,
        reference_splits=reference_splits,
        head_limits=head_limits,
        balance_debug_command_intents=args.balance_debug_command_intents,
        max_command_slot_share=args.max_command_slot_share,
        min_val_target_commands=args.min_val_target_commands,
        max_train_val_target_overlap=args.max_train_val_target_overlap,
        max_semantic_nn=args.max_semantic_nn,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
