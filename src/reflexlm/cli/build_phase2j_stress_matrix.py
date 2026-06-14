from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reflexlm.llm.candidate_features import source_overlap_command_slot_prediction
from reflexlm.llm.receptor_latent import COMMAND_IDENTITY_SLOT_FIELDS, runtime_command_identity_signal
from reflexlm.models.features import candidate_commands
from reflexlm.schema import ActionType, SystemStateFrame


GRADED_METRICS = [
    "task_completion_rate",
    "oracle_step_accuracy",
    "run_command_action_accuracy",
    "command_decision_accuracy",
    "command_slot_match_when_run_command",
    "read_file_decision_accuracy",
    "positive_reward_credit",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _action_type(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("type")
    return str(value) if value is not None else None


def _trace_episode_scores(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    correct = sum(1 for row in rows if row.get("correct") is True)
    total = len(rows)
    completed = any(row.get("done") and float(row.get("reward", 0.0)) > 0.0 for row in rows)
    positive_reward = sum(max(float(row.get("reward", 0.0)), 0.0) for row in rows)
    command_rows = [
        row for row in rows if _action_type(row.get("oracle_action")) == ActionType.RUN_COMMAND.value
    ]
    run_command_action_rows = [
        row for row in command_rows if _action_type(row.get("action")) == ActionType.RUN_COMMAND.value
    ]
    command_slot_matches = [
        row
        for row in run_command_action_rows
        if (row.get("action") or {}).get("command")
        == (row.get("oracle_action") or {}).get("command")
    ]
    read_file_rows = [
        row for row in rows if _action_type(row.get("oracle_action")) == ActionType.READ_FILE.value
    ]
    return {
        "task_completion_rate": 1.0 if completed else 0.0,
        "oracle_step_accuracy": correct / total if total else None,
        "run_command_action_accuracy": (
            len(run_command_action_rows) / len(command_rows)
            if command_rows
            else None
        ),
        "command_decision_accuracy": (
            sum(1 for row in command_rows if row.get("correct") is True) / len(command_rows)
            if command_rows
            else None
        ),
        "command_slot_match_when_run_command": (
            len(command_slot_matches) / len(run_command_action_rows)
            if run_command_action_rows
            else None
        ),
        "read_file_decision_accuracy": (
            sum(1 for row in read_file_rows if row.get("correct") is True) / len(read_file_rows)
            if read_file_rows
            else None
        ),
        "positive_reward_credit": (
            1.0 if completed else positive_reward / max(positive_reward + 1.0, 1.0)
        ),
    }


def _group_by_episode(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("episode_id"))].append(row)
    return dict(grouped)


def _episode_scores_from_eval(eval_json: str | Path) -> dict[str, dict[str, float | None]]:
    payload = _load_json(eval_json)
    run_path = payload.get("run_path")
    if not run_path:
        raise ValueError(f"eval JSON missing run_path: {eval_json}")
    trace_path = Path(run_path) / "trace_rows.jsonl"
    trace_rows = _load_jsonl(trace_path)
    return {
        episode_id: _trace_episode_scores(rows)
        for episode_id, rows in _group_by_episode(trace_rows).items()
    }


def _visible_state_text(state: SystemStateFrame) -> str:
    return "\n".join(
        [
            state.goal.description,
            state.terminal.stdout_delta,
            state.terminal.stderr_delta,
            state.terminal.last_command,
            " ".join(state.filesystem.dirty_files),
            " ".join(state.filesystem.changed_paths),
            " ".join(state.filesystem.watched_paths),
        ]
    )


def _identity_prediction(state: SystemStateFrame) -> tuple[int, float, bool]:
    signal = runtime_command_identity_signal(state)
    scores = [
        float(signal.get(field, 0.0))
        for field in COMMAND_IDENTITY_SLOT_FIELDS
    ]
    best = max(scores) if scores else 0.0
    best_index = scores.index(best) if scores else 0
    return best_index, best, best > 0.0 and scores.count(best) == 1


def _episode_metadata(dataset_jsonl: str | Path) -> dict[str, dict[str, Any]]:
    grouped = _group_by_episode(_load_jsonl(dataset_jsonl))
    metadata: dict[str, dict[str, Any]] = {}
    for episode_id, rows in grouped.items():
        command_rows = [
            row
            for row in rows
            if isinstance(row.get("action"), dict)
            and row["action"].get("type") == ActionType.RUN_COMMAND.value
        ]
        command_row = command_rows[-1] if command_rows else rows[-1]
        state = SystemStateFrame.model_validate(command_row["state"])
        candidates = candidate_commands(state)
        correct_command = (command_row.get("action") or {}).get("command")
        correct_slot = candidates.index(correct_command) if correct_command in candidates else -1
        source_prediction = source_overlap_command_slot_prediction(
            _visible_state_text(state),
            candidates,
        )
        identity_prediction, identity_confidence, identity_unique = _identity_prediction(state)
        source_overlap_correct = correct_slot >= 0 and source_prediction == correct_slot
        identity_correct = correct_slot >= 0 and identity_prediction == correct_slot and identity_unique
        metadata[episode_id] = {
            "episode_id": episode_id,
            "candidate_count": len(candidates),
            "correct_command_slot": correct_slot,
            "source_overlap_prediction": source_prediction,
            "source_overlap_correct": source_overlap_correct,
            "identity_prediction": identity_prediction,
            "identity_confidence": round(identity_confidence, 6),
            "identity_unique": identity_unique,
            "identity_correct": identity_correct,
            "source_overlap_band": (
                "source_overlap_easy" if source_overlap_correct else "source_overlap_hard"
            ),
            "candidate_count_band": f"{len(candidates)}_candidates",
            "identity_confidence_band": (
                "identity_high"
                if identity_confidence >= 0.66
                else "identity_medium"
                if identity_confidence >= 0.33
                else "identity_low"
            ),
        }
    return metadata


def _aggregate_scores(scores: list[dict[str, float | None]]) -> dict[str, Any]:
    return {
        metric: _mean([float(row[metric]) for row in scores if row.get(metric) is not None])
        for metric in GRADED_METRICS
    } | {"episode_count": len(scores)}


def _policy_matrix(
    *,
    episode_metadata: dict[str, dict[str, Any]],
    policy_scores: dict[str, dict[str, dict[str, float | None]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for policy_name, scores_by_episode in policy_scores.items():
        matched = [
            scores
            for episode_id, scores in scores_by_episode.items()
            if episode_id in episode_metadata
        ]
        by_band: dict[str, dict[str, Any]] = {}
        for band_field in [
            "source_overlap_band",
            "candidate_count_band",
            "identity_confidence_band",
        ]:
            buckets: dict[str, list[dict[str, float | None]]] = defaultdict(list)
            for episode_id, scores in scores_by_episode.items():
                meta = episode_metadata.get(episode_id)
                if meta:
                    buckets[str(meta[band_field])].append(scores)
            by_band[band_field] = {
                band: _aggregate_scores(rows)
                for band, rows in sorted(buckets.items())
            }
        output[policy_name] = {
            "overall": _aggregate_scores(matched),
            "by_band": by_band,
        }
    return output


def build_phase2j_stress_matrix(
    *,
    dataset_jsonl: str | Path,
    eval_jsons: dict[str, str | Path],
    min_source_overlap_hard_episodes: int = 8,
    required_full_run_command_action_accuracy: float = 0.95,
    required_full_command_slot_match_when_run_command: float = 0.95,
    required_full_minus_no_nsi_command_accuracy: float = 0.10,
) -> dict[str, Any]:
    episode_metadata = _episode_metadata(dataset_jsonl)
    policy_scores = {
        policy_name: _episode_scores_from_eval(path)
        for policy_name, path in sorted(eval_jsons.items())
    }
    matrix = _policy_matrix(episode_metadata=episode_metadata, policy_scores=policy_scores)
    hard_episodes = [
        meta for meta in episode_metadata.values() if meta["source_overlap_band"] == "source_overlap_hard"
    ]
    candidate_bands = sorted({meta["candidate_count_band"] for meta in episode_metadata.values()})
    full_hard = (
        matrix.get("full", {})
        .get("by_band", {})
        .get("source_overlap_band", {})
        .get("source_overlap_hard", {})
    )
    no_nsi_hard = (
        matrix.get("no_nsi", {})
        .get("by_band", {})
        .get("source_overlap_band", {})
        .get("source_overlap_hard", {})
    )
    full_command = full_hard.get("command_decision_accuracy")
    full_action_gate = full_hard.get("run_command_action_accuracy")
    full_slot_gate = full_hard.get("command_slot_match_when_run_command")
    no_nsi_command = no_nsi_hard.get("command_decision_accuracy")
    delta = (
        float(full_command) - float(no_nsi_command)
        if isinstance(full_command, (int, float)) and isinstance(no_nsi_command, (int, float))
        else None
    )
    checks = {
        "source_overlap_hard_coverage": len(hard_episodes) >= min_source_overlap_hard_episodes,
        "multiple_candidate_count_bands": len(candidate_bands) > 1,
        "graded_metrics_present": all(
            metric in next(iter(matrix.values()))["overall"]
            for metric in GRADED_METRICS
        )
        if matrix
        else False,
        "full_run_command_action_gate_on_hard": (
            isinstance(full_action_gate, (int, float))
            and float(full_action_gate) >= required_full_run_command_action_accuracy
        ),
        "full_command_slot_gate_on_hard": (
            isinstance(full_slot_gate, (int, float))
            and float(full_slot_gate) >= required_full_command_slot_match_when_run_command
        ),
        "full_beats_no_nsi_on_hard_command_accuracy": (
            delta is not None
            and delta >= required_full_minus_no_nsi_command_accuracy
        ),
    }
    return {
        "report_family": "phase2j_stress_matrix",
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_source_overlap_hard_episodes": min_source_overlap_hard_episodes,
            "required_full_run_command_action_accuracy": (
                required_full_run_command_action_accuracy
            ),
            "required_full_command_slot_match_when_run_command": (
                required_full_command_slot_match_when_run_command
            ),
            "required_full_minus_no_nsi_command_accuracy": required_full_minus_no_nsi_command_accuracy,
        },
        "dataset": {
            "path": str(Path(dataset_jsonl)),
            "episode_count": len(episode_metadata),
            "source_overlap_hard_episodes": len(hard_episodes),
            "candidate_count_bands": candidate_bands,
        },
        "mechanism_deltas": {
            "hard_full_minus_no_nsi_command_accuracy": delta,
        },
        "mechanism_metrics": {
            "hard_full_run_command_action_accuracy": full_action_gate,
            "hard_full_command_slot_match_when_run_command": full_slot_gate,
        },
        "matrix": matrix,
        "episode_metadata_examples": list(episode_metadata.values())[:8],
    }


def _parse_eval_jsons(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--eval-json must use NAME=PATH")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError("--eval-json NAME cannot be empty")
        parsed[name] = path.strip()
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2J graded pressure-test matrix.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--eval-json", action="append", default=[], help="Policy eval as NAME=PATH.")
    parser.add_argument("--output-json")
    parser.add_argument("--min-source-overlap-hard-episodes", type=int, default=8)
    parser.add_argument("--required-full-run-command-action-accuracy", type=float, default=0.95)
    parser.add_argument(
        "--required-full-command-slot-match-when-run-command",
        type=float,
        default=0.95,
    )
    parser.add_argument("--required-full-minus-no-nsi-command-accuracy", type=float, default=0.10)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2j_stress_matrix(
        dataset_jsonl=args.dataset_jsonl,
        eval_jsons=_parse_eval_jsons(args.eval_json),
        min_source_overlap_hard_episodes=args.min_source_overlap_hard_episodes,
        required_full_run_command_action_accuracy=(
            args.required_full_run_command_action_accuracy
        ),
        required_full_command_slot_match_when_run_command=(
            args.required_full_command_slot_match_when_run_command
        ),
        required_full_minus_no_nsi_command_accuracy=args.required_full_minus_no_nsi_command_accuracy,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
