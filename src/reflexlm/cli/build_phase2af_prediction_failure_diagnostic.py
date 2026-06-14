from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _slot_index(key: Any) -> int | None:
    if isinstance(key, int):
        return key
    if isinstance(key, str):
        if key.startswith("slot:"):
            key = key.split(":", 1)[1]
        if key.isdigit():
            return int(key)
    return None


def _score_map(record: dict[str, Any]) -> dict[int, float]:
    raw_scores = record.get("command_identity_scores")
    scores: dict[int, float] = {}
    if isinstance(raw_scores, dict):
        for key, value in raw_scores.items():
            idx = _slot_index(key)
            if idx is not None and _is_finite_number(value):
                scores[idx] = float(value)
    elif isinstance(raw_scores, list):
        for idx, value in enumerate(raw_scores):
            if _is_finite_number(value):
                scores[idx] = float(value)
    return scores


def _identity_tie(scores: dict[int, float]) -> tuple[bool, list[int], float | None]:
    if not scores:
        return True, [], None
    max_score = max(scores.values())
    tied_slots = sorted(idx for idx, score in scores.items() if abs(score - max_score) <= 1e-9)
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    return len(tied_slots) > 1 or abs(margin) <= 1e-9, tied_slots, margin


def _repo_id(record: dict[str, Any]) -> str | None:
    trace = _dict(record.get("source_trace"))
    repo = trace.get("repo_id") or trace.get("repo")
    return str(repo) if repo is not None else None


def _candidate_at(record: dict[str, Any], slot: Any) -> Any:
    idx = _slot_index(slot)
    candidates = record.get("candidate_commands")
    if idx is None or not isinstance(candidates, list) or idx < 0 or idx >= len(candidates):
        return None
    return candidates[idx]


def _summarize_failed_record(record: dict[str, Any]) -> dict[str, Any]:
    scores = _score_map(record)
    identity_tie, tied_slots, computed_margin = _identity_tie(scores)
    label = record.get("command_slot_label")
    prediction = record.get("command_slot_prediction")
    return {
        "row_index": record.get("row_index"),
        "episode_id": record.get("episode_id"),
        "repo_id": _repo_id(record),
        "target_slot": label,
        "predicted_slot": prediction,
        "source_overlap_correct": record.get("source_overlap_correct"),
        "identity_tie": identity_tie,
        "identity_tied_slots": tied_slots,
        "identity_margin": record.get("command_identity_margin", computed_margin),
        "computed_identity_margin": computed_margin,
        "identity_confidence": record.get("command_identity_confidence"),
        "identity_scores": {str(key): value for key, value in sorted(scores.items())},
        "candidate_count": len(record.get("candidate_commands") or []),
        "gold_candidate": _candidate_at(record, label),
        "predicted_candidate": _candidate_at(record, prediction),
    }


def _failure_class(failed: list[dict[str, Any]], tie_count: int) -> str:
    if not failed:
        return "no_failed_command_slot_rows"
    if tie_count == len(failed):
        return "identity_tie_candidate_indistinguishability"
    if tie_count >= math.ceil(len(failed) / 2):
        return "mixed_identity_tie_candidate_indistinguishability"
    return "residual_command_slot_prediction_errors"


def build_phase2af_prediction_failure_diagnostic(
    *,
    eval_summary_json: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    summary = _read_json(eval_summary_json)
    records = summary.get("prediction_records")
    if not isinstance(records, list):
        raise ValueError("eval_summary_json must contain prediction_records; rerun eval with --include-prediction-records")

    failed = [record for record in records if record.get("command_slot_correct") is False]
    failed_rows = [_summarize_failed_record(record) for record in failed]
    tie_count = sum(1 for row in failed_rows if row["identity_tie"])
    zero_score_tie_count = sum(
        1
        for row in failed_rows
        if row["identity_tie"] and row["identity_scores"] and set(row["identity_scores"].values()) == {0.0}
    )
    source_overlap_correct_count = sum(1 for row in failed_rows if row["source_overlap_correct"] is True)
    edge_counts = Counter((str(row["target_slot"]), str(row["predicted_slot"])) for row in failed_rows)
    repo_counts = Counter(str(row["repo_id"]) for row in failed_rows)
    failure_class = _failure_class(failed_rows, tie_count)
    passed = not failed_rows
    blocked_actions = [] if passed else [
        "do_not_package_phase2af",
        "do_not_run_sealed_phase2af",
        "do_not_claim_hardened_structural_sidecar_mechanism",
    ]
    if failure_class != "no_failed_command_slot_rows":
        blocked_actions.insert(0, "do_not_train_phase2af_full_from_this_adapter")

    report = {
        "artifact_family": "phase2af_hardened_structural_sidecar_prediction_failure_diagnostic",
        "passed": passed,
        "failure_class": failure_class,
        "claim_upgrade_allowed": False,
        "metrics": {
            "prediction_record_count": len(records),
            "failed_command_slot_rows": len(failed_rows),
            "identity_tie_failed_rows": tie_count,
            "zero_identity_score_tie_failed_rows": zero_score_tie_count,
            "source_overlap_correct_failed_rows": source_overlap_correct_count,
        },
        "failure_distribution": {
            "by_repo": dict(sorted(repo_counts.items())),
            "by_target_predicted_edge": {
                f"{target}->{predicted}": count
                for (target, predicted), count in sorted(edge_counts.items(), key=lambda item: (-item[1], item[0]))
            },
        },
        "diagnosis": {
            "primary_issue": (
                "All command-slot errors are identity-tie cases: the measured command-identity sidecar "
                "does not distinguish the gold candidate from at least one competing candidate."
            )
            if failure_class == "identity_tie_candidate_indistinguishability"
            else (
                "Most command-slot errors are identity-tie cases; inspect remaining non-tie rows before another training run."
                if failure_class == "mixed_identity_tie_candidate_indistinguishability"
                else "Command-slot errors are not dominated by identity ties; inspect model and split behavior before retraining."
            ),
            "interpretation_boundary": (
                "This is not evidence for package-level autonomy or a hardened structural sidecar claim; "
                "the current feature set leaves some same-symbol or structural-hash candidates indistinguishable."
            ),
        },
        "required_next_properties": [
            "add non-label runtime-visible verification or disambiguation state for same-symbol candidates",
            "keep candidate-order and slot-balanced sampling, but do not treat more resampling as sufficient proof",
            "evaluate on a fresh repo-origin-disjoint holdout after any architecture change",
            "do not use sealed failures to design the new disambiguation feature",
        ],
        "failed_rows": failed_rows,
        "blocked_actions": blocked_actions,
        "inputs": {"eval_summary_json": str(Path(eval_summary_json))},
    }
    _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Phase2AF prediction-record failures.")
    parser.add_argument("--eval-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2af_prediction_failure_diagnostic(
        eval_summary_json=args.eval_summary_json,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
