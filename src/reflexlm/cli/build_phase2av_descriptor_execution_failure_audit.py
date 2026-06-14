from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CLAIM_BOUNDARY = (
    "Phase2AV descriptor execution failure audit diagnoses non-sealed bounded "
    "symbolic execution artifacts only. It does not authorize package, sealed "
    "evaluation, freeform patch generation, production autonomy, open-ended "
    "debugging generalization, or epoch-making claims."
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if row.get(key) is True) / len(rows) if rows else 0.0


def _repo_key(row: dict[str, Any]) -> str:
    origin = str(row.get("repo_origin") or "unknown")
    return origin.removesuffix(".git").rstrip("/").split("/")[-1] or origin


def _top(counter: Counter[str], limit: int = 12) -> dict[str, int]:
    return dict(counter.most_common(limit))


def _failure_category(row: dict[str, Any]) -> str:
    if row.get("patch_candidate_selected_correctly") is not True:
        return "candidate_selection_failure"
    if row.get("false_completion") is True:
        return "false_completion"
    if int(row.get("unauthorized_write_count") or 0) > 0:
        return "unauthorized_write"
    if row.get("rollback_failure_restored") is not True:
        return "rollback_safety_failure"
    state = str(row.get("verification_state") or "")
    stop = str(row.get("stop_condition") or "")
    if "failed" in state or "failed" in stop:
        return "selected_candidate_runtime_test_failure"
    return "selected_candidate_runtime_failure"


def _source_artifact_split(row: dict[str, Any]) -> str:
    artifact_paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    source = str(artifact_paths.get("source_patch_artifact") or "").replace("\\", "/")
    for split in ("train", "val", "holdout"):
        if f"/{split}/" in source:
            return split
    return "missing"


def build_phase2av_descriptor_execution_failure_audit(
    *,
    full_execution_jsonl: str | Path,
    control_execution_jsonl: str | Path | None = None,
    execution_gate_json: str | Path | None = None,
    min_full_success_rate: float = 0.85,
) -> dict[str, Any]:
    full_rows = _read_jsonl(full_execution_jsonl)
    control_rows = _read_jsonl(control_execution_jsonl) if control_execution_jsonl else []
    gate = _read_json(execution_gate_json) if execution_gate_json else {}

    failures = [row for row in full_rows if row.get("success") is not True]
    successes = len(full_rows) - len(failures)
    full_success_rate = successes / len(full_rows) if full_rows else 0.0
    full_selection_accuracy = _rate(full_rows, "patch_candidate_selected_correctly")
    control_success_rate = _rate(control_rows, "success") if control_rows else None
    control_selection_accuracy = (
        _rate(control_rows, "patch_candidate_selected_correctly") if control_rows else None
    )

    categories = Counter(_failure_category(row) for row in failures)
    repos = Counter(_repo_key(row) for row in failures)
    stop_conditions = Counter(str(row.get("stop_condition") or "missing") for row in failures)
    verification_states = Counter(
        str(row.get("verification_state") or "missing") for row in failures
    )
    selected_actions = Counter(
        str(row.get("selected_repair_action") or "missing") for row in failures
    )
    source_splits = Counter(_source_artifact_split(row) for row in full_rows)
    failed_source_splits = Counter(_source_artifact_split(row) for row in failures)
    split_clean = set(source_splits) <= {"holdout"}
    control_by_trace = {str(row.get("trace_id") or ""): row for row in control_rows}
    source_split_outcomes: dict[str, dict[str, Any]] = {}
    for split in sorted(source_splits):
        split_rows = [row for row in full_rows if _source_artifact_split(row) == split]
        split_control_rows = [
            control_by_trace[str(row.get("trace_id") or "")]
            for row in split_rows
            if str(row.get("trace_id") or "") in control_by_trace
        ]
        split_full_success = _rate(split_rows, "success")
        split_control_success = (
            _rate(split_control_rows, "success") if split_control_rows else None
        )
        source_split_outcomes[split] = {
            "rows": len(split_rows),
            "full_success_rate": split_full_success,
            "control_success_rate": split_control_success,
            "full_minus_control_success_rate": (
                split_full_success - split_control_success
                if isinstance(split_control_success, float)
                else None
            ),
        }

    selection_is_bottleneck = full_selection_accuracy < min_full_success_rate
    runtime_is_bottleneck = (
        full_selection_accuracy >= min_full_success_rate
        and full_success_rate < min_full_success_rate
    )
    passed = full_success_rate >= min_full_success_rate
    failure_modes: list[str] = []
    if not passed:
        failure_modes.append("full_runtime_execution_success_below_gate")
    if selection_is_bottleneck:
        failure_modes.append("candidate_selection_accuracy_below_gate")
    if runtime_is_bottleneck:
        failure_modes.append("selected_candidate_runtime_execution_below_gate")
    if gate and gate.get("passed") is False:
        failure_modes.append("descriptor_execution_delta_gate_failed")
    if not split_clean:
        failure_modes.append("holdout_execution_uses_mixed_source_artifact_splits")

    return {
        "artifact_family": "phase2av_descriptor_execution_failure_audit",
        "passed": passed,
        "claim_boundary": CLAIM_BOUNDARY,
        "checks": {
            "full_rows_present": bool(full_rows),
            "full_success_rate_gate": full_success_rate >= min_full_success_rate,
            "selection_is_not_primary_bottleneck": not selection_is_bottleneck,
            "runtime_bottleneck_detected": runtime_is_bottleneck,
            "control_rows_present": bool(control_rows),
            "execution_gate_failed_or_absent": not gate or gate.get("passed") is False,
            "holdout_source_artifact_split_clean": split_clean,
        },
        "metrics": {
            "full_rows": len(full_rows),
            "full_successes": successes,
            "full_failures": len(failures),
            "full_success_rate": full_success_rate,
            "full_selection_accuracy": full_selection_accuracy,
            "control_rows": len(control_rows),
            "control_success_rate": control_success_rate,
            "control_selection_accuracy": control_selection_accuracy,
            "full_minus_control_success_rate": (
                full_success_rate - control_success_rate
                if isinstance(control_success_rate, float)
                else None
            ),
            "thresholds": {"min_full_success_rate": min_full_success_rate},
        },
        "failure_modes": failure_modes,
        "failure_breakdown": {
            "category_counts": _top(categories),
            "repo_counts": _top(repos),
            "stop_condition_counts": _top(stop_conditions),
            "verification_state_counts": _top(verification_states),
            "selected_repair_action_counts": _top(selected_actions),
            "source_artifact_split_counts": _top(source_splits),
            "failed_source_artifact_split_counts": _top(failed_source_splits),
            "source_artifact_split_outcomes": source_split_outcomes,
        },
        "diagnosis": (
            "candidate_selection_bottleneck"
            if selection_is_bottleneck
            else "runtime_symbolic_patch_execution_bottleneck"
            if runtime_is_bottleneck
            else "execution_gate_passed_or_no_primary_bottleneck"
        ),
        "recommended_next_actions": [
            "do_not_package_phase2av",
            "inspect_failed_selected_candidate_patch_artifacts",
            "rebuild_candidate_pool_with_split_clean_verified_patch_artifacts",
            "separate invalid_patch_artifact from symbolic_materializer_gap",
            "build_phase2aw_or_successor_runtime_materialization_hardening_before_package",
        ]
        if not passed
        else ["package_gate_may_be_considered_by_a_separate_strict_gate"],
        "unsupported_claims": [
            "phase2av_package_ready",
            "sealed_cross_model_transfer",
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_package_phase2av",
            "do_not_run_sealed_eval_from_phase2av",
            "do_not_claim_runtime_execution_gate_on_full_holdout",
        ],
        "inputs": {
            "full_execution_jsonl": str(Path(full_execution_jsonl)),
            "control_execution_jsonl": str(Path(control_execution_jsonl))
            if control_execution_jsonl
            else None,
            "execution_gate_json": str(Path(execution_gate_json))
            if execution_gate_json
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AV descriptor execution failure audit."
    )
    parser.add_argument("--full-execution-jsonl", required=True)
    parser.add_argument("--control-execution-jsonl")
    parser.add_argument("--execution-gate-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-full-success-rate", type=float, default=0.85)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2av_descriptor_execution_failure_audit(
        full_execution_jsonl=args.full_execution_jsonl,
        control_execution_jsonl=args.control_execution_jsonl,
        execution_gate_json=args.execution_gate_json,
        min_full_success_rate=args.min_full_success_rate,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
