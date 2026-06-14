from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_RESULT_KEYS = {
    "task_id",
    "repo_origin",
    "repo_commit",
    "result_source",
    "native_policy_label",
    "patch_source",
    "policy_open_repair_outputs",
    "patch_proposal",
    "patch_sha256",
    "selected_tests",
    "pre_test_log_sha256",
    "post_test_log_sha256",
    "rollback_safety_decision",
    "verification_state",
    "progress_monitor_trace",
    "oracle_trace_used",
    "sealed_feedback_used",
    "success",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "passed"}
    return False


def _patch_generator(row: dict[str, Any]) -> str:
    return str(row.get("patch_generator") or "unknown").strip() or "unknown"


def _selected_tests(row: dict[str, Any]) -> list[str]:
    tests = row.get("selected_tests")
    return tests if isinstance(tests, list) else []


def _progress_events(row: dict[str, Any]) -> set[str]:
    trace = row.get("progress_monitor_trace")
    if not isinstance(trace, list):
        return set()
    events: set[str] = set()
    for item in trace:
        if isinstance(item, dict):
            events.add(str(item.get("event") or ""))
    return events


def _is_no_patch_control(row: dict[str, Any]) -> bool:
    outputs = row.get("policy_open_repair_outputs")
    return (
        row.get("patch_source") == "package_runtime_no_patch_authorized"
        or (isinstance(outputs, dict) and outputs.get("patch_proposal") == 0)
    )


def _is_generated_test_only(row: dict[str, Any]) -> bool:
    tests = _selected_tests(row)
    return bool(tests) and all("<generated_repair_test>" in str(test) for test in tests)


def audit_phase2y_open_repair_generalization_gap(
    *,
    execution_results_jsonl: str | Path | list[str | Path],
    execution_audit_json: str | Path | list[str | Path] | None = None,
    min_rows: int = 128,
    min_success_rate: float = 0.85,
    min_repo_origins: int = 4,
) -> dict[str, Any]:
    result_paths = (
        [execution_results_jsonl]
        if isinstance(execution_results_jsonl, (str, Path))
        else list(execution_results_jsonl)
    )
    audit_paths: list[str | Path] = []
    if execution_audit_json is not None:
        audit_paths = (
            [execution_audit_json]
            if isinstance(execution_audit_json, (str, Path))
            else list(execution_audit_json)
        )

    rows: list[dict[str, Any]] = []
    for path in result_paths:
        rows.extend(_read_jsonl(path))
    audits = [_read_json(path) for path in audit_paths]

    success_count = sum(1 for row in rows if _as_bool(row.get("success")))
    success_rate = success_count / len(rows) if rows else 0.0
    repo_origins = {str(row.get("repo_origin") or "") for row in rows if row.get("repo_origin")}
    patch_generators = Counter(_patch_generator(row) for row in rows)
    selected_test_counts = Counter(len(_selected_tests(row)) for row in rows)

    row_schema_complete = bool(rows) and all(REQUIRED_RESULT_KEYS.issubset(row) for row in rows)
    execution_audits_passed = not audits or all(audit.get("passed") is True for audit in audits)
    non_oracle_non_sealed = row_schema_complete and all(
        row.get("oracle_trace_used") is False and row.get("sealed_feedback_used") is False
        for row in rows
    )
    no_edit_control_rows = sum(1 for row in rows if _is_no_patch_control(row))
    generated_test_only_rows = sum(1 for row in rows if _is_generated_test_only(row))
    literal_patch_rows = sum(
        1 for row in rows if "literal" in _patch_generator(row).lower()
    )
    non_literal_patch_rows = sum(
        1
        for row in rows
        if row.get("patch_source") == "package_runtime_patch_proposal"
        and "literal" not in _patch_generator(row).lower()
    )
    multi_test_rows = sum(1 for row in rows if len(_selected_tests(row)) >= 2)
    rollback_required_rows = sum(
        1
        for row in rows
        if "rollback_required" in str(row.get("rollback_safety_decision") or "")
        or "rollback_started" in _progress_events(row)
        or "rollback_finished" in _progress_events(row)
    )

    bounded_checks = {
        "execution_audits_passed": execution_audits_passed,
        "row_schema_complete": row_schema_complete,
        "non_oracle_non_sealed": non_oracle_non_sealed,
        "rows_minimum_met": len(rows) >= min_rows,
        "success_rate_minimum_met": success_rate >= min_success_rate,
        "repo_origin_minimum_met": len(repo_origins) >= min_repo_origins,
    }
    open_ended_checks = {
        "non_literal_patch_present": non_literal_patch_rows > 0,
        "multi_test_selection_present": multi_test_rows > 0,
        "rollback_required_path_present": rollback_required_rows > 0,
        "no_edit_control_present": no_edit_control_rows > 0,
        "not_generated_test_only": generated_test_only_rows < len(rows) if rows else False,
        "patch_generator_diversity_present": len(patch_generators) >= 2,
    }
    bounded_execution_supported = all(bounded_checks.values())
    open_ended_claim_ready = bounded_execution_supported and all(open_ended_checks.values())
    blocked_actions = []
    if not open_ended_claim_ready:
        blocked_actions.extend(
            [
                "do_not_claim_open_ended_debugging_generalization_from_phase2x",
                "do_not_claim_production_autonomy_from_literal_repair_results",
                "build_phase2y_nonliteral_multitest_rollback_benchmark_before_stronger_claim",
            ]
        )

    return {
        "artifact_family": "phase2y_open_repair_generalization_gap_audit",
        "passed": open_ended_claim_ready,
        "bounded_execution_supported": bounded_execution_supported,
        "open_ended_claim_ready": open_ended_claim_ready,
        "claim_boundary": (
            "open_ended_repair_candidate_evidence"
            if open_ended_claim_ready
            else "bounded_literal_assertion_repair_only"
        ),
        "checks": {
            **bounded_checks,
            **open_ended_checks,
        },
        "metrics": {
            "row_count": len(rows),
            "success_count": success_count,
            "success_rate": success_rate,
            "repo_origin_count": len(repo_origins),
            "patch_generator_distribution": dict(sorted(patch_generators.items())),
            "selected_test_count_distribution": {
                str(key): value for key, value in sorted(selected_test_counts.items())
            },
            "literal_patch_rows": literal_patch_rows,
            "non_literal_patch_rows": non_literal_patch_rows,
            "multi_test_rows": multi_test_rows,
            "rollback_required_rows": rollback_required_rows,
            "no_edit_control_rows": no_edit_control_rows,
            "generated_test_only_rows": generated_test_only_rows,
        },
        "blocked_actions": blocked_actions,
        "inputs": {
            "execution_results_jsonl": [str(Path(path)) for path in result_paths],
            "execution_audit_json": [str(Path(path)) for path in audit_paths],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether Phase2X open-repair results are enough for stronger Phase2Y claims."
    )
    parser.add_argument("--execution-results-jsonl", action="append", required=True)
    parser.add_argument("--execution-audit-json", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=128)
    parser.add_argument("--min-success-rate", type=float, default=0.85)
    parser.add_argument("--min-repo-origins", type=int, default=4)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = audit_phase2y_open_repair_generalization_gap(
        execution_results_jsonl=args.execution_results_jsonl,
        execution_audit_json=args.execution_audit_json,
        min_rows=args.min_rows,
        min_success_rate=args.min_success_rate,
        min_repo_origins=args.min_repo_origins,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
