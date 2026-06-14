from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_RESULT_FIELDS = (
    "task_id",
    "repo_origin",
    "repo_commit",
    "result_source",
    "native_policy_label",
    "policy_package_manifest_sha256",
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
    "stop_condition",
    "elapsed_seconds",
    "transcript_sha256",
    "oracle_trace_used",
    "sealed_feedback_used",
    "success",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _field_present(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return True


def _hash_fields_valid(row: dict[str, Any]) -> bool:
    fields = (
        "policy_package_manifest_sha256",
        "patch_sha256",
        "pre_test_log_sha256",
        "post_test_log_sha256",
        "transcript_sha256",
    )
    return all(isinstance(row.get(field), str) and SHA256_RE.match(row[field]) for field in fields)


def _non_oracle_policy_execution(row: dict[str, Any]) -> bool:
    return (
        row.get("result_source") == "phase2x_package_runtime_execution"
        and row.get("patch_source")
        in {"package_runtime_patch_proposal", "package_runtime_no_patch_authorized"}
        and isinstance(row.get("native_policy_label"), str)
        and bool(row["native_policy_label"].strip())
        and row.get("oracle_trace_used") is False
        and row.get("sealed_feedback_used") is False
    )


def _policy_control_authorized(row: dict[str, Any]) -> bool:
    outputs = row.get("policy_open_repair_outputs")
    if not isinstance(outputs, dict):
        return False
    return outputs.get("patch_proposal") == 1 and outputs.get("bounded_edit_scope") == 1


def audit_phase2x_open_repair_execution_results(
    *,
    training_readiness_json: str | Path,
    results_jsonl: str | Path,
    runtime_capability_audit_json: str | Path | None = None,
    min_rows: int = 1,
    min_success_rate: float = 0.0,
) -> dict[str, Any]:
    readiness = _read_json(training_readiness_json)
    runtime_audit = _read_json(runtime_capability_audit_json) if runtime_capability_audit_json else {}
    rows = _read_jsonl(results_jsonl)
    missing_by_field = {
        field: sum(1 for row in rows if not _field_present(row, field))
        for field in REQUIRED_RESULT_FIELDS
    }
    hash_valid_rows = sum(1 for row in rows if _hash_fields_valid(row))
    successes = sum(1 for row in rows if row.get("success") is True)
    success_rate = successes / len(rows) if rows else None
    checks = {
        "training_readiness_passed": readiness.get("passed") is True,
        "runtime_capability_audit_passed": runtime_audit.get("passed") is True,
        "rows_present": len(rows) >= min_rows,
        "successes_present": successes > 0,
        "success_rate_threshold_met": success_rate is not None
        and success_rate >= min_success_rate,
        "required_fields_present": all(count == 0 for count in missing_by_field.values()),
        "hash_fields_valid": hash_valid_rows == len(rows) and len(rows) > 0,
        "non_oracle_policy_execution": len(rows) > 0
        and all(_non_oracle_policy_execution(row) for row in rows),
        "policy_control_authorized_patch": len(rows) > 0
        and all(_policy_control_authorized(row) for row in rows if row.get("success") is True),
        "selected_tests_recorded": all(isinstance(row.get("selected_tests"), list) and row["selected_tests"] for row in rows),
        "progress_monitor_trace_recorded": all(isinstance(row.get("progress_monitor_trace"), list) and row["progress_monitor_trace"] for row in rows),
        "elapsed_seconds_positive": all(float(row.get("elapsed_seconds", 0) or 0) > 0 for row in rows),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2x_open_repair_execution_results_audit",
        "passed": passed,
        "checks": checks,
        "rows": len(rows),
        "successes": successes,
        "success_rate": success_rate,
        "min_success_rate": min_success_rate,
        "missing_by_field": missing_by_field,
        "hash_valid_rows": hash_valid_rows,
        "blocked_actions": []
        if passed
        else [
            "do_not_use_phase2x_results_as_real_execution_evidence",
            "do_not_claim_open_ended_repair_until_row_level_execution_artifacts_exist",
            "do_not_substitute_oracle_trace_for_policy_generated_patch",
            "do_not_count_patch_success_when_open_repair_heads_did_not_authorize_patch",
        ],
        "inputs": {
            "training_readiness_json": str(Path(training_readiness_json)),
            "runtime_capability_audit_json": str(Path(runtime_capability_audit_json))
            if runtime_capability_audit_json
            else None,
            "results_jsonl": str(Path(results_jsonl)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2X real open-repair execution results.")
    parser.add_argument("--training-readiness-json", required=True)
    parser.add_argument("--runtime-capability-audit-json")
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-success-rate", type=float, default=0.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2x_open_repair_execution_results(
        training_readiness_json=args.training_readiness_json,
        runtime_capability_audit_json=args.runtime_capability_audit_json,
        results_jsonl=args.results_jsonl,
        min_rows=args.min_rows,
        min_success_rate=args.min_success_rate,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
