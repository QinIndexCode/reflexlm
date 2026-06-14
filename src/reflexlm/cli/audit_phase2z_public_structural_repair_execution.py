from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


RECORDED_BOUNDARY = "public_structural_recorded_patch_runtime_control_only_not_model_patch_generation"
SYMBOLIC_STRUCTURAL_BOUNDARY = (
    "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair"
)
ALLOWED_BOUNDARIES = {RECORDED_BOUNDARY, SYMBOLIC_STRUCTURAL_BOUNDARY}


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


def audit_phase2z_public_structural_repair_execution(
    *,
    execution_results_jsonl: str | Path,
    min_rows: int = 24,
    min_success_rate: float = 0.70,
) -> dict[str, Any]:
    rows = _read_jsonl(execution_results_jsonl)
    successes = [row for row in rows if row.get("success") is True]
    failures = [row for row in rows if row.get("success") is not True]
    failure_reasons: Counter[str] = Counter()
    for row in failures:
        outputs = row.get("policy_open_repair_outputs")
        if not isinstance(outputs, dict) or not outputs:
            failure_reasons["missing_open_repair_head_outputs"] += 1
            continue
        if outputs.get("patch_proposal") != 1:
            failure_reasons["patch_proposal_not_authorized"] += 1
        if outputs.get("bounded_edit_scope") != 1:
            failure_reasons["bounded_edit_scope_not_authorized"] += 1
        if outputs.get("rollback_safety") != 1:
            failure_reasons["rollback_safety_not_authorized"] += 1
        if row.get("full_test_pass_rate") != 1.0:
            failure_reasons["post_patch_test_not_passing"] += 1
        if row.get("rollback_failure_restored") is not True:
            failure_reasons["rollback_failure_not_restored"] += 1
    success_rate = len(successes) / len(rows) if rows else 0.0
    boundaries = {str(row.get("claim_boundary") or "") for row in rows}
    recorded_mode = bool(rows) and boundaries == {RECORDED_BOUNDARY}
    symbolic_structural_mode = bool(rows) and boundaries == {SYMBOLIC_STRUCTURAL_BOUNDARY}
    all_rows_recorded_runtime_control = all(
        row.get("claim_bearing_execution_evidence") is False
        and row.get("recorded_patch_artifact_used") is True
        and row.get("oracle_trace_used") is True
        for row in rows
    )
    all_rows_bounded_symbolic_structural = all(
        row.get("claim_bearing_execution_evidence") is True
        and row.get("recorded_patch_artifact_used") is False
        and row.get("oracle_trace_used") is False
        and row.get("patch_source") == "package_runtime_symbolic_structural_patch_proposal"
        and row.get("patch_generator") == "bounded_symbolic_structural_patch_v1"
        for row in rows
    )
    checks = {
        "row_minimum_met": len(rows) >= min_rows,
        "success_rate_minimum_met": success_rate >= min_success_rate,
        "all_rows_policy_loaded": all(row.get("policy_loaded") is True for row in rows),
        "all_rows_public_repo": all(row.get("source_kind") == "public_repo" for row in rows),
        "single_known_execution_boundary": len(boundaries) == 1
        and boundaries.issubset(ALLOWED_BOUNDARIES),
        "recorded_or_symbolic_boundary_consistent": recorded_mode
        or symbolic_structural_mode,
        "recorded_patch_boundary_valid": (not recorded_mode)
        or all_rows_recorded_runtime_control,
        "symbolic_structural_boundary_valid": (not symbolic_structural_mode)
        or all_rows_bounded_symbolic_structural,
        "no_rows_claim_freeform_model_patch_generation": (
            (recorded_mode and all_rows_recorded_runtime_control)
            or (symbolic_structural_mode and all_rows_bounded_symbolic_structural)
        ),
        "sealed_feedback_absent": all(row.get("sealed_feedback_used") is False for row in rows),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2z_public_structural_repair_execution_audit",
        "passed": passed,
        "claim_boundary": next(iter(boundaries), RECORDED_BOUNDARY),
        "claim_bearing_execution_evidence": symbolic_structural_mode,
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "success_count": len(successes),
            "failure_count": len(failures),
            "success_rate": success_rate,
            "failure_reasons": dict(sorted(failure_reasons.items())),
            "execution_boundaries": sorted(boundaries),
            "recorded_mode": recorded_mode,
            "symbolic_structural_mode": symbolic_structural_mode,
        },
        "thresholds": {
            "min_rows": min_rows,
            "min_success_rate": min_success_rate,
        },
        "blocked_actions": [
            "do_not_claim_freeform_model_generated_patch_repair",
            "do_not_claim_open_ended_debugging_generalization",
        ],
        "inputs": {"execution_results_jsonl": str(Path(execution_results_jsonl))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2Z public structural recorded-patch repair execution."
    )
    parser.add_argument("--execution-results-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=24)
    parser.add_argument("--min-success-rate", type=float, default=0.70)
    args = parser.parse_args()
    report = audit_phase2z_public_structural_repair_execution(
        execution_results_jsonl=args.execution_results_jsonl,
        min_rows=args.min_rows,
        min_success_rate=args.min_success_rate,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
