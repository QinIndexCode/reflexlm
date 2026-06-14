from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_GENERATOR = "bounded_symbolic_structural_patch_v1"
EXPECTED_PATCH_SOURCE = "package_runtime_symbolic_structural_patch_proposal"
EXPECTED_CLAIM_BOUNDARY = (
    "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair"
)


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


def audit_phase2ar_symbolic_structural_execution(
    *,
    execution_results_jsonl: str | Path,
    min_rows: int = 8,
    min_success_rate: float = 1.0,
    required_patch_kinds: list[str] | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(execution_results_jsonl)
    required = set(required_patch_kinds or ["text_membership", "ast_attribute_restoration"])
    successes = [row for row in rows if row.get("success") is True]
    patch_kind_counter: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    for row in rows:
        for kind in row.get("symbolic_patch_kinds") or []:
            patch_kind_counter[str(kind)] += 1
        if row.get("success") is not True:
            failure_reasons["row_failed"] += 1
        if row.get("patch_generator") != EXPECTED_GENERATOR:
            failure_reasons["unexpected_patch_generator"] += 1
        if row.get("patch_source") != EXPECTED_PATCH_SOURCE:
            failure_reasons["unexpected_patch_source"] += 1
        if row.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
            failure_reasons["unexpected_claim_boundary"] += 1
        if row.get("rollback_failure_restored") is not True:
            failure_reasons["rollback_failure_not_restored"] += 1

    success_rate = len(successes) / len(rows) if rows else 0.0
    checks = {
        "row_minimum_met": len(rows) >= min_rows,
        "success_rate_minimum_met": success_rate >= min_success_rate,
        "all_rows_policy_loaded": all(row.get("policy_loaded") is True for row in rows),
        "all_rows_symbolic_structural_generator": all(
            row.get("patch_generator") == EXPECTED_GENERATOR for row in rows
        ),
        "all_rows_runtime_structural_patch_source": all(
            row.get("patch_source") == EXPECTED_PATCH_SOURCE for row in rows
        ),
        "no_rows_use_recorded_patch_as_proposal": all(
            row.get("recorded_patch_artifact_used") is False
            and row.get("recorded_patch_artifact_used_for_fault_injection") is True
            for row in rows
        ),
        "all_rows_claim_bearing_execution_evidence": all(
            row.get("claim_bearing_execution_evidence") is True for row in rows
        ),
        "all_rows_expected_claim_boundary": all(
            row.get("claim_boundary") == EXPECTED_CLAIM_BOUNDARY for row in rows
        ),
        "all_rows_rollback_restore_failure": all(
            row.get("rollback_failure_restored") is True for row in rows
        ),
        "required_patch_kinds_present": required.issubset(set(patch_kind_counter)),
        "sealed_feedback_absent": all(row.get("sealed_feedback_used") is False for row in rows),
        "no_false_completion_or_unauthorized_write": all(
            row.get("false_completion") is False
            and int(row.get("unauthorized_write_count") or 0) == 0
            for row in rows
        ),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2ar_symbolic_structural_execution_audit",
        "passed": passed,
        "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
        "claim_bearing_execution_evidence": passed,
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "success_count": len(successes),
            "failure_count": len(rows) - len(successes),
            "success_rate": success_rate,
            "patch_kind_counts": dict(sorted(patch_kind_counter.items())),
            "failure_reasons": dict(sorted(failure_reasons.items())),
        },
        "thresholds": {
            "min_rows": min_rows,
            "min_success_rate": min_success_rate,
            "required_patch_kinds": sorted(required),
        },
        "supported_claims": [
            "bounded_runtime_symbolic_structural_patch_proposal_holdout_supported"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "inputs": {"execution_results_jsonl": str(Path(execution_results_jsonl))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AR bounded symbolic structural patch execution evidence."
    )
    parser.add_argument("--execution-results-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    args = parser.parse_args()
    report = audit_phase2ar_symbolic_structural_execution(
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
