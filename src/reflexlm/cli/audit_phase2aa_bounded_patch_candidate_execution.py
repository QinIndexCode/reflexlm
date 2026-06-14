from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2aa_bounded_patch_candidates import CLAIM_BOUNDARY


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


def audit_phase2aa_bounded_patch_candidate_execution(
    *,
    execution_results_jsonl: str | Path,
    min_rows: int = 24,
    min_success_rate: float = 0.85,
    min_selection_accuracy: float = 0.85,
) -> dict[str, Any]:
    rows = _read_jsonl(execution_results_jsonl)
    successes = [row for row in rows if row.get("success") is True]
    correct_slots = [row for row in rows if row.get("patch_candidate_selected_correctly") is True]
    failure_reasons: Counter[str] = Counter()
    for row in rows:
        if row.get("patch_candidate_selected_correctly") is not True:
            failure_reasons["wrong_patch_candidate_slot"] += 1
        if row.get("full_test_pass_rate") != 1.0:
            failure_reasons["post_patch_test_not_passing"] += 1
        if row.get("rollback_failure_restored") is not True:
            failure_reasons["rollback_failure_not_restored"] += 1
        outputs = row.get("policy_open_repair_outputs")
        if not isinstance(outputs, dict) or not outputs:
            failure_reasons["missing_open_repair_head_outputs"] += 1
    success_rate = len(successes) / len(rows) if rows else 0.0
    selection_accuracy = len(correct_slots) / len(rows) if rows else 0.0
    checks = {
        "row_minimum_met": len(rows) >= min_rows,
        "success_rate_minimum_met": success_rate >= min_success_rate,
        "selection_accuracy_minimum_met": selection_accuracy >= min_selection_accuracy,
        "all_rows_policy_loaded": all(row.get("policy_loaded") is True for row in rows),
        "all_rows_public_repo": all(row.get("source_kind") == "public_repo" for row in rows),
        "all_rows_boundary_correct": all(row.get("claim_boundary") == CLAIM_BOUNDARY for row in rows),
        "candidate_selection_is_claim_bearing": all(
            row.get("claim_bearing_candidate_selection_evidence") is True for row in rows
        ),
        "no_rows_claim_freeform_patch": all(
            row.get("claim_bearing_freeform_patch_evidence") is False
            and row.get("freeform_patch_generation") is False
            for row in rows
        ),
        "sealed_feedback_absent": all(row.get("sealed_feedback_used") is False for row in rows),
    }
    return {
        "artifact_family": "phase2aa_bounded_patch_candidate_execution_audit",
        "passed": all(checks.values()),
        "claim_boundary": CLAIM_BOUNDARY,
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "success_count": len(successes),
            "success_rate": success_rate,
            "correct_patch_candidate_selections": len(correct_slots),
            "patch_candidate_selection_accuracy": selection_accuracy,
            "failure_reasons": dict(sorted(failure_reasons.items())),
        },
        "thresholds": {
            "min_rows": min_rows,
            "min_success_rate": min_success_rate,
            "min_selection_accuracy": min_selection_accuracy,
        },
        "supported_claim_if_passed": [
            "bounded_patch_candidate_selection_under_public_repo_runtime_evidence"
        ],
        "unsupported_claims": [
            "freeform_patch_generation",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AA bounded patch candidate execution results."
    )
    parser.add_argument("--execution-results-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=24)
    parser.add_argument("--min-success-rate", type=float, default=0.85)
    parser.add_argument("--min-selection-accuracy", type=float, default=0.85)
    args = parser.parse_args()
    report = audit_phase2aa_bounded_patch_candidate_execution(
        execution_results_jsonl=args.execution_results_jsonl,
        min_rows=args.min_rows,
        min_success_rate=args.min_success_rate,
        min_selection_accuracy=args.min_selection_accuracy,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
