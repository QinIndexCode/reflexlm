from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import CLAIM_BOUNDARY


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def audit_phase2ab_identity_ambiguous_execution(
    *,
    execution_results_jsonl: str | Path,
    min_rows: int = 24,
    min_success_rate: float = 0.85,
    min_retry_recoveries: int = 1,
) -> dict[str, Any]:
    rows = _read_jsonl(execution_results_jsonl)
    successes = [row for row in rows if row.get("success") is True]
    final_correct = [
        row for row in rows if row.get("selected_patch_candidate_slot") == row.get("expected_patch_candidate_slot")
    ]
    initial_correct = [
        row
        for row in rows
        if row.get("initial_selected_patch_candidate_slot") == row.get("expected_patch_candidate_slot")
    ]
    retry_recoveries = [
        row
        for row in rows
        if row.get("success") is True
        and row.get("initial_selected_patch_candidate_slot") != row.get("expected_patch_candidate_slot")
        and row.get("selected_patch_candidate_slot") == row.get("expected_patch_candidate_slot")
    ]
    attempt_counts = [
        float(len(row.get("candidate_attempts") or []))
        for row in rows
    ]
    failed_distractor_attempts = sum(
        1
        for row in rows
        for attempt in row.get("candidate_attempts") or []
        if isinstance(attempt, dict)
        and attempt.get("patch_source") == "selected_bounded_distractor_patch_candidate"
        and attempt.get("passed") is False
    )
    checks = {
        "row_minimum_met": len(rows) >= min_rows,
        "success_rate_minimum_met": (len(successes) / len(rows) if rows else 0.0) >= min_success_rate,
        "all_rows_policy_loaded": all(row.get("policy_loaded") is True for row in rows),
        "all_rows_retry_enabled": all(row.get("bounded_candidate_retry_enabled") is True for row in rows),
        "retry_recovery_minimum_met": len(retry_recoveries) >= min_retry_recoveries,
        "final_accuracy_exceeds_initial_accuracy": len(final_correct) > len(initial_correct),
        "distractor_failures_observed": failed_distractor_attempts > 0,
        "all_rows_public_repo": all(row.get("source_kind") == "public_repo" for row in rows),
        "no_rows_claim_freeform_patch": all(
            row.get("claim_bearing_freeform_patch_evidence") is False
            and row.get("freeform_patch_generation") is False
            for row in rows
        ),
        "sealed_feedback_absent": all(row.get("sealed_feedback_used") is False for row in rows),
        "no_false_completion": all(row.get("false_completion") is False for row in rows),
        "patch_observable_or_recorded_tests_used": all(
            row.get("generated_test_used") is True for row in rows
        ),
    }
    row_count = len(rows)
    return {
        "artifact_family": "phase2ab_identity_ambiguous_execution_audit",
        "passed": all(checks.values()),
        "claim_boundary": CLAIM_BOUNDARY,
        "checks": checks,
        "metrics": {
            "row_count": row_count,
            "success_count": len(successes),
            "success_rate": len(successes) / row_count if row_count else 0.0,
            "initial_correct_count": len(initial_correct),
            "initial_selection_accuracy": len(initial_correct) / row_count if row_count else 0.0,
            "final_correct_count": len(final_correct),
            "final_selection_accuracy_after_retry": len(final_correct) / row_count if row_count else 0.0,
            "retry_recovery_count": len(retry_recoveries),
            "failed_distractor_attempts": failed_distractor_attempts,
            "average_candidate_attempts": _mean(attempt_counts),
            "max_candidate_attempts": max(attempt_counts) if attempt_counts else 0.0,
        },
        "thresholds": {
            "min_rows": min_rows,
            "min_success_rate": min_success_rate,
            "min_retry_recoveries": min_retry_recoveries,
        },
        "supported_claim_if_passed": [
            "bounded_verification_retry_recovers_identity_ambiguous_candidate_selection"
        ],
        "unsupported_claims": [
            "single_shot_native_head_necessity",
            "freeform_patch_generation",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AB identity-ambiguous bounded retry execution results."
    )
    parser.add_argument("--execution-results-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=24)
    parser.add_argument("--min-success-rate", type=float, default=0.85)
    parser.add_argument("--min-retry-recoveries", type=int, default=1)
    args = parser.parse_args()
    report = audit_phase2ab_identity_ambiguous_execution(
        execution_results_jsonl=args.execution_results_jsonl,
        min_rows=args.min_rows,
        min_success_rate=args.min_success_rate,
        min_retry_recoveries=args.min_retry_recoveries,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
