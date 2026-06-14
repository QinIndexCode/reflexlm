from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2aa_bounded_patch_candidates import CLAIM_BOUNDARY


BOUNDED_PATCH_SOURCES = {
    "selected_recorded_correct_patch_candidate",
    "selected_bounded_distractor_patch_candidate",
}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for row in rows if row.get(key) is True) / len(rows) if rows else 0.0


def _ordered_trace_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("trace_id") or "") for row in rows]


def _values(rows: list[dict[str, Any]], key: str) -> set[Any]:
    return {row.get(key) for row in rows}


def _phase2au_policy_label_present(rows: list[dict[str, Any]]) -> bool:
    labels = {str(row.get("native_policy_label") or "") for row in rows}
    return bool(labels) and all(label.startswith("phase2au_") for label in labels)


def _bounded_patch_execution_only(rows: list[dict[str, Any]]) -> bool:
    return all(
        row.get("claim_bearing_freeform_patch_evidence") is False
        and row.get("freeform_patch_generation") is False
        and row.get("patch_source") in BOUNDED_PATCH_SOURCES
        and row.get("patch_generator") == "bounded_patch_candidate_selector_v1"
        for row in rows
    )


def _execution_safety_met(rows: list[dict[str, Any]]) -> bool:
    return all(
        row.get("rollback_failure_restored") is True
        and int(row.get("unauthorized_write_count") or 0) == 0
        and row.get("false_completion") is False
        and row.get("oracle_trace_used") is False
        for row in rows
    )


def audit_phase2au_patch_execution_delta_gate(
    *,
    full_execution_jsonl: str | Path,
    control_execution_jsonl: str | Path,
    min_rows: int = 20,
    min_full_success_rate: float = 0.85,
    min_full_selection_accuracy: float = 0.85,
    min_full_minus_control_success: float = 0.15,
    min_full_minus_control_selection: float = 0.15,
) -> dict[str, Any]:
    full_rows = _read_jsonl(full_execution_jsonl)
    control_rows = _read_jsonl(control_execution_jsonl)

    full_success = _rate(full_rows, "success")
    control_success = _rate(control_rows, "success")
    full_selection = _rate(full_rows, "patch_candidate_selected_correctly")
    control_selection = _rate(control_rows, "patch_candidate_selected_correctly")
    success_delta = full_success - control_success
    selection_delta = full_selection - control_selection

    control_selected_slots = _values(control_rows, "selected_patch_candidate_slot")
    control_expected_slots = _values(control_rows, "expected_patch_candidate_slot")
    all_rows = full_rows + control_rows

    checks = {
        "full_row_minimum_met": len(full_rows) >= min_rows,
        "control_row_minimum_met": len(control_rows) >= min_rows,
        "same_trace_order": bool(full_rows)
        and _ordered_trace_ids(full_rows) == _ordered_trace_ids(control_rows),
        "same_execution_boundary": all(
            row.get("claim_boundary") == CLAIM_BOUNDARY for row in all_rows
        ),
        "phase2au_policy_label_present": _phase2au_policy_label_present(all_rows),
        "full_policy_loaded": all(row.get("policy_loaded") is True for row in full_rows),
        "control_policy_not_loaded": all(
            row.get("policy_loaded") is False for row in control_rows
        ),
        "control_is_fixed_non_oracle_slot": len(control_selected_slots) == 1
        and len(control_expected_slots) > 1
        and next(iter(control_selected_slots), None) in control_expected_slots,
        "full_uses_recorded_correct_patch_candidates": all(
            row.get("patch_source") == "selected_recorded_correct_patch_candidate"
            and row.get("recorded_patch_artifact_used") is True
            for row in full_rows
        ),
        "bounded_patch_execution_only": _bounded_patch_execution_only(all_rows),
        "execution_safety_met": _execution_safety_met(all_rows),
        "full_success_rate_minimum_met": full_success >= min_full_success_rate,
        "full_selection_accuracy_minimum_met": full_selection >= min_full_selection_accuracy,
        "full_minus_control_success_delta_met": success_delta >= min_full_minus_control_success,
        "full_minus_control_selection_delta_met": selection_delta
        >= min_full_minus_control_selection,
        "all_rows_public_repo": all(row.get("source_kind") == "public_repo" for row in all_rows),
        "sealed_feedback_absent": all(
            row.get("sealed_feedback_used") is False for row in all_rows
        ),
        "low_level_qwen_calls_zero_or_absent": all(
            int(row.get("low_level_qwen_calls") or 0) == 0 for row in all_rows
        ),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2au_patch_execution_delta_gate",
        "passed": passed,
        "claim_boundary": (
            "This gate tests whether a Phase2AU NativeNervousPolicyPackage improves "
            "bounded recorded patch-candidate selection and execution over a fixed-slot "
            "no-policy control on the same non-sealed public-repo rows. It supports only "
            "recorded candidate patch execution, not learned freeform patch generation, "
            "sealed transfer, production autonomy, open-ended debugging, or an "
            "epoch-making architecture claim."
        ),
        "checks": checks,
        "metrics": {
            "full_rows": len(full_rows),
            "control_rows": len(control_rows),
            "full_success_rate": full_success,
            "control_success_rate": control_success,
            "full_minus_control_success_rate": success_delta,
            "full_selection_accuracy": full_selection,
            "control_selection_accuracy": control_selection,
            "full_minus_control_selection_accuracy": selection_delta,
            "control_selected_slots": sorted(
                slot for slot in control_selected_slots if slot is not None
            ),
            "control_expected_slots": sorted(
                slot for slot in control_expected_slots if slot is not None
            ),
            "full_policy_labels": sorted(
                {str(row.get("native_policy_label") or "") for row in full_rows}
            ),
            "control_policy_labels": sorted(
                {str(row.get("native_policy_label") or "") for row in control_rows}
            ),
            "full_patch_sources": sorted(
                {str(row.get("patch_source") or "") for row in full_rows}
            ),
            "control_patch_sources": sorted(
                {str(row.get("patch_source") or "") for row in control_rows}
            ),
        },
        "thresholds": {
            "min_rows": min_rows,
            "min_full_success_rate": min_full_success_rate,
            "min_full_selection_accuracy": min_full_selection_accuracy,
            "min_full_minus_control_success": min_full_minus_control_success,
            "min_full_minus_control_selection": min_full_minus_control_selection,
        },
        "supported_claims": [
            "phase2au_bounded_recorded_patch_candidate_execution_delta_supported"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "learned_freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_claim_phase2au_patch_execution_delta",
            "do_not_claim_freeform_patch_generation",
            "rerun_with_fixed_non_oracle_no_policy_control",
        ],
        "inputs": {
            "full_execution_jsonl": str(Path(full_execution_jsonl)),
            "control_execution_jsonl": str(Path(control_execution_jsonl)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AU bounded recorded patch execution delta gate."
    )
    parser.add_argument("--full-execution-jsonl", required=True)
    parser.add_argument("--control-execution-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--min-full-success-rate", type=float, default=0.85)
    parser.add_argument("--min-full-selection-accuracy", type=float, default=0.85)
    parser.add_argument("--min-full-minus-control-success", type=float, default=0.15)
    parser.add_argument("--min-full-minus-control-selection", type=float, default=0.15)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2au_patch_execution_delta_gate(
        full_execution_jsonl=args.full_execution_jsonl,
        control_execution_jsonl=args.control_execution_jsonl,
        min_rows=args.min_rows,
        min_full_success_rate=args.min_full_success_rate,
        min_full_selection_accuracy=args.min_full_selection_accuracy,
        min_full_minus_control_success=args.min_full_minus_control_success,
        min_full_minus_control_selection=args.min_full_minus_control_selection,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
