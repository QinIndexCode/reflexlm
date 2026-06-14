from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def audit_phase2bf_natural_receptor_format_shift(
    *,
    phase2be_audit_json: str | Path,
    standard_summary_json: str | Path,
    shifted_summary_json: str | Path,
    wrong_cache_summary_json: str | Path,
    min_rows: int = 12,
) -> dict[str, Any]:
    phase2be = _read(phase2be_audit_json)
    standard = _read(standard_summary_json)
    shifted = _read(shifted_summary_json)
    wrong = _read(wrong_cache_summary_json)
    rows = int(_number(shifted, "rows"))
    checks = {
        "phase2be_natural_receptor_audit_passed": phase2be.get("passed") is True,
        "row_counts_match_and_meet_minimum": rows >= min_rows
        and int(_number(standard, "rows")) == rows,
        "both_use_low_level_only": standard.get("package_nsi_reference_mode")
        == "low_level_only"
        and shifted.get("package_nsi_reference_mode") == "low_level_only",
        "both_use_no_override": _number(standard, "package_nsi_reference_override_rows")
        == 0.0
        and _number(shifted, "package_nsi_reference_override_rows") == 0.0,
        "evidence_labels_are_distinct": standard.get("package_runtime_evidence_label")
        != shifted.get("package_runtime_evidence_label"),
        "standard_success_gate": _number(standard, "success_rate") >= 0.85,
        "format_shift_success_gate": _number(shifted, "success_rate") >= 0.80,
        "format_shift_slot_gate": _number(shifted, "slot_selection_accuracy") >= 0.80,
        "format_shift_selected_repairs_execute": _number(
            shifted, "attempt_success_rate"
        )
        == 1.0,
        "wrong_cache_blocks_execution": _number(wrong, "success_rate") == 0.0
        and _number(wrong, "execution_attempts") == 0.0,
        "no_freeform_patch_generation": _number(
            shifted, "freeform_patch_generation_rows"
        )
        == 0.0,
        "no_sealed_feedback": _number(shifted, "sealed_feedback_used_rows") == 0.0,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2bf_natural_receptor_format_shift",
        "passed": passed,
        "ready_for_bounded_receptor_format_shift_claim": passed,
        "ready_for_open_ended_format_invariance_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "rows": rows,
            "standard_success_rate": _number(standard, "success_rate"),
            "format_shift_success_rate": _number(shifted, "success_rate"),
            "format_shift_slot_selection_accuracy": _number(
                shifted, "slot_selection_accuracy"
            ),
            "wrong_cache_success_rate": _number(wrong, "success_rate"),
        },
        "supported_claims": [
            "bounded low-level receptor latent survives the preregistered runtime-evidence label shift"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "open-ended format invariance",
            "repo-disjoint natural receptor generalization",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bg_repo_disjoint_multiseed_natural_receptor_transfer"
            if passed
            else "repair_phase2bf_format_shift_failure"
        ),
        "inputs": {
            "phase2be_audit_json": str(Path(phase2be_audit_json)),
            "standard_summary_json": str(Path(standard_summary_json)),
            "shifted_summary_json": str(Path(shifted_summary_json)),
            "wrong_cache_summary_json": str(Path(wrong_cache_summary_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2BF natural receptor format shift.")
    parser.add_argument("--phase2be-audit-json", required=True)
    parser.add_argument("--standard-summary-json", required=True)
    parser.add_argument("--shifted-summary-json", required=True)
    parser.add_argument("--wrong-cache-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=12)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2bf_natural_receptor_format_shift(
        phase2be_audit_json=args.phase2be_audit_json,
        standard_summary_json=args.standard_summary_json,
        shifted_summary_json=args.shifted_summary_json,
        wrong_cache_summary_json=args.wrong_cache_summary_json,
        min_rows=args.min_rows,
    )
    _write(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
