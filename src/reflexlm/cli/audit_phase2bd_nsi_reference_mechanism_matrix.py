from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    return int(value) if isinstance(value, (int, float)) else 0


def audit_phase2bd_nsi_reference_mechanism_matrix(
    *,
    runtime_visible_summary_json: str | Path,
    low_level_only_summary_json: str | Path,
    wrong_cache_summary_json: str | Path,
    package_execution_audit_json: str | Path,
    sealed_transfer_report_json: str | Path,
    min_rows: int = 12,
    min_reference_slot_delta: float = 0.25,
    min_reference_success_delta: float = 0.25,
) -> dict[str, Any]:
    full = _read_json(runtime_visible_summary_json)
    low = _read_json(low_level_only_summary_json)
    wrong = _read_json(wrong_cache_summary_json)
    execution_audit = _read_json(package_execution_audit_json)
    sealed = _read_json(sealed_transfer_report_json)

    rows = _int(full, "rows")
    low_rows = _int(low, "rows")
    slot_delta = _float(full, "slot_selection_accuracy") - _float(
        low, "slot_selection_accuracy"
    )
    success_delta = _float(full, "success_rate") - _float(low, "success_rate")
    checks = {
        "phase2bc_package_execution_audit_passed": execution_audit.get("passed") is True,
        "sealed_transfer_report_passed": sealed.get("passed") is True,
        "sealed_report_blocks_epoch_claim": sealed.get(
            "ready_for_epoch_making_architecture_claim"
        )
        is False,
        "row_counts_match_and_meet_minimum": rows == low_rows and rows >= min_rows,
        "runtime_visible_mode_recorded": full.get("package_nsi_reference_mode")
        == "runtime_visible_override",
        "runtime_visible_override_used_for_all_rows": _int(
            full, "package_nsi_reference_override_rows"
        )
        == rows,
        "low_level_only_mode_recorded": low.get("package_nsi_reference_mode")
        == "low_level_only",
        "low_level_only_uses_no_override": _int(low, "package_nsi_reference_override_rows")
        == 0,
        "package_loaded_and_qwen_called_for_all_rows": all(
            _int(payload, "package_policy_loaded_rows") == rows
            and _int(payload, "package_qwen_called_rows") == rows
            for payload in (full, low)
        ),
        "runtime_visible_full_execution_success": _float(full, "success_rate") == 1.0,
        "low_level_selected_repairs_execute_when_selected": _float(
            low, "attempt_success_rate"
        )
        == 1.0,
        "reference_slot_delta_gate": slot_delta >= min_reference_slot_delta,
        "reference_success_delta_gate": success_delta >= min_reference_success_delta,
        "wrong_cache_blocks_execution": _float(wrong, "success_rate") == 0.0
        and _int(wrong, "execution_attempts") == 0,
        "no_freeform_patch_generation": all(
            _int(payload, "freeform_patch_generation_rows") == 0
            for payload in (full, low)
        ),
        "no_sealed_feedback_used": all(
            _int(payload, "sealed_feedback_used_rows") == 0 for payload in (full, low)
        ),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2bd_nsi_reference_mechanism_matrix",
        "passed": passed,
        "natural_perception_status": (
            "runtime_visible_synaptic_signal_is_causal_low_level_nsi_not_yet_sufficient"
            if passed
            else "phase2bd_nsi_reference_mechanism_incomplete"
        ),
        "ready_for_runtime_visible_synaptic_signal_claim": passed,
        "ready_for_low_level_nsi_natural_perception_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "rows": rows,
            "runtime_visible_slot_selection_accuracy": _float(
                full, "slot_selection_accuracy"
            ),
            "low_level_only_slot_selection_accuracy": _float(
                low, "slot_selection_accuracy"
            ),
            "runtime_visible_success_rate": _float(full, "success_rate"),
            "low_level_only_success_rate": _float(low, "success_rate"),
            "low_level_only_attempt_success_rate": _float(low, "attempt_success_rate"),
            "runtime_visible_minus_low_level_slot_accuracy": slot_delta,
            "runtime_visible_minus_low_level_success_rate": success_delta,
            "wrong_cache_success_rate": _float(wrong, "success_rate"),
        },
        "interpretation": (
            "The bounded package-loaded repair loop causally depends on runtime-visible "
            "synaptic reference signals: removing only that reference reduces slot "
            "selection and end-to-end success while selected repairs still execute. "
            "The low-level NSI path therefore does not yet naturally reconstruct the "
            "discriminative prior-runtime signal required by this matrix."
        ),
        "next_required_experiment": {
            "phase": "phase2be_learned_low_level_runtime_receptor_latent",
            "goal": (
                "train or distill the low-level receptor/NSI path to reconstruct "
                "runtime-visible prior-evidence latents without inference-time override"
            ),
            "hard_gates": [
                "low_level_only slot selection accuracy >= 0.85",
                "low_level_only execution success rate >= 0.85",
                "runtime_visible minus low_level_only success <= 0.10",
                "wrong-cache success remains 0.0",
                "no sealed feedback, gold hints, or freeform patch generation",
            ],
        },
        "supported_claims": [
            "runtime-visible synaptic reference signals causally improve bounded package-loaded repair selection and execution",
            "the bounded symbolic repair executor succeeds when the low-level-only policy selects the correct slot",
            "sealed command-level package transfer and nonsealed repair execution are both independently evidenced",
        ]
        if passed
        else [],
        "unsupported_claims": [
            "low-level NSI alone already provides natural perception for this repair matrix",
            "open-ended debugging generalization",
            "production autonomy",
            "freeform patch generation",
            "epoch-making architecture",
        ],
        "inputs": {
            "runtime_visible_summary_json": str(Path(runtime_visible_summary_json)),
            "low_level_only_summary_json": str(Path(low_level_only_summary_json)),
            "wrong_cache_summary_json": str(Path(wrong_cache_summary_json)),
            "package_execution_audit_json": str(Path(package_execution_audit_json)),
            "sealed_transfer_report_json": str(Path(sealed_transfer_report_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2BD NSI reference mechanism matrix.")
    parser.add_argument("--runtime-visible-summary-json", required=True)
    parser.add_argument("--low-level-only-summary-json", required=True)
    parser.add_argument("--wrong-cache-summary-json", required=True)
    parser.add_argument("--package-execution-audit-json", required=True)
    parser.add_argument("--sealed-transfer-report-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=12)
    parser.add_argument("--min-reference-slot-delta", type=float, default=0.25)
    parser.add_argument("--min-reference-success-delta", type=float, default=0.25)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2bd_nsi_reference_mechanism_matrix(
        runtime_visible_summary_json=args.runtime_visible_summary_json,
        low_level_only_summary_json=args.low_level_only_summary_json,
        wrong_cache_summary_json=args.wrong_cache_summary_json,
        package_execution_audit_json=args.package_execution_audit_json,
        sealed_transfer_report_json=args.sealed_transfer_report_json,
        min_rows=args.min_rows,
        min_reference_slot_delta=args.min_reference_slot_delta,
        min_reference_success_delta=args.min_reference_success_delta,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
