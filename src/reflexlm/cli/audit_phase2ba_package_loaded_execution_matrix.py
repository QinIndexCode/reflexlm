from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARTIFACT_FAMILY = "phase2ba_package_loaded_model_predicted_execution_matrix_audit"


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def audit_phase2ba_package_loaded_execution_matrix(
    *,
    package_gate_json: str | Path,
    package_execution_summary_json: str | Path,
    wrong_cache_summary_json: str | Path,
    output_json: str | Path | None = None,
    min_rows: int = 6,
    min_slot_selection_accuracy: float = 0.85,
    min_execution_success_rate: float = 0.75,
) -> dict[str, Any]:
    package_gate = _read_json(package_gate_json)
    package_execution = _read_json(package_execution_summary_json)
    wrong_cache = _read_json(wrong_cache_summary_json)
    rows = int(package_execution.get("rows") or 0)
    slot_accuracy = _float(package_execution.get("slot_selection_accuracy"))
    success_rate = _float(package_execution.get("success_rate"))
    wrong_success_rate = _float(wrong_cache.get("success_rate"))
    checks = {
        "package_gate_passed": package_gate.get("passed") is True
        and package_gate.get("ready_for_phase2az_packaged_adapter_runtime_smoke") is True,
        "package_gate_blocks_epoch_claim": package_gate.get(
            "ready_for_epoch_making_architecture_claim"
        )
        is False,
        "execution_policy_is_package_loaded_native_head": package_execution.get(
            "selection_policy"
        )
        == "package_loaded_native_head",
        "min_rows_met": rows >= min_rows,
        "slot_selection_accuracy_gate": isinstance(slot_accuracy, float)
        and slot_accuracy >= min_slot_selection_accuracy,
        "execution_success_rate_gate": isinstance(success_rate, float)
        and success_rate >= min_execution_success_rate,
        "package_policy_loaded_for_all_rows": int(
            package_execution.get("package_policy_loaded_rows") or 0
        )
        == rows,
        "phase2ax_head_record_visible_state_for_all_rows": int(
            package_execution.get("package_head_record_visible_state_rows") or 0
        )
        == rows,
        "package_open_repair_authorized_for_attempts": int(
            package_execution.get("package_open_repair_authorized_rows") or 0
        )
        >= int(package_execution.get("execution_attempts") or 0),
        "wrong_cache_control_blocks_execution": wrong_cache.get("selection_policy")
        == "wrong_cache"
        and int(wrong_cache.get("execution_attempts") or 0) == 0
        and wrong_success_rate == 0.0,
        "no_recorded_patch_as_generated_evidence": int(
            package_execution.get("recorded_patch_artifact_used_rows") or 0
        )
        == 0,
        "recorded_patch_only_for_fault_injection": int(
            package_execution.get("recorded_patch_artifact_used_for_fault_injection_rows") or 0
        )
        == int(package_execution.get("execution_attempts") or -1),
        "claim_bearing_execution_for_attempts": int(
            package_execution.get("claim_bearing_execution_evidence_rows") or 0
        )
        == int(package_execution.get("execution_attempts") or -1),
        "no_freeform_patch_generation": int(
            package_execution.get("freeform_patch_generation_rows") or 0
        )
        == 0,
        "no_sealed_feedback": int(package_execution.get("sealed_feedback_used_rows") or 0)
        == 0,
        "model_prediction_json_not_used_as_selector": int(
            package_execution.get("model_prediction_records_present_rows") or 0
        )
        == 0,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": ARTIFACT_FAMILY,
        "passed": passed,
        "ready_for_phase2ba_package_loaded_runtime_matrix": passed,
        "ready_for_phase2ax_package": False,
        "ready_for_package_or_execution_claim": False,
        "ready_for_sealed_eval": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "rows": rows,
            "slot_selection_accuracy": slot_accuracy,
            "execution_attempts": package_execution.get("execution_attempts"),
            "success_rate": success_rate,
            "attempt_success_rate": package_execution.get("attempt_success_rate"),
            "wrong_cache_success_rate": wrong_success_rate,
            "package_policy_loaded_rows": package_execution.get(
                "package_policy_loaded_rows"
            ),
            "package_head_record_visible_state_rows": package_execution.get(
                "package_head_record_visible_state_rows"
            ),
            "package_qwen_called_rows": package_execution.get("package_qwen_called_rows"),
            "package_low_level_debug_receptor_observed_rows": package_execution.get(
                "package_low_level_debug_receptor_observed_rows"
            ),
            "package_open_repair_authorized_rows": package_execution.get(
                "package_open_repair_authorized_rows"
            ),
        },
        "claim_boundary": (
            "phase2ba_package_loaded_native_head_runtime_matrix_not_sealed_or_epoch_claim"
            if passed
            else "phase2ba_package_loaded_runtime_matrix_failed_or_incomplete"
        ),
        "supported_claims": [
            "phase2ba_package_loaded_native_head_selects_bounded_repair_slots_and_enables_runtime_execution"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "phase2ax_full_package_claim",
            "sealed_cross_model_transfer",
            "freeform_patch_generation",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": [
            "do_not_claim_sealed_transfer_from_nonsealed_package_loaded_matrix",
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ],
        "next_required_experiment": (
            "phase2bb_expand_package_loaded_runtime_matrix_and_repair_descriptor_failure"
            if passed
            else "repair_phase2ba_package_loaded_runtime_matrix"
        ),
        "thresholds": {
            "min_rows": min_rows,
            "min_slot_selection_accuracy": min_slot_selection_accuracy,
            "min_execution_success_rate": min_execution_success_rate,
        },
        "inputs": {
            "package_gate_json": str(Path(package_gate_json)),
            "package_execution_summary_json": str(Path(package_execution_summary_json)),
            "wrong_cache_summary_json": str(Path(wrong_cache_summary_json)),
        },
        "notes": [
            "This audit requires NativeNervousPolicyPackage-loaded selection, not prediction-record replay.",
            "Execution remains bounded to fixed candidate actions and runtime-symbolic structural repair.",
            "Passing this audit is still nonsealed and does not prove production autonomy or epoch-making architecture.",
        ],
    }
    if output_json is not None:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2BA package-loaded model-predicted execution matrix."
    )
    parser.add_argument("--package-gate-json", required=True)
    parser.add_argument("--package-execution-summary-json", required=True)
    parser.add_argument("--wrong-cache-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=6)
    parser.add_argument("--min-slot-selection-accuracy", type=float, default=0.85)
    parser.add_argument("--min-execution-success-rate", type=float, default=0.75)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ba_package_loaded_execution_matrix(
        package_gate_json=args.package_gate_json,
        package_execution_summary_json=args.package_execution_summary_json,
        wrong_cache_summary_json=args.wrong_cache_summary_json,
        output_json=args.output_json,
        min_rows=args.min_rows,
        min_slot_selection_accuracy=args.min_slot_selection_accuracy,
        min_execution_success_rate=args.min_execution_success_rate,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
