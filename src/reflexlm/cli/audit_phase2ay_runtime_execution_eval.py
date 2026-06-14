from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARTIFACT_FAMILY = "phase2ay_runtime_execution_eval_audit"


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


def audit_phase2ay_runtime_execution_eval(
    *,
    prior_summary_json: str | Path,
    wrong_cache_summary_json: str | Path,
    full_postflight_json: str | Path,
    min_rows: int = 2,
    min_prior_slot_accuracy: float = 0.85,
    min_prior_attempt_success_rate: float = 0.75,
    max_wrong_cache_slot_accuracy: float = 0.25,
    expected_prior_selection_policy: str = "prior_runtime_resolver",
) -> dict[str, Any]:
    prior = _read_json(prior_summary_json)
    wrong = _read_json(wrong_cache_summary_json)
    full_postflight = _read_json(full_postflight_json)

    prior_rows = int(prior.get("rows") or 0)
    wrong_rows = int(wrong.get("rows") or 0)
    prior_slot_accuracy = _float(prior.get("slot_selection_accuracy"))
    wrong_slot_accuracy = _float(wrong.get("slot_selection_accuracy"))
    prior_attempt_success_rate = _float(prior.get("attempt_success_rate"))
    wrong_success_rate = _float(wrong.get("success_rate"))
    prior_attempts = int(prior.get("execution_attempts") or 0)
    wrong_attempts = int(wrong.get("execution_attempts") or 0)

    checks = {
        "full_postflight_passed": full_postflight.get("passed") is True,
        "full_postflight_allows_phase2ay": full_postflight.get(
            "ready_for_phase2ay_runtime_execution_eval"
        )
        is True,
        "prior_summary_present": bool(prior),
        "wrong_cache_summary_present": bool(wrong),
        "prior_policy_matches_expected": prior.get("selection_policy")
        == expected_prior_selection_policy,
        "wrong_cache_policy_is_control": wrong.get("selection_policy") == "wrong_cache",
        "min_rows_met": prior_rows >= min_rows and wrong_rows >= min_rows,
        "row_counts_match": prior_rows == wrong_rows and prior_rows > 0,
        "prior_slot_accuracy_gate": isinstance(prior_slot_accuracy, float)
        and prior_slot_accuracy >= min_prior_slot_accuracy,
        "prior_execution_attempted": prior_attempts > 0,
        "prior_attempt_success_rate_gate": isinstance(prior_attempt_success_rate, float)
        and prior_attempt_success_rate >= min_prior_attempt_success_rate,
        "wrong_cache_slot_accuracy_low": isinstance(wrong_slot_accuracy, float)
        and wrong_slot_accuracy <= max_wrong_cache_slot_accuracy,
        "wrong_cache_no_execution_attempts": wrong_attempts == 0,
        "wrong_cache_no_success": wrong_success_rate == 0.0,
        "prior_uses_runtime_symbolic_patch_not_recorded_patch": int(
            prior.get("recorded_patch_artifact_used_rows") or 0
        )
        == 0
        and int(prior.get("claim_bearing_execution_evidence_rows") or 0) == prior_attempts,
        "model_prediction_records_present_when_required": (
            expected_prior_selection_policy != "model_prediction_records"
            or int(prior.get("model_prediction_records_present_rows") or 0) == prior_rows
        ),
        "recorded_patch_only_for_fault_injection": int(
            prior.get("recorded_patch_artifact_used_for_fault_injection_rows") or 0
        )
        == prior_attempts,
        "no_freeform_patch_generation": int(prior.get("freeform_patch_generation_rows") or 0)
        == 0
        and int(wrong.get("freeform_patch_generation_rows") or 0) == 0,
        "no_sealed_feedback": int(prior.get("sealed_feedback_used_rows") or 0) == 0
        and int(wrong.get("sealed_feedback_used_rows") or 0) == 0,
    }
    passed = all(checks.values())
    return {
        "artifact_family": ARTIFACT_FAMILY,
        "passed": passed,
        "ready_for_phase2ay_expanded_runtime_execution_eval": passed,
        "ready_for_phase2ay_model_prediction_execution_eval": passed
        and expected_prior_selection_policy == "model_prediction_records",
        "ready_for_phase2ax_package": False,
        "ready_for_package_or_execution_claim": False,
        "ready_for_sealed_eval": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "prior_rows": prior_rows,
            "wrong_cache_rows": wrong_rows,
            "prior_slot_selection_accuracy": prior_slot_accuracy,
            "wrong_cache_slot_selection_accuracy": wrong_slot_accuracy,
            "prior_execution_attempts": prior_attempts,
            "wrong_cache_execution_attempts": wrong_attempts,
            "prior_success_rate": _float(prior.get("success_rate")),
            "prior_attempt_success_rate": prior_attempt_success_rate,
            "wrong_cache_success_rate": wrong_success_rate,
            "prior_recorded_patch_artifact_used_rows": prior.get(
                "recorded_patch_artifact_used_rows"
            ),
            "prior_recorded_patch_artifact_used_for_fault_injection_rows": prior.get(
                "recorded_patch_artifact_used_for_fault_injection_rows"
            ),
            "prior_claim_bearing_execution_evidence_rows": prior.get(
                "claim_bearing_execution_evidence_rows"
            ),
            "prior_model_prediction_records_present_rows": prior.get(
                "model_prediction_records_present_rows"
            ),
        },
        "claim_boundary": (
            (
                "phase2ay_smoke_supports_model_predicted_slot_runtime_execution_not_package_or_epoch_claim"
                if expected_prior_selection_policy == "model_prediction_records"
                else "phase2ay_smoke_supports_slot_conditioned_runtime_execution_not_phase2ax_package_or_epoch_claim"
            )
            if passed
            else "phase2ay_runtime_execution_eval_failed_or_incomplete_not_claim_evidence"
        ),
        "next_required_experiment": (
            (
                "phase2az_expand_model_predicted_runtime_execution_repo_disjoint_matrix"
                if expected_prior_selection_policy == "model_prediction_records"
                else "phase2ay_expanded_repo_disjoint_model_prediction_execution_eval"
            )
            if passed
            else "repair_phase2ay_runtime_execution_eval_inputs"
        ),
        "blocked_actions": [
            "do_not_package_phase2ax_from_phase2ay_smoke",
            "do_not_claim_phase2ax_adapter_runtime_execution_without_model_prediction_trace",
            "do_not_run_sealed_v3_from_phase2ay_smoke",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ],
        "unsupported_claims": [
            "phase2ax_package_runtime_execution",
            "model_predicted_patch_descriptor_execution_at_scale",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "thresholds": {
            "min_rows": min_rows,
            "min_prior_slot_accuracy": min_prior_slot_accuracy,
            "min_prior_attempt_success_rate": min_prior_attempt_success_rate,
            "max_wrong_cache_slot_accuracy": max_wrong_cache_slot_accuracy,
            "expected_prior_selection_policy": expected_prior_selection_policy,
        },
        "inputs": {
            "prior_summary_json": str(Path(prior_summary_json)),
            "wrong_cache_summary_json": str(Path(wrong_cache_summary_json)),
            "full_postflight_json": str(Path(full_postflight_json)),
        },
        "notes": [
            "This audit checks slot-conditioned bounded runtime execution with a wrong-cache control.",
            "It is not Phase2AX package evidence; package readiness still requires an explicit packaged adapter/runtime gate.",
            "Recorded patch artifacts are allowed only for fault injection, not as generated repair evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AY runtime execution smoke.")
    parser.add_argument("--prior-summary-json", required=True)
    parser.add_argument("--wrong-cache-summary-json", required=True)
    parser.add_argument("--full-postflight-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=2)
    parser.add_argument("--min-prior-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-prior-attempt-success-rate", type=float, default=0.75)
    parser.add_argument("--max-wrong-cache-slot-accuracy", type=float, default=0.25)
    parser.add_argument(
        "--expected-prior-selection-policy",
        default="prior_runtime_resolver",
        choices=["prior_runtime_resolver", "model_prediction_records"],
    )
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ay_runtime_execution_eval(
        prior_summary_json=args.prior_summary_json,
        wrong_cache_summary_json=args.wrong_cache_summary_json,
        full_postflight_json=args.full_postflight_json,
        min_rows=args.min_rows,
        min_prior_slot_accuracy=args.min_prior_slot_accuracy,
        min_prior_attempt_success_rate=args.min_prior_attempt_success_rate,
        max_wrong_cache_slot_accuracy=args.max_wrong_cache_slot_accuracy,
        expected_prior_selection_policy=args.expected_prior_selection_policy,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
