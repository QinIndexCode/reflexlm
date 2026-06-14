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


def audit_phase2be_natural_receptor_latent_matrix(
    *,
    runtime_visible_summary_json: str | Path,
    pre_receptor_low_level_summary_json: str | Path,
    post_receptor_low_level_summary_json: str | Path,
    wrong_cache_summary_json: str | Path,
    sealed_transfer_report_json: str | Path,
    min_rows: int = 12,
    min_low_level_improvement: float = 0.25,
    max_reference_gap: float = 0.10,
) -> dict[str, Any]:
    reference = _read_json(runtime_visible_summary_json)
    before = _read_json(pre_receptor_low_level_summary_json)
    after = _read_json(post_receptor_low_level_summary_json)
    wrong = _read_json(wrong_cache_summary_json)
    sealed = _read_json(sealed_transfer_report_json)
    rows = _int(after, "rows")
    slot_improvement = _float(after, "slot_selection_accuracy") - _float(
        before, "slot_selection_accuracy"
    )
    success_improvement = _float(after, "success_rate") - _float(before, "success_rate")
    reference_gap = _float(reference, "success_rate") - _float(after, "success_rate")
    checks = {
        "sealed_transfer_passed_without_claim_upgrade": sealed.get("passed") is True
        and sealed.get("ready_for_epoch_making_architecture_claim") is False,
        "row_counts_match_and_meet_minimum": rows >= min_rows
        and _int(reference, "rows") == rows
        and _int(before, "rows") == rows,
        "reference_mode_is_runtime_visible_override": reference.get(
            "package_nsi_reference_mode"
        )
        == "runtime_visible_override",
        "reference_uses_override_for_all_rows": _int(
            reference, "package_nsi_reference_override_rows"
        )
        == rows,
        "before_and_after_are_low_level_only": before.get("package_nsi_reference_mode")
        == "low_level_only"
        and after.get("package_nsi_reference_mode") == "low_level_only",
        "before_and_after_use_no_override": _int(
            before, "package_nsi_reference_override_rows"
        )
        == 0
        and _int(after, "package_nsi_reference_override_rows") == 0,
        "low_level_slot_improvement_gate": slot_improvement >= min_low_level_improvement,
        "low_level_success_improvement_gate": success_improvement
        >= min_low_level_improvement,
        "post_receptor_low_level_slot_accuracy_gate": _float(
            after, "slot_selection_accuracy"
        )
        >= 0.85,
        "post_receptor_low_level_success_gate": _float(after, "success_rate") >= 0.85,
        "post_receptor_matches_reference_gate": abs(reference_gap) <= max_reference_gap,
        "selected_repairs_execute": _float(after, "attempt_success_rate") == 1.0,
        "package_loaded_and_qwen_called_for_all_rows": _int(
            after, "package_policy_loaded_rows"
        )
        == rows
        and _int(after, "package_qwen_called_rows") == rows,
        "wrong_cache_blocks_execution": _float(wrong, "success_rate") == 0.0
        and _int(wrong, "execution_attempts") == 0,
        "no_freeform_patch_generation": _int(after, "freeform_patch_generation_rows")
        == 0,
        "no_sealed_feedback_used": _int(after, "sealed_feedback_used_rows") == 0,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2be_natural_receptor_latent_matrix",
        "passed": passed,
        "natural_perception_status": (
            "low_level_runtime_receptor_latent_matches_explicit_reference_on_bounded_matrix"
            if passed
            else "phase2be_low_level_runtime_receptor_latent_incomplete"
        ),
        "ready_for_bounded_low_level_nsi_natural_perception_claim": passed,
        "ready_for_open_ended_natural_perception_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "rows": rows,
            "pre_receptor_low_level_slot_accuracy": _float(
                before, "slot_selection_accuracy"
            ),
            "post_receptor_low_level_slot_accuracy": _float(
                after, "slot_selection_accuracy"
            ),
            "pre_receptor_low_level_success_rate": _float(before, "success_rate"),
            "post_receptor_low_level_success_rate": _float(after, "success_rate"),
            "runtime_visible_reference_success_rate": _float(reference, "success_rate"),
            "low_level_slot_improvement": slot_improvement,
            "low_level_success_improvement": success_improvement,
            "runtime_visible_minus_post_receptor_success": reference_gap,
            "wrong_cache_success_rate": _float(wrong, "success_rate"),
        },
        "interpretation": (
            "A label-free receptor transformation moved discriminative prior-runtime "
            "structure evidence into the low-level NSI latent path. With inference-time "
            "reference override disabled, the same package now matches the explicit "
            "reference condition on bounded public-repo repair selection and execution."
        ),
        "next_required_experiment": {
            "phase": "phase2bf_natural_receptor_generalization",
            "goal": (
                "test the low-level receptor latent on repo-disjoint, structure-format-"
                "shifted, multi-seed and sealed final-evaluation-only matrices"
            ),
            "hard_gates": [
                "repo-disjoint low-level-only success >= 0.85",
                "format-shifted evidence success >= 0.80",
                "multi-seed mean success >= 0.85",
                "wrong-cache and evidence-erased controls remain diagnostic",
                "no sealed feedback, gold hints, or freeform patch generation",
            ],
        },
        "supported_claims": [
            "bounded low-level NSI natural perception from runtime receptor evidence",
            "low-level receptor latent causally closes the prior 0.5 execution gap without inference-time override",
            "structured motor selection and bounded repair execution remain fully verified on this matrix",
        ]
        if passed
        else [],
        "unsupported_claims": [
            "open-ended natural perception",
            "format-invariant receptor generalization",
            "production autonomy",
            "freeform patch generation",
            "epoch-making architecture",
        ],
        "inputs": {
            "runtime_visible_summary_json": str(Path(runtime_visible_summary_json)),
            "pre_receptor_low_level_summary_json": str(Path(pre_receptor_low_level_summary_json)),
            "post_receptor_low_level_summary_json": str(
                Path(post_receptor_low_level_summary_json)
            ),
            "wrong_cache_summary_json": str(Path(wrong_cache_summary_json)),
            "sealed_transfer_report_json": str(Path(sealed_transfer_report_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2BE natural receptor latent matrix.")
    parser.add_argument("--runtime-visible-summary-json", required=True)
    parser.add_argument("--pre-receptor-low-level-summary-json", required=True)
    parser.add_argument("--post-receptor-low-level-summary-json", required=True)
    parser.add_argument("--wrong-cache-summary-json", required=True)
    parser.add_argument("--sealed-transfer-report-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=12)
    parser.add_argument("--min-low-level-improvement", type=float, default=0.25)
    parser.add_argument("--max-reference-gap", type=float, default=0.10)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2be_natural_receptor_latent_matrix(
        runtime_visible_summary_json=args.runtime_visible_summary_json,
        pre_receptor_low_level_summary_json=args.pre_receptor_low_level_summary_json,
        post_receptor_low_level_summary_json=args.post_receptor_low_level_summary_json,
        wrong_cache_summary_json=args.wrong_cache_summary_json,
        sealed_transfer_report_json=args.sealed_transfer_report_json,
        min_rows=args.min_rows,
        min_low_level_improvement=args.min_low_level_improvement,
        max_reference_gap=args.max_reference_gap,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
