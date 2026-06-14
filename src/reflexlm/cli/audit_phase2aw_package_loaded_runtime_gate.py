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


def _float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def audit_phase2aw_package_loaded_runtime_gate(
    *,
    package_loaded_summary_json: str | Path,
    source_overlap_summary_json: str | Path,
    postpackage_gate_json: str | Path,
    min_full_success_rate: float = 0.85,
    min_full_minus_control_success_rate: float = 0.15,
    min_selection_accuracy: float = 0.85,
) -> dict[str, Any]:
    full = _read_json(package_loaded_summary_json)
    control = _read_json(source_overlap_summary_json)
    postpackage = _read_json(postpackage_gate_json)
    full_success = _float(full.get("success_rate"))
    control_success = _float(control.get("success_rate"))
    full_selection = _float(full.get("patch_candidate_selection_accuracy"))
    control_selection = _float(control.get("selection_accuracy"))
    delta = full_success - control_success
    checks = {
        "postpackage_gate_passed": postpackage.get("passed") is True,
        "postpackage_authorizes_package_loaded_eval": postpackage.get(
            "ready_for_bounded_package_loaded_runtime_eval"
        )
        is True,
        "postpackage_does_not_authorize_sealed_eval": postpackage.get(
            "ready_for_sealed_eval"
        )
        is False,
        "full_summary_is_package_loaded_runner": full.get("artifact_family")
        == "phase2aw_package_loaded_descriptor_execution_runner",
        "full_policy_loaded": full.get("policy_loaded") is True,
        "full_success_rate_min": full_success >= min_full_success_rate,
        "full_selection_accuracy_min": full_selection >= min_selection_accuracy,
        "source_overlap_summary_is_control": control.get("selection_mode")
        == "source_overlap",
        "source_overlap_control_nonzero": control_success > 0.0,
        "source_overlap_control_not_ceiling": control_success <= 0.75,
        "full_minus_control_success_rate_min": delta
        >= min_full_minus_control_success_rate,
        "claim_boundary_bounded": str(full.get("claim_boundary") or "").startswith(
            "phase2aw_package_loaded_bounded_descriptor_execution"
        ),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2aw_package_loaded_nonsealed_runtime_gate",
        "passed": passed,
        "ready_for_sealed_eval_gate_design": passed,
        "ready_for_sealed_eval": False,
        "claim_boundary": (
            "This gate tests only bounded package-loaded non-sealed descriptor "
            "runtime execution. Passing it allows designing the next sealed-eval "
            "authorization gate, but does not itself authorize sealed evaluation "
            "or stronger architecture claims."
        ),
        "checks": checks,
        "metrics": {
            "full_success_rate": full_success,
            "source_overlap_success_rate": control_success,
            "full_minus_source_overlap_success_rate": delta,
            "full_selection_accuracy": full_selection,
            "source_overlap_selection_accuracy": control_selection,
        },
        "supported_claims": [
            "phase2aw_package_loaded_bounded_nonsealed_runtime_delta_supported"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "sealed_cross_model_transfer",
            "learned_freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_design_sealed_eval_gate",
            "do_not_run_sealed_eval",
            "fix_package_loaded_runtime_or_control_evidence",
        ],
        "post_gate_blocked_actions": [
            "do_not_run_sealed_eval_until_explicit_sealed_authorization_gate_passes",
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_epoch_making_architecture",
        ],
        "inputs": {
            "package_loaded_summary_json": str(Path(package_loaded_summary_json)),
            "source_overlap_summary_json": str(Path(source_overlap_summary_json)),
            "postpackage_gate_json": str(Path(postpackage_gate_json)),
        },
        "thresholds": {
            "min_full_success_rate": min_full_success_rate,
            "min_full_minus_control_success_rate": min_full_minus_control_success_rate,
            "min_selection_accuracy": min_selection_accuracy,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AW package-loaded non-sealed runtime gate."
    )
    parser.add_argument("--package-loaded-summary-json", required=True)
    parser.add_argument("--source-overlap-summary-json", required=True)
    parser.add_argument("--postpackage-gate-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-full-success-rate", type=float, default=0.85)
    parser.add_argument("--min-full-minus-control-success-rate", type=float, default=0.15)
    parser.add_argument("--min-selection-accuracy", type=float, default=0.85)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2aw_package_loaded_runtime_gate(
        package_loaded_summary_json=args.package_loaded_summary_json,
        source_overlap_summary_json=args.source_overlap_summary_json,
        postpackage_gate_json=args.postpackage_gate_json,
        min_full_success_rate=args.min_full_success_rate,
        min_full_minus_control_success_rate=args.min_full_minus_control_success_rate,
        min_selection_accuracy=args.min_selection_accuracy,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
