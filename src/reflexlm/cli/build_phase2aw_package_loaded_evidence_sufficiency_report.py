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


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("metrics")
    return value if isinstance(value, dict) else {}


def build_phase2aw_package_loaded_evidence_sufficiency_report(
    *,
    package_authorization_gate_json: str | Path,
    postpackage_gate_json: str | Path,
    package_loaded_runtime_gate_json: str | Path,
    package_loaded_failure_audit_json: str | Path,
) -> dict[str, Any]:
    authorization = _read_json(package_authorization_gate_json)
    postpackage = _read_json(postpackage_gate_json)
    runtime_gate = _read_json(package_loaded_runtime_gate_json)
    failure_audit = _read_json(package_loaded_failure_audit_json)
    runtime_metrics = _metrics(runtime_gate)
    failure_metrics = _metrics(failure_audit)
    checks = {
        "package_authorization_passed": authorization.get("passed") is True,
        "package_authorization_does_not_allow_sealed": authorization.get(
            "ready_for_sealed_eval"
        )
        is False,
        "postpackage_gate_passed": postpackage.get("passed") is True,
        "postpackage_does_not_allow_sealed": postpackage.get("ready_for_sealed_eval")
        is False,
        "package_loaded_runtime_gate_passed": runtime_gate.get("passed") is True,
        "runtime_gate_does_not_allow_sealed": runtime_gate.get("ready_for_sealed_eval")
        is False,
        "failure_audit_passed": failure_audit.get("passed") is True,
        "selection_not_primary_bottleneck": failure_audit.get("checks", {}).get(
            "selection_is_not_primary_bottleneck"
        )
        is True,
        "holdout_split_clean": failure_audit.get("checks", {}).get(
            "holdout_source_artifact_split_clean"
        )
        is True,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2aw_package_loaded_evidence_sufficiency_report",
        "passed": passed,
        "claim_scope": (
            "phase2aw_bounded_nonsealed_package_loaded_descriptor_runtime"
            if passed
            else "phase2aw_package_loaded_evidence_incomplete"
        ),
        "claim_boundary": (
            "Phase2AW supports a bounded non-sealed package-loaded descriptor "
            "runtime mechanism: the NativeNervousPolicyPackage is loaded, selects "
            "bounded descriptor repair candidates, and beats a measured "
            "source-overlap control on split-clean public holdout rows. This does "
            "not prove sealed transfer, freeform patch generation, production "
            "autonomy, open-ended debugging generalization, or an epoch-making "
            "architecture."
        ),
        "checks": checks,
        "metrics": {
            "package_loaded_full_success_rate": runtime_metrics.get("full_success_rate"),
            "source_overlap_success_rate": runtime_metrics.get(
                "source_overlap_success_rate"
            ),
            "full_minus_source_overlap_success_rate": runtime_metrics.get(
                "full_minus_source_overlap_success_rate"
            ),
            "package_loaded_selection_accuracy": runtime_metrics.get(
                "full_selection_accuracy"
            ),
            "failure_audit_full_failures": failure_metrics.get("full_failures"),
            "failure_audit_full_rows": failure_metrics.get("full_rows"),
        },
        "supported_claims": [
            "phase2aw_package_loaded_bounded_nonsealed_runtime_delta_supported",
            "phase2aw_package_loaded_candidate_selection_not_primary_bottleneck",
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
        "blocked_actions": [
            "do_not_run_sealed_eval_without_explicit_sealed_authorization_gate",
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_open_ended_debugging_generalization",
            "do_not_claim_epoch_making_architecture",
        ],
        "next_required_evidence": [
            "design_explicit_sealed_eval_authorization_gate_if_sealed_eval_is_next",
            "run_external_or_cross_repo_package_loaded_controls_before_stronger_claims",
            "add_multiseed_package_loaded_runtime_reproduction_before_architecture_upgrade",
        ],
        "inputs": {
            "package_authorization_gate_json": str(Path(package_authorization_gate_json)),
            "postpackage_gate_json": str(Path(postpackage_gate_json)),
            "package_loaded_runtime_gate_json": str(Path(package_loaded_runtime_gate_json)),
            "package_loaded_failure_audit_json": str(Path(package_loaded_failure_audit_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AW package-loaded evidence sufficiency report."
    )
    parser.add_argument("--package-authorization-gate-json", required=True)
    parser.add_argument("--postpackage-gate-json", required=True)
    parser.add_argument("--package-loaded-runtime-gate-json", required=True)
    parser.add_argument("--package-loaded-failure-audit-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2aw_package_loaded_evidence_sufficiency_report(
        package_authorization_gate_json=args.package_authorization_gate_json,
        postpackage_gate_json=args.postpackage_gate_json,
        package_loaded_runtime_gate_json=args.package_loaded_runtime_gate_json,
        package_loaded_failure_audit_json=args.package_loaded_failure_audit_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
