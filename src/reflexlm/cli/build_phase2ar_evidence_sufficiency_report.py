from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_phase2ar_evidence_sufficiency_report(
    *,
    data_health_json: str | Path,
    execution_audit_json: str | Path,
    control_delta_json: str | Path | None = None,
    reproduction_audit_json: str | Path | None = None,
    multiseed_reproduction_json: str | Path | None = None,
    cross_model_reproduction_json: str | Path | None = None,
) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    execution = _read_json(execution_audit_json)
    control_delta = _read_json(control_delta_json) if control_delta_json else None
    reproduction = _read_json(reproduction_audit_json) if reproduction_audit_json else None
    multiseed = _read_json(multiseed_reproduction_json) if multiseed_reproduction_json else None
    cross_model = (
        _read_json(cross_model_reproduction_json) if cross_model_reproduction_json else None
    )
    checks = {
        "data_health_passed": bool(data_health.get("passed")),
        "execution_audit_passed": bool(execution.get("passed")),
        "diverse_patch_kinds_in_data": bool(
            data_health.get("checks", {}).get("holdout_required_patch_kinds_present")
        ),
        "diverse_patch_kinds_in_execution": bool(
            execution.get("checks", {}).get("required_patch_kinds_present")
        ),
        "runtime_structural_success_rate_perfect": float(
            execution.get("metrics", {}).get("success_rate", 0.0)
        )
        >= 1.0,
        "recorded_patch_not_used_as_proposal": bool(
            execution.get("checks", {}).get("no_rows_use_recorded_patch_as_proposal")
        ),
        "sealed_feedback_absent": bool(data_health.get("checks", {}).get("sealed_feedback_absent"))
        and bool(execution.get("checks", {}).get("sealed_feedback_absent")),
    }
    control_checks = {}
    if control_delta is not None:
        control_checks = {
            "control_delta_passed": bool(control_delta.get("passed")),
            "controls_nonzero": bool(control_delta.get("checks", {}).get("controls_nonzero")),
            "best_control_below_ceiling": bool(
                control_delta.get("checks", {}).get("best_control_below_ceiling")
            ),
            "full_minus_best_control_met": bool(
                control_delta.get("checks", {}).get("full_minus_best_control_met")
            ),
        }
    passed = all(checks.values())
    if control_delta is not None:
        passed = passed and all(control_checks.values())
    supported_claims = [
        "bounded_runtime_symbolic_structural_patch_proposal_diverse_holdout_supported"
    ]
    if control_delta is not None and all(control_checks.values()):
        supported_claims.append(
            "phase2ar_full_symbolic_structural_beats_nonzero_restricted_controls"
        )
    if reproduction is not None and reproduction.get("passed") is True:
        for claim in reproduction.get("supported_claims", []):
            if claim not in supported_claims:
                supported_claims.append(claim)
    if multiseed is not None and multiseed.get("passed") is True:
        for claim in multiseed.get("supported_claims", []):
            if claim not in supported_claims:
                supported_claims.append(claim)
    if cross_model is not None and cross_model.get("passed") is True:
        for claim in cross_model.get("supported_claims", []):
            if claim not in supported_claims:
                supported_claims.append(claim)
    execution_row_count = int(execution.get("metrics", {}).get("row_count") or 0)
    larger_holdout_complete = execution_row_count >= 32
    next_required_evidence = [
        "multi_seed_or_cross_model_reproduction_after_nonsealed_gates",
    ]
    if multiseed is not None and multiseed.get("passed") is True:
        next_required_evidence = [
            item
            for item in next_required_evidence
            if item != "multi_seed_or_cross_model_reproduction_after_nonsealed_gates"
        ]
        next_required_evidence.append("cross_model_reproduction_after_same_model_seed_gate")
    if cross_model is not None and cross_model.get("passed") is True:
        next_required_evidence = [
            item
            for item in next_required_evidence
            if item != "cross_model_reproduction_after_same_model_seed_gate"
        ]
    if not larger_holdout_complete:
        next_required_evidence.insert(0, "larger_nonsealed_public_repo_diverse_patch_holdout")
    if not (control_delta is not None and all(control_checks.values())):
        next_required_evidence.insert(1, "candidate_patch_baselines_with_nonzero_controls")
    blocked_actions = [
        "do_not_claim_freeform_patch_generation",
        "do_not_claim_open_ended_debugging_generalization",
        "do_not_claim_epoch_making_architecture_from_phase2ar_alone",
    ]
    unsupported_claims = [
        "freeform_patch_generation",
        "open_ended_debugging_generalization",
        "production_autonomy",
        "sealed_cross_model_transfer",
        "epoch_making_architecture",
    ]
    if reproduction is not None and reproduction.get("passed") is True:
        unsupported_claims.extend(reproduction.get("unsupported_claims", []))
        blocked_actions.extend(reproduction.get("blocked_actions", []))
    elif reproduction is not None:
        unsupported_claims.extend(
            [
                "phase2ar_cross_package_reproduction_supported",
                "multi_seed_reproduction_3plus",
                "cross_model_reproduction",
            ]
        )
        blocked_actions.extend(
            [
                "do_not_claim_cross_package_reproduction_from_failed_run",
                "do_not_claim_multi_seed_or_cross_model_reproduction",
            ]
        )
    if multiseed is not None and multiseed.get("passed") is True:
        unsupported_claims.extend(multiseed.get("unsupported_claims", []))
        blocked_actions.extend(multiseed.get("blocked_actions", []))
        unsupported_claims = [
            claim
            for claim in unsupported_claims
            if claim != "multi_seed_reproduction_3plus"
        ]
        blocked_actions = [
            action
            for action in blocked_actions
            if action != "do_not_claim_robust_multi_seed_reproduction_until_3plus_seeds"
        ]
    elif multiseed is not None:
        unsupported_claims.append("phase2ar_three_seed_same_model_reproduction_supported")
        blocked_actions.append("do_not_claim_multi_seed_reproduction_from_failed_aggregate")
    if cross_model is not None and cross_model.get("passed") is True:
        unsupported_claims.extend(cross_model.get("unsupported_claims", []))
        blocked_actions.extend(cross_model.get("blocked_actions", []))
        unsupported_claims = [
            claim for claim in unsupported_claims if claim != "cross_model_reproduction"
        ]
        blocked_actions = [
            action
            for action in blocked_actions
            if action != "do_not_claim_cross_model_reproduction_from_same_model_seed_runs"
        ]
    elif cross_model is not None:
        unsupported_claims.append("phase2ar_cross_model_reproduction_supported")
        blocked_actions.append("do_not_claim_cross_model_reproduction_from_failed_run")
    return {
        "artifact_family": "phase2ar_evidence_sufficiency_report",
        "passed": passed,
        "claim_boundary": (
            "Phase2AR supports nonsealed public-repo bounded symbolic structural "
            "patch proposal over text-membership and AST-attribute restoration controls; "
            "it does not support freeform patch generation, production autonomy, or "
            "open-ended debugging generalization."
        ),
        "checks": checks,
        "control_delta_checks": control_checks,
        "supported_claims": supported_claims if passed else [],
        "unsupported_claims": sorted(set(unsupported_claims)),
        "metrics": {
            "rows_by_split": data_health.get("metrics", {}).get("rows_by_split"),
            "data_patch_kinds_by_split": data_health.get("metrics", {}).get(
                "patch_kinds_by_split"
            ),
            "execution_row_count": execution.get("metrics", {}).get("row_count"),
            "execution_success_rate": execution.get("metrics", {}).get("success_rate"),
            "execution_patch_kind_counts": execution.get("metrics", {}).get(
                "patch_kind_counts"
            ),
            "control_delta": control_delta.get("metrics") if control_delta else None,
            "reproduction": reproduction.get("metrics") if reproduction else None,
            "multiseed_reproduction": multiseed.get("metrics") if multiseed else None,
            "cross_model_reproduction": cross_model.get("metrics") if cross_model else None,
        },
        "next_required_evidence": next_required_evidence,
        "blocked_actions": sorted(set(blocked_actions)),
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "execution_audit_json": str(Path(execution_audit_json)),
            "control_delta_json": str(Path(control_delta_json)) if control_delta_json else None,
            "reproduction_audit_json": str(Path(reproduction_audit_json))
            if reproduction_audit_json
            else None,
            "multiseed_reproduction_json": str(Path(multiseed_reproduction_json))
            if multiseed_reproduction_json
            else None,
            "cross_model_reproduction_json": str(Path(cross_model_reproduction_json))
            if cross_model_reproduction_json
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AR evidence sufficiency report.")
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--execution-audit-json", required=True)
    parser.add_argument("--control-delta-json")
    parser.add_argument("--reproduction-audit-json")
    parser.add_argument("--multiseed-reproduction-json")
    parser.add_argument("--cross-model-reproduction-json")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2ar_evidence_sufficiency_report(
        data_health_json=args.data_health_json,
        execution_audit_json=args.execution_audit_json,
        control_delta_json=args.control_delta_json,
        reproduction_audit_json=args.reproduction_audit_json,
        multiseed_reproduction_json=args.multiseed_reproduction_json,
        cross_model_reproduction_json=args.cross_model_reproduction_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
