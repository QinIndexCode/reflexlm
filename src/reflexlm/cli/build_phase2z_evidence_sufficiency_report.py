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


def _latest_val_metrics(training: dict[str, Any]) -> dict[str, Any]:
    history = training.get("history")
    if isinstance(history, list) and history:
        last = history[-1]
        if isinstance(last, dict) and isinstance(last.get("val_metrics"), dict):
            return last["val_metrics"]
    return {}


def build_phase2z_evidence_sufficiency_report(
    *,
    nonliteral_gap_audit_json: str | Path,
    data_health_json: str | Path,
    training_summary_json: str | Path,
    postflight_json: str | Path,
    execution_audit_json: str | Path,
    symbolic_patch_audit_json: str | Path | None = None,
) -> dict[str, Any]:
    nonliteral = _read_json(nonliteral_gap_audit_json)
    data_health = _read_json(data_health_json)
    training = _read_json(training_summary_json)
    postflight = _read_json(postflight_json)
    execution = _read_json(execution_audit_json)
    symbolic_patch = _read_json(symbolic_patch_audit_json) if symbolic_patch_audit_json else None
    val_metrics = _latest_val_metrics(training)
    training_components = {}
    history = training.get("history")
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict) and isinstance(latest.get("train_components"), dict):
            training_components = latest["train_components"]
    open_control_losses_present = all(
        float(training_components.get(name, 0.0)) > 0.0
        for name in [
            "patch_proposal",
            "test_selection",
            "rollback_safety",
            "bounded_edit_scope",
            "progress_monitor",
            "verification_state",
        ]
    )
    supported_claims = []
    unsupported_claims = [
        "model_generated_patch_diff",
        "open_ended_debugging_generalization",
        "production_autonomy",
        "sealed_cross_model_transfer",
        "epoch_making_architecture",
    ]
    checks = {
        "public_structural_nonliteral_split_passed": bool(nonliteral.get("passed")),
        "data_health_passed": bool(data_health.get("passed")),
        "postflight_passed": bool(postflight.get("passed")),
        "execution_audit_passed": bool(execution.get("passed")),
        "training_open_control_losses_present": open_control_losses_present,
        "val_command_slot_accuracy_perfect": float(
            val_metrics.get("command_slot_accuracy", 0.0)
        )
        >= 1.0,
        "runtime_control_success_rate_perfect": float(
            execution.get("metrics", {}).get("success_rate", 0.0)
        )
        >= 1.0,
        "recorded_patch_boundary_preserved": execution.get("claim_bearing_execution_evidence")
        is False,
    }
    symbolic_checks = {}
    if symbolic_patch is not None:
        symbolic_checks = {
            "symbolic_patch_audit_passed": bool(symbolic_patch.get("passed")),
            "symbolic_patch_is_claim_bearing": symbolic_patch.get(
                "claim_bearing_execution_evidence"
            )
            is True,
            "symbolic_patch_holdout24": symbolic_patch.get("evidence_level") == "holdout24",
            "symbolic_patch_success_rate_perfect": float(
                symbolic_patch.get("metrics", {}).get("success_rate", 0.0)
            )
            >= 1.0,
            "symbolic_patch_boundary_preserved": symbolic_patch.get("claim_boundary")
            == "bounded_runtime_symbolic_patch_proposal_only_not_open_ended_repair",
        }
    if all(checks.values()):
        supported_claims.extend(
            [
                "public_repo_structural_nonliteral_repair_runtime_control_supported",
                "native_heads_authorize_bounded_recorded_patch_execution_supported",
                "receptor_to_debug_cortex_two_step_runtime_flow_supported",
            ]
        )
    phase2z_sufficiency_passed = all(checks.values())
    phase2aq_sufficiency_passed = (
        phase2z_sufficiency_passed
        and bool(symbolic_patch is not None)
        and all(symbolic_checks.values())
    )
    if phase2aq_sufficiency_passed:
        supported_claims.append(
            "bounded_runtime_symbolic_text_membership_patch_proposal_holdout24_supported"
        )
    next_required_evidence = [
        "bounded_patch_proposal_motor_or_patch_slot_outputs_not_recorded_diff_replay",
        "candidate_patch_baselines_with_nonzero_controls",
        "multi_seed_or_cross_model_reproduction_after_nonsealed_gates",
    ]
    if phase2aq_sufficiency_passed:
        next_required_evidence = [
            "diverse_patch_proposal_benchmark_beyond_text_membership_assertions",
            "candidate_patch_baselines_with_nonzero_controls",
            "multi_seed_or_cross_model_reproduction_after_nonsealed_gates",
        ]
    return {
        "artifact_family": "phase2z_evidence_sufficiency_report",
        "passed": phase2aq_sufficiency_passed if symbolic_patch is not None else phase2z_sufficiency_passed,
        "claim_boundary": (
            "Phase2Z plus Phase2AQ supports public-repo structural nonliteral "
            "runtime control and bounded symbolic text-membership patch proposal, "
            "not freeform patch generation or open-ended autonomy."
            if phase2aq_sufficiency_passed
            else "Phase2Z supports public-repo structural nonliteral recorded-patch "
            "runtime control, not model-generated patch repair or open-ended autonomy."
        ),
        "checks": checks,
        "phase2aq_checks": symbolic_checks,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "metrics": {
            "structural_nonliteral_rows": nonliteral.get("metrics", {}).get(
                "structural_nonliteral_rows"
            ),
            "multifile_rows": nonliteral.get("metrics", {}).get("multifile_rows"),
            "val_command_slot_accuracy": val_metrics.get("command_slot_accuracy"),
            "source_overlap_val_accuracy": postflight.get("metrics", {}).get(
                "source_overlap_val_accuracy"
            ),
            "model_minus_source_overlap_accuracy": postflight.get("metrics", {}).get(
                "model_minus_source_overlap_accuracy"
            ),
            "runtime_control_success_rate": execution.get("metrics", {}).get("success_rate"),
            "runtime_control_success_count": execution.get("metrics", {}).get("success_count"),
            "runtime_control_row_count": execution.get("metrics", {}).get("row_count"),
            "symbolic_patch_evidence_level": symbolic_patch.get("evidence_level")
            if symbolic_patch
            else None,
            "symbolic_patch_success_rate": symbolic_patch.get("metrics", {}).get("success_rate")
            if symbolic_patch
            else None,
            "symbolic_patch_success_count": symbolic_patch.get("metrics", {}).get(
                "success_count"
            )
            if symbolic_patch
            else None,
            "symbolic_patch_row_count": symbolic_patch.get("metrics", {}).get("row_count")
            if symbolic_patch
            else None,
            "low_level_qwen_calls_target": training.get("low_level_qwen_calls_target"),
            "pairwise_enabled": training.get("use_pairwise_command_reranker"),
            "command_candidate_encoder": training.get("command_candidate_encoder"),
        },
        "next_required_evidence": next_required_evidence,
        "blocked_actions": [
            "do_not_claim_model_generated_patch_repair",
            "do_not_claim_open_ended_debugging_generalization",
            "do_not_claim_epoch_making_architecture_from_phase2z_alone",
        ],
        "inputs": {
            "nonliteral_gap_audit_json": str(Path(nonliteral_gap_audit_json)),
            "data_health_json": str(Path(data_health_json)),
            "training_summary_json": str(Path(training_summary_json)),
            "postflight_json": str(Path(postflight_json)),
            "execution_audit_json": str(Path(execution_audit_json)),
            "symbolic_patch_audit_json": str(Path(symbolic_patch_audit_json))
            if symbolic_patch_audit_json
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2Z evidence sufficiency report.")
    parser.add_argument("--nonliteral-gap-audit-json", required=True)
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--postflight-json", required=True)
    parser.add_argument("--execution-audit-json", required=True)
    parser.add_argument("--symbolic-patch-audit-json")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2z_evidence_sufficiency_report(
        nonliteral_gap_audit_json=args.nonliteral_gap_audit_json,
        data_health_json=args.data_health_json,
        training_summary_json=args.training_summary_json,
        postflight_json=args.postflight_json,
        execution_audit_json=args.execution_audit_json,
        symbolic_patch_audit_json=args.symbolic_patch_audit_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
