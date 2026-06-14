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


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric(report: dict[str, Any], name: str) -> Any:
    return _dict(report.get("metrics")).get(name)


def _unsupported_claims_blocked(report: dict[str, Any]) -> bool:
    unsupported = {str(claim) for claim in report.get("unsupported_claims") or []}
    required = {
        "sealed_cross_model_transfer",
        "freeform_patch_generation",
        "open_ended_debugging_generalization",
        "production_autonomy",
        "epoch_making_architecture",
    }
    return required.issubset(unsupported)


def _phase2av_pretrain_ready(pretrain_gate: dict[str, Any]) -> bool | None:
    for key in (
        "ready_for_phase2av_training",
        "ready_for_phase2av_smoke_training",
        "ready_for_phase2av_full_training",
    ):
        value = pretrain_gate.get(key)
        if isinstance(value, bool):
            return value
    return None


def build_phase2av_evidence_sufficiency_report(
    *,
    pretrain_gate_json: str | Path,
    val_postflight_json: str | Path,
    holdout_postflight_json: str | Path,
    full_readiness_json: str | Path,
    prior_failure_audit_json: str | Path | None = None,
    runtime_execution_gate_json: str | Path | None = None,
    multiseed_report_json: str | Path | None = None,
    cross_model_report_json: str | Path | None = None,
    min_holdout_command_slot_accuracy: float = 0.85,
    min_holdout_model_minus_source_overlap: float = 0.15,
) -> dict[str, Any]:
    pretrain_gate = _read_json(pretrain_gate_json)
    val_postflight = _read_json(val_postflight_json)
    holdout_postflight = _read_json(holdout_postflight_json)
    full_readiness = _read_json(full_readiness_json)
    prior_failure_audit = (
        _read_json(prior_failure_audit_json) if prior_failure_audit_json else {}
    )
    runtime_execution_gate = (
        _read_json(runtime_execution_gate_json) if runtime_execution_gate_json else {}
    )
    multiseed_report = _read_json(multiseed_report_json) if multiseed_report_json else {}
    cross_model_report = (
        _read_json(cross_model_report_json) if cross_model_report_json else {}
    )

    holdout_command = _metric(holdout_postflight, "command_slot_accuracy")
    holdout_delta = _metric(holdout_postflight, "model_minus_source_overlap_accuracy")
    prior_command = _dict(prior_failure_audit.get("metrics")).get("command_slot_accuracy")
    prior_failure_modes = {str(mode) for mode in prior_failure_audit.get("failure_modes") or []}
    prior_failed_below_gate = (
        not prior_failure_audit
        or "command_slot_identity_transfer_below_holdout_gate" in prior_failure_modes
        or "command_slot_accuracy_below_gate" in prior_failure_modes
        or (
            prior_failure_audit.get("passed") is False
            and isinstance(prior_command, (int, float))
            and float(prior_command) < min_holdout_command_slot_accuracy
        )
    )
    checks = {
        "pretrain_gate_passed": pretrain_gate.get("passed") is True,
        "val_postflight_passed": val_postflight.get("passed") is True,
        "holdout_postflight_passed": holdout_postflight.get("passed") is True,
        "holdout_command_slot_gate": isinstance(holdout_command, (int, float))
        and float(holdout_command) >= min_holdout_command_slot_accuracy,
        "holdout_model_delta_gate": isinstance(holdout_delta, (int, float))
        and float(holdout_delta) >= min_holdout_model_minus_source_overlap,
        "full_readiness_passed": full_readiness.get("passed") is True,
        "full_readiness_does_not_authorize_package": full_readiness.get("ready_for_package")
        is False
        and full_readiness.get("ready_for_sealed_eval") is False,
        "boundary_claims_blocked": _unsupported_claims_blocked(full_readiness),
        "prior_failure_recorded_or_not_required": prior_failed_below_gate,
        "runtime_execution_gate_optional_or_passed": not runtime_execution_gate
        or runtime_execution_gate.get("passed") is True,
        "multiseed_report_optional_or_passed": not multiseed_report
        or multiseed_report.get("passed") is True,
        "cross_model_report_optional_or_passed": not cross_model_report
        or cross_model_report.get("passed") is True,
    }
    passed = all(checks.values())
    next_required_evidence = []
    if not multiseed_report or multiseed_report.get("passed") is not True:
        next_required_evidence.append(
            "independent_multiseed_phase2av_v15_or_successor_reproduction"
        )
    if not cross_model_report or cross_model_report.get("passed") is not True:
        next_required_evidence.append(
            "cross_model_nonsealed_descriptor_runtime_reproduction"
        )
    if not runtime_execution_gate or runtime_execution_gate.get("passed") is not True:
        next_required_evidence.append(
            "real_runtime_patch_execution_delta_beyond_descriptor_selection"
        )
    next_required_evidence.append("package_gate_only_after_full_nonsealed_runtime_delta")
    return {
        "artifact_family": "phase2av_evidence_sufficiency_report",
        "passed": passed,
        "claim_scope": (
            "phase2av_bounded_nonsealed_descriptor_runtime_candidate_selection"
            if passed
            else "phase2av_evidence_incomplete"
        ),
        "claim_boundary": (
            "Phase2AV v15 supports bounded descriptor-runtime command-slot and "
            "patch-descriptor selection on a non-sealed repo-operation-disjoint "
            "benchmark when command identity prior is enabled. It does not prove "
            "freeform patch generation, sealed transfer, production autonomy, "
            "open-ended debugging generalization, or an epoch-making architecture."
        ),
        "checks": checks,
        "metrics": {
            "pretrain_ready": _phase2av_pretrain_ready(pretrain_gate),
            "val_command_slot_accuracy": _metric(val_postflight, "command_slot_accuracy"),
            "val_model_minus_source_overlap_accuracy": _metric(
                val_postflight, "model_minus_source_overlap_accuracy"
            ),
            "val_patch_operation_accuracy": _metric(
                val_postflight, "patch_operation_accuracy"
            ),
            "val_patch_template_slot_accuracy": _metric(
                val_postflight, "patch_template_slot_accuracy"
            ),
            "holdout_command_slot_accuracy": holdout_command,
            "holdout_model_minus_source_overlap_accuracy": holdout_delta,
            "holdout_patch_operation_accuracy": _metric(
                holdout_postflight, "patch_operation_accuracy"
            ),
            "holdout_patch_template_slot_accuracy": _metric(
                holdout_postflight, "patch_template_slot_accuracy"
            ),
            "full_readiness_effective_train_examples": _dict(
                full_readiness.get("metrics")
            ).get("effective_train_examples"),
            "full_readiness_split_counts": _dict(full_readiness.get("metrics")).get(
                "split_counts"
            ),
            "prior_failure_modes": prior_failure_audit.get("failure_modes", []),
            "runtime_execution_full_success_rate": _metric(
                runtime_execution_gate, "full_success_rate"
            )
            if runtime_execution_gate
            else None,
            "runtime_execution_control_success_rate": _metric(
                runtime_execution_gate, "control_success_rate"
            )
            if runtime_execution_gate
            else None,
            "runtime_execution_full_minus_control_success_rate": _metric(
                runtime_execution_gate, "full_minus_control_success_rate"
            )
            if runtime_execution_gate
            else None,
            "multiseed_unique_seed_count": _metric(multiseed_report, "unique_seed_count")
            if multiseed_report
            else None,
            "multiseed_holdout_command_slot_accuracy_min": _metric(
                multiseed_report, "holdout_command_slot_accuracy_min"
            )
            if multiseed_report
            else None,
            "cross_model_model_count": _metric(cross_model_report, "model_count")
            if cross_model_report
            else None,
            "cross_model_holdout_command_slot_accuracy_min": _metric(
                cross_model_report, "holdout_command_slot_accuracy_min"
            )
            if cross_model_report
            else None,
        },
        "supported_claims": [
            "phase2av_nonsealed_repo_operation_disjoint_descriptor_runtime_selection_supported",
            "phase2av_command_identity_prior_ablation_resolves_prior_holdout_command_slot_gap",
            *(
                [
                    "phase2av_bounded_descriptor_selected_symbolic_execution_delta_supported"
                ]
                if runtime_execution_gate.get("passed") is True
                else []
            ),
            *(
                ["phase2av_nonsealed_multiseed_descriptor_runtime_reproduced"]
                if multiseed_report.get("passed") is True
                else []
            ),
            *(
                ["phase2av_nonsealed_cross_model_descriptor_runtime_reproduced"]
                if cross_model_report.get("passed") is True
                else []
            ),
        ]
        if passed
        else [],
        "unsupported_claims": [
            "phase2av_package_ready",
            "sealed_cross_model_transfer",
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": [
            "do_not_package_phase2av_from_this_report",
            "do_not_run_sealed_eval_from_this_report",
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_open_ended_debugging_generalization",
            "do_not_claim_epoch_making_architecture",
        ],
        "next_required_evidence": next_required_evidence,
        "inputs": {
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "val_postflight_json": str(Path(val_postflight_json)),
            "holdout_postflight_json": str(Path(holdout_postflight_json)),
            "full_readiness_json": str(Path(full_readiness_json)),
            "prior_failure_audit_json": str(Path(prior_failure_audit_json))
            if prior_failure_audit_json
            else None,
            "runtime_execution_gate_json": str(Path(runtime_execution_gate_json))
            if runtime_execution_gate_json
            else None,
            "multiseed_report_json": str(Path(multiseed_report_json))
            if multiseed_report_json
            else None,
            "cross_model_report_json": str(Path(cross_model_report_json))
            if cross_model_report_json
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AV bounded evidence sufficiency report."
    )
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--val-postflight-json", required=True)
    parser.add_argument("--holdout-postflight-json", required=True)
    parser.add_argument("--full-readiness-json", required=True)
    parser.add_argument("--prior-failure-audit-json")
    parser.add_argument("--runtime-execution-gate-json")
    parser.add_argument("--multiseed-report-json")
    parser.add_argument("--cross-model-report-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2av_evidence_sufficiency_report(
        pretrain_gate_json=args.pretrain_gate_json,
        val_postflight_json=args.val_postflight_json,
        holdout_postflight_json=args.holdout_postflight_json,
        full_readiness_json=args.full_readiness_json,
        prior_failure_audit_json=args.prior_failure_audit_json,
        runtime_execution_gate_json=args.runtime_execution_gate_json,
        multiseed_report_json=args.multiseed_report_json,
        cross_model_report_json=args.cross_model_report_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
