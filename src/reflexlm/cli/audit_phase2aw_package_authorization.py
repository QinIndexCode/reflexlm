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


def audit_phase2aw_package_authorization(
    *,
    split_clean_manifest_json: str | Path,
    train_data_health_json: str | Path,
    val_data_health_json: str | Path,
    holdout_data_health_json: str | Path,
    runtime_execution_gate_json: str | Path,
    verified_candidate_pool_gate_json: str | Path,
    evidence_sufficiency_json: str | Path,
) -> dict[str, Any]:
    split_manifest = _read_json(split_clean_manifest_json)
    train_health = _read_json(train_data_health_json)
    val_health = _read_json(val_data_health_json)
    holdout_health = _read_json(holdout_data_health_json)
    runtime_gate = _read_json(runtime_execution_gate_json)
    verified_gate = _read_json(verified_candidate_pool_gate_json)
    sufficiency = _read_json(evidence_sufficiency_json)

    runtime_metrics = _metrics(runtime_gate)
    sufficiency_metrics = _metrics(sufficiency)
    checks = {
        "split_clean_manifest_passed": split_manifest.get("passed") is True,
        "split_clean_artifacts_rewritten": split_manifest.get("artifact_paths_rewritten")
        is True,
        "source_artifact_split_clean_by_construction": split_manifest.get(
            "source_artifact_split_clean_by_construction"
        )
        is True,
        "train_data_health_passed": train_health.get("passed") is True,
        "val_data_health_passed": val_health.get("passed") is True,
        "holdout_data_health_passed": holdout_health.get("passed") is True,
        "runtime_execution_gate_passed": runtime_gate.get("passed") is True,
        "runtime_full_success_gate_passed": runtime_metrics.get("full_success_rate", 0.0)
        >= 0.85,
        "runtime_control_nonzero_not_ceiling": 0.2
        <= runtime_metrics.get("control_success_rate", 0.0)
        <= 0.75,
        "runtime_delta_gate_passed": runtime_metrics.get(
            "full_minus_control_success_rate", 0.0
        )
        >= 0.15,
        "verified_candidate_pool_gate_passed": verified_gate.get("passed") is True,
        "verified_candidate_pool_ready": verified_gate.get(
            "ready_for_phase2aw_package_or_successor_training"
        )
        is True,
        "evidence_sufficiency_passed": sufficiency.get("passed") is True,
        "evidence_scope_bounded_nonsealed": sufficiency.get("claim_scope")
        == "phase2av_bounded_nonsealed_descriptor_runtime_candidate_selection",
        "multiseed_recorded": int(sufficiency_metrics.get("multiseed_unique_seed_count") or 0)
        >= 2,
        "cross_model_recorded": int(sufficiency_metrics.get("cross_model_model_count") or 0)
        >= 2,
        "package_not_already_claimed_by_evidence_report": "phase2av_package_ready"
        in set(sufficiency.get("unsupported_claims") or []),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2aw_package_authorization_gate",
        "passed": passed,
        "ready_for_package_build": passed,
        "ready_for_sealed_eval": False,
        "claim_boundary": (
            "Phase2AW package authorization only permits constructing a bounded "
            "native package after non-sealed split-clean runtime execution, "
            "multiseed, and cross-model evidence pass. It does not authorize "
            "sealed evaluation, freeform patch generation, production autonomy, "
            "open-ended debugging generalization, or epoch-making claims."
        ),
        "checks": checks,
        "metrics": {
            "full_success_rate": runtime_metrics.get("full_success_rate"),
            "control_success_rate": runtime_metrics.get("control_success_rate"),
            "full_minus_control_success_rate": runtime_metrics.get(
                "full_minus_control_success_rate"
            ),
            "multiseed_unique_seed_count": sufficiency_metrics.get(
                "multiseed_unique_seed_count"
            ),
            "cross_model_model_count": sufficiency_metrics.get("cross_model_model_count"),
        },
        "supported_claims": [
            "phase2aw_package_build_authorized_for_bounded_nonsealed_descriptor_runtime"
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
            "do_not_build_phase2aw_package",
            "do_not_run_sealed_eval",
            "fix_phase2aw_authorization_inputs",
        ],
        "post_authorization_blocked_actions": [
            "do_not_run_sealed_eval_until_postpackage_gate_passes",
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_epoch_making_architecture",
        ],
        "inputs": {
            "split_clean_manifest_json": str(Path(split_clean_manifest_json)),
            "train_data_health_json": str(Path(train_data_health_json)),
            "val_data_health_json": str(Path(val_data_health_json)),
            "holdout_data_health_json": str(Path(holdout_data_health_json)),
            "runtime_execution_gate_json": str(Path(runtime_execution_gate_json)),
            "verified_candidate_pool_gate_json": str(Path(verified_candidate_pool_gate_json)),
            "evidence_sufficiency_json": str(Path(evidence_sufficiency_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AW package authorization.")
    parser.add_argument("--split-clean-manifest-json", required=True)
    parser.add_argument("--train-data-health-json", required=True)
    parser.add_argument("--val-data-health-json", required=True)
    parser.add_argument("--holdout-data-health-json", required=True)
    parser.add_argument("--runtime-execution-gate-json", required=True)
    parser.add_argument("--verified-candidate-pool-gate-json", required=True)
    parser.add_argument("--evidence-sufficiency-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2aw_package_authorization(
        split_clean_manifest_json=args.split_clean_manifest_json,
        train_data_health_json=args.train_data_health_json,
        val_data_health_json=args.val_data_health_json,
        holdout_data_health_json=args.holdout_data_health_json,
        runtime_execution_gate_json=args.runtime_execution_gate_json,
        verified_candidate_pool_gate_json=args.verified_candidate_pool_gate_json,
        evidence_sufficiency_json=args.evidence_sufficiency_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
