from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path(
    "artifacts/reports/phase2j_semantic_command_identity/phase2j_pretrain_gate.json"
)
DEFAULT_PREREGISTRATION = Path(
    "artifacts/reports/phase2j_semantic_command_identity/phase2j_preregistration_check.json"
)
DEFAULT_READINESS = Path(
    "artifacts/reports/phase2j_semantic_command_identity/phase2j_implementation_readiness.json"
)
DEFAULT_DATA_HEALTH = Path(
    "artifacts/reports/phase2j_semantic_command_identity/phase2j_data_health_audit.json"
)
DEFAULT_HEAD_MANIFEST = Path(
    "artifacts/reports/phase2j_semantic_command_identity/phase2j_head_dataset_manifest.json"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload_path = Path(path)
    if not payload_path.exists():
        return {}
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _bool_nested(payload: dict[str, Any], *keys: str) -> bool:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return False
        value = value.get(key)
    return value is True


def build_phase2j_pretrain_gate(
    *,
    preregistration_json: str | Path,
    readiness_json: str | Path,
    data_health_json: str | Path,
    head_manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    preregistration = _load_json(preregistration_json)
    readiness = _load_json(readiness_json)
    data_health = _load_json(data_health_json)
    head_manifest = _load_json(head_manifest_json) if head_manifest_json else {}
    synapse_reference_checks = [
        value
        for key, value in (data_health.get("checks") or {}).items()
        if str(key).endswith("_synapse_reference_present")
    ]
    debug_action_stage_checks = [
        value
        for key, value in (data_health.get("checks") or {}).items()
        if "_debug_action_stage_" in str(key)
    ]

    checks = {
        "phase2j_preregistration_passed": preregistration.get("passed") is True,
        "phase2j_readiness_passed": readiness.get("passed") is True,
        "phase2j_readiness_allows_data_generation_only": (
            readiness.get("ready_for_data_generation") is True
            and readiness.get("ready_for_training") is False
        ),
        "phase2j_data_health_passed": data_health.get("passed") is True,
        "phase2j_effective_split_hashes_present": _bool_nested(
            data_health, "checks", "phase2j_effective_split_hashes_present"
        ),
        "phase2j_train_val_target_overlap_guard_passed": _bool_nested(
            data_health, "checks", "phase2j_train_val_target_overlap"
        ),
        "phase2j_train_val_intent_coverage_passed": _bool_nested(
            data_health, "checks", "phase2j_train_val_command_intent_coverage"
        ),
        "phase2j_source_overlap_hard_baseline_guard_passed": (
            True
            if "phase2j_source_overlap_hard_val_baseline_below_threshold"
            not in data_health.get("checks", {})
            else _bool_nested(
                data_health,
                "checks",
                "phase2j_source_overlap_hard_val_baseline_below_threshold",
            )
        ),
        "phase2j_source_overlap_hard_identity_signal_present": (
            True
            if "phase2j_source_overlap_hard_identity_signal_present"
            not in data_health.get("checks", {})
            else _bool_nested(
                data_health,
                "checks",
                "phase2j_source_overlap_hard_identity_signal_present",
            )
        ),
        "phase2j_synapse_reference_present": (
            all(value is True for value in synapse_reference_checks)
            if synapse_reference_checks
            else True
        ),
        "phase2j_debug_action_stage_present": (
            all(value is True for value in debug_action_stage_checks)
            if debug_action_stage_checks
            else True
        ),
        "sealed_not_used_for_training_or_tuning": (
            data_health.get("sealed_usage", {}).get("sealed_splits_used_for_training") is False
            and data_health.get("sealed_usage", {}).get("sealed_splits_used_for_tuning")
            is False
        ),
        "head_manifest_leakage_passed": (
            True
            if not head_manifest
            else head_manifest.get("leakage_audit", {}).get("passed") is True
        ),
        "head_manifest_coverage_passed": (
            True
            if not head_manifest
            else head_manifest.get("coverage_audit", {}).get("passed") is True
        ),
    }
    blocked_actions: list[str] = []
    if not checks["phase2j_preregistration_passed"]:
        blocked_actions.append("do_not_train_until_phase2j_preregistration_passes")
    if not checks["phase2j_readiness_passed"]:
        blocked_actions.append("do_not_train_until_phase2j_readiness_passes")
    if not checks["phase2j_readiness_allows_data_generation_only"]:
        blocked_actions.append("do_not_train_from_a_training_oriented_readiness_artifact")
    if not checks["phase2j_data_health_passed"]:
        blocked_actions.append("do_not_train_until_phase2j_data_health_passes")
    if not checks["phase2j_effective_split_hashes_present"]:
        blocked_actions.append("do_not_train_without_effective_split_hashes")
    if not checks["phase2j_train_val_target_overlap_guard_passed"]:
        blocked_actions.append("do_not_train_with_train_val_target_command_overlap")
    if not checks["phase2j_train_val_intent_coverage_passed"]:
        blocked_actions.append("do_not_train_without_train_val_intent_coverage")
    if not checks["phase2j_source_overlap_hard_baseline_guard_passed"]:
        blocked_actions.append("do_not_train_when_source_overlap_baseline_solves_phase2j_hard_val")
    if not checks["phase2j_source_overlap_hard_identity_signal_present"]:
        blocked_actions.append("do_not_train_without_phase2j_source_overlap_hard_identity_signal")
    if not checks["phase2j_synapse_reference_present"]:
        blocked_actions.append("do_not_train_without_phase2j_synapse_reference")
    if not checks["phase2j_debug_action_stage_present"]:
        blocked_actions.append("do_not_train_without_phase2j_debug_action_stage")
    if not checks["sealed_not_used_for_training_or_tuning"]:
        blocked_actions.append("do_not_train_with_sealed_inputs_or_tuning_feedback")
    if not checks["head_manifest_leakage_passed"]:
        blocked_actions.append("do_not_train_until_head_manifest_leakage_passes")
    if not checks["head_manifest_coverage_passed"]:
        blocked_actions.append("do_not_train_until_head_manifest_coverage_passes")

    ready_for_smoke_training = all(checks.values())
    return {
        "audit_family": "phase2j_pretrain_gate",
        "passed": ready_for_smoke_training,
        "ready_for_smoke_training": ready_for_smoke_training,
        "ready_for_full_training": False,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "allowed_next_action": (
            "run_nonsealed_phase2j_smoke_training_only"
            if ready_for_smoke_training
            else "fix_phase2j_pretrain_gate_failures_before_training"
        ),
        "blocked_actions": blocked_actions,
        "checks": checks,
        "inputs": {
            "preregistration_json": str(Path(preregistration_json)),
            "readiness_json": str(Path(readiness_json)),
            "data_health_json": str(Path(data_health_json)),
            "head_manifest_json": str(Path(head_manifest_json)) if head_manifest_json else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate Phase2J before any non-sealed smoke training is allowed."
    )
    parser.add_argument("--preregistration-json", default=str(DEFAULT_PREREGISTRATION))
    parser.add_argument("--readiness-json", default=str(DEFAULT_READINESS))
    parser.add_argument("--data-health-json", default=str(DEFAULT_DATA_HEALTH))
    parser.add_argument("--head-manifest-json", default=str(DEFAULT_HEAD_MANIFEST))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = build_phase2j_pretrain_gate(
        preregistration_json=args.preregistration_json,
        readiness_json=args.readiness_json,
        data_health_json=args.data_health_json,
        head_manifest_json=args.head_manifest_json,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
