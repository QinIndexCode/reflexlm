from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.llm.native_cortex import OPEN_REPAIR_CAPABILITY_NAMES
from reflexlm.llm.native_nervous_package import PACKAGE_MANIFEST_NAME


REQUIRED_OPEN_REPAIR_CAPABILITIES = (
    "patch_proposal_head",
    "test_selection_head",
    "rollback_safety_head",
    "stop_condition_head",
    "bounded_edit_scope_policy",
    "progress_monitor_receptors",
    "verification_state_receptors",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _manifest_path(package_path: str | Path) -> Path:
    path = Path(package_path)
    return path if path.is_file() else path / PACKAGE_MANIFEST_NAME


def audit_phase2at_learned_patch_generation_package_gate(
    *,
    package_path: str | Path,
    training_summary_json: str | Path | None = None,
    required_schema_version: str = "phase2at.learned_bounded_patch_candidate.v1",
) -> dict[str, Any]:
    manifest_path = _manifest_path(package_path)
    manifest = _read_json(manifest_path)
    summary = _read_json(training_summary_json) if training_summary_json else {}
    caps = manifest.get("open_repair_capabilities")
    if not isinstance(caps, dict):
        caps = {}
    missing_caps = [
        name for name in REQUIRED_OPEN_REPAIR_CAPABILITIES if caps.get(name) is not True
    ]
    unknown_caps = sorted(set(caps) - set(OPEN_REPAIR_CAPABILITY_NAMES))
    training_mechanism = (
        summary.get("open_repair_training_contract")
        if isinstance(summary.get("open_repair_training_contract"), dict)
        else {}
    )
    checks = {
        "package_family_native_nervous": manifest.get("package_family")
        == "phase2d_native_nervous_package",
        "all_open_repair_capabilities_declared": not missing_caps,
        "no_unknown_open_repair_capabilities": not unknown_caps,
        "patch_strategy_is_learned_bounded_candidate": manifest.get("patch_proposal_strategy")
        == "learned_bounded_candidate",
        "learned_patch_generation_enabled": manifest.get("learned_patch_generation_enabled")
        is True,
        "patch_candidate_schema_version_matches": manifest.get(
            "patch_candidate_schema_version"
        )
        == required_schema_version,
        "no_json_text_target": manifest.get("json_text_target") is False,
        "native_head_calls_enabled": manifest.get("native_head_calls_enabled") is True,
        "sealed_feedback_absent": training_mechanism.get("sealed_feedback_used") is not True,
        "training_summary_records_learned_patch_targets": (
            not training_summary_json
            or training_mechanism.get("learned_patch_candidate_targets") is True
        ),
        "training_summary_blocks_recorded_patch_relabel": (
            not training_summary_json
            or training_mechanism.get("recorded_patch_artifact_as_generation_target")
            is False
        ),
        "training_summary_blocks_symbolic_relabel": (
            not training_summary_json
            or training_mechanism.get("symbolic_generator_as_generation_target") is False
        ),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2at_learned_patch_generation_package_gate",
        "passed": passed,
        "claim_boundary": (
            "This gate only allows Phase2AT learned bounded patch generation "
            "experiments to proceed when the native package manifest and optional "
            "training summary explicitly distinguish learned patch candidate "
            "generation from recorded patches and symbolic runtime generators."
        ),
        "checks": checks,
        "metrics": {
            "missing_open_repair_capabilities": missing_caps,
            "unknown_open_repair_capabilities": unknown_caps,
            "patch_proposal_strategy": manifest.get("patch_proposal_strategy", "none"),
            "learned_patch_generation_enabled": bool(
                manifest.get("learned_patch_generation_enabled", False)
            ),
            "patch_candidate_schema_version": manifest.get("patch_candidate_schema_version"),
        },
        "supported_claims": [
            "phase2at_package_ready_for_learned_bounded_patch_generation_eval"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "learned_patch_generation",
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ]
        if not passed
        else [
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_train_or_eval_phase2at_as_learned_generation_with_old_package_schema",
            "do_not_relabel_recorded_or_symbolic_patch_outputs_as_learned_generation",
        ],
        "inputs": {
            "package_path": str(Path(package_path)),
            "manifest_path": str(manifest_path),
            "training_summary_json": str(Path(training_summary_json))
            if training_summary_json
            else None,
        },
        "thresholds": {"required_schema_version": required_schema_version},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AT learned bounded patch generation package gate."
    )
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--training-summary-json")
    parser.add_argument(
        "--required-schema-version",
        default="phase2at.learned_bounded_patch_candidate.v1",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = audit_phase2at_learned_patch_generation_package_gate(
        package_path=args.package_path,
        training_summary_json=args.training_summary_json,
        required_schema_version=args.required_schema_version,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
