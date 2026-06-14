from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2at_learned_patch_generation_package_gate import (
    REQUIRED_OPEN_REPAIR_CAPABILITIES,
)
from reflexlm.llm.native_nervous_package import PACKAGE_MANIFEST_NAME


REQUIRED_SCHEMA_VERSION = "phase2au.policy_required_patch_descriptor.v1"


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _manifest_path(package_path: str | Path) -> Path:
    path = Path(package_path)
    return path if path.is_file() else path / PACKAGE_MANIFEST_NAME


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _adapter_path_matches(manifest: dict[str, Any], summary: dict[str, Any]) -> bool:
    summary_adapter = str(summary.get("adapter_output_dir") or "")
    manifest_adapter = str(manifest.get("native_head_path") or "")
    if not summary_adapter or not manifest_adapter:
        return False
    return Path(summary_adapter).resolve() == Path(manifest_adapter).resolve()


def audit_phase2au_package_gate(
    *,
    package_path: str | Path,
    training_summary_json: str | Path,
    smoke_postflight_json: str | Path,
    holdout_postflight_json: str | Path,
    required_schema_version: str = REQUIRED_SCHEMA_VERSION,
) -> dict[str, Any]:
    manifest_path = _manifest_path(package_path)
    manifest = _read_json(manifest_path)
    summary = _read_json(training_summary_json)
    smoke = _read_json(smoke_postflight_json)
    holdout = _read_json(holdout_postflight_json)
    caps = _dict(manifest.get("open_repair_capabilities"))
    contract = _dict(summary.get("open_repair_training_contract"))
    missing_caps = [
        name for name in REQUIRED_OPEN_REPAIR_CAPABILITIES if caps.get(name) is not True
    ]
    unsupported_claims = [
        "phase2au_runtime_delta_before_full_no_policy_execution",
        "freeform_patch_generation",
        "sealed_cross_model_transfer",
        "open_ended_debugging_generalization",
        "production_autonomy",
        "epoch_making_architecture",
    ]
    checks = {
        "package_manifest_present": bool(manifest),
        "package_family_native_nervous": manifest.get("package_family")
        == "phase2d_native_nervous_package",
        "policy_label_is_phase2au": "phase2au" in str(manifest.get("policy_label") or ""),
        "adapter_path_matches_training_summary": _adapter_path_matches(manifest, summary),
        "base_model_matches_training_summary": str(manifest.get("base_model_name") or "")
        == str(summary.get("base_model_name") or ""),
        "all_open_repair_capabilities_declared": not missing_caps,
        "patch_strategy_is_learned_bounded_candidate": manifest.get("patch_proposal_strategy")
        == "learned_bounded_candidate",
        "learned_patch_generation_enabled": manifest.get("learned_patch_generation_enabled")
        is True,
        "patch_candidate_schema_version_matches": manifest.get(
            "patch_candidate_schema_version"
        )
        == required_schema_version,
        "no_json_text_target": manifest.get("json_text_target") is False
        and summary.get("no_json_motor_target") is True
        and contract.get("json_text_target") is False
        and contract.get("freeform_patch_text_target") is False,
        "native_head_calls_enabled": manifest.get("native_head_calls_enabled") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0
        and contract.get("low_level_qwen_calls_target") == 0,
        "sealed_feedback_absent": contract.get("sealed_feedback_used") is False,
        "training_summary_records_policy_required_targets": contract.get(
            "learned_patch_candidate_targets"
        )
        is True
        and contract.get("recorded_patch_artifact_as_generation_target") is False
        and contract.get("symbolic_generator_as_generation_target") is False,
        "smoke_postflight_passed": smoke.get("passed") is True,
        "holdout_postflight_passed": holdout.get("passed") is True,
        "holdout_still_not_runtime_delta_claim": holdout.get("claim_boundary")
        == "phase2au_holdout_delta_supported_for_capacity_smoke_not_claim_upgrade",
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2au_policy_required_package_gate",
        "passed": passed,
        "ready_for_phase2au_runtime_delta_eval": passed,
        "claim_boundary": (
            "Phase2AU package gate only verifies that a capacity-smoke native "
            "package is internally consistent and may enter bounded full-vs-no-policy "
            "runtime-control evaluation. It is not runtime-delta evidence."
        ),
        "checks": checks,
        "metrics": {
            "missing_open_repair_capabilities": missing_caps,
            "policy_label": manifest.get("policy_label"),
            "base_model_name": manifest.get("base_model_name"),
            "native_head_path": manifest.get("native_head_path"),
            "patch_candidate_schema_version": manifest.get(
                "patch_candidate_schema_version"
            ),
            "smoke_model_minus_source_overlap": _dict(smoke.get("metrics")).get(
                "model_minus_source_overlap"
            ),
            "holdout_model_minus_source_overlap": _dict(holdout.get("metrics")).get(
                "model_minus_source_overlap"
            ),
        },
        "supported_claims": [
            "phase2au_capacity_package_ready_for_bounded_runtime_control_eval"
        ]
        if passed
        else [],
        "unsupported_claims": unsupported_claims,
        "blocked_actions": []
        if passed
        else [
            "do_not_run_phase2au_runtime_delta_eval_with_this_package",
            "do_not_claim_phase2au_package_runtime_delta",
            "do_not_run_sealed_phase2au",
        ],
        "inputs": {
            "package_path": str(Path(package_path)),
            "manifest_path": str(manifest_path),
            "training_summary_json": str(Path(training_summary_json)),
            "smoke_postflight_json": str(Path(smoke_postflight_json)),
            "holdout_postflight_json": str(Path(holdout_postflight_json)),
        },
        "thresholds": {"required_schema_version": required_schema_version},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AU package gate.")
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--smoke-postflight-json", required=True)
    parser.add_argument("--holdout-postflight-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--required-schema-version",
        default=REQUIRED_SCHEMA_VERSION,
    )
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2au_package_gate(
        package_path=args.package_path,
        training_summary_json=args.training_summary_json,
        smoke_postflight_json=args.smoke_postflight_json,
        holdout_postflight_json=args.holdout_postflight_json,
        required_schema_version=args.required_schema_version,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
