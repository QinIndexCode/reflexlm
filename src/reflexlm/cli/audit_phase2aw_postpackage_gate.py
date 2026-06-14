from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2at_learned_patch_generation_package_gate import (
    REQUIRED_OPEN_REPAIR_CAPABILITIES,
)
from reflexlm.llm.native_nervous_package import PACKAGE_MANIFEST_NAME


REQUIRED_POLICY_LABEL_FRAGMENT = "phase2aw"
REQUIRED_PATCH_SCHEMA_VERSION = "phase2at.learned_bounded_patch_candidate.v1"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _manifest_path(package_path: str | Path) -> Path:
    path = Path(package_path)
    return path if path.is_file() else path / PACKAGE_MANIFEST_NAME


def _path_exists_from_cwd(path_value: Any) -> bool:
    if not isinstance(path_value, str) or not path_value:
        return False
    return Path(path_value).exists()


def audit_phase2aw_postpackage_gate(
    *,
    package_path: str | Path,
    package_authorization_gate_json: str | Path,
    package_schema_gate_json: str | Path,
    evidence_sufficiency_json: str | Path,
) -> dict[str, Any]:
    manifest_path = _manifest_path(package_path)
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    authorization = _read_json(package_authorization_gate_json)
    schema_gate = _read_json(package_schema_gate_json)
    sufficiency = _read_json(evidence_sufficiency_json)
    caps = manifest.get("open_repair_capabilities")
    caps = caps if isinstance(caps, dict) else {}
    missing_caps = [
        name for name in REQUIRED_OPEN_REPAIR_CAPABILITIES if caps.get(name) is not True
    ]
    unsupported = set(sufficiency.get("unsupported_claims") or [])
    checks = {
        "package_manifest_present": manifest_path.exists(),
        "package_family_native_nervous": manifest.get("package_family")
        == "phase2d_native_nervous_package",
        "policy_label_is_phase2aw": REQUIRED_POLICY_LABEL_FRAGMENT
        in str(manifest.get("policy_label", "")).lower(),
        "native_head_path_exists": _path_exists_from_cwd(manifest.get("native_head_path")),
        "low_level_checkpoint_path_exists": _path_exists_from_cwd(
            manifest.get("low_level_checkpoint_path")
        ),
        "json_text_target_disabled": manifest.get("json_text_target") is False,
        "native_head_calls_enabled": manifest.get("native_head_calls_enabled") is True,
        "required_open_repair_capabilities_enabled": not missing_caps,
        "patch_strategy_learned_bounded_candidate": manifest.get("patch_proposal_strategy")
        == "learned_bounded_candidate",
        "learned_patch_generation_enabled": manifest.get("learned_patch_generation_enabled")
        is True,
        "patch_candidate_schema_version_matches": manifest.get(
            "patch_candidate_schema_version"
        )
        == REQUIRED_PATCH_SCHEMA_VERSION,
        "authorization_gate_passed": authorization.get("passed") is True,
        "authorization_ready_for_package_build": authorization.get("ready_for_package_build")
        is True,
        "authorization_does_not_claim_sealed_ready": authorization.get(
            "ready_for_sealed_eval"
        )
        is False,
        "schema_gate_passed": schema_gate.get("passed") is True,
        "schema_gate_is_package_schema_only": schema_gate.get("artifact_family")
        == "phase2at_learned_patch_generation_package_gate",
        "evidence_sufficiency_passed": sufficiency.get("passed") is True,
        "evidence_scope_bounded_nonsealed": sufficiency.get("claim_scope")
        == "phase2av_bounded_nonsealed_descriptor_runtime_candidate_selection",
        "evidence_report_did_not_preclaim_package_ready": "phase2av_package_ready"
        in unsupported,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2aw_postpackage_gate",
        "passed": passed,
        "ready_for_bounded_package_loaded_runtime_eval": passed,
        "ready_for_sealed_eval": False,
        "claim_boundary": (
            "This gate records that a Phase2AW bounded native package exists and "
            "passes package/schema consistency checks. It only authorizes a "
            "package-loaded non-sealed runtime evaluation. It does not authorize "
            "sealed evaluation, freeform patch generation, open-ended debugging "
            "generalization, production autonomy, or epoch-making architecture claims."
        ),
        "checks": checks,
        "metrics": {
            "missing_open_repair_capabilities": missing_caps,
            "policy_label": manifest.get("policy_label"),
            "patch_proposal_strategy": manifest.get("patch_proposal_strategy"),
            "patch_candidate_schema_version": manifest.get("patch_candidate_schema_version"),
        },
        "supported_claims": [
            "phase2aw_package_built_for_bounded_package_loaded_nonsealed_runtime_eval"
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
            "do_not_run_package_loaded_runtime_eval",
            "do_not_run_sealed_eval",
            "fix_phase2aw_postpackage_inputs",
        ],
        "postpackage_blocked_actions": [
            "do_not_run_sealed_eval_until_package_loaded_nonsealed_runtime_gate_passes",
            "do_not_claim_package_loaded_runtime_success_from_eval_summary_only",
            "do_not_claim_epoch_making_architecture",
        ],
        "next_required_artifact": (
            "phase2aw_package_loaded_nonsealed_runtime_gate.json"
            if passed
            else None
        ),
        "inputs": {
            "package_path": str(Path(package_path)),
            "manifest_path": str(manifest_path),
            "package_authorization_gate_json": str(Path(package_authorization_gate_json)),
            "package_schema_gate_json": str(Path(package_schema_gate_json)),
            "evidence_sufficiency_json": str(Path(evidence_sufficiency_json)),
        },
        "thresholds": {
            "required_policy_label_fragment": REQUIRED_POLICY_LABEL_FRAGMENT,
            "required_patch_candidate_schema_version": REQUIRED_PATCH_SCHEMA_VERSION,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AW postpackage gate.")
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--package-authorization-gate-json", required=True)
    parser.add_argument("--package-schema-gate-json", required=True)
    parser.add_argument("--evidence-sufficiency-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2aw_postpackage_gate(
        package_path=args.package_path,
        package_authorization_gate_json=args.package_authorization_gate_json,
        package_schema_gate_json=args.package_schema_gate_json,
        evidence_sufficiency_json=args.evidence_sufficiency_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
