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


def _manifest(path: str | Path) -> dict[str, Any]:
    package_path = Path(path)
    manifest_path = (
        package_path if package_path.is_file() else package_path / "native_nervous_package.json"
    )
    if not manifest_path.exists():
        return {}
    return _read_json(manifest_path)


def _disabled_groups(manifest: dict[str, Any]) -> set[str]:
    value = manifest.get("disabled_command_candidate_feature_groups")
    return {str(item) for item in value} if isinstance(value, list) else set()


def audit_phase2aw_sealed_authorization_gate(
    *,
    package_loaded_sufficiency_json: str | Path,
    full_package_path: str | Path,
    no_nsi_package_path: str | Path,
    native_head_only_package_path: str | Path,
    continuation_only_package_path: str | Path,
    sealed_dataset_path: str | Path,
) -> dict[str, Any]:
    sufficiency = _read_json(package_loaded_sufficiency_json)
    full = _manifest(full_package_path)
    no_nsi = _manifest(no_nsi_package_path)
    native = _manifest(native_head_only_package_path)
    continuation = _manifest(continuation_only_package_path)
    full_label = str(full.get("policy_label") or "")
    expected_labels = {
        "no_nsi": f"{full_label}_no_nsi_latent",
        "native_head_only": f"{full_label}_native_head_only",
        "continuation_only": f"{full_label}_continuation_only",
    }
    no_nsi_groups = _disabled_groups(no_nsi)
    same_artifacts = bool(full) and all(
        manifest.get("base_model_name") == full.get("base_model_name")
        and manifest.get("native_head_path") == full.get("native_head_path")
        and manifest.get("low_level_checkpoint_path") == full.get("low_level_checkpoint_path")
        for manifest in (no_nsi, native, continuation)
    )
    checks = {
        "package_loaded_sufficiency_passed": sufficiency.get("passed") is True,
        "sufficiency_scope_is_phase2aw_package_loaded": sufficiency.get("claim_scope")
        == "phase2aw_bounded_nonsealed_package_loaded_descriptor_runtime",
        "sufficiency_blocks_claim_upgrade": "epoch_making_architecture"
        in set(sufficiency.get("unsupported_claims") or []),
        "sealed_dataset_exists": Path(sealed_dataset_path).exists(),
        "full_package_manifest_exists": bool(full),
        "no_nsi_package_manifest_exists": bool(no_nsi),
        "native_head_only_package_manifest_exists": bool(native),
        "continuation_only_package_manifest_exists": bool(continuation),
        "full_package_family_valid": full.get("package_family")
        == "phase2d_native_nervous_package",
        "full_package_no_json_target": full.get("json_text_target") is False,
        "full_package_native_heads_enabled": full.get("native_head_calls_enabled") is True,
        "full_package_continuation_enabled": full.get("continuation_cache_enabled") is True,
        "full_package_nsi_enabled": full.get("zero_nsi_latent") is False,
        "no_nsi_zeroes_latent": no_nsi.get("zero_nsi_latent") is True,
        "no_nsi_disables_candidate_identity": "candidate_identity" in no_nsi_groups,
        "no_nsi_keeps_native_heads": no_nsi.get("native_head_calls_enabled") is True,
        "no_nsi_keeps_continuation": no_nsi.get("continuation_cache_enabled") is True,
        "native_head_only_keeps_native_heads": native.get("native_head_calls_enabled")
        is True,
        "native_head_only_disables_continuation": native.get("continuation_cache_enabled")
        is False,
        "continuation_only_disables_native_heads": continuation.get(
            "native_head_calls_enabled"
        )
        is False,
        "continuation_only_keeps_continuation": continuation.get(
            "continuation_cache_enabled"
        )
        is True,
        "control_labels_derive_from_full_label": (
            no_nsi.get("policy_label") == expected_labels["no_nsi"]
            and native.get("policy_label") == expected_labels["native_head_only"]
            and continuation.get("policy_label") == expected_labels["continuation_only"]
        ),
        "all_packages_reference_same_model_adapter_checkpoint": same_artifacts,
    }
    passed = all(checks.values())
    blocked_actions: list[str] = []
    if not passed:
        blocked_actions.append("do_not_run_phase2aw_sealed_eval")
    if not checks["no_nsi_disables_candidate_identity"]:
        blocked_actions.append("do_not_evaluate_no_nsi_with_candidate_identity_leak")
    if not checks["control_labels_derive_from_full_label"]:
        blocked_actions.append("do_not_run_sealed_eval_with_mismatched_control_labels")
    if not checks["all_packages_reference_same_model_adapter_checkpoint"]:
        blocked_actions.append("do_not_run_sealed_eval_with_drifted_package_artifacts")
    return {
        "artifact_family": "phase2aw_sealed_authorization_gate",
        "passed": passed,
        "ready_for_sealed_eval": passed,
        "ready_for_claim_upgrade": False,
        "allowed_next_action": (
            "run_phase2aw_sealed_eval_only_no_training_feedback"
            if passed
            else "fix_phase2aw_control_packages_before_sealed_eval"
        ),
        "claim_boundary": (
            "This gate authorizes sealed evaluation only as final evaluation with "
            "no training, sampling, tuning, or failure-feedback use. Passing does "
            "not establish sealed transfer or any stronger architecture claim."
        ),
        "checks": checks,
        "package_paths": {
            "full": str(Path(full_package_path)),
            "no_nsi": str(Path(no_nsi_package_path)),
            "native_head_only": str(Path(native_head_only_package_path)),
            "continuation_only": str(Path(continuation_only_package_path)),
        },
        "sealed_dataset_path": str(Path(sealed_dataset_path)),
        "unsupported_claims": [
            "sealed_cross_model_transfer_not_established_until_sealed_eval_passes",
            "production_autonomy_not_established",
            "open_ended_debugging_generalization_not_established",
            "epoch_making_architecture_claim_not_established",
        ],
        "blocked_actions": sorted(set(blocked_actions)),
        "inputs": {
            "package_loaded_sufficiency_json": str(Path(package_loaded_sufficiency_json))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AW sealed authorization gate.")
    parser.add_argument("--package-loaded-sufficiency-json", required=True)
    parser.add_argument("--full-package-path", required=True)
    parser.add_argument("--no-nsi-package-path", required=True)
    parser.add_argument("--native-head-only-package-path", required=True)
    parser.add_argument("--continuation-only-package-path", required=True)
    parser.add_argument("--sealed-dataset-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2aw_sealed_authorization_gate(
        package_loaded_sufficiency_json=args.package_loaded_sufficiency_json,
        full_package_path=args.full_package_path,
        no_nsi_package_path=args.no_nsi_package_path,
        native_head_only_package_path=args.native_head_only_package_path,
        continuation_only_package_path=args.continuation_only_package_path,
        sealed_dataset_path=args.sealed_dataset_path,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
