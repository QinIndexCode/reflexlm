from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.llm.native_cortex import OPEN_REPAIR_CAPABILITY_NAMES
from reflexlm.llm.native_nervous_package import PACKAGE_MANIFEST_NAME


ARTIFACT_FAMILY = "phase2az_phase2ax_packaged_adapter_gate"
REQUIRED_ADAPTER_ENTRIES = (
    "head_config.json",
    "native_heads.pt",
    "backbone_adapter",
    "tokenizer",
)
REQUIRED_OPEN_REPAIR_CAPABILITIES = tuple(OPEN_REPAIR_CAPABILITY_NAMES)
REQUIRED_SCHEMA_VERSION = "phase2at.learned_bounded_patch_candidate.v1"
HEAVY_PACKAGE_EXTENSIONS = {".bin", ".gguf", ".pt", ".safetensors"}


def _read_json(path: str | Path) -> dict[str, Any]:
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


def _resolve_existing_path(path_value: Any, *, base_dir: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return Path.cwd() / path


def _same_resolved_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def _heavy_files_inside_package(package_path: Path, manifest_path: Path) -> list[str]:
    package_dir = package_path if package_path.is_dir() else manifest_path.parent
    if not package_dir.exists():
        return []
    heavy_files: list[str] = []
    for file in package_dir.rglob("*"):
        if not file.is_file() or file == manifest_path:
            continue
        if file.suffix.lower() in HEAVY_PACKAGE_EXTENSIONS:
            heavy_files.append(str(file))
    return sorted(heavy_files)


def _adapter_entries_present(adapter_path: Path | None) -> dict[str, bool]:
    if adapter_path is None:
        return {name: False for name in REQUIRED_ADAPTER_ENTRIES}
    return {name: (adapter_path / name).exists() for name in REQUIRED_ADAPTER_ENTRIES}


def audit_phase2az_package_gate(
    *,
    package_path: str | Path,
    phase2az_matrix_json: str | Path,
    full_postflight_json: str | Path,
    training_summary_json: str | Path,
    expected_native_head_path: str | Path | None = None,
    output_json: str | Path | None = None,
    required_schema_version: str = REQUIRED_SCHEMA_VERSION,
) -> dict[str, Any]:
    manifest_path = _manifest_path(package_path)
    package_root = manifest_path.parent
    manifest = _read_json(manifest_path)
    matrix = _read_json(phase2az_matrix_json)
    full_postflight = _read_json(full_postflight_json)
    training_summary = _read_json(training_summary_json)

    native_head_path = _resolve_existing_path(
        manifest.get("native_head_path"),
        base_dir=package_root,
    )
    low_level_checkpoint_path = _resolve_existing_path(
        manifest.get("low_level_checkpoint_path"),
        base_dir=package_root,
    )
    expected_head_path = (
        Path(expected_native_head_path)
        if expected_native_head_path is not None
        else _resolve_existing_path(training_summary.get("adapter_output_dir"), base_dir=Path.cwd())
    )
    if expected_head_path is not None and not expected_head_path.is_absolute():
        expected_head_path = Path.cwd() / expected_head_path

    caps = manifest.get("open_repair_capabilities")
    if not isinstance(caps, dict):
        caps = {}
    missing_caps = [
        name for name in REQUIRED_OPEN_REPAIR_CAPABILITIES if caps.get(name) is not True
    ]
    unknown_caps = sorted(set(caps) - set(OPEN_REPAIR_CAPABILITY_NAMES))
    adapter_entries = _adapter_entries_present(native_head_path)
    heavy_files = _heavy_files_inside_package(Path(package_path), manifest_path)
    training_contract = (
        training_summary.get("open_repair_training_contract")
        if isinstance(training_summary.get("open_repair_training_contract"), dict)
        else {}
    )
    training_caps = (
        training_summary.get("open_repair_capabilities")
        if isinstance(training_summary.get("open_repair_capabilities"), dict)
        else {}
    )
    matrix_success = matrix.get("metrics", {}).get("execution_success_rate")

    checks = {
        "manifest_present": manifest_path.exists(),
        "package_family_native_nervous": manifest.get("package_family")
        == "phase2d_native_nervous_package",
        "policy_label_identifies_phase2az_phase2ax": "phase2az"
        in str(manifest.get("policy_label", "")).lower()
        and "phase2ax" in str(manifest.get("policy_label", "")).lower(),
        "phase2az_matrix_passed": matrix.get("passed") is True
        and matrix.get("ready_for_phase2az_package_gate") is True,
        "phase2az_matrix_not_epoch_claim": matrix.get(
            "ready_for_epoch_making_architecture_claim"
        )
        is False,
        "phase2ax_full_postflight_passed": full_postflight.get("passed") is True
        and full_postflight.get("ready_for_phase2ay_runtime_execution_eval") is True,
        "phase2ax_full_postflight_blocks_package_claim": full_postflight.get(
            "ready_for_phase2ax_package"
        )
        is False,
        "phase2ax_full_postflight_blocks_epoch_claim": full_postflight.get(
            "ready_for_epoch_making_architecture_claim"
        )
        is False,
        "base_model_matches_training_summary": manifest.get("base_model_name")
        == training_summary.get("base_model_name"),
        "native_head_path_matches_training_adapter": _same_resolved_path(
            native_head_path,
            expected_head_path,
        ),
        "native_head_adapter_entries_present": all(adapter_entries.values()),
        "low_level_checkpoint_exists": low_level_checkpoint_path is not None
        and low_level_checkpoint_path.exists(),
        "quantization_matches_training_runtime": manifest.get("quantization") == "4bit",
        "max_length_matches_training_runtime": int(manifest.get("max_length") or 0) == 256,
        "all_open_repair_capabilities_declared": not missing_caps,
        "no_unknown_open_repair_capabilities": not unknown_caps,
        "training_summary_records_open_repair_heads": training_summary.get(
            "open_repair_heads_enabled"
        )
        is True,
        "training_summary_capabilities_match_manifest": all(
            training_caps.get(name) is True for name in REQUIRED_OPEN_REPAIR_CAPABILITIES
        ),
        "patch_strategy_is_learned_bounded_candidate": manifest.get(
            "patch_proposal_strategy"
        )
        == "learned_bounded_candidate"
        and training_contract.get("patch_proposal_strategy")
        == "learned_bounded_candidate",
        "learned_patch_generation_enabled": manifest.get(
            "learned_patch_generation_enabled"
        )
        is True,
        "patch_candidate_schema_version_matches": manifest.get(
            "patch_candidate_schema_version"
        )
        == required_schema_version,
        "training_summary_blocks_relabeling": training_contract.get("sealed_feedback_used")
        is False
        and training_contract.get("recorded_patch_artifact_as_generation_target") is False
        and training_contract.get("symbolic_generator_as_generation_target") is False,
        "no_json_text_target": manifest.get("json_text_target") is False
        and training_contract.get("json_text_target") is False,
        "no_freeform_patch_text_target": training_contract.get("freeform_patch_text_target")
        is False,
        "low_level_qwen_calls_target_zero": training_summary.get(
            "low_level_qwen_calls_target"
        )
        == 0,
        "manifest_is_reference_package_without_heavy_model_copies": not heavy_files,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": ARTIFACT_FAMILY,
        "passed": passed,
        "ready_for_phase2az_packaged_adapter_runtime_smoke": passed,
        "ready_for_phase2ax_package": False,
        "ready_for_package_or_execution_claim": False,
        "ready_for_sealed_eval": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "missing_open_repair_capabilities": missing_caps,
            "unknown_open_repair_capabilities": unknown_caps,
            "adapter_entries_present": adapter_entries,
            "heavy_files_inside_package": heavy_files,
            "phase2az_matrix_execution_success_rate": matrix_success,
            "phase2az_model_slot_accuracy": (matrix.get("metrics") or {}).get(
                "model_slot_accuracy"
            ),
            "phase2az_repo_count": (matrix.get("metrics") or {}).get("repo_count"),
            "native_head_path": str(native_head_path) if native_head_path else None,
            "expected_native_head_path": str(expected_head_path)
            if expected_head_path
            else None,
            "low_level_checkpoint_path": str(low_level_checkpoint_path)
            if low_level_checkpoint_path
            else None,
        },
        "claim_boundary": (
            "phase2az_packaged_adapter_manifest_validated_not_loaded_runtime_or_epoch_claim"
            if passed
            else "phase2az_packaged_adapter_gate_failed_or_incomplete"
        ),
        "supported_claims": [
            "phase2az_phase2ax_adapter_manifest_is_ready_for_package_loaded_runtime_smoke"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "actual_package_loaded_runtime_execution_matrix",
            "phase2ax_package_claim",
            "sealed_cross_model_transfer",
            "freeform_patch_generation",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": [
            "do_not_claim_phase2ax_package_from_manifest_only",
            "do_not_claim_sealed_transfer_from_nonsealed_phase2az_matrix",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ],
        "next_required_experiment": (
            "phase2ba_package_loaded_model_predicted_execution_matrix"
            if passed
            else "repair_phase2az_package_manifest_or_evidence_chain"
        ),
        "secondary_required_fix": "repair_phase2ax_pair_00024_descriptor_execution",
        "inputs": {
            "package_path": str(Path(package_path)),
            "manifest_path": str(manifest_path),
            "phase2az_matrix_json": str(Path(phase2az_matrix_json)),
            "full_postflight_json": str(Path(full_postflight_json)),
            "training_summary_json": str(Path(training_summary_json)),
            "expected_native_head_path": str(Path(expected_native_head_path))
            if expected_native_head_path is not None
            else None,
        },
        "notes": [
            "This gate validates a reference package manifest that points at the trained Phase2AX adapter.",
            "It does not instantiate NativeNervousPolicyPackage or rerun the runtime matrix through the package loader.",
            "The current Phase2AZ matrix still has a bounded execution failure, so the next gate must be package-loaded runtime evidence.",
        ],
    }
    if output_json is not None:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AZ packaged Phase2AX adapter manifest gate."
    )
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--phase2az-matrix-json", required=True)
    parser.add_argument("--full-postflight-json", required=True)
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--expected-native-head-path")
    parser.add_argument(
        "--required-schema-version",
        default=REQUIRED_SCHEMA_VERSION,
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2az_package_gate(
        package_path=args.package_path,
        phase2az_matrix_json=args.phase2az_matrix_json,
        full_postflight_json=args.full_postflight_json,
        training_summary_json=args.training_summary_json,
        expected_native_head_path=args.expected_native_head_path,
        output_json=args.output_json,
        required_schema_version=args.required_schema_version,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
