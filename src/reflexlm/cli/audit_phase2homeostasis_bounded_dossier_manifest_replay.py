from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

from reflexlm.cli.audit_phase2homeostasis_bounded_dossier_manifest import (
    REQUIRED_REPRODUCTION_STEPS,
    REQUIRED_SOURCE_REPORT_ROLES,
    REQUIRED_STATE_ARTIFACT_ROLES,
    _sha256,
    validate_phase2homeostasis_bounded_dossier_manifest,
)
from reflexlm.cli.audit_phase2homeostasis_publication_bundle import (
    _read_json,
    _write_json,
)


OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_exact_cross_runtime_homeostatic_dynamics_claim",
    "ready_for_unbounded_long_term_memory_claim",
    "ready_for_general_runtime_interpreter_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _entries(manifest: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = manifest.get(key, [])
    return (
        [item for item in values if isinstance(item, dict)]
        if isinstance(values, list)
        else []
    )


def _copy_entries(
    *,
    entries: list[dict[str, Any]],
    target_dir: Path,
) -> list[dict[str, Any]]:
    copied_entries: list[dict[str, Any]] = []
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in entries:
        source = Path(str(item.get("path", "")))
        if not source.exists():
            raise ValueError(f"source path missing for replay: {source}")
        target = target_dir / source.name
        shutil.copy2(source, target)
        copied = deepcopy(item)
        copied["path"] = str(target)
        copied["bytes"] = target.stat().st_size
        copied["sha256"] = _sha256(target)
        copied_entries.append(copied)
    return copied_entries


def _rewrite_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [_rewrite_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, path_map) for key, item in value.items()}
    return value


def _role_set(entries: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("role")) for item in entries}


def validate_phase2homeostasis_bounded_dossier_manifest_replay(
    report: dict[str, Any],
) -> dict[str, Any]:
    replay_report_path = report.get("evidence", {}).get(
        "replayed_reproducibility_report"
    )
    replay_report_readable = False
    replay_validation: dict[str, Any] = {}
    if replay_report_path:
        try:
            replay_report = _read_json(replay_report_path)
            replay_report_readable = True
            replay_validation = validate_phase2homeostasis_bounded_dossier_manifest(
                replay_report
            )
        except (OSError, json.JSONDecodeError):
            replay_report_readable = False
    checks = {
        "artifact_family_matches": (
            report.get("artifact_family")
            == "phase2homeostasis_bounded_dossier_manifest_replay"
        ),
        "top_level_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_homeostasis_dossier_manifest_replay_claim")
            is True
            and all(report.get(flag) is not True for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_manifest_report_passed": (
            report.get("source_summary", {}).get("source_manifest_report_passed")
            is True
        ),
        "replayed_report_readable": replay_report_readable,
        "replayed_manifest_validation_passed": replay_validation.get("passed") is True,
        "replayed_source_report_roles_complete": set(
            REQUIRED_SOURCE_REPORT_ROLES
        ).issubset(set(report.get("replay_summary", {}).get("source_report_roles", []))),
        "replayed_state_artifact_roles_complete": set(
            REQUIRED_STATE_ARTIFACT_ROLES
        ).issubset(set(report.get("replay_summary", {}).get("state_artifact_roles", []))),
        "replayed_reproduction_steps_complete": set(
            REQUIRED_REPRODUCTION_STEPS
        ).issubset(
            set(report.get("replay_summary", {}).get("reproduction_step_ids", []))
        ),
        "replay_directory_is_distinct": (
            report.get("replay_summary", {}).get("source_manifest_parent")
            != report.get("replay_summary", {}).get("replay_manifest_parent")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "replayed_source_report_count": report.get("replay_summary", {}).get(
                "source_report_count", 0
            ),
            "replayed_state_artifact_count": report.get("replay_summary", {}).get(
                "state_artifact_count", 0
            ),
            "replayed_supporting_artifact_count": report.get("replay_summary", {}).get(
                "supporting_artifact_count", 0
            ),
            "replayed_reproduction_step_count": report.get("replay_summary", {}).get(
                "reproduction_step_count", 0
            ),
        },
    }


def audit_phase2homeostasis_bounded_dossier_manifest_replay(
    *,
    reproducibility_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    source_report = _read_json(reproducibility_report_json)
    source_manifest_path = Path(
        str(source_report.get("evidence", {}).get("reproducibility_manifest", ""))
    )
    if not source_manifest_path.exists():
        raise ValueError("replay requires a readable bounded dossier manifest")
    source_manifest = _read_json(source_manifest_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    replay_source_reports = _copy_entries(
        entries=_entries(source_manifest, "source_reports"),
        target_dir=output_root / "source_reports",
    )
    replay_state_artifacts = _copy_entries(
        entries=_entries(source_manifest, "state_artifacts"),
        target_dir=output_root / "state_artifacts",
    )
    replay_supporting_artifacts = _copy_entries(
        entries=_entries(source_manifest, "supporting_artifacts"),
        target_dir=output_root / "supporting_artifacts",
    )
    original_entries = (
        _entries(source_manifest, "source_reports")
        + _entries(source_manifest, "state_artifacts")
        + _entries(source_manifest, "supporting_artifacts")
    )
    replay_entries = (
        replay_source_reports + replay_state_artifacts + replay_supporting_artifacts
    )
    path_map = {
        str(original.get("path")): str(replayed.get("path"))
        for original, replayed in zip(original_entries, replay_entries, strict=True)
    }
    replay_manifest = deepcopy(source_manifest)
    replay_manifest["source_reports"] = replay_source_reports
    replay_manifest["state_artifacts"] = replay_state_artifacts
    replay_manifest["supporting_artifacts"] = replay_supporting_artifacts
    replay_manifest["reproduction_steps"] = _rewrite_paths(
        source_manifest.get("reproduction_steps", []),
        path_map,
    )
    replay_manifest_path = output_root / "replayed_bounded_dossier_manifest.json"
    _write_json(replay_manifest_path, replay_manifest)
    replay_report = deepcopy(source_report)
    replay_report.setdefault("evidence", {})[
        "reproducibility_manifest"
    ] = str(replay_manifest_path)
    replay_report_path = output_root / "replayed_bounded_dossier_manifest_report.json"
    _write_json(replay_report_path, replay_report)
    replay_validation = validate_phase2homeostasis_bounded_dossier_manifest(
        replay_report
    )
    replay_validation_path = output_root / "replayed_bounded_dossier_validation.json"
    _write_json(replay_validation_path, replay_validation)
    reproduction_step_ids = [
        str(item.get("step_id"))
        for item in replay_manifest.get("reproduction_steps", [])
        if isinstance(item, dict)
    ]
    checks = {
        "source_reproducibility_report_passed": source_report.get("passed") is True,
        "replay_manifest_written": replay_manifest_path.exists(),
        "replay_report_written": replay_report_path.exists(),
        "replay_validation_passed": replay_validation.get("passed") is True,
        "source_report_roles_preserved": set(REQUIRED_SOURCE_REPORT_ROLES).issubset(
            _role_set(replay_source_reports)
        ),
        "state_artifact_roles_preserved": set(REQUIRED_STATE_ARTIFACT_ROLES).issubset(
            _role_set(replay_state_artifacts)
        ),
        "reproduction_steps_preserved": set(REQUIRED_REPRODUCTION_STEPS).issubset(
            set(reproduction_step_ids)
        ),
        "replay_directory_is_distinct": source_manifest_path.parent.resolve()
        != output_root.resolve(),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_bounded_dossier_manifest_replay",
        "passed": passed,
        "ready_for_bounded_homeostasis_dossier_manifest_replay_claim": passed,
        "ready_for_exact_cross_runtime_homeostatic_dynamics_claim": False,
        "ready_for_unbounded_long_term_memory_claim": False,
        "ready_for_general_runtime_interpreter_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "replayed_source_report_count": len(replay_source_reports),
            "replayed_state_artifact_count": len(replay_state_artifacts),
            "replayed_supporting_artifact_count": len(replay_supporting_artifacts),
            "replayed_reproduction_step_count": len(reproduction_step_ids),
        },
        "source_summary": {
            "source_manifest_report_passed": source_report.get("passed") is True,
            "source_reproducibility_report_json": str(reproducibility_report_json),
            "source_manifest_json": str(source_manifest_path),
        },
        "replay_summary": {
            "source_manifest_parent": str(source_manifest_path.parent.resolve()),
            "replay_manifest_parent": str(output_root.resolve()),
            "source_report_count": len(replay_source_reports),
            "state_artifact_count": len(replay_state_artifacts),
            "supporting_artifact_count": len(replay_supporting_artifacts),
            "reproduction_step_count": len(reproduction_step_ids),
            "source_report_roles": sorted(_role_set(replay_source_reports)),
            "state_artifact_roles": sorted(_role_set(replay_state_artifacts)),
            "reproduction_step_ids": reproduction_step_ids,
        },
        "claim_boundary": (
            "This replay verifies that the bounded HMAC homeostatic dossier "
            "manifest can be copied to a distinct directory and revalidated by "
            "hash. It is not a fresh model rerun and does not support exact "
            "cross-runtime homeostatic microdynamics, unbounded memory, "
            "production autonomy, or epoch-making architecture."
        ),
        "supported_claims": [
            "bounded cross-directory replay of the homeostatic dossier manifest"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "fresh model rerun",
            "exact cross-runtime homeostatic microdynamics",
            "unbounded or semantic long-term memory",
            "free-form shell autonomy",
            "general runtime interpreter invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "external_machine_replay_or_deterministic_microdynamics_calibration"
            if passed
            else "repair_bounded_dossier_manifest_replay"
        ),
        "evidence": {
            "source_reproducibility_report": str(reproducibility_report_json),
            "replayed_reproducibility_manifest": str(replay_manifest_path),
            "replayed_reproducibility_report": str(replay_report_path),
            "replayed_reproducibility_validation": str(replay_validation_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a bounded Phase2Homeostasis dossier manifest."
    )
    parser.add_argument("--reproducibility-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_bounded_dossier_manifest_replay(
        reproducibility_report_json=args.reproducibility_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
