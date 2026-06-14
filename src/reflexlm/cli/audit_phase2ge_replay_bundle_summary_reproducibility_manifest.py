from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2dk_latex_candidate_publication_bundle import _sha256
from reflexlm.cli.audit_phase2gc_replay_bundle_summary_replay_bundle import (
    REQUIRED_ARTIFACT_ROLES,
    validate_phase2gc_replay_bundle_summary_replay_bundle,
)


REQUIRED_SOURCE_REPORT_ROLES: tuple[str, ...] = (
    "phase2ga_report",
    "phase2gb_negative_control_report",
    "phase2gc_replay_bundle_report",
    "phase2gd_negative_control_report",
)

REQUIRED_REPRODUCTION_STEPS: tuple[str, ...] = (
    "phase2ga_replay_bundle_summary_cross_directory_replay",
    "phase2gb_replay_bundle_summary_replay_negative_controls",
    "phase2gc_replay_bundle_summary_replay_bundle",
    "phase2gd_replay_bundle_summary_replay_bundle_negative_controls",
)

CONTROL_ARTIFACT_ROLES: set[str] = {
    "replayed_control_report",
    "replayed_control_validation",
}

OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_general_shell_autonomy_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _file_entry(*, role: str, path: str | Path, content_type: str) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "role": role,
        "path": str(file_path),
        "content_type": content_type,
        "bytes": file_path.stat().st_size,
        "sha256": _sha256(file_path),
    }


def _entries(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _entry_roles(entries: Any) -> set[str]:
    return {str(item.get("role")) for item in _entries(entries)}


def _entries_exist(entries: list[dict[str, Any]]) -> bool:
    return bool(entries) and all(Path(str(item.get("path", ""))).exists() for item in entries)


def _entry_hashes_match(entries: list[dict[str, Any]]) -> bool:
    if not entries:
        return False
    matches = []
    for item in entries:
        path = Path(str(item.get("path", "")))
        matches.append(path.exists() and _sha256(path) == item.get("sha256"))
    return all(matches)


def _bundle_artifact_entries(phase2gc: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = _read_json(phase2gc.get("evidence", {}).get("bundle_manifest"))
    artifacts = manifest.get("artifacts", [])
    entries = []
    for item in _entries(artifacts):
        path = Path(str(item.get("path", "")))
        current_sha256 = _sha256(path) if path.exists() else None
        entries.append(
            {
                "role": item.get("role"),
                "control_id": item.get("control_id"),
                "path": str(path),
                "content_type": item.get("content_type"),
                "bytes": path.stat().st_size if path.exists() else item.get("bytes"),
                "sha256": item.get("sha256"),
                "current_sha256": current_sha256,
                "hash_matches_bundle_manifest": current_sha256 == item.get("sha256"),
            }
        )
    return entries


def _control_artifact_count(entries: list[dict[str, Any]]) -> int:
    return sum(item.get("role") in CONTROL_ARTIFACT_ROLES for item in entries)


def _reproduction_steps(
    *,
    phase2ga: dict[str, Any],
    phase2gb: dict[str, Any],
    phase2gc: dict[str, Any],
    phase2ga_report_json: str | Path,
    phase2gb_report_json: str | Path,
    phase2gc_report_json: str | Path,
    phase2gd_report_json: str | Path,
) -> list[dict[str, Any]]:
    return [
        {
            "step_id": "phase2ga_replay_bundle_summary_cross_directory_replay",
            "module": "reflexlm.cli.audit_phase2ga_replay_bundle_summary_cross_directory_replay",
            "inputs": {
                "phase2fz_report_json": phase2ga.get("evidence", {}).get(
                    "phase2fz_report_json"
                )
            },
            "outputs": {
                "output_report_json": str(phase2ga_report_json),
                "output_dir": phase2ga.get("evidence", {}).get("replay_dir"),
            },
        },
        {
            "step_id": "phase2gb_replay_bundle_summary_replay_negative_controls",
            "module": "reflexlm.cli.audit_phase2gb_replay_bundle_summary_replay_negative_controls",
            "inputs": {"phase2ga_report_json": str(phase2ga_report_json)},
            "outputs": {
                "output_report_json": str(phase2gb_report_json),
                "output_dir": phase2gb.get("evidence", {}).get(
                    "negative_control_output_dir"
                ),
            },
        },
        {
            "step_id": "phase2gc_replay_bundle_summary_replay_bundle",
            "module": "reflexlm.cli.audit_phase2gc_replay_bundle_summary_replay_bundle",
            "inputs": {"phase2gb_report_json": str(phase2gb_report_json)},
            "outputs": {
                "output_report_json": str(phase2gc_report_json),
                "output_dir": phase2gc.get("evidence", {}).get("bundle_dir"),
            },
        },
        {
            "step_id": "phase2gd_replay_bundle_summary_replay_bundle_negative_controls",
            "module": "reflexlm.cli.audit_phase2gd_replay_bundle_summary_replay_bundle_negative_controls",
            "inputs": {"phase2gc_report_json": str(phase2gc_report_json)},
            "outputs": {
                "output_report_json": str(phase2gd_report_json),
            },
        },
    ]


def validate_phase2ge_replay_bundle_summary_reproducibility_manifest(
    report: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = report.get("evidence", {}).get("reproducibility_manifest")
    manifest_readable = False
    manifest: dict[str, Any] = {}
    if manifest_path:
        try:
            manifest = _read_json(manifest_path)
            manifest_readable = True
        except (OSError, json.JSONDecodeError):
            manifest_readable = False
    source_reports = _entries(manifest.get("source_reports", []))
    bundle_artifacts = _entries(manifest.get("bundle_artifacts", []))
    reproduction_steps = _entries(manifest.get("reproduction_steps", []))
    bundle_artifact_hash_matches = [
        item.get("hash_matches_bundle_manifest") is True
        and Path(str(item.get("path", ""))).exists()
        and _sha256(Path(str(item.get("path")))) == item.get("current_sha256")
        for item in bundle_artifacts
    ]
    checks = {
        "artifact_family_matches_phase2ge": (
            report.get("artifact_family")
            == "phase2ge_replay_bundle_summary_reproducibility_manifest"
        ),
        "top_level_phase2ge_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_reproducibility_manifest_claim") is True
            and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2gc_and_phase2gd_passed": (
            report.get("source_summary", {}).get("phase2gc_passed") is True
            and report.get("source_summary", {}).get("phase2gd_passed") is True
        ),
        "source_phase2gc_validation_passed": (
            report.get("source_summary", {}).get("phase2gc_validation_passed") is True
        ),
        "manifest_readable": manifest_readable,
        "source_report_roles_complete": set(REQUIRED_SOURCE_REPORT_ROLES).issubset(
            _entry_roles(source_reports)
        ),
        "source_reports_exist": _entries_exist(source_reports),
        "source_report_hashes_match": _entry_hashes_match(source_reports),
        "bundle_artifact_roles_complete": set(REQUIRED_ARTIFACT_ROLES).issubset(
            _entry_roles(bundle_artifacts)
        ),
        "bundle_artifacts_exist": _entries_exist(bundle_artifacts),
        "bundle_artifact_hashes_match": bool(bundle_artifact_hash_matches)
        and all(bundle_artifact_hash_matches),
        "control_artifact_count_preserved": _control_artifact_count(bundle_artifacts)
        == report.get("source_summary", {}).get("phase2gc_copied_control_artifact_count"),
        "reproduction_steps_complete": set(REQUIRED_REPRODUCTION_STEPS).issubset(
            {str(item.get("step_id")) for item in reproduction_steps}
        ),
        "reproduction_steps_have_modules_and_paths": bool(reproduction_steps)
        and all(
            item.get("module")
            and isinstance(item.get("inputs"), dict)
            and isinstance(item.get("outputs"), dict)
            for item in reproduction_steps
        ),
        "manifest_contains_bounded_boundary": (
            "not free-form shell autonomy" in manifest.get("claim_boundary", "")
            and "not an epoch-making architecture" in manifest.get("claim_boundary", "")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "source_report_count": len(source_reports),
            "bundle_artifact_count": len(bundle_artifacts),
            "bundle_artifact_hash_match_count": sum(bundle_artifact_hash_matches),
            "control_artifact_count": _control_artifact_count(bundle_artifacts),
            "reproduction_step_count": len(reproduction_steps),
        },
    }


def audit_phase2ge_replay_bundle_summary_reproducibility_manifest(
    *,
    phase2gd_report_json: str | Path,
    output_manifest_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2gd = _read_json(phase2gd_report_json)
    phase2gc_report_json = phase2gd.get("evidence", {}).get("phase2gc_report_json")
    if not phase2gc_report_json:
        raise ValueError("Phase2GE requires Phase2GD evidence.phase2gc_report_json")
    phase2gc = _read_json(phase2gc_report_json)
    phase2gb_report_json = phase2gc.get("evidence", {}).get("phase2gb_report_json")
    phase2ga_report_json = phase2gc.get("evidence", {}).get("phase2ga_report_json")
    if not phase2gb_report_json or not phase2ga_report_json:
        raise ValueError("Phase2GE requires Phase2GC source report evidence paths")
    phase2gb = _read_json(phase2gb_report_json)
    phase2ga = _read_json(phase2ga_report_json)
    phase2gc_validation = validate_phase2gc_replay_bundle_summary_replay_bundle(
        phase2gc
    )
    source_reports = [
        _file_entry(role="phase2ga_report", path=phase2ga_report_json, content_type="application/json"),
        _file_entry(
            role="phase2gb_negative_control_report",
            path=phase2gb_report_json,
            content_type="application/json",
        ),
        _file_entry(
            role="phase2gc_replay_bundle_report",
            path=phase2gc_report_json,
            content_type="application/json",
        ),
        _file_entry(
            role="phase2gd_negative_control_report",
            path=phase2gd_report_json,
            content_type="application/json",
        ),
    ]
    bundle_artifacts = _bundle_artifact_entries(phase2gc)
    reproduction_steps = _reproduction_steps(
        phase2ga=phase2ga,
        phase2gb=phase2gb,
        phase2gc=phase2gc,
        phase2ga_report_json=phase2ga_report_json,
        phase2gb_report_json=phase2gb_report_json,
        phase2gc_report_json=phase2gc_report_json,
        phase2gd_report_json=phase2gd_report_json,
    )
    manifest = {
        "artifact_family": "phase2ge_replay_bundle_summary_reproducibility_manifest",
        "manifest_schema_version": 1,
        "source_reports": source_reports,
        "bundle_artifacts": bundle_artifacts,
        "reproduction_steps": reproduction_steps,
        "claim_boundary": (
            "bounded package-internal structured runtime evidence only; not "
            "free-form shell autonomy, not general runtime invariance, not "
            "open-ended native perception, not production autonomy, and not an "
            "epoch-making architecture"
        ),
    }
    output_manifest_path = Path(output_manifest_json)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_manifest_path, manifest)
    checks = {
        "source_phase2gd_passed": phase2gd.get("passed") is True,
        "source_phase2gc_passed": phase2gc.get("passed") is True,
        "source_phase2gc_validation_passed": phase2gc_validation.get("passed") is True,
        "source_phase2gd_negative_controls_complete": (
            phase2gd.get("metrics", {}).get("negative_control_count")
            == phase2gd.get("metrics", {}).get("negative_controls_failed")
        ),
        "source_reports_readable_and_hashed": _entries_exist(source_reports)
        and _entry_hashes_match(source_reports),
        "bundle_artifact_roles_complete": set(REQUIRED_ARTIFACT_ROLES).issubset(
            _entry_roles(bundle_artifacts)
        ),
        "bundle_artifacts_present": _entries_exist(bundle_artifacts),
        "bundle_artifact_hashes_match": all(
            item.get("hash_matches_bundle_manifest") is True
            for item in bundle_artifacts
        ),
        "control_artifact_count_preserved": _control_artifact_count(bundle_artifacts)
        == phase2gc.get("metrics", {}).get("copied_control_artifact_count"),
        "required_reproduction_steps_present": set(REQUIRED_REPRODUCTION_STEPS).issubset(
            {item["step_id"] for item in reproduction_steps}
        ),
        "reproduction_steps_have_modules_and_paths": all(
            item.get("module")
            and isinstance(item.get("inputs"), dict)
            and isinstance(item.get("outputs"), dict)
            for item in reproduction_steps
        ),
        "reproducibility_manifest_written": output_manifest_path.exists(),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ge_replay_bundle_summary_reproducibility_manifest",
        "passed": passed,
        "ready_for_bounded_reproducibility_manifest_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "source_report_count": len(source_reports),
            "bundle_artifact_count": len(bundle_artifacts),
            "bundle_artifact_hash_match_count": sum(
                item.get("hash_matches_bundle_manifest") is True
                for item in bundle_artifacts
            ),
            "control_artifact_count": _control_artifact_count(bundle_artifacts),
            "reproduction_step_count": len(reproduction_steps),
            "phase2gd_negative_control_count": phase2gd.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2gd_negative_controls_failed": phase2gd.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "source_summary": {
            "phase2gc_passed": phase2gc.get("passed") is True,
            "phase2gd_passed": phase2gd.get("passed") is True,
            "phase2gc_validation_passed": phase2gc_validation.get("passed") is True,
            "phase2gc_copied_control_artifact_count": phase2gc.get("metrics", {}).get(
                "copied_control_artifact_count"
            ),
            "phase2gd_negative_control_count": phase2gd.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2gd_negative_controls_failed": phase2gd.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "supported_claims": [
            (
                "reproducibility manifest for the bounded Phase2GC replay bundle "
                "and Phase2GD negative-control evidence"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "free-form shell autonomy",
            "arbitrary shell/environment generalization",
            "general runtime invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2gf_replay_bundle_summary_manifest_negative_controls"
            if passed
            else "repair_phase2ge_replay_bundle_summary_reproducibility_manifest"
        ),
        "evidence": {
            "phase2gd_report_json": str(phase2gd_report_json),
            "phase2gc_report_json": str(phase2gc_report_json),
            "phase2gb_report_json": str(phase2gb_report_json),
            "phase2ga_report_json": str(phase2ga_report_json),
            "reproducibility_manifest": str(output_manifest_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2GE replay bundle summary reproducibility manifest."
    )
    parser.add_argument("--phase2gd-report-json", required=True)
    parser.add_argument("--output-manifest-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ge_replay_bundle_summary_reproducibility_manifest(
        phase2gd_report_json=args.phase2gd_report_json,
        output_manifest_json=args.output_manifest_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
