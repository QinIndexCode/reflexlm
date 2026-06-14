from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2dk_latex_candidate_publication_bundle import (
    REQUIRED_ARTIFACT_ROLES,
    _sha256,
    validate_phase2dk_latex_candidate_publication_bundle,
)


REQUIRED_SOURCE_REPORT_ROLES: tuple[str, ...] = (
    "phase2di_report",
    "phase2dj_negative_control_report",
    "phase2dk_bundle_report",
    "phase2dl_negative_control_report",
)

REQUIRED_REPRODUCTION_STEPS: tuple[str, ...] = (
    "phase2di_publication_table_latex_candidate",
    "phase2dj_latex_candidate_negative_controls",
    "phase2dk_latex_candidate_publication_bundle",
    "phase2dl_publication_bundle_negative_controls",
)

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


def _entry_roles(entries: Any) -> set[str]:
    if not isinstance(entries, list):
        return set()
    return {str(item.get("role")) for item in entries if isinstance(item, dict)}


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


def _bundle_artifact_entries(phase2dk: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = phase2dk.get("evidence", {}).get("bundle_manifest")
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        return []
    entries = []
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path", "")))
        if not path.exists():
            entries.append(
                {
                    "role": item.get("role"),
                    "path": str(path),
                    "content_type": item.get("content_type"),
                    "bytes": item.get("bytes"),
                    "sha256": item.get("sha256"),
                    "current_sha256": None,
                    "hash_matches_bundle_manifest": False,
                }
            )
            continue
        current_sha256 = _sha256(path)
        entries.append(
            {
                "role": item.get("role"),
                "path": str(path),
                "content_type": item.get("content_type"),
                "bytes": path.stat().st_size,
                "sha256": item.get("sha256"),
                "current_sha256": current_sha256,
                "hash_matches_bundle_manifest": current_sha256 == item.get("sha256"),
            }
        )
    return entries


def _reproduction_steps(
    *,
    phase2di: dict[str, Any],
    phase2dj: dict[str, Any],
    phase2dk: dict[str, Any],
    phase2dl: dict[str, Any],
    phase2dk_report_json: str | Path,
    phase2dl_report_json: str | Path,
) -> list[dict[str, Any]]:
    phase2di_report = phase2dj.get("evidence", {}).get("phase2di_report_json")
    return [
        {
            "step_id": "phase2di_publication_table_latex_candidate",
            "module": "reflexlm.cli.audit_phase2di_publication_table_latex_candidate",
            "inputs": {
                "phase2dh_report_json": phase2di.get("evidence", {}).get(
                    "phase2dh_report_json"
                )
            },
            "outputs": {
                "output_report_json": phase2di_report,
                "output_latex": phase2di.get("evidence", {}).get(
                    "latex_candidate_path"
                ),
            },
        },
        {
            "step_id": "phase2dj_latex_candidate_negative_controls",
            "module": "reflexlm.cli.audit_phase2dj_latex_candidate_negative_controls",
            "inputs": {"phase2di_report_json": phase2di_report},
            "outputs": {
                "output_report_json": phase2dk.get("evidence", {}).get(
                    "phase2dj_report_json"
                ),
                "output_dir": phase2dj.get("evidence", {}).get(
                    "negative_control_output_dir"
                ),
            },
        },
        {
            "step_id": "phase2dk_latex_candidate_publication_bundle",
            "module": "reflexlm.cli.audit_phase2dk_latex_candidate_publication_bundle",
            "inputs": {
                "phase2dj_report_json": phase2dk.get("evidence", {}).get(
                    "phase2dj_report_json"
                )
            },
            "outputs": {
                "output_report_json": str(phase2dk_report_json),
                "output_dir": phase2dk.get("evidence", {}).get("bundle_dir"),
            },
        },
        {
            "step_id": "phase2dl_publication_bundle_negative_controls",
            "module": "reflexlm.cli.audit_phase2dl_publication_bundle_negative_controls",
            "inputs": {"phase2dk_report_json": str(phase2dk_report_json)},
            "outputs": {
                "output_report_json": str(phase2dl_report_json),
                "output_dir": phase2dl.get("evidence", {}).get(
                    "negative_control_output_dir"
                ),
            },
        },
    ]


def validate_phase2dm_publication_bundle_reproducibility_manifest(
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
    source_reports = manifest.get("source_reports", [])
    if not isinstance(source_reports, list):
        source_reports = []
    bundle_artifacts = manifest.get("bundle_artifacts", [])
    if not isinstance(bundle_artifacts, list):
        bundle_artifacts = []
    reproduction_steps = manifest.get("reproduction_steps", [])
    if not isinstance(reproduction_steps, list):
        reproduction_steps = []
    bundle_artifact_hash_matches = [
        item.get("hash_matches_bundle_manifest") is True
        and Path(str(item.get("path", ""))).exists()
        and _sha256(Path(str(item.get("path")))) == item.get("current_sha256")
        for item in bundle_artifacts
        if isinstance(item, dict)
    ]
    checks = {
        "artifact_family_matches_phase2dm": (
            report.get("artifact_family")
            == "phase2dm_publication_bundle_reproducibility_manifest"
        ),
        "top_level_phase2dm_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_reproducibility_manifest_claim") is True
            and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2dk_and_phase2dl_passed": (
            report.get("source_summary", {}).get("phase2dk_passed") is True
            and report.get("source_summary", {}).get("phase2dl_passed") is True
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
        "reproduction_steps_complete": set(REQUIRED_REPRODUCTION_STEPS).issubset(
            {
                str(item.get("step_id"))
                for item in reproduction_steps
                if isinstance(item, dict)
            }
        ),
        "reproduction_steps_have_modules_and_paths": bool(reproduction_steps)
        and all(
            item.get("module")
            and isinstance(item.get("inputs"), dict)
            and isinstance(item.get("outputs"), dict)
            for item in reproduction_steps
            if isinstance(item, dict)
        ),
        "manifest_contains_bounded_boundary": (
            "not free-form shell autonomy" in manifest.get("claim_boundary", "")
            and "not an epoch-making architecture"
            in manifest.get("claim_boundary", "")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "source_report_count": len(source_reports),
            "bundle_artifact_count": len(bundle_artifacts),
            "bundle_artifact_hash_match_count": sum(bundle_artifact_hash_matches),
            "reproduction_step_count": len(reproduction_steps),
        },
    }


def audit_phase2dm_publication_bundle_reproducibility_manifest(
    *,
    phase2dl_report_json: str | Path,
    output_manifest_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2dl = _read_json(phase2dl_report_json)
    phase2dk_report_json = phase2dl.get("evidence", {}).get("phase2dk_report_json")
    if not phase2dk_report_json:
        raise ValueError("Phase2DM requires Phase2DL evidence.phase2dk_report_json")
    phase2dk = _read_json(phase2dk_report_json)
    phase2dj_report_json = phase2dk.get("evidence", {}).get("phase2dj_report_json")
    phase2di_report_json = phase2dk.get("evidence", {}).get("phase2di_report_json")
    if not phase2dj_report_json or not phase2di_report_json:
        raise ValueError("Phase2DM requires Phase2DK source report evidence paths")
    phase2dj = _read_json(phase2dj_report_json)
    phase2di = _read_json(phase2di_report_json)
    source_reports = [
        _file_entry(
            role="phase2di_report",
            path=phase2di_report_json,
            content_type="application/json",
        ),
        _file_entry(
            role="phase2dj_negative_control_report",
            path=phase2dj_report_json,
            content_type="application/json",
        ),
        _file_entry(
            role="phase2dk_bundle_report",
            path=phase2dk_report_json,
            content_type="application/json",
        ),
        _file_entry(
            role="phase2dl_negative_control_report",
            path=phase2dl_report_json,
            content_type="application/json",
        ),
    ]
    bundle_artifacts = _bundle_artifact_entries(phase2dk)
    reproduction_steps = _reproduction_steps(
        phase2di=phase2di,
        phase2dj=phase2dj,
        phase2dk=phase2dk,
        phase2dl=phase2dl,
        phase2dk_report_json=phase2dk_report_json,
        phase2dl_report_json=phase2dl_report_json,
    )
    manifest = {
        "artifact_family": "phase2dm_publication_bundle_reproducibility_manifest",
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
    phase2dk_validation = validate_phase2dk_latex_candidate_publication_bundle(
        phase2dk
    )
    checks = {
        "source_phase2dl_passed": phase2dl.get("passed") is True,
        "source_phase2dk_passed": phase2dk.get("passed") is True,
        "source_phase2dk_validation_passed": phase2dk_validation.get("passed")
        is True,
        "source_phase2dl_negative_controls_complete": (
            phase2dl.get("metrics", {}).get("negative_control_count")
            == phase2dl.get("metrics", {}).get("negative_controls_failed")
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
        "required_reproduction_steps_present": set(
            REQUIRED_REPRODUCTION_STEPS
        ).issubset({item["step_id"] for item in reproduction_steps}),
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
        "artifact_family": "phase2dm_publication_bundle_reproducibility_manifest",
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
            "reproduction_step_count": len(reproduction_steps),
            "phase2dl_negative_control_count": phase2dl.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2dl_negative_controls_failed": phase2dl.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "source_summary": {
            "phase2dk_passed": phase2dk.get("passed") is True,
            "phase2dl_passed": phase2dl.get("passed") is True,
            "phase2dl_negative_control_count": phase2dl.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2dl_negative_controls_failed": phase2dl.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "supported_claims": [
            (
                "reproducibility manifest for the bounded Phase2DK publication "
                "bundle and Phase2DL negative-control evidence"
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
            "phase2dn_reproducibility_manifest_negative_controls"
            if passed
            else "repair_phase2dm_publication_bundle_reproducibility_manifest"
        ),
        "evidence": {
            "phase2dl_report_json": str(phase2dl_report_json),
            "phase2dk_report_json": str(phase2dk_report_json),
            "phase2dj_report_json": str(phase2dj_report_json),
            "phase2di_report_json": str(phase2di_report_json),
            "reproducibility_manifest": str(output_manifest_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2DM publication bundle reproducibility manifest."
    )
    parser.add_argument("--phase2dl-report-json", required=True)
    parser.add_argument("--output-manifest-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2dm_publication_bundle_reproducibility_manifest(
        phase2dl_report_json=args.phase2dl_report_json,
        output_manifest_json=args.output_manifest_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
