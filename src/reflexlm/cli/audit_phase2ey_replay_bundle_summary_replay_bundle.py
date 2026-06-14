from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2dk_latex_candidate_publication_bundle import _sha256
from reflexlm.cli.audit_phase2ew_replay_bundle_summary_cross_directory_replay import (
    validate_phase2ew_replay_bundle_summary_cross_directory_replay,
)


REQUIRED_ARTIFACT_ROLES: tuple[str, ...] = (
    "replayed_bundle_summary_markdown",
    "replayed_phase2eu_report",
    "replayed_phase2eu_validation",
    "replayed_control_results_json",
    "phase2ew_report",
    "phase2ex_negative_control_report",
    "bundle_readme",
)

OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_general_shell_autonomy_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _copy_artifact(
    *,
    role: str,
    source_path: str | Path,
    target_path: Path,
    content_type: str,
) -> dict[str, Any]:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"missing source artifact for {role}: {source}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_path)
    return {
        "role": role,
        "path": str(target_path),
        "source_path": str(source),
        "content_type": content_type,
        "bytes": target_path.stat().st_size,
        "sha256": _sha256(target_path),
    }


def _control_artifact_entries(
    *,
    phase2ew: dict[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    artifacts = []
    rows = phase2ew.get("replayed_control_results", [])
    if not isinstance(rows, list):
        return artifacts
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        control_id = str(row.get("control_id", f"control_{index:02d}"))
        case_dir = output_root / "control_results" / f"c{index:02d}_{control_id}"
        artifacts.append(
            _copy_artifact(
                role="replayed_control_report",
                source_path=row.get("replayed_control_report_json", ""),
                target_path=case_dir / "phase2eu_control_report.json",
                content_type="application/json",
            )
        )
        artifacts[-1]["control_id"] = control_id
        artifacts.append(
            _copy_artifact(
                role="replayed_control_validation",
                source_path=row.get("replayed_validation_report_json", ""),
                target_path=case_dir / "phase2eu_validation.json",
                content_type="application/json",
            )
        )
        artifacts[-1]["control_id"] = control_id
    return artifacts


def _write_readme(path: Path, *, phase2ew: dict[str, Any], phase2ex: dict[str, Any]) -> dict[str, Any]:
    text = "\n".join(
        [
            "# Phase2EY Replay Bundle Summary Replay Bundle",
            "",
            "This bundle contains bounded cross-directory replay evidence for "
            "the Phase2EU replay-bundle summary portability summary and the "
            "Phase2EX negative-control audit of the Phase2EW replay gate.",
            "",
            "Included artifacts:",
            "- replayed Phase2EU summary Markdown",
            "- replayed Phase2EU report",
            "- replayed Phase2EU validation",
            "- replayed Phase2EV control result summary",
            "- copied Phase2EV control reports and validations",
            "- Phase2EW source report",
            "- Phase2EX negative-control report",
            "",
            "Boundary:",
            "- not free-form shell autonomy",
            "- not general runtime invariance",
            "- not open-ended native perception",
            "- not production autonomy",
            "- not an epoch-making architecture",
            "",
            f"Phase2EW passed: {phase2ew.get('passed') is True}",
            f"Phase2EX passed: {phase2ex.get('passed') is True}",
            "Phase2EW replayed summary rows: "
            f"{phase2ew.get('metrics', {}).get('replayed_summary_row_count')}",
            "Phase2EX negative controls failed: "
            f"{phase2ex.get('metrics', {}).get('negative_controls_failed')}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "role": "bundle_readme",
        "path": str(path),
        "source_path": None,
        "content_type": "text/markdown",
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _manifest_roles(manifest: dict[str, Any]) -> set[str]:
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        return set()
    return {str(item.get("role")) for item in artifacts if isinstance(item, dict)}


def validate_phase2ey_replay_bundle_summary_replay_bundle(
    report: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = report.get("evidence", {}).get("bundle_manifest")
    manifest_readable = False
    manifest: dict[str, Any] = {}
    if manifest_path:
        try:
            manifest = _read_json(manifest_path)
            manifest_readable = True
        except (OSError, json.JSONDecodeError):
            manifest_readable = False
    artifacts = manifest.get("artifacts", []) if isinstance(manifest, dict) else []
    if not isinstance(artifacts, list):
        artifacts = []
    existing_artifacts = [
        item
        for item in artifacts
        if isinstance(item, dict) and Path(str(item.get("path", ""))).exists()
    ]
    hash_matches = []
    for item in existing_artifacts:
        path = Path(str(item["path"]))
        hash_matches.append(_sha256(path) == item.get("sha256"))
    readme_text = ""
    readme_entries = [item for item in artifacts if item.get("role") == "bundle_readme"]
    if readme_entries:
        readme = Path(str(readme_entries[0].get("path", "")))
        if readme.exists():
            readme_text = readme.read_text(encoding="utf-8")
    control_artifacts = [
        item
        for item in artifacts
        if item.get("role") in {"replayed_control_report", "replayed_control_validation"}
    ]
    checks = {
        "artifact_family_matches_phase2ey": (
            report.get("artifact_family")
            == "phase2ey_replay_bundle_summary_replay_bundle"
        ),
        "top_level_phase2ey_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_replay_bundle_claim") is True
            and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2ew_and_phase2ex_passed": (
            report.get("source_summary", {}).get("phase2ew_passed") is True
            and report.get("source_summary", {}).get("phase2ex_passed") is True
        ),
        "source_phase2ew_validation_passed": (
            report.get("source_summary", {}).get("phase2ew_validation_passed") is True
        ),
        "source_negative_controls_complete": (
            report.get("source_summary", {}).get("phase2ex_negative_control_count")
            == report.get("source_summary", {}).get("phase2ex_negative_controls_failed")
        ),
        "manifest_readable": manifest_readable,
        "manifest_roles_complete": set(REQUIRED_ARTIFACT_ROLES).issubset(
            _manifest_roles(manifest)
        ),
        "all_manifest_artifacts_exist": len(existing_artifacts) == len(artifacts)
        and len(artifacts) >= len(REQUIRED_ARTIFACT_ROLES),
        "all_manifest_hashes_match": bool(hash_matches) and all(hash_matches),
        "readme_contains_bounded_boundary": (
            "not free-form shell autonomy" in readme_text
            and "not an epoch-making architecture" in readme_text
        ),
        "replay_artifact_count_preserved": (
            report.get("metrics", {}).get("replay_artifact_count") == 4
        ),
        "control_artifact_count_preserved": len(control_artifacts)
        == report.get("metrics", {}).get("copied_control_artifact_count"),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "manifest_entry_count": len(artifacts),
            "existing_artifact_count": len(existing_artifacts),
            "hash_match_count": sum(hash_matches),
            "control_artifact_count": len(control_artifacts),
        },
    }


def audit_phase2ey_replay_bundle_summary_replay_bundle(
    *,
    phase2ex_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2ex = _read_json(phase2ex_report_json)
    phase2ew_report_json = phase2ex.get("evidence", {}).get("phase2ew_report_json")
    if not phase2ew_report_json:
        raise ValueError("Phase2EY requires Phase2EX evidence.phase2ew_report_json")
    phase2ew = _read_json(phase2ew_report_json)
    phase2ew_validation = validate_phase2ew_replay_bundle_summary_cross_directory_replay(
        phase2ew
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _copy_artifact(
            role="replayed_bundle_summary_markdown",
            source_path=phase2ew["evidence"]["replayed_markdown"],
            target_path=output_root / "phase2eu_replayed_bundle_portability_summary.md",
            content_type="text/markdown",
        ),
        _copy_artifact(
            role="replayed_phase2eu_report",
            source_path=phase2ew["evidence"]["replayed_phase2eu_report"],
            target_path=output_root / "phase2eu_replayed_report.json",
            content_type="application/json",
        ),
        _copy_artifact(
            role="replayed_phase2eu_validation",
            source_path=phase2ew["evidence"]["replayed_phase2eu_validation"],
            target_path=output_root / "phase2eu_replayed_validation.json",
            content_type="application/json",
        ),
        _copy_artifact(
            role="replayed_control_results_json",
            source_path=phase2ew["evidence"]["replayed_control_results_json"],
            target_path=output_root / "phase2ev_replayed_control_results.json",
            content_type="application/json",
        ),
        _copy_artifact(
            role="phase2ew_report",
            source_path=phase2ew_report_json,
            target_path=output_root / "phase2ew_replay_bundle_summary_cross_directory_replay.json",
            content_type="application/json",
        ),
        _copy_artifact(
            role="phase2ex_negative_control_report",
            source_path=phase2ex_report_json,
            target_path=output_root / "phase2ex_replay_bundle_summary_replay_negative_controls.json",
            content_type="application/json",
        ),
    ]
    control_artifacts = _control_artifact_entries(
        phase2ew=phase2ew,
        output_root=output_root,
    )
    artifacts.extend(control_artifacts)
    artifacts.append(_write_readme(output_root / "README.md", phase2ew=phase2ew, phase2ex=phase2ex))
    manifest = {
        "artifact_family": "phase2ey_replay_bundle_summary_replay_bundle_manifest",
        "bundle_schema_version": 1,
        "artifacts": artifacts,
        "claim_boundary": (
            "bounded package-internal structured runtime evidence only; not "
            "free-form shell autonomy, not general runtime invariance, not "
            "open-ended native perception, not production autonomy, and not an "
            "epoch-making architecture"
        ),
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    checks = {
        "source_phase2ew_passed": phase2ew.get("passed") is True,
        "source_phase2ex_passed": phase2ex.get("passed") is True,
        "source_phase2ew_validation_passed": phase2ew_validation.get("passed") is True,
        "source_phase2ex_negative_controls_complete": (
            phase2ex.get("metrics", {}).get("negative_control_count")
            == phase2ex.get("metrics", {}).get("negative_controls_failed")
        ),
        "required_artifact_roles_present": set(REQUIRED_ARTIFACT_ROLES).issubset(
            {item["role"] for item in artifacts}
        ),
        "all_bundle_artifacts_written": all(Path(item["path"]).exists() for item in artifacts),
        "all_bundle_artifact_hashes_recorded": all(
            isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64
            for item in artifacts
        ),
        "control_artifacts_copied": len(control_artifacts)
        == phase2ew.get("metrics", {}).get("replayed_control_count", 0) * 2,
        "manifest_written": manifest_path.exists(),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ey_replay_bundle_summary_replay_bundle",
        "passed": passed,
        "ready_for_bounded_replay_bundle_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "bundle_artifact_count": len(artifacts),
            "manifest_entry_count": len(manifest["artifacts"]),
            "replay_artifact_count": 4,
            "copied_control_artifact_count": len(control_artifacts),
            "phase2ew_replayed_summary_row_count": phase2ew.get("metrics", {}).get(
                "replayed_summary_row_count"
            ),
            "phase2ew_replayed_control_count": phase2ew.get("metrics", {}).get(
                "replayed_control_count"
            ),
            "phase2ex_negative_control_count": phase2ex.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2ex_negative_controls_failed": phase2ex.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "source_summary": {
            "phase2ew_passed": phase2ew.get("passed") is True,
            "phase2ex_passed": phase2ex.get("passed") is True,
            "phase2ew_validation_passed": phase2ew_validation.get("passed") is True,
            "phase2ex_negative_control_count": phase2ex.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2ex_negative_controls_failed": phase2ex.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "supported_claims": [
            (
                "artifact-level replay bundle for the bounded Phase2EW replay-bundle "
                "summary cross-directory replay and Phase2EX replay negative-control "
                "evidence, including copied EV control reports and validations"
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
            "phase2ez_replay_bundle_summary_replay_bundle_negative_controls"
            if passed
            else "repair_phase2ey_replay_bundle_summary_replay_bundle"
        ),
        "evidence": {
            "phase2ex_report_json": str(phase2ex_report_json),
            "phase2ew_report_json": str(phase2ew_report_json),
            "bundle_dir": str(output_root),
            "bundle_manifest": str(manifest_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2EY replay bundle summary replay bundle."
    )
    parser.add_argument("--phase2ex-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ey_replay_bundle_summary_replay_bundle(
        phase2ex_report_json=args.phase2ex_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
