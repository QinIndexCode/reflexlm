from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2di_publication_table_latex_candidate import (
    _is_under_docs_tables,
    validate_phase2di_publication_table_latex_candidate,
)


REQUIRED_ARTIFACT_ROLES: tuple[str, ...] = (
    "latex_candidate",
    "compact_markdown_table",
    "phase2di_report",
    "phase2dj_negative_control_report",
    "bundle_readme",
)

OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_general_shell_autonomy_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _write_readme(path: Path, *, phase2di: dict[str, Any], phase2dj: dict[str, Any]) -> dict[str, Any]:
    text = "\n".join(
        [
            "# Phase2DK LaTeX Candidate Publication Bundle",
            "",
            "This bundle contains an artifact-level candidate table for bounded "
            "package-internal structured runtime evidence.",
            "",
            "Included artifacts:",
            "- Phase2DI LaTeX candidate table",
            "- Phase2DG compact Markdown table",
            "- Phase2DI source report",
            "- Phase2DJ negative-control report",
            "",
            "Boundary:",
            "- not free-form shell autonomy",
            "- not general runtime invariance",
            "- not open-ended native perception",
            "- not production autonomy",
            "- not an epoch-making architecture",
            "",
            f"Phase2DI passed: {phase2di.get('passed') is True}",
            f"Phase2DJ passed: {phase2dj.get('passed') is True}",
            f"Phase2DI row count: {phase2di.get('metrics', {}).get('row_count')}",
            "Phase2DJ negative controls failed: "
            f"{phase2dj.get('metrics', {}).get('negative_controls_failed')}",
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


def validate_phase2dk_latex_candidate_publication_bundle(
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
    latex_entries = [item for item in artifacts if item.get("role") == "latex_candidate"]
    latex_not_in_main_tables = bool(latex_entries) and all(
        not _is_under_docs_tables(str(item.get("path", ""))) for item in latex_entries
    )
    checks = {
        "artifact_family_matches_phase2dk": (
            report.get("artifact_family")
            == "phase2dk_latex_candidate_publication_bundle"
        ),
        "top_level_phase2dk_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_publication_bundle_claim") is True
            and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2di_and_phase2dj_passed": (
            report.get("source_summary", {}).get("phase2di_passed") is True
            and report.get("source_summary", {}).get("phase2dj_passed") is True
        ),
        "manifest_readable": manifest_readable,
        "manifest_roles_complete": set(REQUIRED_ARTIFACT_ROLES).issubset(
            _manifest_roles(manifest)
        ),
        "all_manifest_artifacts_exist": len(existing_artifacts) == len(artifacts)
        and len(artifacts) >= len(REQUIRED_ARTIFACT_ROLES),
        "all_manifest_hashes_match": bool(hash_matches) and all(hash_matches),
        "latex_candidate_not_in_main_tables_dir": latex_not_in_main_tables,
        "readme_contains_bounded_boundary": (
            "not free-form shell autonomy" in readme_text
            and "not an epoch-making architecture" in readme_text
        ),
        "source_negative_controls_complete": (
            report.get("source_summary", {}).get("phase2dj_negative_control_count")
            == report.get("source_summary", {}).get("phase2dj_negative_controls_failed")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "manifest_entry_count": len(artifacts),
            "existing_artifact_count": len(existing_artifacts),
            "hash_match_count": sum(hash_matches),
        },
    }


def audit_phase2dk_latex_candidate_publication_bundle(
    *,
    phase2dj_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2dj = _read_json(phase2dj_report_json)
    phase2di_report_json = phase2dj.get("evidence", {}).get("phase2di_report_json")
    if not phase2di_report_json:
        raise ValueError("Phase2DK requires Phase2DJ evidence.phase2di_report_json")
    phase2di = _read_json(phase2di_report_json)
    phase2dg_report_json = phase2di.get("evidence", {}).get("phase2dg_report_json")
    if not phase2dg_report_json:
        raise ValueError("Phase2DK requires Phase2DI evidence.phase2dg_report_json")
    phase2dg = _read_json(phase2dg_report_json)
    markdown_path = phase2dg.get("evidence", {}).get("publication_table_markdown")
    latex_path = phase2di.get("evidence", {}).get("latex_candidate_path")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _copy_artifact(
            role="latex_candidate",
            source_path=latex_path,
            target_path=output_root / "phase2di_publication_table_latex_candidate.tex",
            content_type="text/x-tex",
        ),
        _copy_artifact(
            role="compact_markdown_table",
            source_path=markdown_path,
            target_path=output_root / "phase2dg_compact_rollup_publication_table.md",
            content_type="text/markdown",
        ),
        _copy_artifact(
            role="phase2di_report",
            source_path=phase2di_report_json,
            target_path=output_root / "phase2di_publication_table_latex_candidate.json",
            content_type="application/json",
        ),
        _copy_artifact(
            role="phase2dj_negative_control_report",
            source_path=phase2dj_report_json,
            target_path=output_root / "phase2dj_latex_candidate_negative_controls.json",
            content_type="application/json",
        ),
        _write_readme(output_root / "README.md", phase2di=phase2di, phase2dj=phase2dj),
    ]
    manifest = {
        "artifact_family": "phase2dk_latex_candidate_publication_bundle_manifest",
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
    phase2di_validation = validate_phase2di_publication_table_latex_candidate(phase2di)
    checks = {
        "source_phase2di_passed": phase2di.get("passed") is True,
        "source_phase2dj_passed": phase2dj.get("passed") is True,
        "source_phase2di_validation_passed": phase2di_validation.get("passed") is True,
        "source_phase2dj_negative_controls_complete": (
            phase2dj.get("metrics", {}).get("negative_control_count")
            == phase2dj.get("metrics", {}).get("negative_controls_failed")
        ),
        "required_artifact_roles_present": set(REQUIRED_ARTIFACT_ROLES).issubset(
            {item["role"] for item in artifacts}
        ),
        "all_bundle_artifacts_written": all(Path(item["path"]).exists() for item in artifacts),
        "all_bundle_artifact_hashes_recorded": all(
            isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64
            for item in artifacts
        ),
        "latex_candidate_not_in_main_tables_dir": not _is_under_docs_tables(
            output_root / "phase2di_publication_table_latex_candidate.tex"
        ),
        "manifest_written": manifest_path.exists(),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2dk_latex_candidate_publication_bundle",
        "passed": passed,
        "ready_for_bounded_publication_bundle_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "bundle_artifact_count": len(artifacts),
            "manifest_entry_count": len(manifest["artifacts"]),
            "phase2di_row_count": phase2di.get("metrics", {}).get("row_count"),
            "phase2dj_negative_control_count": phase2dj.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2dj_negative_controls_failed": phase2dj.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "source_summary": {
            "phase2di_passed": phase2di.get("passed") is True,
            "phase2dj_passed": phase2dj.get("passed") is True,
            "phase2di_row_count": phase2di.get("metrics", {}).get("row_count"),
            "phase2dj_negative_control_count": phase2dj.get("metrics", {}).get(
                "negative_control_count"
            ),
            "phase2dj_negative_controls_failed": phase2dj.get("metrics", {}).get(
                "negative_controls_failed"
            ),
        },
        "supported_claims": [
            (
                "artifact-level publication bundle for the bounded Phase2DI "
                "LaTeX candidate and Phase2DJ negative-control evidence"
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
            "phase2dl_publication_bundle_negative_controls"
            if passed
            else "repair_phase2dk_latex_candidate_publication_bundle"
        ),
        "evidence": {
            "phase2dj_report_json": str(phase2dj_report_json),
            "phase2di_report_json": str(phase2di_report_json),
            "phase2dg_report_json": str(phase2dg_report_json),
            "bundle_dir": str(output_root),
            "bundle_manifest": str(manifest_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2DK LaTeX candidate publication bundle."
    )
    parser.add_argument("--phase2dj-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2dk_latex_candidate_publication_bundle(
        phase2dj_report_json=args.phase2dj_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
