from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any, Callable

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2gc_replay_bundle_summary_replay_bundle import (
    validate_phase2gc_replay_bundle_summary_replay_bundle,
)


Mutation = Callable[[dict[str, Any], Path], None]


def _manifest_path(report: dict[str, Any]) -> Path:
    return Path(str(report.get("evidence", {}).get("bundle_manifest", "")))


def _read_manifest(report: dict[str, Any]) -> dict[str, Any]:
    return _read_json(_manifest_path(report))


def _write_manifest(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    _write_json(_manifest_path(report), manifest)


def _manifest_artifacts(report: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = _read_manifest(report)
    artifacts = manifest.get("artifacts", [])
    return [item for item in artifacts if isinstance(item, dict)] if isinstance(artifacts, list) else []


def _artifact_by_role(report: dict[str, Any], role: str) -> dict[str, Any]:
    for item in _manifest_artifacts(report):
        if item.get("role") == role:
            return item
    raise ValueError(f"missing GC bundle artifact role: {role}")


def _first_control_artifact(report: dict[str, Any], role: str) -> dict[str, Any]:
    for item in _manifest_artifacts(report):
        if item.get("role") == role:
            return item
    raise ValueError(f"missing GC control artifact role: {role}")


def _materialize_control_report(
    *,
    phase2gc_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    control_report = deepcopy(phase2gc_report)
    source_manifest_path = _manifest_path(phase2gc_report)
    if not source_manifest_path.exists():
        raise ValueError("Phase2GD requires a readable Phase2GC bundle manifest")
    source_manifest = _read_json(source_manifest_path)
    source_artifacts = source_manifest.get("artifacts", [])
    if not isinstance(source_artifacts, list):
        raise ValueError("Phase2GD requires a list-valued Phase2GC manifest.artifacts")
    source_bundle_dir = source_manifest_path.parent
    bundle_dir = case_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied_artifacts = []
    for item in source_artifacts:
        if not isinstance(item, dict):
            continue
        copied = deepcopy(item)
        source_path = item.get("path")
        if source_path is not None:
            source = Path(str(source_path))
            if not source.exists():
                raise ValueError(f"Phase2GD source artifact missing: {source}")
            try:
                relative = source.relative_to(source_bundle_dir)
            except ValueError:
                relative = Path(source.name)
            target = bundle_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied["path"] = str(target)
        copied_artifacts.append(copied)
    control_manifest = deepcopy(source_manifest)
    control_manifest["artifacts"] = copied_artifacts
    target_manifest_path = bundle_dir / "manifest.json"
    _write_json(target_manifest_path, control_manifest)
    control_report.setdefault("evidence", {})["bundle_dir"] = str(bundle_dir)
    control_report.setdefault("evidence", {})["bundle_manifest"] = str(
        target_manifest_path
    )
    return control_report


def _mutate_source_summary_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2ga_passed"] = False


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["all_bundle_artifacts_written"] = False


def _mutate_source_phase2ga_validation_failed(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2ga_validation_passed"] = False


def _mutate_negative_controls_incomplete(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2gb_negative_controls_failed"] = 0


def _mutate_missing_manifest(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    manifest = _manifest_path(report)
    if manifest.exists():
        manifest.unlink()


def _mutate_manifest_missing_role(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    manifest = _read_manifest(report)
    manifest["artifacts"] = [
        item
        for item in manifest.get("artifacts", [])
        if item.get("role") != "replayed_phase2fy_validation"
    ]
    _write_manifest(report, manifest)


def _mutate_manifest_artifacts_not_list(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    manifest = _read_manifest(report)
    manifest["artifacts"] = {"role": "replayed_bundle_summary_markdown"}
    _write_manifest(report, manifest)


def _mutate_missing_replayed_markdown(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    markdown = Path(str(_artifact_by_role(report, "replayed_bundle_summary_markdown")["path"]))
    if markdown.exists():
        markdown.unlink()


def _mutate_tampered_replayed_markdown(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    markdown = Path(str(_artifact_by_role(report, "replayed_bundle_summary_markdown")["path"]))
    markdown.write_text("tampered\n", encoding="utf-8")


def _mutate_missing_control_artifact(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    artifact = Path(str(_first_control_artifact(report, "replayed_control_validation")["path"]))
    if artifact.exists():
        artifact.unlink()


def _mutate_tampered_control_artifact(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    artifact = Path(str(_first_control_artifact(report, "replayed_control_report")["path"]))
    artifact.write_text("tampered\n", encoding="utf-8")


def _mutate_manifest_missing_control_artifact(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    manifest = _read_manifest(report)
    removed = False
    artifacts = []
    for item in manifest.get("artifacts", []):
        if not removed and item.get("role") == "replayed_control_validation":
            removed = True
            continue
        artifacts.append(item)
    manifest["artifacts"] = artifacts
    _write_manifest(report, manifest)


def _mutate_readme_missing_boundary(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    readme = Path(str(_artifact_by_role(report, "bundle_readme")["path"]))
    readme.write_text("Phase2GC replay bundle\n", encoding="utf-8")


def _mutate_replay_artifact_count_collapsed(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("metrics", {})["replay_artifact_count"] = 0


def _mutate_control_artifact_count_collapsed(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("metrics", {})["copied_control_artifact_count"] = 0


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2gc_bundle",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_source_summary_failed",
        "mutation": _mutate_source_summary_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2ga_and_phase2gb_passed"],
    },
    {
        "control_id": "negative_recorded_check_false",
        "mutation": _mutate_recorded_check_false,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
    {
        "control_id": "negative_source_phase2ga_validation_failed",
        "mutation": _mutate_source_phase2ga_validation_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2ga_validation_passed"],
    },
    {
        "control_id": "negative_source_negative_controls_incomplete",
        "mutation": _mutate_negative_controls_incomplete,
        "expected_passed": False,
        "expected_failed_checks": ["source_negative_controls_complete"],
    },
    {
        "control_id": "negative_missing_manifest",
        "mutation": _mutate_missing_manifest,
        "expected_passed": False,
        "expected_failed_checks": [
            "manifest_readable",
            "manifest_roles_complete",
            "all_manifest_artifacts_exist",
            "all_manifest_hashes_match",
        ],
    },
    {
        "control_id": "negative_manifest_missing_role",
        "mutation": _mutate_manifest_missing_role,
        "expected_passed": False,
        "expected_failed_checks": ["manifest_roles_complete"],
    },
    {
        "control_id": "negative_manifest_artifacts_not_list",
        "mutation": _mutate_manifest_artifacts_not_list,
        "expected_passed": False,
        "expected_failed_checks": [
            "manifest_roles_complete",
            "all_manifest_artifacts_exist",
            "all_manifest_hashes_match",
        ],
    },
    {
        "control_id": "negative_missing_replayed_markdown",
        "mutation": _mutate_missing_replayed_markdown,
        "expected_passed": False,
        "expected_failed_checks": ["all_manifest_artifacts_exist"],
    },
    {
        "control_id": "negative_tampered_replayed_markdown",
        "mutation": _mutate_tampered_replayed_markdown,
        "expected_passed": False,
        "expected_failed_checks": ["all_manifest_hashes_match"],
    },
    {
        "control_id": "negative_missing_control_artifact",
        "mutation": _mutate_missing_control_artifact,
        "expected_passed": False,
        "expected_failed_checks": ["all_manifest_artifacts_exist"],
    },
    {
        "control_id": "negative_tampered_control_artifact",
        "mutation": _mutate_tampered_control_artifact,
        "expected_passed": False,
        "expected_failed_checks": ["all_manifest_hashes_match"],
    },
    {
        "control_id": "negative_manifest_missing_control_artifact",
        "mutation": _mutate_manifest_missing_control_artifact,
        "expected_passed": False,
        "expected_failed_checks": ["control_artifact_count_preserved"],
    },
    {
        "control_id": "negative_readme_missing_boundary",
        "mutation": _mutate_readme_missing_boundary,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_manifest_hashes_match",
            "readme_contains_bounded_boundary",
        ],
    },
    {
        "control_id": "negative_replay_artifact_count_collapsed",
        "mutation": _mutate_replay_artifact_count_collapsed,
        "expected_passed": False,
        "expected_failed_checks": ["replay_artifact_count_preserved"],
    },
    {
        "control_id": "negative_control_artifact_count_collapsed",
        "mutation": _mutate_control_artifact_count_collapsed,
        "expected_passed": False,
        "expected_failed_checks": ["control_artifact_count_preserved"],
    },
    {
        "control_id": "negative_overstated_epoch_claim",
        "mutation": _mutate_overstated_epoch_claim,
        "expected_passed": False,
        "expected_failed_checks": ["top_level_ready_claim_is_bounded"],
    },
)


def _run_control(
    *,
    control_spec: dict[str, Any],
    control_index: int,
    phase2gc_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"c{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2gc_report=phase2gc_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2gc_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2gc_replay_bundle_summary_replay_bundle(control_report)
    validation_report_json = case_dir / "phase2gc_validation.json"
    _write_json(validation_report_json, validation)
    expected_passed = bool(control_spec["expected_passed"])
    expected_failed_checks = list(control_spec["expected_failed_checks"])
    failed_checks = [
        name
        for name, passed in validation.get("checks", {}).items()
        if passed is False
    ]
    return {
        "control_id": control_id,
        "expected_passed": expected_passed,
        "observed_passed": validation.get("passed") is True,
        "pass_expectation_met": (validation.get("passed") is True) == expected_passed,
        "expected_failed_checks": expected_failed_checks,
        "failed_checks": failed_checks,
        "expected_failed_checks_observed": all(
            check in failed_checks for check in expected_failed_checks
        ),
        "control_report_json": str(control_report_json),
        "validation_report_json": str(validation_report_json),
    }


def audit_phase2gd_replay_bundle_summary_replay_bundle_negative_controls(
    *,
    phase2gc_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2gc_report = _read_json(phase2gc_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2gc_report=phase2gc_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2gc_bundle"
    ]
    checks = {
        "source_phase2gc_passed": phase2gc_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2gc_bundle"
            and row["observed_passed"] is True
            and row["pass_expectation_met"] is True
            for row in control_rows
        ),
        "minimum_negative_control_count_met": len(negative_rows) >= 16,
        "all_negative_controls_failed": bool(negative_rows)
        and all(row["observed_passed"] is False for row in negative_rows),
        "all_pass_expectations_met": all(
            row["pass_expectation_met"] for row in control_rows
        ),
        "all_expected_failed_checks_observed": all(
            row["expected_failed_checks_observed"] for row in control_rows
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": (
            "phase2gd_replay_bundle_summary_replay_bundle_negative_controls"
        ),
        "passed": passed,
        "ready_for_phase2gc_gate_strictness_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "control_count": len(control_rows),
            "negative_control_count": len(negative_rows),
            "negative_controls_failed": sum(
                row["observed_passed"] is False for row in negative_rows
            ),
            "expected_failed_check_assertions": sum(
                len(row["expected_failed_checks"]) for row in control_rows
            ),
        },
        "control_results": control_rows,
        "supported_claims": [
            (
                "the Phase2GC replay bundle gate rejects source summary tampering, "
                "recorded check tampering, failed upstream GA validation, incomplete "
                "upstream GB negative controls, missing or malformed manifests, "
                "missing or tampered replay and copied-control artifacts, README "
                "boundary deletion, replay or control artifact count collapse, and "
                "overstated epoch claims"
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
            "phase2ge_replay_bundle_summary_reproducibility_manifest"
            if passed
            else "repair_phase2gd_replay_bundle_summary_replay_bundle_negative_controls"
        ),
        "evidence": {
            "phase2gc_report_json": str(phase2gc_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2GD replay bundle summary replay bundle negative controls."
    )
    parser.add_argument("--phase2gc-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2gd_replay_bundle_summary_replay_bundle_negative_controls(
        phase2gc_report_json=args.phase2gc_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
