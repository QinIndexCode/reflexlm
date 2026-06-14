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
from reflexlm.cli.audit_phase2dm_publication_bundle_reproducibility_manifest import (
    validate_phase2dm_publication_bundle_reproducibility_manifest,
)


Mutation = Callable[[dict[str, Any], Path], None]


def _manifest_path(report: dict[str, Any]) -> Path:
    return Path(str(report.get("evidence", {}).get("reproducibility_manifest", "")))


def _read_manifest(report: dict[str, Any]) -> dict[str, Any]:
    return _read_json(_manifest_path(report))


def _write_manifest(report: dict[str, Any], manifest: dict[str, Any]) -> None:
    _write_json(_manifest_path(report), manifest)


def _entries(manifest: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = manifest.get(key, [])
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def _entry_by_role(report: dict[str, Any], key: str, role: str) -> dict[str, Any]:
    for item in _entries(_read_manifest(report), key):
        if item.get("role") == role:
            return item
    raise ValueError(f"missing manifest {key} role: {role}")


def _copy_entries(entries: list[dict[str, Any]], target_dir: Path) -> list[dict[str, Any]]:
    copied_entries = []
    for item in entries:
        copied = deepcopy(item)
        source = Path(str(item.get("path", "")))
        if not source.exists():
            raise ValueError(f"Phase2DN source path missing: {source}")
        target = target_dir / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied["path"] = str(target)
        copied_entries.append(copied)
    return copied_entries


def _materialize_control_report(
    *,
    phase2dm_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    control_report = deepcopy(phase2dm_report)
    source_manifest_path = _manifest_path(phase2dm_report)
    if not source_manifest_path.exists():
        raise ValueError("Phase2DN requires a readable Phase2DM manifest")
    source_manifest = _read_json(source_manifest_path)
    manifest_dir = case_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    control_manifest = deepcopy(source_manifest)
    control_manifest["source_reports"] = _copy_entries(
        _entries(source_manifest, "source_reports"),
        manifest_dir / "source_reports",
    )
    control_manifest["bundle_artifacts"] = _copy_entries(
        _entries(source_manifest, "bundle_artifacts"),
        manifest_dir / "bundle_artifacts",
    )
    target_manifest_path = manifest_dir / "phase2dm_manifest.json"
    _write_json(target_manifest_path, control_manifest)
    control_report.setdefault("evidence", {})[
        "reproducibility_manifest"
    ] = str(target_manifest_path)
    return control_report


def _mutate_source_summary_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2dk_passed"] = False


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["bundle_artifacts_present"] = False


def _mutate_missing_manifest(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    manifest = _manifest_path(report)
    if manifest.exists():
        manifest.unlink()


def _mutate_source_reports_not_list(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    manifest = _read_manifest(report)
    manifest["source_reports"] = {"role": "phase2di_report"}
    _write_manifest(report, manifest)


def _mutate_missing_source_report_role(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    manifest = _read_manifest(report)
    manifest["source_reports"] = [
        item
        for item in manifest.get("source_reports", [])
        if item.get("role") != "phase2di_report"
    ]
    _write_manifest(report, manifest)


def _mutate_missing_source_report_file(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    source = Path(str(_entry_by_role(report, "source_reports", "phase2di_report")["path"]))
    if source.exists():
        source.unlink()


def _mutate_tampered_source_report_hash(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    source = Path(
        str(_entry_by_role(report, "source_reports", "phase2di_report")["path"])
    )
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")


def _mutate_bundle_artifacts_not_list(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    manifest = _read_manifest(report)
    manifest["bundle_artifacts"] = {"role": "latex_candidate"}
    _write_manifest(report, manifest)


def _mutate_missing_bundle_artifact_role(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    manifest = _read_manifest(report)
    manifest["bundle_artifacts"] = [
        item
        for item in manifest.get("bundle_artifacts", [])
        if item.get("role") != "latex_candidate"
    ]
    _write_manifest(report, manifest)


def _mutate_missing_bundle_artifact_file(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    artifact = Path(str(_entry_by_role(report, "bundle_artifacts", "latex_candidate")["path"]))
    if artifact.exists():
        artifact.unlink()


def _mutate_tampered_bundle_artifact_hash(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    artifact = Path(
        str(_entry_by_role(report, "bundle_artifacts", "bundle_readme")["path"])
    )
    artifact.write_text("tampered\n", encoding="utf-8")


def _mutate_missing_reproduction_step(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    manifest = _read_manifest(report)
    manifest["reproduction_steps"] = manifest.get("reproduction_steps", [])[:-1]
    _write_manifest(report, manifest)


def _mutate_reproduction_step_missing_module(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    manifest = _read_manifest(report)
    manifest["reproduction_steps"][0]["module"] = ""
    _write_manifest(report, manifest)


def _mutate_missing_boundary(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    manifest = _read_manifest(report)
    manifest["claim_boundary"] = "epoch-making architecture"
    _write_manifest(report, manifest)


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2dm_manifest",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_source_summary_failed",
        "mutation": _mutate_source_summary_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2dk_and_phase2dl_passed"],
    },
    {
        "control_id": "negative_recorded_check_false",
        "mutation": _mutate_recorded_check_false,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
    {
        "control_id": "negative_missing_manifest",
        "mutation": _mutate_missing_manifest,
        "expected_passed": False,
        "expected_failed_checks": [
            "manifest_readable",
            "source_report_roles_complete",
            "bundle_artifact_roles_complete",
            "reproduction_steps_complete",
        ],
    },
    {
        "control_id": "negative_source_reports_not_list",
        "mutation": _mutate_source_reports_not_list,
        "expected_passed": False,
        "expected_failed_checks": [
            "source_report_roles_complete",
            "source_reports_exist",
            "source_report_hashes_match",
        ],
    },
    {
        "control_id": "negative_missing_source_report_role",
        "mutation": _mutate_missing_source_report_role,
        "expected_passed": False,
        "expected_failed_checks": ["source_report_roles_complete"],
    },
    {
        "control_id": "negative_missing_source_report_file",
        "mutation": _mutate_missing_source_report_file,
        "expected_passed": False,
        "expected_failed_checks": [
            "source_reports_exist",
            "source_report_hashes_match",
        ],
    },
    {
        "control_id": "negative_tampered_source_report_hash",
        "mutation": _mutate_tampered_source_report_hash,
        "expected_passed": False,
        "expected_failed_checks": ["source_report_hashes_match"],
    },
    {
        "control_id": "negative_bundle_artifacts_not_list",
        "mutation": _mutate_bundle_artifacts_not_list,
        "expected_passed": False,
        "expected_failed_checks": [
            "bundle_artifact_roles_complete",
            "bundle_artifacts_exist",
            "bundle_artifact_hashes_match",
        ],
    },
    {
        "control_id": "negative_missing_bundle_artifact_role",
        "mutation": _mutate_missing_bundle_artifact_role,
        "expected_passed": False,
        "expected_failed_checks": ["bundle_artifact_roles_complete"],
    },
    {
        "control_id": "negative_missing_bundle_artifact_file",
        "mutation": _mutate_missing_bundle_artifact_file,
        "expected_passed": False,
        "expected_failed_checks": [
            "bundle_artifacts_exist",
            "bundle_artifact_hashes_match",
        ],
    },
    {
        "control_id": "negative_tampered_bundle_artifact_hash",
        "mutation": _mutate_tampered_bundle_artifact_hash,
        "expected_passed": False,
        "expected_failed_checks": ["bundle_artifact_hashes_match"],
    },
    {
        "control_id": "negative_missing_reproduction_step",
        "mutation": _mutate_missing_reproduction_step,
        "expected_passed": False,
        "expected_failed_checks": ["reproduction_steps_complete"],
    },
    {
        "control_id": "negative_reproduction_step_missing_module",
        "mutation": _mutate_reproduction_step_missing_module,
        "expected_passed": False,
        "expected_failed_checks": ["reproduction_steps_have_modules_and_paths"],
    },
    {
        "control_id": "negative_missing_boundary",
        "mutation": _mutate_missing_boundary,
        "expected_passed": False,
        "expected_failed_checks": ["manifest_contains_bounded_boundary"],
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
    phase2dm_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"c{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2dm_report=phase2dm_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2dm_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2dm_publication_bundle_reproducibility_manifest(
        control_report
    )
    validation_report_json = case_dir / "phase2dm_validation.json"
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


def audit_phase2dn_reproducibility_manifest_negative_controls(
    *,
    phase2dm_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2dm_report = _read_json(phase2dm_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2dm_report=phase2dm_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2dm_manifest"
    ]
    checks = {
        "source_phase2dm_passed": phase2dm_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2dm_manifest"
            and row["observed_passed"] is True
            and row["pass_expectation_met"] is True
            for row in control_rows
        ),
        "minimum_negative_control_count_met": len(negative_rows) >= 15,
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
        "artifact_family": "phase2dn_reproducibility_manifest_negative_controls",
        "passed": passed,
        "ready_for_phase2dm_gate_strictness_claim": passed,
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
                "the Phase2DM reproducibility-manifest gate rejects source tampering, "
                "recorded check tampering, missing or malformed manifests, missing "
                "or tampered source reports, missing or tampered bundle artifacts, "
                "incomplete reproduction steps, boundary deletion, and overstated "
                "epoch claims"
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
            "phase2do_reproducibility_manifest_cross_directory_replay"
            if passed
            else "repair_phase2dn_reproducibility_manifest_negative_controls"
        ),
        "evidence": {
            "phase2dm_report_json": str(phase2dm_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2DN reproducibility manifest negative controls."
    )
    parser.add_argument("--phase2dm-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2dn_reproducibility_manifest_negative_controls(
        phase2dm_report_json=args.phase2dm_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
