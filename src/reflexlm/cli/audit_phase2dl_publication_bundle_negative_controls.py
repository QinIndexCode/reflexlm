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
from reflexlm.cli.audit_phase2dk_latex_candidate_publication_bundle import (
    validate_phase2dk_latex_candidate_publication_bundle,
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
    raise ValueError(f"missing bundle artifact role: {role}")


def _materialize_control_report(
    *,
    phase2dk_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    control_report = deepcopy(phase2dk_report)
    source_manifest_path = _manifest_path(phase2dk_report)
    if not source_manifest_path.exists():
        raise ValueError("Phase2DL requires a readable Phase2DK bundle manifest")
    source_manifest = _read_json(source_manifest_path)
    source_artifacts = source_manifest.get("artifacts", [])
    if not isinstance(source_artifacts, list):
        raise ValueError("Phase2DL requires a list-valued Phase2DK manifest.artifacts")
    bundle_dir = case_dir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    copied_artifacts = []
    for item in source_artifacts:
        if not isinstance(item, dict):
            continue
        source_path = Path(str(item.get("path", "")))
        if not source_path.exists():
            raise ValueError(f"Phase2DL source artifact missing: {source_path}")
        copied = deepcopy(item)
        target_path = bundle_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied["path"] = str(target_path)
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
    report.setdefault("source_summary", {})["phase2di_passed"] = False


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["all_bundle_artifacts_written"] = False


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
        if item.get("role") != "compact_markdown_table"
    ]
    _write_manifest(report, manifest)


def _mutate_manifest_artifacts_not_list(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    manifest = _read_manifest(report)
    manifest["artifacts"] = {"role": "latex_candidate"}
    _write_manifest(report, manifest)


def _mutate_missing_latex_artifact(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    latex = Path(str(_artifact_by_role(report, "latex_candidate")["path"]))
    if latex.exists():
        latex.unlink()


def _mutate_tampered_latex_hash(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    latex = Path(str(_artifact_by_role(report, "latex_candidate")["path"]))
    latex.write_text(latex.read_text(encoding="utf-8") + "% tampered\n", encoding="utf-8")


def _mutate_latex_in_main_tables_dir(report: dict[str, Any], case_dir: Path) -> None:
    manifest = _read_manifest(report)
    target = case_dir / "docs" / "paper_b" / "tables" / "table.tex"
    target.parent.mkdir(parents=True, exist_ok=True)
    latex_entry = next(
        item for item in manifest["artifacts"] if item.get("role") == "latex_candidate"
    )
    shutil.copy2(latex_entry["path"], target)
    latex_entry["path"] = str(target)
    _write_manifest(report, manifest)


def _mutate_readme_missing_boundary(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    readme = Path(str(_artifact_by_role(report, "bundle_readme")["path"]))
    readme.write_text("Phase2DK publication bundle\n", encoding="utf-8")


def _mutate_negative_controls_incomplete(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2dj_negative_controls_failed"] = 0


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2dk_bundle",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_source_summary_failed",
        "mutation": _mutate_source_summary_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2di_and_phase2dj_passed"],
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
            "latex_candidate_not_in_main_tables_dir",
        ],
    },
    {
        "control_id": "negative_missing_latex_artifact",
        "mutation": _mutate_missing_latex_artifact,
        "expected_passed": False,
        "expected_failed_checks": ["all_manifest_artifacts_exist"],
    },
    {
        "control_id": "negative_tampered_latex_hash",
        "mutation": _mutate_tampered_latex_hash,
        "expected_passed": False,
        "expected_failed_checks": ["all_manifest_hashes_match"],
    },
    {
        "control_id": "negative_latex_in_main_tables_dir",
        "mutation": _mutate_latex_in_main_tables_dir,
        "expected_passed": False,
        "expected_failed_checks": ["latex_candidate_not_in_main_tables_dir"],
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
        "control_id": "negative_source_negative_controls_incomplete",
        "mutation": _mutate_negative_controls_incomplete,
        "expected_passed": False,
        "expected_failed_checks": ["source_negative_controls_complete"],
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
    phase2dk_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"c{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2dk_report=phase2dk_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2dk_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2dk_latex_candidate_publication_bundle(control_report)
    validation_report_json = case_dir / "phase2dk_validation.json"
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


def audit_phase2dl_publication_bundle_negative_controls(
    *,
    phase2dk_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2dk_report = _read_json(phase2dk_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2dk_report=phase2dk_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2dk_bundle"
    ]
    checks = {
        "source_phase2dk_passed": phase2dk_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2dk_bundle"
            and row["observed_passed"] is True
            and row["pass_expectation_met"] is True
            for row in control_rows
        ),
        "minimum_negative_control_count_met": len(negative_rows) >= 11,
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
        "artifact_family": "phase2dl_publication_bundle_negative_controls",
        "passed": passed,
        "ready_for_phase2dk_gate_strictness_claim": passed,
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
                "the Phase2DK bundle gate rejects source tampering, recorded check "
                "tampering, missing or malformed manifests, missing or tampered "
                "artifacts, main-paper table directory leakage, boundary deletion, "
                "incomplete upstream negative controls, and overstated epoch claims"
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
            "phase2dm_publication_bundle_reproducibility_manifest"
            if passed
            else "repair_phase2dl_publication_bundle_negative_controls"
        ),
        "evidence": {
            "phase2dk_report_json": str(phase2dk_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2DL publication bundle negative controls."
    )
    parser.add_argument("--phase2dk-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2dl_publication_bundle_negative_controls(
        phase2dk_report_json=args.phase2dk_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
