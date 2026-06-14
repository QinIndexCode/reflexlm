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
from reflexlm.cli.audit_phase2ec_replay_bundle_summary_cross_directory_replay import (
    validate_phase2ec_replay_bundle_summary_cross_directory_replay,
)


Mutation = Callable[[dict[str, Any], Path], None]


def _copy_replay_tree(
    *,
    phase2ec_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, str]:
    source_replay_dir = Path(str(phase2ec_report.get("evidence", {}).get("replay_dir", "")))
    if not source_replay_dir.exists():
        raise ValueError("Phase2ED requires a readable Phase2EC replay directory")
    target_replay_dir = case_dir / "replay"
    if target_replay_dir.exists():
        shutil.rmtree(target_replay_dir)
    shutil.copytree(source_replay_dir, target_replay_dir)
    path_map = {str(source_replay_dir): str(target_replay_dir)}
    for source_path in source_replay_dir.rglob("*"):
        relative = source_path.relative_to(source_replay_dir)
        path_map[str(source_path)] = str(target_replay_dir / relative)
    return path_map


def _rewrite_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [_rewrite_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, path_map) for key, item in value.items()}
    return value


def _materialize_control_report(
    *,
    phase2ec_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    control_report = deepcopy(phase2ec_report)
    path_map = _copy_replay_tree(phase2ec_report=phase2ec_report, case_dir=case_dir)
    control_report["evidence"] = _rewrite_paths(control_report.get("evidence", {}), path_map)
    control_report["replay_summary"] = _rewrite_paths(
        control_report.get("replay_summary", {}),
        path_map,
    )
    control_report["replayed_control_results"] = _rewrite_paths(
        control_report.get("replayed_control_results", []),
        path_map,
    )
    replayed_report_path = Path(
        str(control_report.get("evidence", {}).get("replayed_phase2ea_report", ""))
    )
    replayed_report = _rewrite_paths(_read_json(replayed_report_path), path_map)
    _write_json(replayed_report_path, replayed_report)
    replayed_controls_path = Path(
        str(control_report.get("evidence", {}).get("replayed_control_results_json", ""))
    )
    if replayed_controls_path.exists():
        replayed_controls = _rewrite_paths(_read_json(replayed_controls_path), path_map)
        _write_json(replayed_controls_path, replayed_controls)
    return control_report


def _replayed_report_path(report: dict[str, Any]) -> Path:
    return Path(str(report.get("evidence", {}).get("replayed_phase2ea_report", "")))


def _replayed_markdown_path(report: dict[str, Any]) -> Path:
    return Path(str(report.get("evidence", {}).get("replayed_markdown", "")))


def _first_replayed_control_path(report: dict[str, Any], key: str) -> Path:
    rows = report.get("replayed_control_results", [])
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise ValueError("Phase2ED requires replayed control results")
    return Path(str(rows[0].get(key, "")))


def _mutate_source_summary_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2ea_passed"] = False


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["replay_markdown_hash_matches_source"] = False


def _mutate_source_phase2ea_validation_failed(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2ea_validation_passed"] = False


def _mutate_source_negative_controls_incomplete(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2eb_negative_controls_failed"] = 0


def _mutate_missing_replayed_report(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    replayed_report = _replayed_report_path(report)
    if replayed_report.exists():
        replayed_report.unlink()


def _mutate_replayed_report_bad_json(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    _replayed_report_path(report).write_text("{bad json", encoding="utf-8")


def _mutate_missing_replayed_markdown(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    markdown = _replayed_markdown_path(report)
    if markdown.exists():
        markdown.unlink()


def _mutate_tampered_replayed_markdown(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    _replayed_markdown_path(report).write_text("tampered\n", encoding="utf-8")


def _mutate_summary_rows_collapsed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("replay_summary", {})["replayed_summary_row_count"] = 0


def _mutate_summary_columns_changed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("replay_summary", {})["replayed_summary_columns"] = [
        "Dimension",
        "Observed evidence",
        "Boundary",
        "Unbounded claim",
    ]


def _mutate_control_count_collapsed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("replay_summary", {})["replayed_control_count"] = 0


def _mutate_negative_failures_changed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("replay_summary", {})["replayed_negative_controls_failed"] = 0


def _mutate_expected_assertions_changed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("replay_summary", {})[
        "replayed_expected_failed_check_assertions"
    ] = 0


def _mutate_missing_copied_control_validation(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    validation_path = _first_replayed_control_path(report, "replayed_validation_report_json")
    if validation_path.exists():
        validation_path.unlink()


def _mutate_tampered_copied_control_report(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    _first_replayed_control_path(report, "replayed_control_report_json").write_text(
        "tampered\n",
        encoding="utf-8",
    )


def _mutate_replay_directory_not_distinct(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("replay_summary", {})["replay_markdown_parent"] = report.get(
        "replay_summary", {}
    ).get("source_markdown_parent")


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2ec_replay",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_source_summary_failed",
        "mutation": _mutate_source_summary_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2ea_and_phase2eb_passed"],
    },
    {
        "control_id": "negative_recorded_check_false",
        "mutation": _mutate_recorded_check_false,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
    {
        "control_id": "negative_source_phase2ea_validation_failed",
        "mutation": _mutate_source_phase2ea_validation_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2ea_validation_passed"],
    },
    {
        "control_id": "negative_source_negative_controls_incomplete",
        "mutation": _mutate_source_negative_controls_incomplete,
        "expected_passed": False,
        "expected_failed_checks": ["source_negative_controls_complete"],
    },
    {
        "control_id": "negative_missing_replayed_report",
        "mutation": _mutate_missing_replayed_report,
        "expected_passed": False,
        "expected_failed_checks": [
            "replayed_phase2ea_report_readable",
            "replayed_phase2ea_validation_passed",
        ],
    },
    {
        "control_id": "negative_replayed_report_bad_json",
        "mutation": _mutate_replayed_report_bad_json,
        "expected_passed": False,
        "expected_failed_checks": [
            "replayed_phase2ea_report_readable",
            "replayed_phase2ea_validation_passed",
        ],
    },
    {
        "control_id": "negative_missing_replayed_markdown",
        "mutation": _mutate_missing_replayed_markdown,
        "expected_passed": False,
        "expected_failed_checks": [
            "replayed_phase2ea_validation_passed",
            "replayed_markdown_hash_matches_source",
        ],
    },
    {
        "control_id": "negative_tampered_replayed_markdown",
        "mutation": _mutate_tampered_replayed_markdown,
        "expected_passed": False,
        "expected_failed_checks": [
            "replayed_phase2ea_validation_passed",
            "replayed_markdown_hash_matches_source",
        ],
    },
    {
        "control_id": "negative_summary_rows_collapsed",
        "mutation": _mutate_summary_rows_collapsed,
        "expected_passed": False,
        "expected_failed_checks": ["summary_rows_preserved"],
    },
    {
        "control_id": "negative_summary_columns_changed",
        "mutation": _mutate_summary_columns_changed,
        "expected_passed": False,
        "expected_failed_checks": ["summary_columns_preserved"],
    },
    {
        "control_id": "negative_control_count_collapsed",
        "mutation": _mutate_control_count_collapsed,
        "expected_passed": False,
        "expected_failed_checks": ["control_results_count_preserved"],
    },
    {
        "control_id": "negative_control_failures_changed",
        "mutation": _mutate_negative_failures_changed,
        "expected_passed": False,
        "expected_failed_checks": ["negative_control_failures_preserved"],
    },
    {
        "control_id": "negative_expected_assertions_changed",
        "mutation": _mutate_expected_assertions_changed,
        "expected_passed": False,
        "expected_failed_checks": ["expected_failed_assertions_preserved"],
    },
    {
        "control_id": "negative_missing_copied_control_validation",
        "mutation": _mutate_missing_copied_control_validation,
        "expected_passed": False,
        "expected_failed_checks": ["copied_control_files_complete"],
    },
    {
        "control_id": "negative_tampered_copied_control_report",
        "mutation": _mutate_tampered_copied_control_report,
        "expected_passed": False,
        "expected_failed_checks": ["copied_control_files_complete"],
    },
    {
        "control_id": "negative_replay_directory_not_distinct",
        "mutation": _mutate_replay_directory_not_distinct,
        "expected_passed": False,
        "expected_failed_checks": ["replay_directory_is_distinct"],
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
    phase2ec_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"c{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2ec_report=phase2ec_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2ec_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2ec_replay_bundle_summary_cross_directory_replay(
        control_report
    )
    validation_report_json = case_dir / "phase2ec_validation.json"
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


def audit_phase2ed_replay_bundle_summary_replay_negative_controls(
    *,
    phase2ec_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2ec_report = _read_json(phase2ec_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2ec_report=phase2ec_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2ec_replay"
    ]
    checks = {
        "source_phase2ec_passed": phase2ec_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2ec_replay"
            and row["observed_passed"] is True
            and row["pass_expectation_met"] is True
            for row in control_rows
        ),
        "minimum_negative_control_count_met": len(negative_rows) >= 17,
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
        "artifact_family": "phase2ed_replay_bundle_summary_replay_negative_controls",
        "passed": passed,
        "ready_for_phase2ec_gate_strictness_claim": passed,
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
                "the Phase2EC replay gate rejects source summary tampering, "
                "recorded check tampering, failed upstream EA validation, "
                "incomplete upstream EB negative controls, missing or malformed "
                "replayed reports, missing or tampered replayed markdown, collapsed "
                "summary rows or columns, control count drift, negative-control "
                "failure drift, expected assertion drift, copied-control file loss "
                "or tampering, non-distinct replay directories, and overstated "
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
            "phase2ee_replay_bundle_summary_replay_bundle"
            if passed
            else "repair_phase2ed_replay_bundle_summary_replay_negative_controls"
        ),
        "evidence": {
            "phase2ec_report_json": str(phase2ec_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2ED replay bundle summary replay negative controls."
    )
    parser.add_argument("--phase2ec-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ed_replay_bundle_summary_replay_negative_controls(
        phase2ec_report_json=args.phase2ec_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
