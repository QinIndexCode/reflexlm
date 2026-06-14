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
from reflexlm.cli.audit_phase2ds_portability_summary_cross_directory_replay import (
    validate_phase2ds_portability_summary_cross_directory_replay,
)


Mutation = Callable[[dict[str, Any], Path], None]


def _copy_replay_tree(
    *,
    phase2ds_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, str]:
    source_replay_dir = Path(str(phase2ds_report.get("evidence", {}).get("replay_dir", "")))
    if not source_replay_dir.exists():
        raise ValueError("Phase2DT requires a readable Phase2DS replay directory")
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
    phase2ds_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    control_report = deepcopy(phase2ds_report)
    path_map = _copy_replay_tree(phase2ds_report=phase2ds_report, case_dir=case_dir)
    control_report["evidence"] = _rewrite_paths(
        control_report.get("evidence", {}),
        path_map,
    )
    control_report["replay_summary"] = _rewrite_paths(
        control_report.get("replay_summary", {}),
        path_map,
    )
    replayed_report_path = Path(
        str(control_report.get("evidence", {}).get("replayed_phase2dq_report", ""))
    )
    replayed_report = _read_json(replayed_report_path)
    replayed_report = _rewrite_paths(replayed_report, path_map)
    _write_json(replayed_report_path, replayed_report)
    return control_report


def _replayed_report_path(report: dict[str, Any]) -> Path:
    return Path(str(report.get("evidence", {}).get("replayed_phase2dq_report", "")))


def _replayed_markdown_path(report: dict[str, Any]) -> Path:
    return Path(str(report.get("evidence", {}).get("replayed_markdown", "")))


def _mutate_source_summary_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2dq_passed"] = False


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["replay_markdown_hash_matches_source"] = False


def _mutate_source_phase2dq_validation_failed(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2dq_validation_passed"] = False


def _mutate_source_negative_controls_incomplete(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2dr_negative_controls_failed"] = 0


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
        "control_id": "positive_control_original_phase2ds_replay",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_source_summary_failed",
        "mutation": _mutate_source_summary_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2dq_and_phase2dr_passed"],
    },
    {
        "control_id": "negative_recorded_check_false",
        "mutation": _mutate_recorded_check_false,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
    {
        "control_id": "negative_source_phase2dq_validation_failed",
        "mutation": _mutate_source_phase2dq_validation_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2dq_validation_passed"],
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
            "replayed_phase2dq_report_readable",
            "replayed_phase2dq_validation_passed",
        ],
    },
    {
        "control_id": "negative_replayed_report_bad_json",
        "mutation": _mutate_replayed_report_bad_json,
        "expected_passed": False,
        "expected_failed_checks": [
            "replayed_phase2dq_report_readable",
            "replayed_phase2dq_validation_passed",
        ],
    },
    {
        "control_id": "negative_missing_replayed_markdown",
        "mutation": _mutate_missing_replayed_markdown,
        "expected_passed": False,
        "expected_failed_checks": [
            "replayed_phase2dq_validation_passed",
            "replayed_markdown_hash_matches_source",
        ],
    },
    {
        "control_id": "negative_tampered_replayed_markdown",
        "mutation": _mutate_tampered_replayed_markdown,
        "expected_passed": False,
        "expected_failed_checks": [
            "replayed_phase2dq_validation_passed",
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
    phase2ds_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"c{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2ds_report=phase2ds_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2ds_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2ds_portability_summary_cross_directory_replay(
        control_report
    )
    validation_report_json = case_dir / "phase2ds_validation.json"
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


def audit_phase2dt_portability_summary_replay_negative_controls(
    *,
    phase2ds_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2ds_report = _read_json(phase2ds_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2ds_report=phase2ds_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2ds_replay"
    ]
    checks = {
        "source_phase2ds_passed": phase2ds_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2ds_replay"
            and row["observed_passed"] is True
            and row["pass_expectation_met"] is True
            for row in control_rows
        ),
        "minimum_negative_control_count_met": len(negative_rows) >= 12,
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
        "artifact_family": "phase2dt_portability_summary_replay_negative_controls",
        "passed": passed,
        "ready_for_phase2ds_gate_strictness_claim": passed,
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
                "the Phase2DS portability-summary replay gate rejects source summary "
                "tampering, recorded check tampering, failed upstream DQ validation, "
                "incomplete upstream DR negative controls, missing or malformed "
                "replayed reports, missing or tampered replayed markdown, collapsed "
                "summary rows, changed summary columns, non-distinct replay "
                "directories, and overstated epoch claims"
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
            "phase2du_portability_summary_replay_bundle"
            if passed
            else "repair_phase2dt_portability_summary_replay_negative_controls"
        ),
        "evidence": {
            "phase2ds_report_json": str(phase2ds_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2DT portability summary replay negative controls."
    )
    parser.add_argument("--phase2ds-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2dt_portability_summary_replay_negative_controls(
        phase2ds_report_json=args.phase2ds_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
