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
from reflexlm.cli.audit_phase2fe_replay_bundle_summary_portability_summary import (
    validate_phase2fe_replay_bundle_summary_portability_summary,
)


Mutation = Callable[[dict[str, Any], Path], None]


def _markdown_path(report: dict[str, Any]) -> Path:
    return Path(str(report.get("evidence", {}).get("portability_summary_markdown", "")))


def _materialize_control_report(
    *,
    phase2fe_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    control_report = deepcopy(phase2fe_report)
    source_markdown = _markdown_path(phase2fe_report)
    if not source_markdown.exists():
        raise ValueError("Phase2FF requires a readable Phase2FE markdown summary")
    target_markdown = case_dir / "phase2fe_replay_bundle_summary_portability_summary.md"
    target_markdown.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_markdown, target_markdown)
    control_report.setdefault("evidence", {})["portability_summary_markdown"] = str(
        target_markdown
    )
    return control_report


def _mutate_source_summary_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2fd_passed"] = False


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["summary_markdown_written"] = False


def _mutate_phase2fc_validation_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2fc_validation_passed"] = False


def _mutate_phase2fb_controls_incomplete(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2fb_negative_controls_failed"] = 0


def _mutate_phase2fd_controls_incomplete(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2fd_negative_controls_failed"] = 0


def _mutate_missing_summary_row(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    rows = report.setdefault("summary_table", {}).get("rows", [])
    report["summary_table"]["rows"] = rows[:-1] if isinstance(rows, list) else []


def _mutate_missing_required_dimension(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    rows = report.setdefault("summary_table", {}).get("rows", [])
    if isinstance(rows, list):
        report["summary_table"]["rows"] = [
            row
            for row in rows
            if not (
                isinstance(row, dict)
                and row.get("Dimension") == "control artifacts"
            )
        ]


def _mutate_wrong_summary_columns(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("summary_table", {})["columns"] = [
        "Dimension",
        "Observed evidence",
        "Boundary",
        "Unbounded claim",
    ]


def _mutate_missing_markdown(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    markdown = _markdown_path(report)
    if markdown.exists():
        markdown.unlink()


def _mutate_markdown_missing_dimension(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    rows = report.get("summary_table", {}).get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("Phase2FF requires summary rows for markdown dimension control")
    dimension = str(rows[0].get("Dimension"))
    markdown = _markdown_path(report)
    markdown.write_text(
        markdown.read_text(encoding="utf-8").replace(dimension, "redacted dimension"),
        encoding="utf-8",
    )


def _mutate_markdown_missing_boundary(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    markdown = _markdown_path(report)
    text = markdown.read_text(encoding="utf-8")
    text = text.replace("not free-form shell autonomy", "bounded shell evidence")
    text = text.replace(
        "not an epoch-making architecture", "bounded architecture evidence"
    )
    markdown.write_text(text, encoding="utf-8")


def _mutate_bundle_artifact_count_mismatch(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("metrics", {})["replayed_bundle_artifact_count"] = 999


def _mutate_bundle_hash_count_mismatch(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("metrics", {})["replayed_bundle_artifact_hash_match_count"] = 0


def _mutate_control_artifact_count_mismatch(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("metrics", {})["replayed_control_artifact_count"] = 0


def _mutate_reproduction_step_count_mismatch(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("metrics", {})["replayed_reproduction_step_count"] = 0


def _mutate_source_report_count_mismatch(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    report.setdefault("metrics", {})["replayed_source_report_count"] = 0


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2fe_summary",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_source_summary_failed",
        "mutation": _mutate_source_summary_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2fc_and_phase2fd_passed"],
    },
    {
        "control_id": "negative_recorded_check_false",
        "mutation": _mutate_recorded_check_false,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
    {
        "control_id": "negative_phase2fc_validation_failed",
        "mutation": _mutate_phase2fc_validation_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2fc_validation_passed"],
    },
    {
        "control_id": "negative_phase2fb_controls_incomplete",
        "mutation": _mutate_phase2fb_controls_incomplete,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2fb_negative_controls_complete"],
    },
    {
        "control_id": "negative_phase2fd_controls_incomplete",
        "mutation": _mutate_phase2fd_controls_incomplete,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2fd_negative_controls_complete"],
    },
    {
        "control_id": "negative_missing_summary_row",
        "mutation": _mutate_missing_summary_row,
        "expected_passed": False,
        "expected_failed_checks": ["summary_rows_present"],
    },
    {
        "control_id": "negative_missing_required_dimension",
        "mutation": _mutate_missing_required_dimension,
        "expected_passed": False,
        "expected_failed_checks": ["summary_dimensions_complete"],
    },
    {
        "control_id": "negative_wrong_summary_columns",
        "mutation": _mutate_wrong_summary_columns,
        "expected_passed": False,
        "expected_failed_checks": ["summary_columns_match"],
    },
    {
        "control_id": "negative_missing_markdown",
        "mutation": _mutate_missing_markdown,
        "expected_passed": False,
        "expected_failed_checks": ["markdown_summary_readable"],
    },
    {
        "control_id": "negative_markdown_missing_dimension",
        "mutation": _mutate_markdown_missing_dimension,
        "expected_passed": False,
        "expected_failed_checks": ["markdown_contains_all_dimensions"],
    },
    {
        "control_id": "negative_markdown_missing_boundary",
        "mutation": _mutate_markdown_missing_boundary,
        "expected_passed": False,
        "expected_failed_checks": ["markdown_contains_bounded_boundary"],
    },
    {
        "control_id": "negative_bundle_artifact_count_mismatch",
        "mutation": _mutate_bundle_artifact_count_mismatch,
        "expected_passed": False,
        "expected_failed_checks": ["metrics_match_source"],
    },
    {
        "control_id": "negative_bundle_hash_count_mismatch",
        "mutation": _mutate_bundle_hash_count_mismatch,
        "expected_passed": False,
        "expected_failed_checks": ["metrics_match_source"],
    },
    {
        "control_id": "negative_control_artifact_count_mismatch",
        "mutation": _mutate_control_artifact_count_mismatch,
        "expected_passed": False,
        "expected_failed_checks": ["metrics_match_source"],
    },
    {
        "control_id": "negative_reproduction_step_count_mismatch",
        "mutation": _mutate_reproduction_step_count_mismatch,
        "expected_passed": False,
        "expected_failed_checks": ["metrics_match_source"],
    },
    {
        "control_id": "negative_source_report_count_mismatch",
        "mutation": _mutate_source_report_count_mismatch,
        "expected_passed": False,
        "expected_failed_checks": ["metrics_match_source"],
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
    phase2fe_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"c{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2fe_report=phase2fe_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2fe_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2fe_replay_bundle_summary_portability_summary(
        control_report
    )
    validation_report_json = case_dir / "phase2fe_validation.json"
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


def audit_phase2ff_replay_bundle_summary_portability_summary_negative_controls(
    *,
    phase2fe_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2fe_report = _read_json(phase2fe_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2fe_report=phase2fe_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2fe_summary"
    ]
    checks = {
        "source_phase2fe_passed": phase2fe_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2fe_summary"
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
        "artifact_family": (
            "phase2ff_replay_bundle_summary_portability_summary_negative_controls"
        ),
        "passed": passed,
        "ready_for_phase2fe_gate_strictness_claim": passed,
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
                "the Phase2FE replay-bundle summary portability-summary gate "
                "rejects source summary tampering, recorded check tampering, "
                "failed FC validation, incomplete FB or FD negative controls, "
                "missing or malformed summary rows or columns, missing or weakened "
                "markdown evidence, source-report, bundle-hash, control-artifact, "
                "and reproduction-step metric mismatch, and overstated epoch claims"
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
            "phase2fg_replay_bundle_summary_cross_directory_replay"
            if passed
            else "repair_phase2ff_replay_bundle_summary_portability_summary_negative_controls"
        ),
        "evidence": {
            "phase2fe_report_json": str(phase2fe_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Phase2FF replay bundle summary portability summary negative controls."
        )
    )
    parser.add_argument("--phase2fe-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ff_replay_bundle_summary_portability_summary_negative_controls(
        phase2fe_report_json=args.phase2fe_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
