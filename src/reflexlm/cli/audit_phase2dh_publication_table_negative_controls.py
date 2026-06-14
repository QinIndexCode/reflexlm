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
from reflexlm.cli.audit_phase2dg_compact_rollup_publication_table import (
    TABLE_COLUMNS,
    validate_phase2dg_compact_rollup_publication_table,
)


Mutation = Callable[[dict[str, Any], Path], None]


def _table_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    table = report.get("table", {})
    rows = table.get("rows", []) if isinstance(table, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _materialize_control_report(
    *,
    phase2dg_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    control_report = deepcopy(phase2dg_report)
    source_markdown = phase2dg_report.get("evidence", {}).get(
        "publication_table_markdown"
    )
    if not source_markdown:
        raise ValueError("Phase2DH requires Phase2DG evidence.publication_table_markdown")
    target_markdown = case_dir / "table.md"
    target_markdown.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_markdown, target_markdown)
    control_report.setdefault("evidence", {})[
        "publication_table_markdown"
    ] = str(target_markdown)
    return control_report


def _mutate_source_summary_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2df_passed"] = False


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["all_rows_passed_and_bounded"] = False


def _mutate_column_schema(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("table", {})["columns"] = list(TABLE_COLUMNS[:-1])


def _mutate_missing_row(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("table", {})["rows"] = _table_rows(report)[:-1]


def _mutate_row_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    _table_rows(report)[0]["passed"] = False


def _mutate_row_unbounded(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    _table_rows(report)[0]["bounded_claim_ok"] = False


def _mutate_positive_rows_collapsed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    rows = [
        row
        for row in _table_rows(report)
        if row.get("evidence_role") != "positive stability evidence"
        or row.get("phase_id") == "phase2cp"
    ]
    report.setdefault("table", {})["rows"] = rows


def _mutate_negative_rows_collapsed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    rows = [
        row
        for row in _table_rows(report)
        if row.get("evidence_role") != "negative-control gate"
        or row.get("phase_id") == "phase2cq"
    ]
    report.setdefault("table", {})["rows"] = rows


def _mutate_missing_key_metrics(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    _table_rows(report)[0]["key_metrics"] = ""


def _mutate_missing_next_step(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    _table_rows(report)[0]["next_required_experiment"] = None


def _mutate_missing_markdown(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    markdown = Path(str(report.get("evidence", {}).get("publication_table_markdown")))
    if markdown.exists():
        markdown.unlink()


def _mutate_markdown_missing_phase(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    markdown = Path(str(report.get("evidence", {}).get("publication_table_markdown")))
    text = markdown.read_text(encoding="utf-8")
    phase_id = str(_table_rows(report)[0]["phase_id"])
    markdown.write_text(text.replace(phase_id, "phase2xx"), encoding="utf-8")


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2dg_report",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_source_summary_failed",
        "mutation": _mutate_source_summary_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2de_and_phase2df_passed"],
    },
    {
        "control_id": "negative_recorded_check_false",
        "mutation": _mutate_recorded_check_false,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
    {
        "control_id": "negative_column_schema",
        "mutation": _mutate_column_schema,
        "expected_passed": False,
        "expected_failed_checks": ["columns_match_publication_schema"],
    },
    {
        "control_id": "negative_missing_row",
        "mutation": _mutate_missing_row,
        "expected_passed": False,
        "expected_failed_checks": [
            "row_count_matches_source_phase_count",
            "positive_and_negative_rows_present",
        ],
    },
    {
        "control_id": "negative_row_failed",
        "mutation": _mutate_row_failed,
        "expected_passed": False,
        "expected_failed_checks": ["all_rows_passed_and_bounded"],
    },
    {
        "control_id": "negative_row_unbounded",
        "mutation": _mutate_row_unbounded,
        "expected_passed": False,
        "expected_failed_checks": ["all_rows_passed_and_bounded"],
    },
    {
        "control_id": "negative_positive_rows_collapsed",
        "mutation": _mutate_positive_rows_collapsed,
        "expected_passed": False,
        "expected_failed_checks": [
            "row_count_matches_source_phase_count",
            "positive_and_negative_rows_present",
        ],
    },
    {
        "control_id": "negative_negative_rows_collapsed",
        "mutation": _mutate_negative_rows_collapsed,
        "expected_passed": False,
        "expected_failed_checks": [
            "row_count_matches_source_phase_count",
            "positive_and_negative_rows_present",
        ],
    },
    {
        "control_id": "negative_missing_key_metrics",
        "mutation": _mutate_missing_key_metrics,
        "expected_passed": False,
        "expected_failed_checks": ["all_rows_have_key_metrics_and_next_step"],
    },
    {
        "control_id": "negative_missing_next_step",
        "mutation": _mutate_missing_next_step,
        "expected_passed": False,
        "expected_failed_checks": ["all_rows_have_key_metrics_and_next_step"],
    },
    {
        "control_id": "negative_missing_markdown",
        "mutation": _mutate_missing_markdown,
        "expected_passed": False,
        "expected_failed_checks": [
            "markdown_table_readable",
            "markdown_table_contains_all_phase_ids",
        ],
    },
    {
        "control_id": "negative_markdown_missing_phase",
        "mutation": _mutate_markdown_missing_phase,
        "expected_passed": False,
        "expected_failed_checks": ["markdown_table_contains_all_phase_ids"],
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
    phase2dg_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"c{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2dg_report=phase2dg_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2dg_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2dg_compact_rollup_publication_table(control_report)
    validation_report_json = case_dir / "phase2dg_validation.json"
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


def audit_phase2dh_publication_table_negative_controls(
    *,
    phase2dg_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2dg_report = _read_json(phase2dg_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2dg_report=phase2dg_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2dg_report"
    ]
    checks = {
        "source_phase2dg_passed": phase2dg_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2dg_report"
            and row["observed_passed"] is True
            and row["pass_expectation_met"] is True
            for row in control_rows
        ),
        "minimum_negative_control_count_met": len(negative_rows) >= 13,
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
        "artifact_family": "phase2dh_publication_table_negative_controls",
        "passed": passed,
        "ready_for_phase2dg_gate_strictness_claim": passed,
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
                "the Phase2DG publication-table gate rejects overclaims for source "
                "summary tampering, recorded check tampering, schema changes, row "
                "loss, failed or unbounded rows, collapsed evidence roles, missing "
                "row metadata, missing Markdown, incomplete Markdown, and overstated "
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
            "phase2di_publication_table_latex_candidate"
            if passed
            else "repair_phase2dh_publication_table_negative_controls"
        ),
        "evidence": {
            "phase2dg_report_json": str(phase2dg_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2DH publication table negative controls."
    )
    parser.add_argument("--phase2dg-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2dh_publication_table_negative_controls(
        phase2dg_report_json=args.phase2dg_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
