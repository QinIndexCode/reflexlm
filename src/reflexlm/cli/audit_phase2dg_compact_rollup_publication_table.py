from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)


TABLE_COLUMNS: tuple[str, ...] = (
    "phase_id",
    "evidence_role",
    "artifact_family",
    "passed",
    "bounded_claim_ok",
    "key_metrics",
    "next_required_experiment",
)

OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_general_shell_autonomy_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _metric_summary(metrics: dict[str, Any]) -> str:
    if not isinstance(metrics, dict) or not metrics:
        return "no compact metrics"
    preferred_keys = (
        "runtime_count",
        "seed_count",
        "perturbation_count",
        "fresh_runtime_execution_count",
        "runtime_signature_mismatch_count",
        "control_count",
        "negative_control_count",
        "negative_controls_failed",
        "expected_failed_check_assertions",
        "order_count",
        "passed_order_validations",
        "order_validation_signature_mismatch_count",
    )
    parts = [
        f"{key}={metrics[key]}"
        for key in preferred_keys
        if key in metrics
    ]
    return "; ".join(parts) if parts else "compact metrics recorded"


def _evidence_role(category: str) -> str:
    if category == "negative_control":
        return "negative-control gate"
    if category.startswith("positive"):
        return "positive stability evidence"
    return "supporting evidence"


def _table_row(phase_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase_id": phase_row.get("phase_id"),
        "evidence_role": _evidence_role(str(phase_row.get("category", ""))),
        "artifact_family": phase_row.get("artifact_family"),
        "passed": phase_row.get("passed") is True,
        "bounded_claim_ok": phase_row.get("bounded_claim_ok") is True,
        "key_metrics": _metric_summary(phase_row.get("compact_metrics", {})),
        "next_required_experiment": phase_row.get("next_required_experiment"),
    }


def _markdown_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(table: dict[str, Any]) -> str:
    columns = table["columns"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table["rows"]:
        lines.append(
            "| "
            + " | ".join(_markdown_escape(row.get(column)) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def _ready_boundary_ok(report: dict[str, Any]) -> bool:
    return (
        report.get("ready_for_bounded_compact_rollup_publication_table_claim")
        is True
        and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
    )


def validate_phase2dg_compact_rollup_publication_table(
    report: dict[str, Any],
) -> dict[str, Any]:
    table = report.get("table", {})
    rows = table.get("rows", []) if isinstance(table, dict) else []
    columns = table.get("columns", []) if isinstance(table, dict) else []
    if not isinstance(rows, list):
        rows = []
    if not isinstance(columns, list):
        columns = []
    positive_rows = [
        row
        for row in rows
        if row.get("evidence_role") == "positive stability evidence"
    ]
    negative_rows = [
        row
        for row in rows
        if row.get("evidence_role") == "negative-control gate"
    ]
    markdown_path = report.get("evidence", {}).get("publication_table_markdown")
    markdown_readable = False
    markdown_has_all_phase_ids = False
    if markdown_path:
        try:
            markdown_text = Path(markdown_path).read_text(encoding="utf-8")
            markdown_readable = True
            markdown_has_all_phase_ids = all(
                str(row.get("phase_id")) in markdown_text for row in rows
            )
        except OSError:
            markdown_readable = False
    checks = {
        "artifact_family_matches_phase2dg": (
            report.get("artifact_family")
            == "phase2dg_compact_rollup_publication_table"
        ),
        "top_level_phase2dg_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": _ready_boundary_ok(report),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2de_and_phase2df_passed": (
            report.get("source_summary", {}).get("phase2de_passed") is True
            and report.get("source_summary", {}).get("phase2df_passed") is True
        ),
        "columns_match_publication_schema": tuple(columns) == TABLE_COLUMNS,
        "row_count_matches_source_phase_count": len(rows)
        == int(report.get("source_summary", {}).get("phase_count", -1)),
        "all_rows_passed_and_bounded": bool(rows)
        and all(
            row.get("passed") is True and row.get("bounded_claim_ok") is True
            for row in rows
        ),
        "positive_and_negative_rows_present": len(positive_rows) >= 8
        and len(negative_rows) >= 7,
        "all_rows_have_key_metrics_and_next_step": bool(rows)
        and all(
            bool(row.get("key_metrics")) and row.get("next_required_experiment")
            for row in rows
        ),
        "markdown_table_readable": markdown_readable,
        "markdown_table_contains_all_phase_ids": markdown_has_all_phase_ids,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "positive_row_count": len(positive_rows),
            "negative_row_count": len(negative_rows),
            "markdown_readable": markdown_readable,
        },
    }


def audit_phase2dg_compact_rollup_publication_table(
    *,
    phase2df_report_json: str | Path,
    output_report_json: str | Path,
    output_markdown: str | Path,
) -> dict[str, Any]:
    phase2df = _read_json(phase2df_report_json)
    phase2de_report_json = phase2df.get("evidence", {}).get("phase2de_report_json")
    if not phase2de_report_json:
        raise ValueError("Phase2DG requires Phase2DF evidence.phase2de_report_json")
    phase2de = _read_json(phase2de_report_json)
    source_rows = phase2de.get("phase_results", [])
    if not isinstance(source_rows, list):
        source_rows = []
    rows = [_table_row(row) for row in source_rows if isinstance(row, dict)]
    table = {
        "table_family": "phase2dg_compact_rollup_publication_table",
        "columns": list(TABLE_COLUMNS),
        "rows": rows,
    }
    markdown_path = Path(output_markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_table(table) + "\n", encoding="utf-8")
    positive_rows = [
        row for row in rows if row["evidence_role"] == "positive stability evidence"
    ]
    negative_rows = [
        row for row in rows if row["evidence_role"] == "negative-control gate"
    ]
    checks = {
        "source_phase2df_passed": phase2df.get("passed") is True,
        "source_phase2de_passed": phase2de.get("passed") is True,
        "row_count_matches_phase2de_phase_count": len(rows)
        == int(phase2de.get("metrics", {}).get("phase_count", -1)),
        "all_rows_passed_and_bounded": bool(rows)
        and all(row["passed"] is True and row["bounded_claim_ok"] is True for row in rows),
        "positive_and_negative_rows_present": len(positive_rows) >= 8
        and len(negative_rows) >= 7,
        "markdown_table_written": markdown_path.exists(),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2dg_compact_rollup_publication_table",
        "passed": passed,
        "ready_for_bounded_compact_rollup_publication_table_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "positive_row_count": len(positive_rows),
            "negative_row_count": len(negative_rows),
        },
        "source_summary": {
            "phase2df_passed": phase2df.get("passed") is True,
            "phase2de_passed": phase2de.get("passed") is True,
            "phase_count": phase2de.get("metrics", {}).get("phase_count"),
            "positive_phase_count": phase2de.get("metrics", {}).get(
                "positive_phase_count"
            ),
            "negative_control_phase_count": phase2de.get("metrics", {}).get(
                "negative_control_phase_count"
            ),
        },
        "table": table,
        "claim_boundary": (
            "This publication table summarizes the bounded Phase2CP-DF evidence "
            "chain for package-internal structured runtime recovery. It is a "
            "presentation artifact only; it does not establish free-form shell "
            "autonomy, general runtime invariance, open-ended native perception, "
            "production autonomy, or an epoch-making architecture."
        ),
        "supported_claims": [
            "publication-ready compact table for bounded structured-runtime evidence"
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
            "phase2dh_publication_table_negative_controls"
            if passed
            else "repair_phase2dg_compact_rollup_publication_table"
        ),
        "evidence": {
            "phase2df_report_json": str(phase2df_report_json),
            "phase2de_report_json": str(phase2de_report_json),
            "publication_table_markdown": str(markdown_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2DG compact rollup publication table."
    )
    parser.add_argument("--phase2df-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2dg_compact_rollup_publication_table(
        phase2df_report_json=args.phase2df_report_json,
        output_report_json=args.output_report_json,
        output_markdown=args.output_markdown,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
