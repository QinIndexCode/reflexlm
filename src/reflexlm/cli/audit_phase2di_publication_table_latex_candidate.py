from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)


LATEX_COLUMNS: tuple[str, ...] = (
    "Phase",
    "Role",
    "Status",
    "Key metrics",
    "Boundary / next evidence",
)

OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_general_shell_autonomy_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _status(row: dict[str, Any]) -> str:
    if row.get("passed") is True and row.get("bounded_claim_ok") is True:
        return "passed; bounded"
    return "not claim-bearing"


def _boundary_next(row: dict[str, Any]) -> str:
    return f"next: {row.get('next_required_experiment')}; no autonomy overclaim"


def build_latex_table(table: dict[str, Any]) -> str:
    rows = table.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{1pt}",
        r"\caption{Compact bounded evidence chain for structured runtime recovery.}",
        r"\label{tab:phase2dg-compact-rollup-candidate}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.09\linewidth}>{\raggedright\arraybackslash}p{0.13\linewidth}>{\raggedright\arraybackslash}p{0.10\linewidth}>{\raggedright\arraybackslash}p{0.31\linewidth}>{\raggedright\arraybackslash}p{0.28\linewidth}@{}}",
        r"\toprule",
        " & ".join(_latex_escape(column) for column in LATEX_COLUMNS) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = [
            row.get("phase_id"),
            row.get("evidence_role"),
            _status(row),
            row.get("key_metrics"),
            _boundary_next(row),
        ]
        lines.append(" & ".join(_latex_escape(value) for value in values) + r" \\")
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{5}{@{}p{0.95\linewidth}@{}}{\textbf{Boundary:} bounded package-internal structured runtime evidence only; not free-form shell autonomy, general runtime invariance, open-ended native perception, production autonomy, or epoch-making architecture.} \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _ready_boundary_ok(report: dict[str, Any]) -> bool:
    return (
        report.get("ready_for_bounded_publication_table_latex_candidate_claim")
        is True
        and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
    )


def _is_under_docs_tables(path: str | Path) -> bool:
    normalized = str(Path(path)).replace("\\", "/").lower()
    return "/docs/paper_b/tables/" in f"/{normalized}"


def validate_phase2di_publication_table_latex_candidate(
    report: dict[str, Any],
) -> dict[str, Any]:
    rows = report.get("table_summary", {}).get("rows", [])
    if not isinstance(rows, list):
        rows = []
    latex_path = report.get("evidence", {}).get("latex_candidate_path")
    latex_readable = False
    latex_text = ""
    if latex_path:
        try:
            latex_text = Path(latex_path).read_text(encoding="utf-8")
            latex_readable = True
        except OSError:
            latex_readable = False
    checks = {
        "artifact_family_matches_phase2di": (
            report.get("artifact_family")
            == "phase2di_publication_table_latex_candidate"
        ),
        "top_level_phase2di_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": _ready_boundary_ok(report),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "source_phase2dg_and_phase2dh_passed": (
            report.get("source_summary", {}).get("phase2dg_passed") is True
            and report.get("source_summary", {}).get("phase2dh_passed") is True
        ),
        "row_count_matches_source": len(rows)
        == int(report.get("source_summary", {}).get("row_count", -1)),
        "latex_candidate_readable": latex_readable,
        "latex_candidate_not_in_main_tables_dir": bool(latex_path)
        and not _is_under_docs_tables(latex_path),
        "latex_contains_table_environment": "\\begin{table}" in latex_text
        and "\\end{table}" in latex_text,
        "latex_contains_caption_and_label": "\\caption{" in latex_text
        and "\\label{tab:phase2dg-compact-rollup-candidate}" in latex_text,
        "latex_contains_all_phase_ids": bool(rows)
        and all(str(row.get("phase_id")) in latex_text for row in rows),
        "latex_contains_bounded_boundary": (
            "not free-form shell autonomy" in latex_text
            and "not free-form shell autonomy" in report.get("claim_boundary", "")
        ),
        "all_rows_passed_and_bounded": bool(rows)
        and all(
            row.get("passed") is True and row.get("bounded_claim_ok") is True
            for row in rows
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "latex_readable": latex_readable,
            "latex_bytes": len(latex_text.encode("utf-8")) if latex_text else 0,
        },
    }


def audit_phase2di_publication_table_latex_candidate(
    *,
    phase2dh_report_json: str | Path,
    output_report_json: str | Path,
    output_latex: str | Path,
) -> dict[str, Any]:
    phase2dh = _read_json(phase2dh_report_json)
    phase2dg_report_json = phase2dh.get("evidence", {}).get("phase2dg_report_json")
    if not phase2dg_report_json:
        raise ValueError("Phase2DI requires Phase2DH evidence.phase2dg_report_json")
    phase2dg = _read_json(phase2dg_report_json)
    table = phase2dg.get("table", {})
    rows = table.get("rows", []) if isinstance(table, dict) else []
    if not isinstance(rows, list):
        rows = []
    latex_path = Path(output_latex)
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(build_latex_table(table), encoding="utf-8")
    checks = {
        "source_phase2dh_passed": phase2dh.get("passed") is True,
        "source_phase2dg_passed": phase2dg.get("passed") is True,
        "row_count_matches_phase2dg": len(rows)
        == int(phase2dg.get("metrics", {}).get("row_count", -1)),
        "all_rows_passed_and_bounded": bool(rows)
        and all(
            row.get("passed") is True and row.get("bounded_claim_ok") is True
            for row in rows
        ),
        "latex_candidate_written": latex_path.exists(),
        "latex_candidate_not_in_main_tables_dir": not _is_under_docs_tables(latex_path),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2di_publication_table_latex_candidate",
        "passed": passed,
        "ready_for_bounded_publication_table_latex_candidate_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "latex_candidate_written": latex_path.exists(),
        },
        "source_summary": {
            "phase2dh_passed": phase2dh.get("passed") is True,
            "phase2dg_passed": phase2dg.get("passed") is True,
            "row_count": phase2dg.get("metrics", {}).get("row_count"),
            "positive_row_count": phase2dg.get("metrics", {}).get(
                "positive_row_count"
            ),
            "negative_row_count": phase2dg.get("metrics", {}).get(
                "negative_row_count"
            ),
        },
        "table_summary": {
            "columns": list(LATEX_COLUMNS),
            "rows": [
                {
                    "phase_id": row.get("phase_id"),
                    "passed": row.get("passed") is True,
                    "bounded_claim_ok": row.get("bounded_claim_ok") is True,
                }
                for row in rows
                if isinstance(row, dict)
            ],
        },
        "claim_boundary": (
            "This LaTeX candidate is an artifact-level table for bounded "
            "package-internal structured runtime evidence only; it is not inserted "
            "into the main paper tables and is not free-form shell "
            "autonomy, not general runtime invariance, not open-ended native perception, "
            "not production autonomy, and not an epoch-making architecture."
        ),
        "supported_claims": [
            "artifact-level LaTeX candidate for bounded compact evidence table"
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
            "phase2dj_latex_candidate_negative_controls"
            if passed
            else "repair_phase2di_publication_table_latex_candidate"
        ),
        "evidence": {
            "phase2dh_report_json": str(phase2dh_report_json),
            "phase2dg_report_json": str(phase2dg_report_json),
            "latex_candidate_path": str(latex_path),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2DI publication table LaTeX candidate."
    )
    parser.add_argument("--phase2dh-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--output-latex", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2di_publication_table_latex_candidate(
        phase2dh_report_json=args.phase2dh_report_json,
        output_report_json=args.output_report_json,
        output_latex=args.output_latex,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
