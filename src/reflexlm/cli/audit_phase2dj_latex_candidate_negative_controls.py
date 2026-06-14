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
from reflexlm.cli.audit_phase2di_publication_table_latex_candidate import (
    validate_phase2di_publication_table_latex_candidate,
)


Mutation = Callable[[dict[str, Any], Path], None]


def _summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("table_summary", {}).get("rows", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _latex_path(report: dict[str, Any]) -> Path:
    return Path(str(report.get("evidence", {}).get("latex_candidate_path", "")))


def _materialize_control_report(
    *,
    phase2di_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    control_report = deepcopy(phase2di_report)
    source_latex = _latex_path(phase2di_report)
    if not source_latex.exists():
        raise ValueError("Phase2DJ requires a readable Phase2DI LaTeX candidate")
    target_latex = case_dir / "table.tex"
    target_latex.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_latex, target_latex)
    control_report.setdefault("evidence", {})[
        "latex_candidate_path"
    ] = str(target_latex)
    return control_report


def _mutate_source_summary_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("source_summary", {})["phase2dh_passed"] = False


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["all_rows_passed_and_bounded"] = False


def _mutate_missing_summary_row(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("table_summary", {})["rows"] = _summary_rows(report)[:-1]


def _mutate_summary_row_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    _summary_rows(report)[0]["passed"] = False


def _mutate_summary_row_unbounded(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    _summary_rows(report)[0]["bounded_claim_ok"] = False


def _mutate_missing_latex(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    latex = _latex_path(report)
    if latex.exists():
        latex.unlink()


def _mutate_latex_missing_phase(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    latex = _latex_path(report)
    phase_id = str(_summary_rows(report)[0]["phase_id"])
    latex.write_text(
        latex.read_text(encoding="utf-8").replace(phase_id, "phase2xx"),
        encoding="utf-8",
    )


def _mutate_latex_missing_table_environment(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    latex = _latex_path(report)
    text = latex.read_text(encoding="utf-8")
    text = text.replace(r"\begin{table}[ht]", r"\begin{center}")
    text = text.replace(r"\end{table}", r"\end{center}")
    latex.write_text(text, encoding="utf-8")


def _mutate_latex_missing_caption_label(
    report: dict[str, Any], case_dir: Path
) -> None:
    del case_dir
    latex = _latex_path(report)
    text = latex.read_text(encoding="utf-8")
    text = text.replace(
        r"\label{tab:phase2dg-compact-rollup-candidate}",
        r"\label{tab:overclaimed-main-result}",
    )
    latex.write_text(text, encoding="utf-8")


def _mutate_latex_missing_boundary(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    latex = _latex_path(report)
    text = latex.read_text(encoding="utf-8")
    text = text.replace(
        "not free-form shell autonomy, general runtime invariance, open-ended "
        "native perception, production autonomy, or epoch-making architecture",
        "free-form shell autonomy and epoch-making architecture",
    )
    latex.write_text(text, encoding="utf-8")


def _mutate_latex_in_main_tables_dir(report: dict[str, Any], case_dir: Path) -> None:
    target_latex = case_dir / "docs" / "paper_b" / "tables" / "table.tex"
    target_latex.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_latex_path(report), target_latex)
    report.setdefault("evidence", {})["latex_candidate_path"] = str(target_latex)


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2di_report",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_source_summary_failed",
        "mutation": _mutate_source_summary_failed,
        "expected_passed": False,
        "expected_failed_checks": ["source_phase2dg_and_phase2dh_passed"],
    },
    {
        "control_id": "negative_recorded_check_false",
        "mutation": _mutate_recorded_check_false,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
    {
        "control_id": "negative_missing_summary_row",
        "mutation": _mutate_missing_summary_row,
        "expected_passed": False,
        "expected_failed_checks": ["row_count_matches_source"],
    },
    {
        "control_id": "negative_summary_row_failed",
        "mutation": _mutate_summary_row_failed,
        "expected_passed": False,
        "expected_failed_checks": ["all_rows_passed_and_bounded"],
    },
    {
        "control_id": "negative_summary_row_unbounded",
        "mutation": _mutate_summary_row_unbounded,
        "expected_passed": False,
        "expected_failed_checks": ["all_rows_passed_and_bounded"],
    },
    {
        "control_id": "negative_missing_latex",
        "mutation": _mutate_missing_latex,
        "expected_passed": False,
        "expected_failed_checks": [
            "latex_candidate_readable",
            "latex_contains_table_environment",
            "latex_contains_caption_and_label",
            "latex_contains_all_phase_ids",
            "latex_contains_bounded_boundary",
        ],
    },
    {
        "control_id": "negative_latex_missing_phase",
        "mutation": _mutate_latex_missing_phase,
        "expected_passed": False,
        "expected_failed_checks": ["latex_contains_all_phase_ids"],
    },
    {
        "control_id": "negative_latex_missing_table_environment",
        "mutation": _mutate_latex_missing_table_environment,
        "expected_passed": False,
        "expected_failed_checks": ["latex_contains_table_environment"],
    },
    {
        "control_id": "negative_latex_missing_caption_label",
        "mutation": _mutate_latex_missing_caption_label,
        "expected_passed": False,
        "expected_failed_checks": ["latex_contains_caption_and_label"],
    },
    {
        "control_id": "negative_latex_missing_boundary",
        "mutation": _mutate_latex_missing_boundary,
        "expected_passed": False,
        "expected_failed_checks": ["latex_contains_bounded_boundary"],
    },
    {
        "control_id": "negative_latex_in_main_tables_dir",
        "mutation": _mutate_latex_in_main_tables_dir,
        "expected_passed": False,
        "expected_failed_checks": ["latex_candidate_not_in_main_tables_dir"],
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
    phase2di_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"c{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2di_report=phase2di_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2di_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2di_publication_table_latex_candidate(control_report)
    validation_report_json = case_dir / "phase2di_validation.json"
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


def audit_phase2dj_latex_candidate_negative_controls(
    *,
    phase2di_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2di_report = _read_json(phase2di_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2di_report=phase2di_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2di_report"
    ]
    checks = {
        "source_phase2di_passed": phase2di_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2di_report"
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
        "artifact_family": "phase2dj_latex_candidate_negative_controls",
        "passed": passed,
        "ready_for_phase2di_gate_strictness_claim": passed,
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
                "the Phase2DI LaTeX-candidate gate rejects source tampering, "
                "recorded check tampering, row loss, failed or unbounded rows, "
                "missing or incomplete LaTeX, boundary deletion, main-paper table "
                "directory leakage, and overstated epoch claims"
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
            "phase2dk_latex_candidate_publication_bundle"
            if passed
            else "repair_phase2dj_latex_candidate_negative_controls"
        ),
        "evidence": {
            "phase2di_report_json": str(phase2di_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2DJ LaTeX candidate negative controls."
    )
    parser.add_argument("--phase2di-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2dj_latex_candidate_negative_controls(
        phase2di_report_json=args.phase2di_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
