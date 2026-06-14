from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2de_compact_evidence_rollup import (
    validate_phase2de_compact_evidence_rollup,
)


Mutation = Callable[[dict[str, Any], Path], None]


def _phase_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("phase_results", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["all_phase_claims_are_bounded"] = False


def _mutate_missing_phase_row(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["phase_results"] = _phase_rows(report)[:-1]
    report.setdefault("metrics", {})["phase_count"] = len(_phase_rows(report))


def _mutate_phase_order_swapped(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    rows = _phase_rows(report)
    if len(rows) >= 2:
        rows[0], rows[1] = rows[1], rows[0]
    report["phase_results"] = rows


def _mutate_phase_unreadable(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    row = _phase_rows(report)[0]
    row["readable"] = False
    row["read_error"] = "SyntheticMissingReport"


def _mutate_phase_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    row = _phase_rows(report)[0]
    row["passed"] = False


def _mutate_phase_claim_unbounded(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    row = _phase_rows(report)[0]
    row["bounded_claim_ok"] = False
    row.setdefault("ready_flags", {})["ready_for_epoch_making_architecture_claim"] = True


def _mutate_positive_phase_count_collapsed(
    report: dict[str, Any],
    case_dir: Path,
) -> None:
    del case_dir
    rows = [
        row
        for row in _phase_rows(report)
        if not str(row.get("category", "")).startswith("positive")
        or row.get("phase_id") == "phase2cp"
    ]
    report["phase_results"] = rows
    report.setdefault("metrics", {})["positive_phase_count"] = 1
    report.setdefault("metrics", {})["phase_count"] = len(rows)


def _mutate_negative_control_count_collapsed(
    report: dict[str, Any],
    case_dir: Path,
) -> None:
    del case_dir
    rows = [
        row
        for row in _phase_rows(report)
        if row.get("category") != "negative_control"
        or row.get("phase_id") == "phase2cq"
    ]
    report["phase_results"] = rows
    report.setdefault("metrics", {})["negative_control_phase_count"] = 1
    report.setdefault("metrics", {})["phase_count"] = len(rows)


def _mutate_compact_metrics_invalid(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    row = _phase_rows(report)[0]
    row["compact_metrics"] = None


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2de_report",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_recorded_check_false",
        "mutation": _mutate_recorded_check_false,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
    {
        "control_id": "negative_missing_phase_row",
        "mutation": _mutate_missing_phase_row,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_expected_phase_reports_present",
            "phase_order_matches_expected_chain",
        ],
    },
    {
        "control_id": "negative_phase_order_swapped",
        "mutation": _mutate_phase_order_swapped,
        "expected_passed": False,
        "expected_failed_checks": ["phase_order_matches_expected_chain"],
    },
    {
        "control_id": "negative_phase_unreadable",
        "mutation": _mutate_phase_unreadable,
        "expected_passed": False,
        "expected_failed_checks": ["all_phase_reports_readable"],
    },
    {
        "control_id": "negative_phase_failed",
        "mutation": _mutate_phase_failed,
        "expected_passed": False,
        "expected_failed_checks": ["all_phase_reports_passed"],
    },
    {
        "control_id": "negative_phase_claim_unbounded",
        "mutation": _mutate_phase_claim_unbounded,
        "expected_passed": False,
        "expected_failed_checks": ["all_phase_claims_are_bounded"],
    },
    {
        "control_id": "negative_positive_phase_count_collapsed",
        "mutation": _mutate_positive_phase_count_collapsed,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_expected_phase_reports_present",
            "phase_order_matches_expected_chain",
            "positive_phase_count_met",
        ],
    },
    {
        "control_id": "negative_negative_control_count_collapsed",
        "mutation": _mutate_negative_control_count_collapsed,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_expected_phase_reports_present",
            "phase_order_matches_expected_chain",
            "negative_control_phase_count_met",
        ],
    },
    {
        "control_id": "negative_compact_metrics_invalid",
        "mutation": _mutate_compact_metrics_invalid,
        "expected_passed": False,
        "expected_failed_checks": ["compact_metrics_present_for_all_phases"],
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
    phase2de_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"c{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = deepcopy(phase2de_report)
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2de_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2de_compact_evidence_rollup(control_report)
    validation_report_json = case_dir / "phase2de_validation.json"
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


def audit_phase2df_compact_rollup_negative_controls(
    *,
    phase2de_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2de_report = _read_json(phase2de_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2de_report=phase2de_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2de_report"
    ]
    checks = {
        "source_phase2de_passed": phase2de_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2de_report"
            and row["observed_passed"] is True
            and row["pass_expectation_met"] is True
            for row in control_rows
        ),
        "minimum_negative_control_count_met": len(negative_rows) >= 10,
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
        "artifact_family": "phase2df_compact_rollup_negative_controls",
        "passed": passed,
        "ready_for_phase2de_gate_strictness_claim": passed,
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
                "the Phase2DE compact rollup gate rejects overclaims for recorded "
                "check tampering, missing or reordered phases, unreadable or failed "
                "phase rows, unbounded phase claims, collapsed evidence categories, "
                "invalid compact metrics, and overstated epoch claims"
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
            "phase2dg_compact_rollup_publication_table"
            if passed
            else "repair_phase2df_compact_rollup_negative_controls"
        ),
        "evidence": {
            "phase2de_report_json": str(phase2de_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2DF compact rollup negative controls."
    )
    parser.add_argument("--phase2de-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2df_compact_rollup_negative_controls(
        phase2de_report_json=args.phase2de_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
