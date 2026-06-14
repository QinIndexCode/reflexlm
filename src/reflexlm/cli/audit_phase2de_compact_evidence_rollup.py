from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)


REPORT_SPECS: tuple[dict[str, str], ...] = (
    {
        "phase_id": "phase2cp",
        "filename": "phase2cp_cross_runtime_environment_stress_recovery_matrix.json",
        "category": "positive_runtime_matrix",
    },
    {
        "phase_id": "phase2cq",
        "filename": "phase2cq_cross_runtime_stress_negative_controls.json",
        "category": "negative_control",
    },
    {
        "phase_id": "phase2cr",
        "filename": "phase2cr_stress_recovery_repetition_stability.json",
        "category": "positive_repetition_stability",
    },
    {
        "phase_id": "phase2cs",
        "filename": "phase2cs_fresh_runtime_execution_repetition_stability.json",
        "category": "positive_fresh_execution",
    },
    {
        "phase_id": "phase2ct",
        "filename": "phase2ct_fresh_runtime_execution_negative_controls.json",
        "category": "negative_control",
    },
    {
        "phase_id": "phase2cu",
        "filename": "phase2cu_fresh_execution_runtime_perturbation_matrix.json",
        "category": "positive_runtime_perturbation",
    },
    {
        "phase_id": "phase2cv",
        "filename": "phase2cv_runtime_perturbation_negative_controls.json",
        "category": "negative_control",
    },
    {
        "phase_id": "phase2cw",
        "filename": "phase2cw_runtime_perturbation_recovery_stress_expansion.json",
        "category": "positive_expanded_recovery",
    },
    {
        "phase_id": "phase2cx",
        "filename": "phase2cx_expanded_recovery_stress_negative_controls.json",
        "category": "negative_control",
    },
    {
        "phase_id": "phase2cy",
        "filename": "phase2cy_expanded_recovery_cross_seed_stability.json",
        "category": "positive_cross_seed",
    },
    {
        "phase_id": "phase2cz",
        "filename": "phase2cz_cross_seed_negative_controls.json",
        "category": "negative_control",
    },
    {
        "phase_id": "phase2da",
        "filename": "phase2da_cross_seed_runtime_perturbation_composition.json",
        "category": "positive_composed_grid",
    },
    {
        "phase_id": "phase2db",
        "filename": "phase2db_composed_grid_negative_controls.json",
        "category": "negative_control",
    },
    {
        "phase_id": "phase2dc",
        "filename": "phase2dc_composed_grid_cross_order_stability.json",
        "category": "positive_cross_order",
    },
    {
        "phase_id": "phase2dd",
        "filename": "phase2dd_cross_order_negative_controls.json",
        "category": "negative_control",
    },
)


OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_general_shell_autonomy_claim",
    "ready_for_general_runtime_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _bounded_ready_flags(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key.startswith("ready_for_") and key not in OVERCLAIM_READY_FLAGS
    }


def _bounded_claim_ok(report: dict[str, Any]) -> bool:
    bounded_flags = _bounded_ready_flags(report)
    return (
        bool(bounded_flags)
        and any(value is True for value in bounded_flags.values())
        and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
    )


def _compact_metrics(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict):
        return {}
    keys = (
        "runtime_count",
        "runtime_rows",
        "runtime_interpreters",
        "seed_count",
        "perturbation_count",
        "fresh_runtime_execution_count",
        "passed_runtime_reports",
        "runtime_signature_mismatch_count",
        "control_count",
        "negative_control_count",
        "negative_controls_failed",
        "expected_failed_check_assertions",
        "order_count",
        "passed_order_validations",
        "order_validation_signature_mismatch_count",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _read_report_row(report_dir: Path, spec: dict[str, str]) -> dict[str, Any]:
    path = report_dir / spec["filename"]
    try:
        report = _read_json(path)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        return {
            "phase_id": spec["phase_id"],
            "category": spec["category"],
            "report_json": str(path),
            "readable": False,
            "read_error": type(exc).__name__,
            "passed": False,
            "artifact_family": None,
            "bounded_claim_ok": False,
            "compact_metrics": {},
        }
    return {
        "phase_id": spec["phase_id"],
        "category": spec["category"],
        "report_json": str(path),
        "readable": True,
        "passed": report.get("passed") is True,
        "artifact_family": report.get("artifact_family"),
        "bounded_claim_ok": _bounded_claim_ok(report),
        "ready_flags": {
            **_bounded_ready_flags(report),
            **{flag: report.get(flag) for flag in OVERCLAIM_READY_FLAGS},
        },
        "compact_metrics": _compact_metrics(report),
        "next_required_experiment": report.get("next_required_experiment"),
    }


def _phase_order_matches(rows: list[dict[str, Any]]) -> bool:
    return [row.get("phase_id") for row in rows] == [
        spec["phase_id"] for spec in REPORT_SPECS
    ]


def validate_phase2de_compact_evidence_rollup(
    report: dict[str, Any],
) -> dict[str, Any]:
    rows = report.get("phase_results", [])
    if not isinstance(rows, list):
        rows = []
    category_counts: dict[str, int] = {}
    for row in rows:
        category = str(row.get("category", ""))
        category_counts[category] = category_counts.get(category, 0) + 1
    expected_phase_count = len(REPORT_SPECS)
    positive_rows = [
        row for row in rows if str(row.get("category", "")).startswith("positive")
    ]
    negative_rows = [row for row in rows if row.get("category") == "negative_control"]
    checks = {
        "artifact_family_matches_phase2de": (
            report.get("artifact_family") == "phase2de_compact_evidence_rollup"
        ),
        "top_level_phase2de_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_compact_evidence_rollup_claim") is True
            and all(report.get(flag) is False for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "phase_order_matches_expected_chain": _phase_order_matches(rows),
        "all_expected_phase_reports_present": len(rows) == expected_phase_count,
        "all_phase_reports_readable": bool(rows)
        and all(row.get("readable") is True for row in rows),
        "all_phase_reports_passed": bool(rows)
        and all(row.get("passed") is True for row in rows),
        "all_phase_claims_are_bounded": bool(rows)
        and all(row.get("bounded_claim_ok") is True for row in rows),
        "positive_phase_count_met": len(positive_rows) >= 8,
        "negative_control_phase_count_met": len(negative_rows) >= 7,
        "compact_metrics_present_for_all_phases": bool(rows)
        and all(isinstance(row.get("compact_metrics"), dict) for row in rows),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "phase_count": len(rows),
            "expected_phase_count": expected_phase_count,
            "positive_phase_count": len(positive_rows),
            "negative_control_phase_count": len(negative_rows),
            "category_counts": category_counts,
        },
    }


def audit_phase2de_compact_evidence_rollup(
    *,
    phase2dd_report_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2dd_path = Path(phase2dd_report_json)
    report_dir = phase2dd_path.parent
    phase_rows = [_read_report_row(report_dir, spec) for spec in REPORT_SPECS]
    positive_rows = [
        row
        for row in phase_rows
        if str(row.get("category", "")).startswith("positive")
    ]
    negative_rows = [
        row for row in phase_rows if row.get("category") == "negative_control"
    ]
    checks = {
        "source_phase2dd_path_matches_expected_filename": (
            phase2dd_path.name == REPORT_SPECS[-1]["filename"]
        ),
        "all_expected_phase_reports_present": len(phase_rows) == len(REPORT_SPECS),
        "all_phase_reports_readable": all(
            row.get("readable") is True for row in phase_rows
        ),
        "all_phase_reports_passed": all(
            row.get("passed") is True for row in phase_rows
        ),
        "all_phase_claims_are_bounded": all(
            row.get("bounded_claim_ok") is True for row in phase_rows
        ),
        "positive_and_negative_evidence_present": len(positive_rows) >= 8
        and len(negative_rows) >= 7,
        "phase_order_matches_expected_chain": _phase_order_matches(phase_rows),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2de_compact_evidence_rollup",
        "passed": passed,
        "ready_for_bounded_compact_evidence_rollup_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "phase_count": len(phase_rows),
            "positive_phase_count": len(positive_rows),
            "negative_control_phase_count": len(negative_rows),
            "passed_phase_count": sum(row.get("passed") is True for row in phase_rows),
            "bounded_phase_count": sum(
                row.get("bounded_claim_ok") is True for row in phase_rows
            ),
        },
        "phase_results": phase_rows,
        "claim_boundary": (
            "Phase2CP-DD supports a bounded package-internal structured runtime "
            "cortex evidence chain for expanded recovery under cross-runtime, "
            "cross-seed, runtime-perturbation, composed-grid, and cross-order "
            "audits plus negative controls. It does not prove free-form shell "
            "autonomy, general runtime invariance, open-ended native perception, "
            "production autonomy, or an epoch-making architecture."
        ),
        "supported_claims": [
            (
                "bounded compact evidence rollup across positive recovery stability "
                "audits and negative-control gates"
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
            "phase2df_compact_rollup_negative_controls"
            if passed
            else "repair_phase2de_compact_evidence_rollup"
        ),
        "evidence": {
            "phase2dd_report_json": str(phase2dd_report_json),
            "report_dir": str(report_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2DE compact evidence rollup from Phase2CP-DD reports."
    )
    parser.add_argument("--phase2dd-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2de_compact_evidence_rollup(
        phase2dd_report_json=args.phase2dd_report_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
