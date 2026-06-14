from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _stable_matrix_signature,
    _write_json,
    audit_phase2cs_fresh_runtime_execution_repetition_stability,
    validate_phase2cs_fresh_runtime_execution_report,
)


DEFAULT_PERTURBATIONS: tuple[dict[str, Any], ...] = (
    {
        "perturbation_id": "baseline_budget",
        "timeout_seconds": 5.0,
        "max_extra_steps": 5,
    },
    {
        "perturbation_id": "extended_timeout_budget",
        "timeout_seconds": 7.5,
        "max_extra_steps": 5,
    },
    {
        "perturbation_id": "extended_step_budget",
        "timeout_seconds": 5.0,
        "max_extra_steps": 6,
    },
)


def _matrix_signatures_from_phase2cs(report: dict[str, Any]) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    for repetition in report.get("repetition_results", []):
        matrix_path = repetition.get("matrix_audit_json")
        try:
            matrix = _read_json(matrix_path)
        except (OSError, TypeError, json.JSONDecodeError):
            matrix = {}
        signatures.append(_stable_matrix_signature(matrix))
    return signatures


def _runtime_interpreters(report: dict[str, Any]) -> list[str]:
    rows = report.get("metrics", {}).get("runtime_interpreters", [])
    return [str(row) for row in rows] if isinstance(rows, list) else []


def _core_signature(report: dict[str, Any]) -> dict[str, Any]:
    matrix_signatures = _matrix_signatures_from_phase2cs(report)
    return {
        "runtime_interpreters": _runtime_interpreters(report),
        "runtime_count": report.get("metrics", {}).get("runtime_count"),
        "fresh_runtime_execution_count": report.get("metrics", {}).get(
            "fresh_runtime_execution_count"
        ),
        "passed_runtime_reports": report.get("metrics", {}).get(
            "passed_runtime_reports"
        ),
        "passed_matrix_audits": report.get("metrics", {}).get(
            "passed_matrix_audits"
        ),
        "matrix_signature_mismatch_count": report.get("metrics", {}).get(
            "matrix_signature_mismatch_count"
        ),
        "runtime_signature_mismatch_count": report.get("metrics", {}).get(
            "runtime_signature_mismatch_count"
        ),
        "matrix_signatures": matrix_signatures,
        "ready_for_fresh_runtime_execution_repetition_stability_claim": report.get(
            "ready_for_fresh_runtime_execution_repetition_stability_claim"
        ),
        "ready_for_general_shell_autonomy_claim": report.get(
            "ready_for_general_shell_autonomy_claim"
        ),
        "ready_for_general_runtime_invariance_claim": report.get(
            "ready_for_general_runtime_invariance_claim"
        ),
        "ready_for_open_ended_native_perception_claim": report.get(
            "ready_for_open_ended_native_perception_claim"
        ),
        "ready_for_production_autonomy_claim": report.get(
            "ready_for_production_autonomy_claim"
        ),
        "ready_for_epoch_making_architecture_claim": report.get(
            "ready_for_epoch_making_architecture_claim"
        ),
    }


def validate_phase2cu_fresh_execution_runtime_perturbation_matrix(
    report: dict[str, Any],
) -> dict[str, Any]:
    perturbation_rows = report.get("perturbation_results", [])
    if not isinstance(perturbation_rows, list):
        perturbation_rows = []
    recomputed_rows: list[dict[str, Any]] = []
    phase2cs_read_failures = 0
    validation_read_failures = 0
    for row in perturbation_rows:
        try:
            phase2cs = _read_json(row.get("phase2cs_report_json"))
        except (OSError, TypeError, json.JSONDecodeError):
            phase2cs_read_failures += 1
            phase2cs = {}
        try:
            validation = _read_json(row.get("validation_report_json"))
        except (OSError, TypeError, json.JSONDecodeError):
            validation_read_failures += 1
            validation = {}
        recomputed_rows.append(
            {
                "phase2cs_passed": phase2cs.get("passed") is True,
                "validation_passed": validation.get("passed") is True,
                "core_signature": _core_signature(phase2cs),
            }
        )

    reference_signature = (
        recomputed_rows[0]["core_signature"] if recomputed_rows else {}
    )
    signature_mismatch_count = sum(
        row["core_signature"] != reference_signature for row in recomputed_rows
    )
    checks = {
        "artifact_family_matches_phase2cu": (
            report.get("artifact_family")
            == "phase2cu_fresh_execution_runtime_perturbation_matrix"
        ),
        "top_level_phase2cu_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_fresh_execution_runtime_perturbation_claim")
            is True
            and report.get("ready_for_general_shell_autonomy_claim") is False
            and report.get("ready_for_general_runtime_invariance_claim") is False
            and report.get("ready_for_open_ended_native_perception_claim") is False
            and report.get("ready_for_production_autonomy_claim") is False
            and report.get("ready_for_epoch_making_architecture_claim") is False
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "minimum_three_perturbations_present": len(perturbation_rows) >= 3,
        "all_phase2cs_reports_readable": bool(perturbation_rows)
        and phase2cs_read_failures == 0,
        "all_validation_reports_readable": bool(perturbation_rows)
        and validation_read_failures == 0,
        "all_recomputed_phase2cs_reports_passed": bool(recomputed_rows)
        and all(row["phase2cs_passed"] is True for row in recomputed_rows),
        "all_recomputed_validations_passed": bool(recomputed_rows)
        and all(row["validation_passed"] is True for row in recomputed_rows),
        "all_recomputed_core_signatures_match": bool(recomputed_rows)
        and signature_mismatch_count == 0,
        "all_recomputed_perturbations_keep_bounded_claims_only": bool(
            recomputed_rows
        )
        and all(
            row["core_signature"].get(
                "ready_for_fresh_runtime_execution_repetition_stability_claim"
            )
            is True
            and row["core_signature"].get("ready_for_general_shell_autonomy_claim")
            is False
            and row["core_signature"].get(
                "ready_for_general_runtime_invariance_claim"
            )
            is False
            and row["core_signature"].get(
                "ready_for_open_ended_native_perception_claim"
            )
            is False
            and row["core_signature"].get("ready_for_production_autonomy_claim")
            is False
            and row["core_signature"].get(
                "ready_for_epoch_making_architecture_claim"
            )
            is False
            for row in recomputed_rows
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "perturbation_count": len(perturbation_rows),
            "phase2cs_read_failures": phase2cs_read_failures,
            "validation_read_failures": validation_read_failures,
            "recomputed_phase2cs_passed": sum(
                row["phase2cs_passed"] is True for row in recomputed_rows
            ),
            "recomputed_validations_passed": sum(
                row["validation_passed"] is True for row in recomputed_rows
            ),
            "recomputed_core_signature_mismatch_count": signature_mismatch_count,
        },
    }


def _phase2cs_source_from_phase2ct(phase2ct: dict[str, Any]) -> dict[str, str]:
    phase2cs_report_json = phase2ct.get("evidence", {}).get("phase2cs_report_json")
    if not phase2cs_report_json:
        raise ValueError("Phase2CU requires Phase2CT evidence.phase2cs_report_json")
    phase2cs = _read_json(phase2cs_report_json)
    evidence = phase2cs.get("evidence", {})
    phase2cp_report_json = evidence.get("phase2cp_report_json")
    suite_json = evidence.get("suite_json")
    if not phase2cp_report_json or not suite_json:
        raise ValueError("Phase2CU requires Phase2CS evidence phase2cp_report_json and suite_json")
    return {
        "phase2cs_report_json": str(phase2cs_report_json),
        "phase2cp_report_json": str(phase2cp_report_json),
        "suite_json": str(suite_json),
    }


def audit_phase2cu_fresh_execution_runtime_perturbation_matrix(
    *,
    phase2ct_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    repetition_count: int = 2,
    perturbations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    phase2ct = _read_json(phase2ct_report_json)
    source = _phase2cs_source_from_phase2ct(phase2ct)
    perturbation_specs = list(perturbations or DEFAULT_PERTURBATIONS)
    perturbation_rows: list[dict[str, Any]] = []

    for index, spec in enumerate(perturbation_specs):
        perturbation_id = str(spec["perturbation_id"])
        timeout_seconds = float(spec["timeout_seconds"])
        max_extra_steps = int(spec["max_extra_steps"])
        case_dir = Path(output_dir) / f"{index:02d}_{perturbation_id}"
        phase2cs_report_json = case_dir / "phase2cs_fresh_execution_report.json"
        phase2cs_report = audit_phase2cs_fresh_runtime_execution_repetition_stability(
            phase2cp_report_json=source["phase2cp_report_json"],
            suite_json=source["suite_json"],
            output_dir=case_dir / "fresh_execution",
            output_report_json=phase2cs_report_json,
            repetition_count=repetition_count,
            timeout_seconds=timeout_seconds,
            max_extra_steps=max_extra_steps,
        )
        validation = validate_phase2cs_fresh_runtime_execution_report(phase2cs_report)
        validation_report_json = case_dir / "phase2cs_validation.json"
        _write_json(validation_report_json, validation)
        perturbation_rows.append(
            {
                "perturbation_id": perturbation_id,
                "timeout_seconds": timeout_seconds,
                "max_extra_steps": max_extra_steps,
                "phase2cs_report_json": str(phase2cs_report_json),
                "validation_report_json": str(validation_report_json),
                "phase2cs_passed": phase2cs_report.get("passed") is True,
                "validation_passed": validation.get("passed") is True,
                "core_signature": _core_signature(phase2cs_report),
            }
        )

    reference_signature = (
        perturbation_rows[0]["core_signature"] if perturbation_rows else {}
    )
    signature_mismatch_count = sum(
        row["core_signature"] != reference_signature for row in perturbation_rows
    )
    checks = {
        "source_phase2ct_passed": phase2ct.get("passed") is True,
        "minimum_three_perturbations_met": len(perturbation_rows) >= 3,
        "all_perturbation_phase2cs_reports_passed": bool(perturbation_rows)
        and all(row["phase2cs_passed"] is True for row in perturbation_rows),
        "all_perturbation_phase2cs_validations_passed": bool(perturbation_rows)
        and all(row["validation_passed"] is True for row in perturbation_rows),
        "all_perturbation_core_signatures_match": bool(perturbation_rows)
        and signature_mismatch_count == 0,
        "all_perturbations_keep_bounded_claims_only": bool(perturbation_rows)
        and all(
            row["core_signature"].get(
                "ready_for_fresh_runtime_execution_repetition_stability_claim"
            )
            is True
            and row["core_signature"].get("ready_for_general_shell_autonomy_claim")
            is False
            and row["core_signature"].get(
                "ready_for_general_runtime_invariance_claim"
            )
            is False
            and row["core_signature"].get(
                "ready_for_open_ended_native_perception_claim"
            )
            is False
            and row["core_signature"].get("ready_for_production_autonomy_claim")
            is False
            and row["core_signature"].get(
                "ready_for_epoch_making_architecture_claim"
            )
            is False
            for row in perturbation_rows
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cu_fresh_execution_runtime_perturbation_matrix",
        "passed": passed,
        "ready_for_bounded_fresh_execution_runtime_perturbation_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "perturbation_count": len(perturbation_rows),
            "passed_phase2cs_reports": sum(
                row["phase2cs_passed"] is True for row in perturbation_rows
            ),
            "passed_phase2cs_validations": sum(
                row["validation_passed"] is True for row in perturbation_rows
            ),
            "core_signature_mismatch_count": signature_mismatch_count,
            "repetition_count_per_perturbation": repetition_count,
            "fresh_runtime_execution_count": sum(
                int(row["core_signature"].get("fresh_runtime_execution_count") or 0)
                for row in perturbation_rows
            ),
        },
        "perturbation_results": perturbation_rows,
        "supported_claims": [
            (
                "bounded package-internal structured-runtime cortex fresh execution "
                "remained stable across safe runtime execution-budget perturbations "
                "while preserving Phase2CS validation and cross-runtime matrix "
                "signatures"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "free-form shell autonomy",
            "arbitrary shell/environment generalization",
            "general runtime invariance beyond safe budget perturbations on the recorded CPython matrix",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cv_runtime_perturbation_negative_controls"
            if passed
            else "repair_phase2cu_fresh_execution_runtime_perturbation_matrix"
        ),
        "evidence": {
            "phase2ct_report_json": str(phase2ct_report_json),
            **source,
            "runtime_perturbation_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2CU fresh execution runtime perturbation matrix."
    )
    parser.add_argument("--phase2ct-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--repetition-count", type=int, default=2)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2cu_fresh_execution_runtime_perturbation_matrix(
        phase2ct_report_json=args.phase2ct_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
        repetition_count=args.repetition_count,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
