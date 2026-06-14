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
    validate_phase2cs_fresh_runtime_execution_report,
)


Mutation = Callable[[dict[str, Any], Path], None]


def _runtime_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for repetition in report.get("repetition_results", [])
        for row in repetition.get("runtime_results", [])
        if isinstance(row, dict)
    ]


def _matrix_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in report.get("repetition_results", [])
        if isinstance(row, dict)
    ]


def _rewrite_paths(payload: Any, *, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(payload, dict):
        return {
            key: _rewrite_paths(value, old_prefix=old_prefix, new_prefix=new_prefix)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [
            _rewrite_paths(value, old_prefix=old_prefix, new_prefix=new_prefix)
            for value in payload
        ]
    if isinstance(payload, str):
        return payload.replace(old_prefix, new_prefix)
    return payload


def _materialize_control_report(
    *,
    phase2cs_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    evidence = phase2cs_report.get("evidence", {})
    source_output_dir = evidence.get("fresh_execution_output_dir")
    if not source_output_dir:
        raise ValueError("Phase2CT requires Phase2CS evidence.fresh_execution_output_dir")
    copied_output_dir = case_dir / "fresh_execution_output"
    if copied_output_dir.exists():
        shutil.rmtree(copied_output_dir)
    shutil.copytree(source_output_dir, copied_output_dir)
    copied_report = deepcopy(phase2cs_report)
    copied_report = _rewrite_paths(
        copied_report,
        old_prefix=str(source_output_dir),
        new_prefix=str(copied_output_dir),
    )
    copied_report.setdefault("evidence", {})[
        "fresh_execution_output_dir"
    ] = str(copied_output_dir)
    return copied_report


def _mutate_subprocess_nonzero(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    row = _runtime_rows(report)[0]
    row.setdefault("subprocess", {})["returncode"] = 17


def _mutate_missing_runtime_report(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    row = _runtime_rows(report)[0]
    row["report_exists"] = False
    report_path = Path(str(row["report_json"]))
    if report_path.exists():
        report_path.unlink()


def _mutate_runtime_report_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    row = _runtime_rows(report)[0]
    row["report_passed"] = False
    runtime_report = _read_json(row["report_json"])
    runtime_report["passed"] = False
    runtime_report["ready_for_bounded_environment_stress_failure_recovery_claim"] = False
    _write_json(row["report_json"], runtime_report)


def _mutate_matrix_audit_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    row = _matrix_rows(report)[0]
    row["matrix_passed"] = False
    matrix = _read_json(row["matrix_audit_json"])
    matrix["passed"] = False
    matrix["checks"]["all_runtime_reports_passed"] = False
    _write_json(row["matrix_audit_json"], matrix)


def _mutate_runtime_signature_drift(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    rows = _runtime_rows(report)
    row = rows[min(4, len(rows) - 1)]
    runtime_report = _read_json(row["report_json"])
    metrics = runtime_report.setdefault("metrics", {})
    metrics["executed_actions"] = int(metrics.get("executed_actions", 0)) + 1
    _write_json(row["report_json"], runtime_report)


def _mutate_matrix_signature_drift(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    rows = _matrix_rows(report)
    row = rows[min(1, len(rows) - 1)]
    matrix = _read_json(row["matrix_audit_json"])
    matrix.setdefault("metrics", {})["distinct_python_versions"] = 1
    _write_json(row["matrix_audit_json"], matrix)


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


def _mutate_output_outside_repetition_dir(report: dict[str, Any], case_dir: Path) -> None:
    row = _runtime_rows(report)[0]
    row["generated_manifest_dir_under_repetition_dir"] = False
    runtime_report = _read_json(row["report_json"])
    runtime_report["generated_manifest_dir"] = str(case_dir / "outside_manifests")
    _write_json(row["report_json"], runtime_report)


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2cs_report",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_subprocess_nonzero",
        "mutation": _mutate_subprocess_nonzero,
        "expected_passed": False,
        "expected_failed_checks": ["all_runtime_subprocesses_recorded_zero"],
    },
    {
        "control_id": "negative_missing_runtime_report",
        "mutation": _mutate_missing_runtime_report,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_runtime_reports_readable",
            "all_runtime_reports_exist_flags_true",
        ],
    },
    {
        "control_id": "negative_runtime_report_failed",
        "mutation": _mutate_runtime_report_failed,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_runtime_reports_passed_flags_true",
            "all_runtime_report_signatures_match_first_repetition",
        ],
    },
    {
        "control_id": "negative_matrix_audit_failed",
        "mutation": _mutate_matrix_audit_failed,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_matrix_audits_passed",
            "all_matrix_signatures_match_first_repetition",
        ],
    },
    {
        "control_id": "negative_runtime_signature_drift",
        "mutation": _mutate_runtime_signature_drift,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_runtime_report_signatures_match_first_repetition"
        ],
    },
    {
        "control_id": "negative_matrix_signature_drift",
        "mutation": _mutate_matrix_signature_drift,
        "expected_passed": False,
        "expected_failed_checks": ["all_matrix_signatures_match_first_repetition"],
    },
    {
        "control_id": "negative_overstated_epoch_claim",
        "mutation": _mutate_overstated_epoch_claim,
        "expected_passed": False,
        "expected_failed_checks": ["top_level_ready_claim_is_bounded"],
    },
    {
        "control_id": "negative_output_outside_repetition_dir",
        "mutation": _mutate_output_outside_repetition_dir,
        "expected_passed": False,
        "expected_failed_checks": ["all_generated_manifests_under_repetition_dirs"],
    },
)


def _run_control(
    *,
    control_spec: dict[str, Any],
    phase2cs_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / control_id
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2cs_report=phase2cs_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)

    control_report_json = case_dir / "phase2cs_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2cs_fresh_runtime_execution_report(control_report)
    validation_report_json = case_dir / "phase2cs_validation.json"
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


def audit_phase2ct_fresh_runtime_execution_negative_controls(
    *,
    phase2cs_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2cs_report = _read_json(phase2cs_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            phase2cs_report=phase2cs_report,
            output_dir=output_root,
        )
        for control_spec in CONTROL_SPECS
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2cs_report"
    ]
    checks = {
        "source_phase2cs_passed": phase2cs_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2cs_report"
            and row["observed_passed"] is True
            and row["pass_expectation_met"] is True
            for row in control_rows
        ),
        "minimum_negative_control_count_met": len(negative_rows) >= 8,
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
        "artifact_family": "phase2ct_fresh_runtime_execution_negative_controls",
        "passed": passed,
        "ready_for_phase2cs_gate_strictness_claim": passed,
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
                "the Phase2CS report-level gate rejects fresh-execution overclaims for "
                "subprocess failure, missing runtime reports, failed runtime reports, "
                "failed matrix audits, runtime signature drift, matrix signature drift, "
                "overstated epoch claims, and output-directory provenance drift"
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
            "phase2cu_fresh_execution_runtime_perturbation_matrix"
            if passed
            else "repair_phase2ct_fresh_runtime_execution_negative_controls"
        ),
        "evidence": {
            "phase2cs_report_json": str(phase2cs_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2CT fresh-runtime execution negative controls."
    )
    parser.add_argument("--phase2cs-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ct_fresh_runtime_execution_negative_controls(
        phase2cs_report_json=args.phase2cs_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
