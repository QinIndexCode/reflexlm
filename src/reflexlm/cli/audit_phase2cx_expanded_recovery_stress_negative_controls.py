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
from reflexlm.cli.audit_phase2cw_runtime_perturbation_recovery_stress_expansion import (
    validate_phase2cw_runtime_perturbation_recovery_stress_expansion,
)
from reflexlm.cli.run_phase2cw_runtime_perturbation_recovery_stress_expansion import (
    PHASE2CW_EXTRA_STRESS_IDS,
)


Mutation = Callable[[dict[str, Any], Path], None]


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


def _runtime_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for perturbation in report.get("perturbation_results", [])
        for row in perturbation.get("runtime_results", [])
        if isinstance(row, dict)
    ]


def _perturbation_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("perturbation_results", [])
    return rows if isinstance(rows, list) else []


def _materialize_control_report(
    *,
    phase2cw_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    source_output_dir = phase2cw_report.get("evidence", {}).get(
        "expanded_recovery_output_dir"
    )
    if not source_output_dir:
        raise ValueError("Phase2CX requires Phase2CW evidence.expanded_recovery_output_dir")
    copied_output_dir = case_dir / "expanded_recovery_output"
    if copied_output_dir.exists():
        shutil.rmtree(copied_output_dir)
    shutil.copytree(source_output_dir, copied_output_dir)
    copied_report = deepcopy(phase2cw_report)
    copied_report = _rewrite_paths(
        copied_report,
        old_prefix=str(source_output_dir),
        new_prefix=str(copied_output_dir),
    )
    copied_report.setdefault("evidence", {})[
        "expanded_recovery_output_dir"
    ] = str(copied_output_dir)
    return copied_report


def _mutate_subprocess_nonzero(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    row = _runtime_rows(report)[0]
    row.setdefault("subprocess", {})["returncode"] = 23


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
    runtime_report["ready_for_bounded_expanded_recovery_stress_claim"] = False
    _write_json(row["report_json"], runtime_report)


def _mutate_extra_stress_missing(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    row = _runtime_rows(report)[0]
    runtime_report = _read_json(row["report_json"])
    stress_id = PHASE2CW_EXTRA_STRESS_IDS[0]
    runtime_report.setdefault("metrics", {}).setdefault("stress_counts", {})[
        stress_id
    ] = 2
    _write_json(row["report_json"], runtime_report)


def _mutate_recovery_rate_drop(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    row = _runtime_rows(report)[0]
    runtime_report = _read_json(row["report_json"])
    metrics = runtime_report.setdefault("metrics", {})
    metrics["observed_recoveries_after_failure"] = int(
        metrics.get("observed_recoveries_after_failure", 1)
    ) - 1
    metrics["failure_recovery_success_rate"] = metrics[
        "observed_recoveries_after_failure"
    ] / max(int(metrics.get("failure_episodes", 1)), 1)
    _write_json(row["report_json"], runtime_report)


def _mutate_signature_drift(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    rows = _runtime_rows(report)
    row = rows[min(4, len(rows) - 1)]
    runtime_report = _read_json(row["report_json"])
    runtime_report.setdefault("metrics", {})["executed_actions"] = int(
        runtime_report.get("metrics", {}).get("executed_actions", 0)
    ) + 1
    _write_json(row["report_json"], runtime_report)


def _mutate_perturbation_count_collapsed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["perturbation_results"] = _perturbation_rows(report)[:2]
    report.setdefault("metrics", {})["perturbation_count"] = 2


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["all_extra_stress_ids_present"] = False


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2cw_report",
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
            "all_recomputed_runtime_reports_passed",
            "all_runtime_signatures_stable_across_perturbations",
        ],
    },
    {
        "control_id": "negative_extra_stress_missing",
        "mutation": _mutate_extra_stress_missing,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_recomputed_extra_stress_ids_present",
            "all_runtime_signatures_stable_across_perturbations",
        ],
    },
    {
        "control_id": "negative_recovery_rate_drop",
        "mutation": _mutate_recovery_rate_drop,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_recomputed_failure_recovery_rates_are_perfect",
            "all_runtime_signatures_stable_across_perturbations",
        ],
    },
    {
        "control_id": "negative_signature_drift",
        "mutation": _mutate_signature_drift,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_runtime_signatures_stable_across_perturbations"
        ],
    },
    {
        "control_id": "negative_perturbation_count_collapsed",
        "mutation": _mutate_perturbation_count_collapsed,
        "expected_passed": False,
        "expected_failed_checks": ["minimum_three_perturbations_recorded"],
    },
    {
        "control_id": "negative_overstated_epoch_claim",
        "mutation": _mutate_overstated_epoch_claim,
        "expected_passed": False,
        "expected_failed_checks": ["top_level_ready_claim_is_bounded"],
    },
    {
        "control_id": "negative_recorded_check_false",
        "mutation": _mutate_recorded_check_false,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
)


def _run_control(
    *,
    control_spec: dict[str, Any],
    control_index: int,
    phase2cw_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"control_{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2cw_report=phase2cw_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2cw_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2cw_runtime_perturbation_recovery_stress_expansion(
        control_report
    )
    validation_report_json = case_dir / "phase2cw_validation.json"
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


def audit_phase2cx_expanded_recovery_stress_negative_controls(
    *,
    phase2cw_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2cw_report = _read_json(phase2cw_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2cw_report=phase2cw_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2cw_report"
    ]
    checks = {
        "source_phase2cw_passed": phase2cw_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2cw_report"
            and row["observed_passed"] is True
            and row["pass_expectation_met"] is True
            for row in control_rows
        ),
        "minimum_negative_control_count_met": len(negative_rows) >= 9,
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
        "artifact_family": "phase2cx_expanded_recovery_stress_negative_controls",
        "passed": passed,
        "ready_for_phase2cw_gate_strictness_claim": passed,
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
                "the Phase2CW report-level gate rejects expanded-recovery-stress "
                "overclaims for subprocess failures, missing or failed runtime "
                "reports, missing extra stress coverage, degraded recovery rates, "
                "runtime signature drift, collapsed perturbation coverage, overstated "
                "epoch claims, and false recorded checks"
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
            "phase2cy_expanded_recovery_cross_seed_stability"
            if passed
            else "repair_phase2cx_expanded_recovery_stress_negative_controls"
        ),
        "evidence": {
            "phase2cw_report_json": str(phase2cw_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2CX expanded recovery stress negative controls."
    )
    parser.add_argument("--phase2cw-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2cx_expanded_recovery_stress_negative_controls(
        phase2cw_report_json=args.phase2cw_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
