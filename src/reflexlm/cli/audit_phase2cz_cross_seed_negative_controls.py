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
from reflexlm.cli.audit_phase2cy_expanded_recovery_cross_seed_stability import (
    _runtime_rows,
    _seed_rows,
    validate_phase2cy_expanded_recovery_cross_seed_stability,
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


def _copy_referenced_json(old_path: str, *, old_prefix: str, new_prefix: str) -> str:
    new_path = old_path.replace(old_prefix, new_prefix)
    source = Path(old_path)
    target = Path(new_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return new_path


def _materialize_control_report(
    *,
    phase2cy_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    source_output_dir = phase2cy_report.get("evidence", {}).get(
        "cross_seed_output_dir"
    )
    if not source_output_dir:
        raise ValueError("Phase2CZ requires Phase2CY evidence.cross_seed_output_dir")
    copied_output_dir = case_dir / "cross_seed_output"
    if copied_output_dir.exists():
        shutil.rmtree(copied_output_dir)
    for seed_row in _seed_rows(phase2cy_report):
        _copy_referenced_json(
            str(seed_row["suite_json"]),
            old_prefix=str(source_output_dir),
            new_prefix=str(copied_output_dir),
        )
    for runtime_row in _runtime_rows(phase2cy_report):
        _copy_referenced_json(
            str(runtime_row["report_json"]),
            old_prefix=str(source_output_dir),
            new_prefix=str(copied_output_dir),
        )
    copied_report = deepcopy(phase2cy_report)
    copied_report = _rewrite_paths(
        copied_report,
        old_prefix=str(source_output_dir),
        new_prefix=str(copied_output_dir),
    )
    copied_report.setdefault("evidence", {})["cross_seed_output_dir"] = str(
        copied_output_dir
    )
    return copied_report


def _mutate_source_phase2cx_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["source_phase2cx_passed"] = False


def _mutate_missing_seed_suite(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    suite_path = Path(str(_seed_rows(report)[0]["suite_json"]))
    if suite_path.exists():
        suite_path.unlink()


def _mutate_seed_suite_mismatch(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    seed_row = _seed_rows(report)[0]
    suite_path = Path(str(seed_row["suite_json"]))
    suite = _read_json(suite_path)
    suite["seed"] = int(seed_row["seed"]) + 999
    _write_json(suite_path, suite)


def _mutate_seed_count_collapsed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["seed_results"] = _seed_rows(report)[:2]
    report.setdefault("metrics", {})["seed_count"] = 2


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
    row = rows[min(3, len(rows) - 1)]
    runtime_report = _read_json(row["report_json"])
    runtime_report.setdefault("metrics", {})["executed_actions"] = int(
        runtime_report.get("metrics", {}).get("executed_actions", 0)
    ) + 1
    _write_json(row["report_json"], runtime_report)


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["all_runtime_signatures_stable_across_seeds"] = (
        False
    )


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2cy_report",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_source_phase2cx_failed",
        "mutation": _mutate_source_phase2cx_failed,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
    {
        "control_id": "negative_missing_seed_suite",
        "mutation": _mutate_missing_seed_suite,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_seed_suites_readable",
            "all_seed_suites_match_recorded_seed",
        ],
    },
    {
        "control_id": "negative_seed_suite_mismatch",
        "mutation": _mutate_seed_suite_mismatch,
        "expected_passed": False,
        "expected_failed_checks": ["all_seed_suites_match_recorded_seed"],
    },
    {
        "control_id": "negative_seed_count_collapsed",
        "mutation": _mutate_seed_count_collapsed,
        "expected_passed": False,
        "expected_failed_checks": ["minimum_three_seeds_recorded"],
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
            "all_runtime_reports_exist_flags_true",
            "all_runtime_reports_readable",
        ],
    },
    {
        "control_id": "negative_runtime_report_failed",
        "mutation": _mutate_runtime_report_failed,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_runtime_reports_passed_flags_true",
            "all_recomputed_runtime_reports_passed",
            "all_runtime_signatures_stable_across_seeds",
        ],
    },
    {
        "control_id": "negative_extra_stress_missing",
        "mutation": _mutate_extra_stress_missing,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_recomputed_extra_stress_ids_present",
            "all_runtime_signatures_stable_across_seeds",
        ],
    },
    {
        "control_id": "negative_recovery_rate_drop",
        "mutation": _mutate_recovery_rate_drop,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_recomputed_failure_recovery_rates_are_perfect",
            "all_runtime_signatures_stable_across_seeds",
        ],
    },
    {
        "control_id": "negative_signature_drift",
        "mutation": _mutate_signature_drift,
        "expected_passed": False,
        "expected_failed_checks": ["all_runtime_signatures_stable_across_seeds"],
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
    phase2cy_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"control_{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2cy_report=phase2cy_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2cy_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2cy_expanded_recovery_cross_seed_stability(
        control_report
    )
    validation_report_json = case_dir / "phase2cy_validation.json"
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


def audit_phase2cz_cross_seed_negative_controls(
    *,
    phase2cy_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2cy_report = _read_json(phase2cy_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2cy_report=phase2cy_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2cy_report"
    ]
    checks = {
        "source_phase2cy_passed": phase2cy_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2cy_report"
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
        "artifact_family": "phase2cz_cross_seed_negative_controls",
        "passed": passed,
        "ready_for_phase2cy_gate_strictness_claim": passed,
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
                "the Phase2CY report-level gate rejects cross-seed overclaims "
                "for source-gate failure, seed-suite loss or mismatch, collapsed "
                "seed coverage, subprocess failures, missing or failed runtime "
                "reports, stress coverage loss, degraded recovery, signature drift, "
                "and overstated epoch claims"
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
            "phase2da_cross_seed_runtime_perturbation_composition"
            if passed
            else "repair_phase2cz_cross_seed_negative_controls"
        ),
        "evidence": {
            "phase2cy_report_json": str(phase2cy_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2CZ cross-seed stability negative controls."
    )
    parser.add_argument("--phase2cy-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2cz_cross_seed_negative_controls(
        phase2cy_report_json=args.phase2cy_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
