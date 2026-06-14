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
)
from reflexlm.cli.audit_phase2dc_composed_grid_cross_order_stability import (
    validate_phase2dc_composed_grid_cross_order_stability,
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


def _copy_json(old_path: str, *, old_prefix: str, new_prefix: str) -> str:
    new_path = old_path.replace(old_prefix, new_prefix)
    source = Path(old_path)
    target = Path(new_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return new_path


def _copy_order_dependencies(
    *,
    order_report: dict[str, Any],
    old_prefix: str,
    new_prefix: str,
) -> None:
    for seed_row in _seed_rows(order_report):
        _copy_json(
            str(seed_row["suite_json"]),
            old_prefix=old_prefix,
            new_prefix=new_prefix,
        )
    for runtime_row in _runtime_rows(order_report):
        _copy_json(
            str(runtime_row["report_json"]),
            old_prefix=old_prefix,
            new_prefix=new_prefix,
        )


def _materialize_control_report(
    *,
    phase2dc_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    source_output_dir = phase2dc_report.get("evidence", {}).get(
        "cross_order_output_dir"
    )
    if not source_output_dir:
        raise ValueError("Phase2DD requires Phase2DC evidence.cross_order_output_dir")
    copied_output_dir = case_dir / "x"
    if copied_output_dir.exists():
        shutil.rmtree(copied_output_dir)
    for order_row in phase2dc_report.get("order_results", []):
        if not isinstance(order_row, dict):
            continue
        order_report_path = str(order_row["order_report_json"])
        validation_report_path = str(order_row["validation_report_json"])
        order_report = _read_json(order_report_path)
        _copy_json(
            order_report_path,
            old_prefix=str(source_output_dir),
            new_prefix=str(copied_output_dir),
        )
        _copy_json(
            validation_report_path,
            old_prefix=str(source_output_dir),
            new_prefix=str(copied_output_dir),
        )
        _copy_order_dependencies(
            order_report=order_report,
            old_prefix=str(source_output_dir),
            new_prefix=str(copied_output_dir),
        )
    copied_report = deepcopy(phase2dc_report)
    copied_report = _rewrite_paths(
        copied_report,
        old_prefix=str(source_output_dir),
        new_prefix=str(copied_output_dir),
    )
    copied_report.setdefault("evidence", {})["cross_order_output_dir"] = str(
        copied_output_dir
    )
    return copied_report


def _order_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("order_results", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _mutate_source_phase2db_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["source_phase2db_passed"] = False


def _mutate_missing_order_report(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    order_path = Path(str(_order_rows(report)[0]["order_report_json"]))
    if order_path.exists():
        order_path.unlink()


def _mutate_missing_validation_report(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    validation_path = Path(str(_order_rows(report)[0]["validation_report_json"]))
    if validation_path.exists():
        validation_path.unlink()


def _mutate_recorded_validation_failed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    validation_path = Path(str(_order_rows(report)[0]["validation_report_json"]))
    validation = _read_json(validation_path)
    validation["passed"] = False
    validation.setdefault("checks", {})["top_level_phase2da_passed"] = False
    _write_json(validation_path, validation)


def _mutate_order_count_collapsed(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["order_results"] = _order_rows(report)[:3]
    report.setdefault("metrics", {})["order_count"] = 3


def _mutate_order_report_drift(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    order_path = Path(str(_order_rows(report)[1]["order_report_json"]))
    order_report = _read_json(order_path)
    order_report["seed_results"][0]["perturbation_results"] = order_report[
        "seed_results"
    ][0]["perturbation_results"][:2]
    _write_json(order_path, order_report)


def _mutate_validation_signature_drift(
    report: dict[str, Any],
    case_dir: Path,
) -> None:
    del case_dir
    validation_path = Path(str(_order_rows(report)[1]["validation_report_json"]))
    validation = _read_json(validation_path)
    validation.setdefault("metrics", {})["runtime_rows"] = int(
        validation.get("metrics", {}).get("runtime_rows", 0)
    ) + 1
    _write_json(validation_path, validation)


def _mutate_overstated_epoch_claim(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report["ready_for_epoch_making_architecture_claim"] = True


def _mutate_recorded_check_false(report: dict[str, Any], case_dir: Path) -> None:
    del case_dir
    report.setdefault("checks", {})["all_order_validation_signatures_stable"] = False


CONTROL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "control_id": "positive_control_original_phase2dc_report",
        "mutation": None,
        "expected_passed": True,
        "expected_failed_checks": [],
    },
    {
        "control_id": "negative_source_phase2db_failed",
        "mutation": _mutate_source_phase2db_failed,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_checks_true"],
    },
    {
        "control_id": "negative_missing_order_report",
        "mutation": _mutate_missing_order_report,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_order_reports_readable",
            "all_recomputed_validations_passed",
            "all_recomputed_order_signatures_match",
        ],
    },
    {
        "control_id": "negative_missing_validation_report",
        "mutation": _mutate_missing_validation_report,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_validation_reports_readable",
            "all_recorded_validations_passed",
            "all_recorded_validations_match_recomputed",
        ],
    },
    {
        "control_id": "negative_recorded_validation_failed",
        "mutation": _mutate_recorded_validation_failed,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_recorded_validations_passed",
            "all_recorded_validations_match_recomputed",
        ],
    },
    {
        "control_id": "negative_order_count_collapsed",
        "mutation": _mutate_order_count_collapsed,
        "expected_passed": False,
        "expected_failed_checks": ["minimum_four_orderings_recorded"],
    },
    {
        "control_id": "negative_order_report_drift",
        "mutation": _mutate_order_report_drift,
        "expected_passed": False,
        "expected_failed_checks": [
            "all_recomputed_validations_passed",
            "all_recorded_validations_match_recomputed",
            "all_recomputed_order_signatures_match",
        ],
    },
    {
        "control_id": "negative_validation_signature_drift",
        "mutation": _mutate_validation_signature_drift,
        "expected_passed": False,
        "expected_failed_checks": ["all_recorded_validations_match_recomputed"],
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
    phase2dc_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    control_id = str(control_spec["control_id"])
    case_dir = output_dir / f"c{control_index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    control_report = _materialize_control_report(
        phase2dc_report=phase2dc_report,
        case_dir=case_dir,
    )
    mutation: Mutation | None = control_spec["mutation"]
    if mutation is not None:
        mutation(control_report, case_dir)
    control_report_json = case_dir / "phase2dc_control_report.json"
    _write_json(control_report_json, control_report)
    validation = validate_phase2dc_composed_grid_cross_order_stability(
        control_report
    )
    validation_report_json = case_dir / "phase2dc_validation.json"
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


def audit_phase2dd_cross_order_negative_controls(
    *,
    phase2dc_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2dc_report = _read_json(phase2dc_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    control_rows = [
        _run_control(
            control_spec=control_spec,
            control_index=index,
            phase2dc_report=phase2dc_report,
            output_dir=output_root,
        )
        for index, control_spec in enumerate(CONTROL_SPECS)
    ]
    negative_rows = [
        row
        for row in control_rows
        if row["control_id"] != "positive_control_original_phase2dc_report"
    ]
    checks = {
        "source_phase2dc_passed": phase2dc_report.get("passed") is True,
        "positive_control_still_passes": any(
            row["control_id"] == "positive_control_original_phase2dc_report"
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
        "artifact_family": "phase2dd_cross_order_negative_controls",
        "passed": passed,
        "ready_for_phase2dc_gate_strictness_claim": passed,
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
                "the Phase2DC cross-order gate rejects overclaims for source-gate "
                "failure, missing order or validation reports, validation tampering, "
                "collapsed order coverage, order-report drift, and overstated epoch "
                "claims"
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
            "phase2de_compact_evidence_rollup"
            if passed
            else "repair_phase2dd_cross_order_negative_controls"
        ),
        "evidence": {
            "phase2dc_report_json": str(phase2dc_report_json),
            "negative_control_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2DD cross-order negative controls."
    )
    parser.add_argument("--phase2dc-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2dd_cross_order_negative_controls(
        phase2dc_report_json=args.phase2dc_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
