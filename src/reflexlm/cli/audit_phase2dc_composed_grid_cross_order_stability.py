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
from reflexlm.cli.audit_phase2da_cross_seed_runtime_perturbation_composition import (
    validate_phase2da_cross_seed_runtime_perturbation_composition,
)


OrderTransform = Callable[[dict[str, Any]], dict[str, Any]]


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


def _materialize_order_report(
    *,
    phase2da_report: dict[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    source_output_dir = phase2da_report.get("evidence", {}).get(
        "composed_grid_output_dir"
    )
    if not source_output_dir:
        raise ValueError("Phase2DC requires Phase2DA evidence.composed_grid_output_dir")
    copied_output_dir = case_dir / "composed_grid_output"
    if copied_output_dir.exists():
        shutil.rmtree(copied_output_dir)
    for seed_row in _seed_rows(phase2da_report):
        _copy_referenced_json(
            str(seed_row["suite_json"]),
            old_prefix=str(source_output_dir),
            new_prefix=str(copied_output_dir),
        )
    for runtime_row in _runtime_rows(phase2da_report):
        _copy_referenced_json(
            str(runtime_row["report_json"]),
            old_prefix=str(source_output_dir),
            new_prefix=str(copied_output_dir),
        )
    copied_report = deepcopy(phase2da_report)
    copied_report = _rewrite_paths(
        copied_report,
        old_prefix=str(source_output_dir),
        new_prefix=str(copied_output_dir),
    )
    copied_report.setdefault("evidence", {})["composed_grid_output_dir"] = str(
        copied_output_dir
    )
    return copied_report


def _rotate_left(rows: list[Any]) -> list[Any]:
    return rows[1:] + rows[:1] if rows else rows


def _identity(report: dict[str, Any]) -> dict[str, Any]:
    return report


def _reverse_seed_order(report: dict[str, Any]) -> dict[str, Any]:
    report["seed_results"] = list(reversed(_seed_rows(report)))
    return report


def _reverse_perturbation_order(report: dict[str, Any]) -> dict[str, Any]:
    for seed_row in _seed_rows(report):
        rows = seed_row.get("perturbation_results", [])
        seed_row["perturbation_results"] = list(reversed(rows)) if isinstance(rows, list) else []
    return report


def _reverse_runtime_order(report: dict[str, Any]) -> dict[str, Any]:
    for seed_row in _seed_rows(report):
        for perturbation in seed_row.get("perturbation_results", []):
            rows = perturbation.get("runtime_results", [])
            perturbation["runtime_results"] = (
                list(reversed(rows)) if isinstance(rows, list) else []
            )
    return report


def _rotate_all_axes(report: dict[str, Any]) -> dict[str, Any]:
    report["seed_results"] = _rotate_left(_seed_rows(report))
    for seed_row in _seed_rows(report):
        perturbations = seed_row.get("perturbation_results", [])
        seed_row["perturbation_results"] = (
            _rotate_left(perturbations) if isinstance(perturbations, list) else []
        )
        for perturbation in seed_row.get("perturbation_results", []):
            runtimes = perturbation.get("runtime_results", [])
            perturbation["runtime_results"] = (
                _rotate_left(runtimes) if isinstance(runtimes, list) else []
            )
    return report


ORDER_SPECS: tuple[dict[str, Any], ...] = (
    {"order_id": "original_order", "transform": _identity},
    {"order_id": "reverse_seed_order", "transform": _reverse_seed_order},
    {
        "order_id": "reverse_perturbation_order",
        "transform": _reverse_perturbation_order,
    },
    {"order_id": "reverse_runtime_order", "transform": _reverse_runtime_order},
    {"order_id": "rotate_all_axes_order", "transform": _rotate_all_axes},
)


def _validation_signature(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": validation.get("passed"),
        "checks": validation.get("checks", {}),
        "metrics": validation.get("metrics", {}),
    }


def _run_order(
    *,
    order_spec: dict[str, Any],
    order_index: int,
    phase2da_report: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    order_id = str(order_spec["order_id"])
    case_dir = output_dir / f"order_{order_index:02d}_{order_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    order_report = _materialize_order_report(
        phase2da_report=phase2da_report,
        case_dir=case_dir,
    )
    transform: OrderTransform = order_spec["transform"]
    order_report = transform(order_report)
    order_report_json = case_dir / "phase2da_order_report.json"
    _write_json(order_report_json, order_report)
    validation = validate_phase2da_cross_seed_runtime_perturbation_composition(
        order_report
    )
    validation_report_json = case_dir / "phase2da_validation.json"
    _write_json(validation_report_json, validation)
    return {
        "order_id": order_id,
        "order_report_json": str(order_report_json),
        "validation_report_json": str(validation_report_json),
        "observed_passed": validation.get("passed") is True,
        "validation_signature": _validation_signature(validation),
    }


def _bounded_claims_only(report: dict[str, Any]) -> bool:
    return (
        report.get(
            "ready_for_bounded_composed_grid_cross_order_stability_claim"
        )
        is True
        and report.get("ready_for_general_shell_autonomy_claim") is False
        and report.get("ready_for_general_runtime_invariance_claim") is False
        and report.get("ready_for_open_ended_native_perception_claim") is False
        and report.get("ready_for_production_autonomy_claim") is False
        and report.get("ready_for_epoch_making_architecture_claim") is False
    )


def validate_phase2dc_composed_grid_cross_order_stability(
    report: dict[str, Any],
) -> dict[str, Any]:
    order_rows = report.get("order_results", [])
    if not isinstance(order_rows, list):
        order_rows = []
    validation_read_failures = 0
    order_report_read_failures = 0
    recomputed_rows: list[dict[str, Any]] = []
    for row in order_rows:
        try:
            order_report = _read_json(row.get("order_report_json"))
        except (OSError, TypeError, json.JSONDecodeError):
            order_report_read_failures += 1
            order_report = {}
        try:
            recorded_validation = _read_json(row.get("validation_report_json"))
        except (OSError, TypeError, json.JSONDecodeError):
            validation_read_failures += 1
            recorded_validation = {}
        recomputed_validation = (
            validate_phase2da_cross_seed_runtime_perturbation_composition(
                order_report
            )
            if order_report
            else {}
        )
        recomputed_rows.append(
            {
                "recorded_validation_passed": recorded_validation.get("passed")
                is True,
                "recomputed_validation_passed": recomputed_validation.get("passed")
                is True,
                "recorded_signature": _validation_signature(recorded_validation),
                "recomputed_signature": _validation_signature(recomputed_validation),
                "recorded_matches_recomputed": _validation_signature(
                    recorded_validation
                )
                == _validation_signature(recomputed_validation),
            }
        )
    reference_signature = (
        recomputed_rows[0]["recomputed_signature"] if recomputed_rows else {}
    )
    signature_mismatch_count = sum(
        row["recomputed_signature"] != reference_signature
        for row in recomputed_rows
    )
    order_count = int(report.get("metrics", {}).get("order_count", 0) or 0)
    checks = {
        "artifact_family_matches_phase2dc": (
            report.get("artifact_family")
            == "phase2dc_composed_grid_cross_order_stability"
        ),
        "top_level_phase2dc_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": _bounded_claims_only(report),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "minimum_four_orderings_recorded": order_count >= 4
        and len(order_rows) >= 4,
        "all_order_reports_readable": bool(order_rows)
        and order_report_read_failures == 0,
        "all_validation_reports_readable": bool(order_rows)
        and validation_read_failures == 0,
        "all_recorded_validations_passed": bool(recomputed_rows)
        and all(row["recorded_validation_passed"] is True for row in recomputed_rows),
        "all_recomputed_validations_passed": bool(recomputed_rows)
        and all(row["recomputed_validation_passed"] is True for row in recomputed_rows),
        "all_recorded_validations_match_recomputed": bool(recomputed_rows)
        and all(row["recorded_matches_recomputed"] is True for row in recomputed_rows),
        "all_recomputed_order_signatures_match": bool(recomputed_rows)
        and signature_mismatch_count == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "order_rows": len(order_rows),
            "order_report_read_failures": order_report_read_failures,
            "validation_read_failures": validation_read_failures,
            "order_signature_mismatch_count": signature_mismatch_count,
            "recomputed_validations_passed": sum(
                row["recomputed_validation_passed"] is True
                for row in recomputed_rows
            ),
        },
    }


def audit_phase2dc_composed_grid_cross_order_stability(
    *,
    phase2db_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    phase2db = _read_json(phase2db_report_json)
    phase2da_report_json = phase2db.get("evidence", {}).get("phase2da_report_json")
    if not phase2da_report_json:
        raise ValueError("Phase2DC requires Phase2DB evidence.phase2da_report_json")
    phase2da_report = _read_json(phase2da_report_json)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    order_rows = [
        _run_order(
            order_spec=order_spec,
            order_index=index,
            phase2da_report=phase2da_report,
            output_dir=output_root,
        )
        for index, order_spec in enumerate(ORDER_SPECS)
    ]
    reference_signature = (
        order_rows[0]["validation_signature"] if order_rows else {}
    )
    signature_mismatch_count = sum(
        row["validation_signature"] != reference_signature for row in order_rows
    )
    checks = {
        "source_phase2db_passed": phase2db.get("passed") is True,
        "source_phase2da_passed": phase2da_report.get("passed") is True,
        "minimum_four_orderings_met": len(order_rows) >= 4,
        "all_order_validations_passed": bool(order_rows)
        and all(row["observed_passed"] is True for row in order_rows),
        "all_order_validation_signatures_stable": bool(order_rows)
        and signature_mismatch_count == 0,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2dc_composed_grid_cross_order_stability",
        "passed": passed,
        "ready_for_bounded_composed_grid_cross_order_stability_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "order_count": len(order_rows),
            "order_validation_signature_mismatch_count": signature_mismatch_count,
            "passed_order_validations": sum(
                row["observed_passed"] is True for row in order_rows
            ),
        },
        "order_results": order_rows,
        "supported_claims": [
            (
                "the composed seed/runtime-perturbation grid gate remains stable "
                "under seed, perturbation, runtime, and combined order permutations"
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
            "phase2dd_cross_order_negative_controls"
            if passed
            else "repair_phase2dc_composed_grid_cross_order_stability"
        ),
        "evidence": {
            "phase2db_report_json": str(phase2db_report_json),
            "phase2da_report_json": str(phase2da_report_json),
            "cross_order_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2DC composed-grid cross-order stability."
    )
    parser.add_argument("--phase2db-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2dc_composed_grid_cross_order_stability(
        phase2db_report_json=args.phase2db_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
