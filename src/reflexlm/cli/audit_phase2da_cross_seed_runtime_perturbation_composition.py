from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2cu_fresh_execution_runtime_perturbation_matrix import (
    DEFAULT_PERTURBATIONS,
)
from reflexlm.cli.audit_phase2cw_runtime_perturbation_recovery_stress_expansion import (
    _report_signature,
    _run_phase2cw_subprocess,
)
from reflexlm.cli.audit_phase2cy_expanded_recovery_cross_seed_stability import (
    _default_seed_values,
    _runtime_report_checks,
    _runtime_rows,
    _seed_rows,
    _source_from_phase2cx,
    _write_seed_suite,
)
from reflexlm.cli.run_phase2cw_runtime_perturbation_recovery_stress_expansion import (
    PHASE2CW_EXTRA_STRESS_IDS,
    STRESS_IDS,
)


def _source_from_phase2cz(phase2cz: dict[str, Any]) -> dict[str, Any]:
    phase2cy_report_json = phase2cz.get("evidence", {}).get("phase2cy_report_json")
    if not phase2cy_report_json:
        raise ValueError("Phase2DA requires Phase2CZ evidence.phase2cy_report_json")
    phase2cy = _read_json(phase2cy_report_json)
    phase2cx_report_json = phase2cy.get("evidence", {}).get("phase2cx_report_json")
    if not phase2cx_report_json:
        raise ValueError("Phase2DA requires Phase2CY evidence.phase2cx_report_json")
    source = _source_from_phase2cx(_read_json(phase2cx_report_json))
    source["phase2cy_report_json"] = str(phase2cy_report_json)
    source["phase2cx_report_json"] = str(phase2cx_report_json)
    return source


def _perturbation_rows(seed_row: dict[str, Any]) -> list[dict[str, Any]]:
    rows = seed_row.get("perturbation_results", [])
    return rows if isinstance(rows, list) else []


def _validate_suite_rows(seed_rows: list[dict[str, Any]]) -> dict[str, int]:
    suite_read_failures = 0
    suite_seed_mismatches = 0
    for seed_row in seed_rows:
        expected_seed = int(seed_row.get("seed", -1))
        try:
            suite = _read_json(seed_row.get("suite_json"))
        except (OSError, TypeError, json.JSONDecodeError):
            suite_read_failures += 1
            suite = {}
        if suite.get("seed") != expected_seed:
            suite_seed_mismatches += 1
    return {
        "suite_read_failures": suite_read_failures,
        "suite_seed_mismatches": suite_seed_mismatches,
    }


def _recompute_runtime_rows(
    runtime_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime_reference_signatures: dict[str, dict[str, Any]] = {}
    runtime_signature_mismatches = 0
    runtime_report_read_failures = 0
    recomputed_rows: list[dict[str, Any]] = []
    for row in runtime_rows:
        try:
            runtime_report = _read_json(row.get("report_json"))
        except (OSError, TypeError, json.JSONDecodeError):
            runtime_report_read_failures += 1
            runtime_report = {}
        signature = _report_signature(runtime_report)
        runtime = str(row.get("runtime_interpreter", ""))
        reference = runtime_reference_signatures.setdefault(runtime, signature)
        if signature != reference:
            runtime_signature_mismatches += 1
        recomputed_rows.append(_runtime_report_checks(runtime_report))
    return {
        "runtime_report_read_failures": runtime_report_read_failures,
        "runtime_signature_mismatch_count": runtime_signature_mismatches,
        "recomputed_rows": recomputed_rows,
    }


def validate_phase2da_cross_seed_runtime_perturbation_composition(
    report: dict[str, Any],
) -> dict[str, Any]:
    seed_rows = _seed_rows(report)
    runtime_rows = _runtime_rows(report)
    suite_metrics = _validate_suite_rows(seed_rows)
    runtime_metrics = _recompute_runtime_rows(runtime_rows)
    recomputed_rows = runtime_metrics["recomputed_rows"]

    seed_count = int(report.get("metrics", {}).get("seed_count", 0) or 0)
    runtime_count = int(report.get("metrics", {}).get("runtime_count", 0) or 0)
    perturbation_count = int(
        report.get("metrics", {}).get("perturbation_count", 0) or 0
    )
    expected_runtime_rows = seed_count * runtime_count * perturbation_count
    checks = {
        "artifact_family_matches_phase2da": (
            report.get("artifact_family")
            == "phase2da_cross_seed_runtime_perturbation_composition"
        ),
        "top_level_phase2da_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get(
                "ready_for_bounded_cross_seed_runtime_perturbation_composition_claim"
            )
            is True
            and report.get("ready_for_general_shell_autonomy_claim") is False
            and report.get("ready_for_general_runtime_invariance_claim") is False
            and report.get("ready_for_open_ended_native_perception_claim") is False
            and report.get("ready_for_production_autonomy_claim") is False
            and report.get("ready_for_epoch_making_architecture_claim") is False
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "minimum_three_seeds_recorded": seed_count >= 3
        and len({row.get("seed") for row in seed_rows}) >= 3,
        "minimum_three_runtime_interpreters_recorded": runtime_count >= 3,
        "minimum_three_perturbations_recorded": perturbation_count >= 3
        and all(len(_perturbation_rows(row)) >= 3 for row in seed_rows),
        "all_seed_suites_readable": bool(seed_rows)
        and suite_metrics["suite_read_failures"] == 0,
        "all_seed_suites_match_recorded_seed": bool(seed_rows)
        and suite_metrics["suite_seed_mismatches"] == 0,
        "all_expected_runtime_rows_present": bool(runtime_rows)
        and len(runtime_rows) == expected_runtime_rows,
        "all_runtime_subprocesses_recorded_zero": bool(runtime_rows)
        and all(row.get("subprocess", {}).get("returncode") == 0 for row in runtime_rows),
        "all_runtime_reports_exist_flags_true": bool(runtime_rows)
        and all(row.get("report_exists") is True for row in runtime_rows),
        "all_runtime_reports_passed_flags_true": bool(runtime_rows)
        and all(row.get("report_passed") is True for row in runtime_rows),
        "all_runtime_reports_readable": bool(runtime_rows)
        and runtime_metrics["runtime_report_read_failures"] == 0,
        "all_recomputed_runtime_reports_passed": bool(recomputed_rows)
        and all(row["report_passed"] is True for row in recomputed_rows),
        "all_recomputed_expanded_stress_ids_present": bool(recomputed_rows)
        and all(row["all_stress_ids_present"] is True for row in recomputed_rows),
        "all_recomputed_extra_stress_ids_present": bool(recomputed_rows)
        and all(row["extra_stress_ids_present"] is True for row in recomputed_rows),
        "all_recomputed_failure_recovery_rates_are_perfect": bool(recomputed_rows)
        and all(row["failure_recovery_success_rate"] == 1.0 for row in recomputed_rows),
        "all_recomputed_runtime_claims_are_bounded": bool(recomputed_rows)
        and all(row["bounded_claim_only"] is True for row in recomputed_rows),
        "all_runtime_signatures_stable_across_seed_perturbation_grid": bool(
            runtime_rows
        )
        and runtime_metrics["runtime_signature_mismatch_count"] == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "seed_rows": len(seed_rows),
            "runtime_rows": len(runtime_rows),
            "suite_read_failures": suite_metrics["suite_read_failures"],
            "suite_seed_mismatches": suite_metrics["suite_seed_mismatches"],
            "runtime_report_read_failures": runtime_metrics[
                "runtime_report_read_failures"
            ],
            "runtime_signature_mismatch_count": runtime_metrics[
                "runtime_signature_mismatch_count"
            ],
            "recomputed_runtime_reports_passed": sum(
                row["report_passed"] is True for row in recomputed_rows
            ),
        },
    }


def audit_phase2da_cross_seed_runtime_perturbation_composition(
    *,
    phase2cz_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    seeds: list[int] | None = None,
    perturbations: list[dict[str, Any]] | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(cwd or Path.cwd())
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    phase2cz = _read_json(phase2cz_report_json)
    source = _source_from_phase2cz(phase2cz)
    source_suite = _read_json(str(source["suite_json"]))
    seed_values = list(seeds or _default_seed_values(source_suite))
    perturbation_specs = list(perturbations or DEFAULT_PERTURBATIONS)
    runtimes = list(source["runtime_interpreters"])
    runtime_reference_signatures: dict[str, dict[str, Any]] = {}
    signature_mismatch_count = 0
    seed_rows: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seed_values):
        suite_json = _write_seed_suite(
            source_suite=source_suite,
            seed=int(seed),
            output_dir=output_root,
        )
        perturbation_rows: list[dict[str, Any]] = []
        for perturbation_index, spec in enumerate(perturbation_specs):
            perturbation_id = str(spec["perturbation_id"])
            timeout_seconds = float(spec["timeout_seconds"])
            max_extra_steps = int(spec["max_extra_steps"])
            runtime_rows: list[dict[str, Any]] = []
            for runtime_index, runtime in enumerate(runtimes):
                runtime_dir = (
                    output_root
                    / f"seed_{seed_index:02d}_{int(seed)}"
                    / f"{perturbation_index:02d}_{perturbation_id}"
                    / f"runtime_{runtime_index:02d}"
                )
                report_json = runtime_dir / "phase2cw_report.json"
                run = _run_phase2cw_subprocess(
                    runtime_interpreter=runtime,
                    package_path=str(source["package_path"]),
                    suite_json=str(suite_json),
                    output_dir=runtime_dir,
                    output_report_json=report_json,
                    timeout_seconds=timeout_seconds,
                    max_extra_steps=max_extra_steps,
                    cwd=repo_root,
                )
                report_exists = report_json.exists()
                runtime_report = _read_json(report_json) if report_exists else {}
                signature = _report_signature(runtime_report)
                reference = runtime_reference_signatures.setdefault(runtime, signature)
                signature_matches_reference = signature == reference
                if not signature_matches_reference:
                    signature_mismatch_count += 1
                runtime_checks = _runtime_report_checks(runtime_report)
                runtime_rows.append(
                    {
                        "seed": int(seed),
                        "runtime_index": runtime_index,
                        "runtime_interpreter": runtime,
                        "report_json": str(report_json),
                        "output_dir": str(runtime_dir),
                        "subprocess": run,
                        "report_exists": report_exists,
                        "report_passed": runtime_checks["report_passed"],
                        "all_stress_ids_present": runtime_checks[
                            "all_stress_ids_present"
                        ],
                        "extra_stress_ids_present": runtime_checks[
                            "extra_stress_ids_present"
                        ],
                        "failure_recovery_success_rate": runtime_checks[
                            "failure_recovery_success_rate"
                        ],
                        "signature_matches_reference": signature_matches_reference,
                    }
                )
            perturbation_rows.append(
                {
                    "perturbation_id": perturbation_id,
                    "timeout_seconds": timeout_seconds,
                    "max_extra_steps": max_extra_steps,
                    "runtime_results": runtime_rows,
                }
            )
        seed_rows.append(
            {
                "seed_index": seed_index,
                "seed": int(seed),
                "suite_json": str(suite_json),
                "perturbation_results": perturbation_rows,
            }
        )

    all_runtime_rows = _runtime_rows({"seed_results": seed_rows})
    checks = {
        "source_phase2cz_passed": phase2cz.get("passed") is True,
        "minimum_three_seeds_met": len(set(seed_values)) >= 3,
        "minimum_three_runtime_interpreters_met": len(runtimes) >= 3,
        "minimum_three_perturbations_met": len(perturbation_specs) >= 3,
        "all_subprocesses_returned_zero": bool(all_runtime_rows)
        and all(row["subprocess"]["returncode"] == 0 for row in all_runtime_rows),
        "all_runtime_reports_exist": bool(all_runtime_rows)
        and all(row["report_exists"] is True for row in all_runtime_rows),
        "all_runtime_reports_passed": bool(all_runtime_rows)
        and all(row["report_passed"] is True for row in all_runtime_rows),
        "all_expanded_stress_ids_present": bool(all_runtime_rows)
        and all(row["all_stress_ids_present"] is True for row in all_runtime_rows),
        "all_extra_stress_ids_present": bool(all_runtime_rows)
        and all(row["extra_stress_ids_present"] is True for row in all_runtime_rows),
        "all_failure_recovery_rates_are_perfect": bool(all_runtime_rows)
        and all(row["failure_recovery_success_rate"] == 1.0 for row in all_runtime_rows),
        "all_runtime_signatures_stable_across_seed_perturbation_grid": bool(
            all_runtime_rows
        )
        and signature_mismatch_count == 0,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2da_cross_seed_runtime_perturbation_composition",
        "passed": passed,
        "ready_for_bounded_cross_seed_runtime_perturbation_composition_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "seed_count": len(set(seed_values)),
            "runtime_count": len(runtimes),
            "perturbation_count": len(perturbation_specs),
            "fresh_runtime_execution_count": len(all_runtime_rows),
            "passed_runtime_reports": sum(
                row["report_passed"] is True for row in all_runtime_rows
            ),
            "runtime_signature_mismatch_count": signature_mismatch_count,
            "seeds": [int(seed) for seed in seed_values],
            "perturbation_ids": [
                str(spec["perturbation_id"]) for spec in perturbation_specs
            ],
            "stress_ids": list(STRESS_IDS),
            "extra_stress_ids": list(PHASE2CW_EXTRA_STRESS_IDS),
        },
        "seed_results": seed_rows,
        "supported_claims": [
            (
                "bounded package-internal structured-runtime cortex retained "
                "expanded recovery behavior across the composed seed and runtime "
                "budget perturbation grid under the recorded runtime interpreters"
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
            "phase2db_composed_grid_negative_controls"
            if passed
            else "repair_phase2da_cross_seed_runtime_perturbation_composition"
        ),
        "evidence": {
            "phase2cz_report_json": str(phase2cz_report_json),
            **{key: value for key, value in source.items() if key != "runtime_interpreters"},
            "runtime_interpreters": runtimes,
            "composed_grid_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2DA composed cross-seed/runtime-perturbation grid."
    )
    parser.add_argument("--phase2cz-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2da_cross_seed_runtime_perturbation_composition(
        phase2cz_report_json=args.phase2cz_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
