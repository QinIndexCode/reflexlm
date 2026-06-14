from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability import (
    _read_json,
    _write_json,
)
from reflexlm.cli.audit_phase2cu_fresh_execution_runtime_perturbation_matrix import (
    DEFAULT_PERTURBATIONS,
)
from reflexlm.cli.run_phase2cw_runtime_perturbation_recovery_stress_expansion import (
    PHASE2CW_EXTRA_STRESS_IDS,
    STRESS_IDS,
)


def _runtime_paths_from_phase2cp(phase2cp: dict[str, Any]) -> list[str]:
    rows = phase2cp.get("metrics", {}).get("runtime_paths", [])
    return [str(row) for row in rows] if isinstance(rows, list) else []


def _source_from_phase2cv(phase2cv: dict[str, Any]) -> dict[str, str | list[str]]:
    phase2cu_report_json = phase2cv.get("evidence", {}).get("phase2cu_report_json")
    if not phase2cu_report_json:
        raise ValueError("Phase2CW requires Phase2CV evidence.phase2cu_report_json")
    phase2cu = _read_json(phase2cu_report_json)
    phase2cs_report_json = phase2cu.get("evidence", {}).get("phase2cs_report_json")
    phase2cp_report_json = phase2cu.get("evidence", {}).get("phase2cp_report_json")
    suite_json = phase2cu.get("evidence", {}).get("suite_json")
    if not phase2cs_report_json or not phase2cp_report_json or not suite_json:
        raise ValueError("Phase2CW requires Phase2CU evidence phase2cs/phase2cp/suite paths")
    phase2cs = _read_json(phase2cs_report_json)
    package_build_report_json = phase2cs.get("evidence", {}).get(
        "package_build_report_json"
    )
    if not package_build_report_json:
        phase2cp = _read_json(phase2cp_report_json)
        package_build_report_json = phase2cp.get("evidence", {}).get(
            "package_build_report_json"
        )
    if not package_build_report_json:
        raise ValueError("Phase2CW requires package build report evidence")
    package_build = _read_json(package_build_report_json)
    package_path = package_build.get("package_path")
    if not package_path:
        raise ValueError("Phase2CW requires package build package_path")
    phase2cp = _read_json(phase2cp_report_json)
    runtimes = _runtime_paths_from_phase2cp(phase2cp)
    if not runtimes:
        raise ValueError("Phase2CW requires Phase2CP runtime paths")
    return {
        "phase2cu_report_json": str(phase2cu_report_json),
        "phase2cs_report_json": str(phase2cs_report_json),
        "phase2cp_report_json": str(phase2cp_report_json),
        "package_build_report_json": str(package_build_report_json),
        "package_path": str(package_path),
        "suite_json": str(suite_json),
        "runtime_interpreters": runtimes,
    }


def _run_phase2cw_subprocess(
    *,
    runtime_interpreter: str,
    package_path: str | Path,
    suite_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float,
    max_extra_steps: int,
    cwd: str | Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    src_path = str(Path(cwd) / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    command = [
        runtime_interpreter,
        "-m",
        "reflexlm.cli.run_phase2cw_runtime_perturbation_recovery_stress_expansion",
        "--package-path",
        str(package_path),
        "--suite-json",
        str(suite_json),
        "--output-dir",
        str(output_dir),
        "--output-report-json",
        str(output_report_json),
        "--timeout-seconds",
        str(timeout_seconds),
        "--max-extra-steps",
        str(max_extra_steps),
    ]
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _report_signature(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": report.get("passed"),
        "runtime_interpreter": report.get("runtime_interpreter"),
        "runtime_environment": report.get("runtime_environment"),
        "stress_ids": report.get("stress_ids"),
        "extra_stress_ids": report.get("extra_stress_ids"),
        "checks": report.get("checks", {}),
        "metrics": report.get("metrics", {}),
        "ready_for_bounded_expanded_recovery_stress_claim": report.get(
            "ready_for_bounded_expanded_recovery_stress_claim"
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


def validate_phase2cw_runtime_perturbation_recovery_stress_expansion(
    report: dict[str, Any],
) -> dict[str, Any]:
    perturbation_rows = report.get("perturbation_results", [])
    if not isinstance(perturbation_rows, list):
        perturbation_rows = []
    runtime_rows = [
        row
        for perturbation in perturbation_rows
        for row in perturbation.get("runtime_results", [])
        if isinstance(row, dict)
    ]
    runtime_report_read_failures = 0
    runtime_reference_signatures: dict[str, dict[str, Any]] = {}
    runtime_signature_mismatches = 0
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
        stress_counts = runtime_report.get("metrics", {}).get("stress_counts", {})
        recomputed_rows.append(
            {
                "report_passed": runtime_report.get("passed") is True,
                "all_stress_ids_present": set(stress_counts) == set(STRESS_IDS),
                "extra_stress_ids_present": all(
                    stress_counts.get(stress_id, 0) >= 3
                    for stress_id in PHASE2CW_EXTRA_STRESS_IDS
                ),
                "failure_recovery_success_rate": runtime_report.get(
                    "metrics", {}
                ).get("failure_recovery_success_rate"),
                "bounded_claim_only": (
                    runtime_report.get(
                        "ready_for_bounded_expanded_recovery_stress_claim"
                    )
                    is True
                    and runtime_report.get("ready_for_general_shell_autonomy_claim")
                    is False
                    and runtime_report.get(
                        "ready_for_general_runtime_invariance_claim"
                    )
                    is False
                    and runtime_report.get(
                        "ready_for_open_ended_native_perception_claim"
                    )
                    is False
                    and runtime_report.get("ready_for_production_autonomy_claim")
                    is False
                    and runtime_report.get(
                        "ready_for_epoch_making_architecture_claim"
                    )
                    is False
                ),
            }
        )
    runtime_count = int(report.get("metrics", {}).get("runtime_count", 0) or 0)
    perturbation_count = int(
        report.get("metrics", {}).get("perturbation_count", 0) or 0
    )
    checks = {
        "artifact_family_matches_phase2cw": (
            report.get("artifact_family")
            == "phase2cw_runtime_perturbation_recovery_stress_expansion"
        ),
        "top_level_phase2cw_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_expanded_recovery_stress_matrix_claim")
            is True
            and report.get("ready_for_general_shell_autonomy_claim") is False
            and report.get("ready_for_general_runtime_invariance_claim") is False
            and report.get("ready_for_open_ended_native_perception_claim") is False
            and report.get("ready_for_production_autonomy_claim") is False
            and report.get("ready_for_epoch_making_architecture_claim") is False
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "minimum_three_runtime_interpreters_recorded": runtime_count >= 3,
        "minimum_three_perturbations_recorded": perturbation_count >= 3,
        "all_expected_runtime_rows_present": bool(runtime_rows)
        and len(runtime_rows) == runtime_count * perturbation_count,
        "all_runtime_subprocesses_recorded_zero": bool(runtime_rows)
        and all(row.get("subprocess", {}).get("returncode") == 0 for row in runtime_rows),
        "all_runtime_reports_readable": bool(runtime_rows)
        and runtime_report_read_failures == 0,
        "all_runtime_reports_exist_flags_true": bool(runtime_rows)
        and all(row.get("report_exists") is True for row in runtime_rows),
        "all_runtime_reports_passed_flags_true": bool(runtime_rows)
        and all(row.get("report_passed") is True for row in runtime_rows),
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
        "all_runtime_signatures_stable_across_perturbations": bool(runtime_rows)
        and runtime_signature_mismatches == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "runtime_rows": len(runtime_rows),
            "runtime_report_read_failures": runtime_report_read_failures,
            "runtime_signature_mismatch_count": runtime_signature_mismatches,
            "recomputed_runtime_reports_passed": sum(
                row["report_passed"] is True for row in recomputed_rows
            ),
        },
    }


def audit_phase2cw_runtime_perturbation_recovery_stress_expansion(
    *,
    phase2cv_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    perturbations: list[dict[str, Any]] | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(cwd or Path.cwd())
    phase2cv = _read_json(phase2cv_report_json)
    source = _source_from_phase2cv(phase2cv)
    runtimes = list(source["runtime_interpreters"])
    perturbation_specs = list(perturbations or DEFAULT_PERTURBATIONS)
    perturbation_rows: list[dict[str, Any]] = []
    runtime_reference_signatures: dict[str, dict[str, Any]] = {}
    signature_mismatch_count = 0

    for perturbation_index, spec in enumerate(perturbation_specs):
        perturbation_id = str(spec["perturbation_id"])
        timeout_seconds = float(spec["timeout_seconds"])
        max_extra_steps = int(spec["max_extra_steps"])
        runtime_rows: list[dict[str, Any]] = []
        for runtime_index, runtime in enumerate(runtimes):
            runtime_dir = (
                Path(output_dir)
                / f"{perturbation_index:02d}_{perturbation_id}"
                / f"runtime_{runtime_index:02d}"
            )
            report_json = runtime_dir / "phase2cw_report.json"
            run = _run_phase2cw_subprocess(
                runtime_interpreter=runtime,
                package_path=str(source["package_path"]),
                suite_json=str(source["suite_json"]),
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
            stress_counts = runtime_report.get("metrics", {}).get("stress_counts", {})
            runtime_rows.append(
                {
                    "runtime_index": runtime_index,
                    "runtime_interpreter": runtime,
                    "report_json": str(report_json),
                    "output_dir": str(runtime_dir),
                    "subprocess": run,
                    "report_exists": report_exists,
                    "report_passed": runtime_report.get("passed") is True,
                    "all_stress_ids_present": set(stress_counts) == set(STRESS_IDS),
                    "extra_stress_ids_present": all(
                        stress_counts.get(stress_id, 0) >= 3
                        for stress_id in PHASE2CW_EXTRA_STRESS_IDS
                    ),
                    "failure_recovery_success_rate": runtime_report.get(
                        "metrics", {}
                    ).get("failure_recovery_success_rate"),
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

    all_runtime_rows = [
        row
        for perturbation in perturbation_rows
        for row in perturbation["runtime_results"]
    ]
    checks = {
        "source_phase2cv_passed": phase2cv.get("passed") is True,
        "minimum_three_runtime_interpreters_met": len(runtimes) >= 3,
        "minimum_three_perturbations_met": len(perturbation_rows) >= 3,
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
        "all_runtime_signatures_stable_across_perturbations": bool(all_runtime_rows)
        and signature_mismatch_count == 0,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cw_runtime_perturbation_recovery_stress_expansion",
        "passed": passed,
        "ready_for_bounded_expanded_recovery_stress_matrix_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "runtime_count": len(runtimes),
            "perturbation_count": len(perturbation_rows),
            "fresh_runtime_execution_count": len(all_runtime_rows),
            "passed_runtime_reports": sum(
                row["report_passed"] is True for row in all_runtime_rows
            ),
            "runtime_signature_mismatch_count": signature_mismatch_count,
            "stress_ids": list(STRESS_IDS),
            "extra_stress_ids": list(PHASE2CW_EXTRA_STRESS_IDS),
        },
        "perturbation_results": perturbation_rows,
        "supported_claims": [
            "bounded package-internal structured-runtime cortex recovered from expanded file and argument stress families across safe runtime budget perturbations"
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
            "phase2cx_expanded_recovery_stress_negative_controls"
            if passed
            else "repair_phase2cw_runtime_perturbation_recovery_stress_expansion"
        ),
        "evidence": {
            "phase2cv_report_json": str(phase2cv_report_json),
            **{key: value for key, value in source.items() if key != "runtime_interpreters"},
            "runtime_interpreters": runtimes,
            "expanded_recovery_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2CW expanded recovery stress under runtime perturbations."
    )
    parser.add_argument("--phase2cv-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2cw_runtime_perturbation_recovery_stress_expansion(
        phase2cv_report_json=args.phase2cv_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
