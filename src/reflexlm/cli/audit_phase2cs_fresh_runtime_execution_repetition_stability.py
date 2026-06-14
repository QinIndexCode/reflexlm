from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from reflexlm.cli.audit_phase2cp_cross_runtime_environment_stress_recovery_matrix import (
    audit_phase2cp_cross_runtime_environment_stress_recovery_matrix,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _runtime_paths_from_phase2cp(phase2cp: dict[str, Any]) -> list[str]:
    rows = phase2cp.get("metrics", {}).get("runtime_paths", [])
    if isinstance(rows, list) and rows:
        return [str(row) for row in rows]
    evidence_rows = phase2cp.get("evidence", {}).get("runtime_report_jsons", [])
    paths: list[str] = []
    if isinstance(evidence_rows, list):
        for report_path in evidence_rows:
            report = _read_json(report_path)
            paths.append(str(report.get("runtime_interpreter", "")))
    return [path for path in paths if path]


def _stable_matrix_signature(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": audit.get("passed"),
        "ready_for_bounded_cross_runtime_environment_stress_recovery_claim": audit.get(
            "ready_for_bounded_cross_runtime_environment_stress_recovery_claim"
        ),
        "ready_for_general_shell_autonomy_claim": audit.get(
            "ready_for_general_shell_autonomy_claim"
        ),
        "ready_for_general_runtime_invariance_claim": audit.get(
            "ready_for_general_runtime_invariance_claim"
        ),
        "ready_for_open_ended_native_perception_claim": audit.get(
            "ready_for_open_ended_native_perception_claim"
        ),
        "ready_for_production_autonomy_claim": audit.get(
            "ready_for_production_autonomy_claim"
        ),
        "ready_for_epoch_making_architecture_claim": audit.get(
            "ready_for_epoch_making_architecture_claim"
        ),
        "checks": audit.get("checks", {}),
        "metrics": audit.get("metrics", {}),
        "supported_claims": audit.get("supported_claims", []),
        "unsupported_claims": audit.get("unsupported_claims", []),
        "next_required_experiment": audit.get("next_required_experiment"),
    }


def _runtime_report_signature(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_family": report.get("artifact_family"),
        "passed": report.get("passed"),
        "runtime_interpreter": report.get("runtime_interpreter"),
        "runtime_environment": report.get("runtime_environment"),
        "seed": report.get("seed"),
        "stress_ids": report.get("stress_ids"),
        "checks": report.get("checks", {}),
        "metrics": report.get("metrics", {}),
        "generated_contract_signatures": report.get(
            "generated_contract_signatures", []
        ),
        "ready_for_bounded_environment_stress_failure_recovery_claim": report.get(
            "ready_for_bounded_environment_stress_failure_recovery_claim"
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


def validate_phase2cs_fresh_runtime_execution_report(
    report: dict[str, Any],
) -> dict[str, Any]:
    repetition_rows = report.get("repetition_results", [])
    if not isinstance(repetition_rows, list):
        repetition_rows = []
    runtime_rows = [
        row
        for repetition in repetition_rows
        for row in repetition.get("runtime_results", [])
        if isinstance(row, dict)
    ]
    matrix_audits: list[dict[str, Any]] = []
    matrix_signatures: list[dict[str, Any]] = []
    runtime_reference_signatures: dict[str, dict[str, Any]] = {}
    runtime_signature_mismatches = 0
    runtime_report_read_failures = 0
    matrix_audit_read_failures = 0

    for row in runtime_rows:
        report_path = row.get("report_json")
        try:
            runtime_report = _read_json(report_path)
        except (OSError, TypeError, json.JSONDecodeError):
            runtime_report_read_failures += 1
            runtime_report = {}
        runtime_signature = _runtime_report_signature(runtime_report)
        runtime = str(row.get("runtime_interpreter", ""))
        reference = runtime_reference_signatures.setdefault(runtime, runtime_signature)
        if runtime_signature != reference:
            runtime_signature_mismatches += 1

    for repetition in repetition_rows:
        matrix_path = repetition.get("matrix_audit_json")
        try:
            matrix_audit = _read_json(matrix_path)
        except (OSError, TypeError, json.JSONDecodeError):
            matrix_audit_read_failures += 1
            matrix_audit = {}
        matrix_audits.append(matrix_audit)
        matrix_signatures.append(_stable_matrix_signature(matrix_audit))

    reference_matrix_signature = matrix_signatures[0] if matrix_signatures else {}
    matrix_signature_mismatches = sum(
        signature != reference_matrix_signature for signature in matrix_signatures
    )
    runtime_count = int(report.get("metrics", {}).get("runtime_count", 0) or 0)
    repetition_count = int(report.get("metrics", {}).get("repetition_count", 0) or 0)
    checks = {
        "artifact_family_matches_phase2cs": (
            report.get("artifact_family")
            == "phase2cs_fresh_runtime_execution_repetition_stability"
        ),
        "top_level_phase2cs_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_fresh_runtime_execution_repetition_stability_claim")
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
        "minimum_two_repetitions_recorded": repetition_count >= 2,
        "all_expected_runtime_rows_present": bool(runtime_rows)
        and len(runtime_rows) == runtime_count * repetition_count,
        "all_runtime_subprocesses_recorded_zero": bool(runtime_rows)
        and all(row.get("subprocess", {}).get("returncode") == 0 for row in runtime_rows),
        "all_runtime_reports_readable": bool(runtime_rows)
        and runtime_report_read_failures == 0,
        "all_runtime_reports_exist_flags_true": bool(runtime_rows)
        and all(row.get("report_exists") is True for row in runtime_rows),
        "all_runtime_reports_passed_flags_true": bool(runtime_rows)
        and all(row.get("report_passed") is True for row in runtime_rows),
        "all_runtime_report_signatures_match_first_repetition": bool(runtime_rows)
        and runtime_signature_mismatches == 0,
        "all_matrix_audits_readable": bool(repetition_rows)
        and matrix_audit_read_failures == 0,
        "all_matrix_audits_passed": bool(matrix_audits)
        and all(audit.get("passed") is True for audit in matrix_audits),
        "all_matrix_signatures_match_first_repetition": bool(matrix_signatures)
        and matrix_signature_mismatches == 0,
        "all_reports_under_repetition_dirs": bool(runtime_rows)
        and all(row.get("report_under_repetition_dir") is True for row in runtime_rows),
        "all_generated_manifests_under_repetition_dirs": bool(runtime_rows)
        and all(
            row.get("generated_manifest_dir_under_repetition_dir") is True
            for row in runtime_rows
        ),
        "bounded_claim_true_only_for_all_matrix_audits": bool(matrix_signatures)
        and all(
            signature.get(
                "ready_for_bounded_cross_runtime_environment_stress_recovery_claim"
            )
            is True
            and signature.get("ready_for_general_shell_autonomy_claim") is False
            and signature.get("ready_for_general_runtime_invariance_claim") is False
            and signature.get("ready_for_open_ended_native_perception_claim") is False
            and signature.get("ready_for_production_autonomy_claim") is False
            and signature.get("ready_for_epoch_making_architecture_claim") is False
            for signature in matrix_signatures
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "runtime_rows": len(runtime_rows),
            "matrix_audits": len(matrix_audits),
            "runtime_report_read_failures": runtime_report_read_failures,
            "matrix_audit_read_failures": matrix_audit_read_failures,
            "runtime_signature_mismatch_count": runtime_signature_mismatches,
            "matrix_signature_mismatch_count": matrix_signature_mismatches,
        },
    }


def _is_under(path: str | Path, root: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _run_phase2co_subprocess(
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
        "reflexlm.cli.run_phase2co_environment_stress_with_failure_recovery",
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


def audit_phase2cs_fresh_runtime_execution_repetition_stability(
    *,
    phase2cp_report_json: str | Path,
    suite_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    runtime_interpreters: list[str] | None = None,
    repetition_count: int = 2,
    timeout_seconds: float = 5.0,
    max_extra_steps: int = 5,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    if repetition_count < 1:
        raise ValueError("Phase2CS repetition_count must be positive")
    repo_root = Path(cwd or Path.cwd())
    phase2cp = _read_json(phase2cp_report_json)
    evidence = phase2cp.get("evidence", {})
    package_build_report_json = evidence.get("package_build_report_json")
    if not package_build_report_json:
        raise ValueError("Phase2CS requires Phase2CP evidence.package_build_report_json")
    package_build = _read_json(package_build_report_json)
    package_path = package_build.get("package_path")
    if not package_path:
        raise ValueError("Phase2CS requires package build package_path")

    runtimes = runtime_interpreters or _runtime_paths_from_phase2cp(phase2cp)
    if not runtimes:
        raise ValueError("Phase2CS requires at least one runtime interpreter")

    repetition_rows: list[dict[str, Any]] = []
    matrix_signatures: list[dict[str, Any]] = []
    runtime_reference_signatures: dict[str, dict[str, Any]] = {}
    runtime_signature_mismatches = 0

    for repetition_index in range(repetition_count):
        repetition_dir = Path(output_dir) / f"repetition_{repetition_index:02d}"
        runtime_report_jsons: list[str] = []
        runtime_rows: list[dict[str, Any]] = []
        for runtime_index, runtime_interpreter in enumerate(runtimes):
            runtime_dir = repetition_dir / f"runtime_{runtime_index:02d}"
            runtime_report_json = runtime_dir / "phase2co_report.json"
            run = _run_phase2co_subprocess(
                runtime_interpreter=runtime_interpreter,
                package_path=package_path,
                suite_json=suite_json,
                output_dir=runtime_dir,
                output_report_json=runtime_report_json,
                timeout_seconds=timeout_seconds,
                max_extra_steps=max_extra_steps,
                cwd=repo_root,
            )
            report_exists = runtime_report_json.exists()
            runtime_report = _read_json(runtime_report_json) if report_exists else {}
            runtime_signature = (
                _runtime_report_signature(runtime_report) if report_exists else {}
            )
            reference_signature = runtime_reference_signatures.setdefault(
                runtime_interpreter,
                runtime_signature,
            )
            signature_matches_reference = runtime_signature == reference_signature
            if not signature_matches_reference:
                runtime_signature_mismatches += 1
            runtime_report_jsons.append(str(runtime_report_json))
            runtime_rows.append(
                {
                    "runtime_index": runtime_index,
                    "runtime_interpreter": runtime_interpreter,
                    "report_json": str(runtime_report_json),
                    "output_dir": str(runtime_dir),
                    "subprocess": run,
                    "report_exists": report_exists,
                    "report_passed": runtime_report.get("passed") is True,
                    "report_under_repetition_dir": _is_under(
                        runtime_report_json,
                        repetition_dir,
                    ),
                    "generated_manifest_dir_under_repetition_dir": _is_under(
                        runtime_report.get("generated_manifest_dir", ""),
                        repetition_dir,
                    )
                    if report_exists
                    else False,
                    "signature_matches_first_repetition_for_runtime": (
                        signature_matches_reference
                    ),
                }
            )

        matrix_audit_json = repetition_dir / "phase2cp_fresh_execution_audit.json"
        matrix_audit = audit_phase2cp_cross_runtime_environment_stress_recovery_matrix(
            runtime_report_jsons=runtime_report_jsons,
            package_build_report_json=package_build_report_json,
            output_report_json=matrix_audit_json,
        )
        matrix_signature = _stable_matrix_signature(matrix_audit)
        matrix_signatures.append(matrix_signature)
        repetition_rows.append(
            {
                "repetition_index": repetition_index,
                "matrix_audit_json": str(matrix_audit_json),
                "matrix_passed": matrix_audit.get("passed") is True,
                "matrix_signature": matrix_signature,
                "runtime_results": runtime_rows,
            }
        )

    reference_matrix_signature = matrix_signatures[0] if matrix_signatures else {}
    matrix_signature_mismatches = sum(
        signature != reference_matrix_signature for signature in matrix_signatures
    )
    all_runtime_rows = [
        row
        for repetition in repetition_rows
        for row in repetition.get("runtime_results", [])
    ]
    checks = {
        "source_phase2cp_passed": phase2cp.get("passed") is True,
        "minimum_three_runtime_interpreters_met": len(runtimes) >= 3,
        "minimum_two_repetitions_met": repetition_count >= 2,
        "all_subprocesses_returned_zero": bool(all_runtime_rows)
        and all(row["subprocess"]["returncode"] == 0 for row in all_runtime_rows),
        "all_fresh_runtime_reports_exist": bool(all_runtime_rows)
        and all(row["report_exists"] is True for row in all_runtime_rows),
        "all_fresh_runtime_reports_passed": bool(all_runtime_rows)
        and all(row["report_passed"] is True for row in all_runtime_rows),
        "all_reports_written_under_repetition_dirs": bool(all_runtime_rows)
        and all(row["report_under_repetition_dir"] is True for row in all_runtime_rows),
        "all_generated_manifests_written_under_repetition_dirs": bool(all_runtime_rows)
        and all(
            row["generated_manifest_dir_under_repetition_dir"] is True
            for row in all_runtime_rows
        ),
        "all_repetition_matrix_audits_passed": bool(repetition_rows)
        and all(row["matrix_passed"] is True for row in repetition_rows),
        "all_repetition_matrix_signatures_match": bool(matrix_signatures)
        and matrix_signature_mismatches == 0,
        "all_runtime_signatures_match_first_repetition": bool(all_runtime_rows)
        and runtime_signature_mismatches == 0,
        "bounded_claim_true_only_for_all_repetitions": bool(matrix_signatures)
        and all(
            signature.get(
                "ready_for_bounded_cross_runtime_environment_stress_recovery_claim"
            )
            is True
            and signature.get("ready_for_general_shell_autonomy_claim") is False
            and signature.get("ready_for_general_runtime_invariance_claim") is False
            and signature.get("ready_for_open_ended_native_perception_claim") is False
            and signature.get("ready_for_production_autonomy_claim") is False
            and signature.get("ready_for_epoch_making_architecture_claim") is False
            for signature in matrix_signatures
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cs_fresh_runtime_execution_repetition_stability",
        "passed": passed,
        "ready_for_fresh_runtime_execution_repetition_stability_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "runtime_count": len(runtimes),
            "repetition_count": repetition_count,
            "fresh_runtime_execution_count": len(all_runtime_rows),
            "passed_runtime_reports": sum(
                row["report_passed"] is True for row in all_runtime_rows
            ),
            "passed_matrix_audits": sum(
                row["matrix_passed"] is True for row in repetition_rows
            ),
            "matrix_signature_mismatch_count": matrix_signature_mismatches,
            "runtime_signature_mismatch_count": runtime_signature_mismatches,
            "runtime_interpreters": runtimes,
        },
        "repetition_results": repetition_rows,
        "supported_claims": [
            (
                "bounded package-internal structured-runtime cortex fresh-executed "
                "the Phase2CO environment stress-recovery suite repeatedly across "
                "the recorded Phase2CP runtime matrix, preserving per-runtime reports "
                "and cross-runtime Phase2CP audit signatures"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "free-form shell autonomy",
            "arbitrary shell/environment generalization",
            "general runtime invariance beyond the recorded CPython matrix",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2ct_fresh_runtime_execution_negative_controls"
            if passed
            else "repair_phase2cs_fresh_runtime_execution_repetition_stability"
        ),
        "evidence": {
            "phase2cp_report_json": str(phase2cp_report_json),
            "package_build_report_json": str(package_build_report_json),
            "package_path": str(package_path),
            "suite_json": str(suite_json),
            "fresh_execution_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2CS fresh runtime execution repetition stability."
    )
    parser.add_argument("--phase2cp-report-json", required=True)
    parser.add_argument("--suite-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--runtime-interpreter", action="append")
    parser.add_argument("--repetition-count", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--max-extra-steps", type=int, default=5)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2cs_fresh_runtime_execution_repetition_stability(
        phase2cp_report_json=args.phase2cp_report_json,
        suite_json=args.suite_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
        runtime_interpreters=args.runtime_interpreter,
        repetition_count=args.repetition_count,
        timeout_seconds=args.timeout_seconds,
        max_extra_steps=args.max_extra_steps,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
