from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cp_cross_runtime_environment_stress_recovery_matrix import (
    audit_phase2cp_cross_runtime_environment_stress_recovery_matrix,
)
from reflexlm.cli.audit_phase2cq_cross_runtime_stress_negative_controls import (
    _materialize_case_inputs,
    _read_json,
    _rewrite_case_inputs,
    _write_json,
)


def _stable_signature(audit: dict[str, Any]) -> dict[str, Any]:
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


def _runtime_paths(audit: dict[str, Any]) -> list[str]:
    rows = audit.get("metrics", {}).get("runtime_paths", [])
    return [str(row) for row in rows] if isinstance(rows, list) else []


def _python_versions(audit: dict[str, Any]) -> list[str]:
    rows = audit.get("metrics", {}).get("python_versions", [])
    return [str(row) for row in rows] if isinstance(rows, list) else []


def _inject_metrics_drift(reports: list[dict[str, Any]]) -> None:
    if len(reports) < 2:
        raise ValueError("Phase2CR drift injection requires at least two reports")
    metrics = reports[1].setdefault("metrics", {})
    observed = int(metrics.get("observed_recoveries_after_failure", 0))
    metrics["observed_recoveries_after_failure"] = max(0, observed - 1)
    failure_episodes = int(metrics.get("failure_episodes", observed or 1))
    metrics["failure_recovery_success_rate"] = (
        metrics["observed_recoveries_after_failure"] / failure_episodes
    )


def audit_phase2cr_stress_recovery_repetition_stability(
    *,
    phase2cp_report_json: str | Path,
    output_dir: str | Path,
    output_report_json: str | Path,
    repetition_count: int = 3,
    inject_drift_repetition_index: int | None = None,
) -> dict[str, Any]:
    if repetition_count < 1:
        raise ValueError("Phase2CR repetition_count must be positive")

    phase2cp = _read_json(phase2cp_report_json)
    evidence = phase2cp.get("evidence", {})
    source_runtime_report_jsons = evidence.get("runtime_report_jsons", [])
    package_build_report_json = evidence.get("package_build_report_json")
    if not isinstance(source_runtime_report_jsons, list) or not source_runtime_report_jsons:
        raise ValueError("Phase2CR requires Phase2CP evidence.runtime_report_jsons")
    if not package_build_report_json:
        raise ValueError("Phase2CR requires Phase2CP evidence.package_build_report_json")

    repetition_rows: list[dict[str, Any]] = []
    for index in range(repetition_count):
        case_dir = Path(output_dir) / f"repetition_{index:02d}"
        runtime_report_jsons, manifest_groups, subreport_groups = _materialize_case_inputs(
            source_runtime_report_jsons=source_runtime_report_jsons,
            case_dir=case_dir,
        )
        reports = [_read_json(path) for path in runtime_report_jsons]
        if inject_drift_repetition_index == index:
            _inject_metrics_drift(reports)
            _rewrite_case_inputs(
                runtime_report_jsons=runtime_report_jsons,
                reports=reports,
                manifest_groups=manifest_groups,
                subreport_groups=subreport_groups,
            )

        audit_report_json = case_dir / "phase2cp_repetition_audit.json"
        audit = audit_phase2cp_cross_runtime_environment_stress_recovery_matrix(
            runtime_report_jsons=runtime_report_jsons,
            package_build_report_json=package_build_report_json,
            output_report_json=audit_report_json,
        )
        repetition_rows.append(
            {
                "repetition_index": index,
                "audit_report_json": str(audit_report_json),
                "runtime_report_jsons": [str(path) for path in runtime_report_jsons],
                "passed": audit.get("passed") is True,
                "checks_passed": all(
                    value is True for value in audit.get("checks", {}).values()
                ),
                "runtime_paths": _runtime_paths(audit),
                "python_versions": _python_versions(audit),
                "signature": _stable_signature(audit),
            }
        )

    reference_signature = repetition_rows[0]["signature"] if repetition_rows else {}
    reference_runtime_paths = _runtime_paths(phase2cp)
    reference_python_versions = _python_versions(phase2cp)
    checks = {
        "source_phase2cp_passed": phase2cp.get("passed") is True,
        "minimum_three_repetitions_met": repetition_count >= 3,
        "all_repetition_audits_passed": bool(repetition_rows)
        and all(row["passed"] is True for row in repetition_rows),
        "all_repetition_checks_passed": bool(repetition_rows)
        and all(row["checks_passed"] is True for row in repetition_rows),
        "all_repetition_signatures_match_reference": bool(repetition_rows)
        and all(row["signature"] == reference_signature for row in repetition_rows),
        "all_repetition_runtime_paths_match_source": bool(repetition_rows)
        and all(row["runtime_paths"] == reference_runtime_paths for row in repetition_rows),
        "all_repetition_python_versions_match_source": bool(repetition_rows)
        and all(
            row["python_versions"] == reference_python_versions
            for row in repetition_rows
        ),
        "bounded_claim_true_only_for_all_repetitions": bool(repetition_rows)
        and all(
            row["signature"].get(
                "ready_for_bounded_cross_runtime_environment_stress_recovery_claim"
            )
            is True
            and row["signature"].get("ready_for_general_shell_autonomy_claim") is False
            and row["signature"].get("ready_for_general_runtime_invariance_claim")
            is False
            and row["signature"].get("ready_for_open_ended_native_perception_claim")
            is False
            and row["signature"].get("ready_for_production_autonomy_claim") is False
            and row["signature"].get("ready_for_epoch_making_architecture_claim")
            is False
            for row in repetition_rows
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cr_stress_recovery_repetition_stability",
        "passed": passed,
        "ready_for_phase2cp_repetition_stability_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "repetition_count": repetition_count,
            "passed_repetitions": sum(row["passed"] is True for row in repetition_rows),
            "signature_mismatch_count": sum(
                row["signature"] != reference_signature for row in repetition_rows
            ),
            "runtime_path_mismatch_count": sum(
                row["runtime_paths"] != reference_runtime_paths
                for row in repetition_rows
            ),
            "python_version_mismatch_count": sum(
                row["python_versions"] != reference_python_versions
                for row in repetition_rows
            ),
            "runtime_paths": reference_runtime_paths,
            "python_versions": reference_python_versions,
        },
        "repetition_results": repetition_rows,
        "supported_claims": [
            (
                "the Phase2CP cross-runtime environment stress-recovery gate is "
                "stable across repeated re-materialized audits of the same bounded "
                "runtime evidence, preserving checks, metrics, runtime identities, "
                "Python-version diversity, and bounded-only claim flags"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "fresh re-execution stability of runtime commands",
            "free-form shell autonomy",
            "arbitrary shell/environment generalization",
            "general runtime invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cs_fresh_runtime_execution_repetition_stability"
            if passed
            else "repair_phase2cr_stress_recovery_repetition_stability"
        ),
        "evidence": {
            "phase2cp_report_json": str(phase2cp_report_json),
            "source_runtime_report_jsons": [
                str(path) for path in source_runtime_report_jsons
            ],
            "package_build_report_json": str(package_build_report_json),
            "repetition_output_dir": str(output_dir),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2CR stress-recovery repetition stability."
    )
    parser.add_argument("--phase2cp-report-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--repetition-count", type=int, default=3)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2cr_stress_recovery_repetition_stability(
        phase2cp_report_json=args.phase2cp_report_json,
        output_dir=args.output_dir,
        output_report_json=args.output_report_json,
        repetition_count=args.repetition_count,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
