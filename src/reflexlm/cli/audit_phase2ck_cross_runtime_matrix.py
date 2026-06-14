from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cj_runtime_interpreter_invariance import (
    _all_repo_checks,
    _all_repo_package_internal_runtime,
    _repo_matrix_signature,
    _repo_policy_metadata,
    _top_level_success,
)


MAPPING_SCOPE = "command_executable_prefix"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _all_repo_mapping_scope(report: dict[str, Any]) -> bool:
    metadata_rows = _repo_policy_metadata(report)
    return bool(metadata_rows) and all(
        metadata.get("python_identity_mapping_scope") == MAPPING_SCOPE
        for metadata in metadata_rows
    )


def _canonicalization_matches_runtime(
    report: dict[str, Any],
    *,
    training_identity: str,
) -> bool:
    runtime = str(report.get("runtime_interpreter", ""))
    expected = runtime.lower() != training_identity.lower()
    metadata_rows = _repo_policy_metadata(report)
    return bool(metadata_rows) and all(
        metadata.get("python_identity_canonicalization") is expected
        and metadata.get("runtime_python") == runtime
        and metadata.get("python_identity") == training_identity
        for metadata in metadata_rows
    )


def audit_phase2ck_cross_runtime_matrix(
    *,
    runtime_report_jsons: list[str | Path],
    package_build_report_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    reports = [_read_json(path) for path in runtime_report_jsons]
    build = _read_json(package_build_report_json)
    training_identity = str(build.get("structured_runtime_cortex_python_identity", ""))
    reference = reports[0] if reports else {}
    reference_matrix = _repo_matrix_signature(reference)
    runtimes = [str(report.get("runtime_interpreter", "")) for report in reports]
    versions = [
        str(report.get("runtime_environment", {}).get("version", ""))
        for report in reports
    ]
    implementations = [
        str(report.get("runtime_environment", {}).get("implementation", ""))
        for report in reports
    ]
    checks = {
        "minimum_three_runtime_reports_met": len(reports) >= 3,
        "package_build_passed": build.get("passed") is True,
        "package_training_python_identity_recorded": bool(training_identity),
        "minimum_three_distinct_runtime_paths_met": len(set(runtimes)) >= 3,
        "minimum_two_distinct_python_versions_met": len(set(versions)) >= 2,
        "all_runtime_environments_are_cpython": (
            bool(implementations)
            and all(implementation == "CPython" for implementation in implementations)
        ),
        "all_runtime_environment_executables_match_reports": all(
            report.get("runtime_environment", {}).get("executable")
            == report.get("runtime_interpreter")
            for report in reports
        ),
        "all_runtime_reports_passed": bool(reports)
        and all(_top_level_success(report) for report in reports),
        "all_runtime_reports_share_seed": bool(reports)
        and len({report.get("seed") for report in reports}) == 1,
        "all_runtime_reports_share_timeout_recovery_window": bool(reports)
        and len(
            {
                report.get("timeout_recovery_command_timeout_seconds")
                for report in reports
            }
        )
        == 1,
        "all_runtime_reports_share_task_matrix": bool(reference_matrix)
        and all(_repo_matrix_signature(report) == reference_matrix for report in reports),
        "all_runtime_reports_share_core_metrics": bool(reports)
        and all(
            {
                "repositories": report.get("metrics", {}).get("repositories"),
                "generated_episode_templates": report.get("metrics", {}).get(
                    "generated_episode_templates"
                ),
                "episodes": report.get("metrics", {}).get("episodes"),
                "executed_actions": report.get("metrics", {}).get("executed_actions"),
                "task_completion_successes": report.get("metrics", {}).get(
                    "task_completion_successes"
                ),
            }
            == {
                "repositories": reference.get("metrics", {}).get("repositories"),
                "generated_episode_templates": reference.get("metrics", {}).get(
                    "generated_episode_templates"
                ),
                "episodes": reference.get("metrics", {}).get("episodes"),
                "executed_actions": reference.get("metrics", {}).get(
                    "executed_actions"
                ),
                "task_completion_successes": reference.get("metrics", {}).get(
                    "task_completion_successes"
                ),
            }
            for report in reports
        ),
        "all_reports_used_structured_runtime_only_package_view": bool(reports)
        and all(
            report.get("package_metadata", {}).get("native_head_policy_loaded") is False
            and report.get("package_metadata", {}).get("verification_cortex_loaded")
            is False
            and report.get("package_metadata", {}).get(
                "structured_runtime_cortex_packaged"
            )
            is True
            for report in reports
        ),
        "all_repositories_used_package_internal_runtime_cortex": bool(reports)
        and all(_all_repo_package_internal_runtime(report) for report in reports),
        "all_repositories_used_command_prefix_identity_mapping": bool(reports)
        and all(_all_repo_mapping_scope(report) for report in reports),
        "canonicalization_activation_matches_each_runtime": bool(training_identity)
        and bool(reports)
        and all(
            _canonicalization_matches_runtime(
                report,
                training_identity=training_identity,
            )
            for report in reports
        ),
        "all_repository_actions_allowlisted": bool(reports)
        and all(
            _all_repo_checks(report, "all_model_selected_actions_were_allowlisted")
            for report in reports
        ),
        "all_repository_completion_predicates_satisfied": bool(reports)
        and all(
            _all_repo_checks(report, "all_task_completion_predicates_satisfied")
            for report in reports
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ck_cross_runtime_path_and_python_version_matrix",
        "passed": passed,
        "ready_for_bounded_cross_runtime_path_and_python_version_claim": passed,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "runtime_paths": runtimes,
            "python_versions": versions,
            "distinct_runtime_paths": len(set(runtimes)),
            "distinct_python_versions": len(set(versions)),
            "training_python_identity": training_identity,
            "repositories_per_runtime": reference.get("metrics", {}).get(
                "repositories"
            ),
            "episodes_per_runtime": reference.get("metrics", {}).get("episodes"),
            "executed_actions_per_runtime": reference.get("metrics", {}).get(
                "executed_actions"
            ),
            "task_completion_success_rate_per_runtime": reference.get(
                "metrics", {}
            ).get("task_completion_success_rate"),
            "timeout_recovery_command_timeout_seconds": reference.get(
                "timeout_recovery_command_timeout_seconds"
            ),
        },
        "supported_claims": [
            (
                "bounded package-internal structured-runtime cortex completed the "
                "same generated repository-disjoint task matrix across three recorded "
                "Python executable paths and two recorded CPython versions using "
                "command-executable-prefix identity mapping without loading unrelated "
                "packaged cortical experts"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "general Python-version invariance",
            "non-CPython runtime invariance",
            "operating-system invariance",
            "shell/environment invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cl_runtime_environment_and_shell_perturbation_matrix"
            if passed
            else "repair_phase2ck_cross_runtime_path_and_python_version_matrix"
        ),
        "evidence": {
            "runtime_report_jsons": [str(path) for path in runtime_report_jsons],
            "package_build_report_json": str(package_build_report_json),
        },
    }
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Phase2CK runtime matrix.")
    parser.add_argument("--runtime-report-json", action="append", required=True)
    parser.add_argument("--package-build-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ck_cross_runtime_matrix(
        runtime_report_jsons=args.runtime_report_json,
        package_build_report_json=args.package_build_report_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
