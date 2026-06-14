from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2cj_runtime_interpreter_invariance import (
    _all_repo_checks,
    _all_repo_package_internal_runtime,
    _repo_policy_metadata,
    _top_level_success,
)
from reflexlm.cli.audit_phase2ck_cross_runtime_matrix import (
    MAPPING_SCOPE,
    _all_repo_mapping_scope,
)
from reflexlm.cli.audit_phase2cl_runtime_environment_shell_matrix import (
    PERTURBATION_IDS,
    _all_cmd_wrappers_are_explicit_argv,
    _all_cwd_probes_are_generated_subdirs,
    _all_env_overlays_are_manifest_declared,
    _all_failure_recovery_gates_not_applicable,
    _all_manifests_contract_bounded,
    _all_manifests_have_expected_perturbations,
    _load_generated_manifests,
    _repository_reports,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _runtime_paths(reports: list[dict[str, Any]]) -> list[str]:
    return [str(report.get("runtime_interpreter", "")) for report in reports]


def _runtime_versions(reports: list[dict[str, Any]]) -> list[str]:
    return [
        str(report.get("runtime_environment", {}).get("version", ""))
        for report in reports
    ]


def _manifest_signature(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repo in sorted(_repository_reports(report), key=lambda row: row["repository_id"]):
        manifest = _read_json(repo["generated_manifest_json"])
        rows.append(
            {
                "repository_id": repo.get("repository_id"),
                "origin": repo.get("provenance", {}).get("origin"),
                "head": repo.get("provenance", {}).get("head"),
                "perturbation_ids": list(repo.get("perturbation_ids", [])),
                "contract_signatures": list(repo.get("contract_signatures", [])),
                "episodes": [
                    {
                        "perturbation_id": episode.get("generator", {}).get(
                            "perturbation_id"
                        ),
                        "permissions": episode.get("permissions"),
                        "completion_requirements": episode.get(
                            "completion_requirements"
                        ),
                        "requires_failure": episode.get("requires_failure"),
                    }
                    for episode in manifest.get("episodes", [])
                ],
            }
        )
    return rows


def _metrics_signature(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {})
    return {
        "repositories": metrics.get("repositories"),
        "perturbation_counts": metrics.get("perturbation_counts"),
        "episodes": metrics.get("episodes"),
        "executed_actions": metrics.get("executed_actions"),
        "task_completion_successes": metrics.get("task_completion_successes"),
        "task_completion_success_rate": metrics.get("task_completion_success_rate"),
    }


def _all_reports_phase2cl_family(reports: list[dict[str, Any]]) -> bool:
    return bool(reports) and all(
        report.get("artifact_family")
        == "phase2cl_runtime_environment_and_shell_perturbation_matrix"
        for report in reports
    )


def _all_runtime_environment_executables_match_reports(
    reports: list[dict[str, Any]],
) -> bool:
    return bool(reports) and all(
        report.get("runtime_environment", {}).get("executable")
        == report.get("runtime_interpreter")
        for report in reports
    )


def _all_reports_structured_runtime_only_package_view(
    reports: list[dict[str, Any]],
) -> bool:
    return bool(reports) and all(
        report.get("package_metadata", {}).get("structured_runtime_cortex_packaged")
        is True
        and report.get("package_metadata", {}).get("native_head_policy_loaded")
        is False
        and report.get("package_metadata", {}).get("verification_cortex_loaded")
        is False
        for report in reports
    )


def _all_report_manifests_pass_phase2cl_shape(reports: list[dict[str, Any]]) -> bool:
    for report in reports:
        manifests = _load_generated_manifests(report)
        if len(manifests) != len(_repository_reports(report)):
            return False
        if not _all_manifests_have_expected_perturbations(manifests):
            return False
        if not _all_manifests_contract_bounded(manifests):
            return False
        if not _all_env_overlays_are_manifest_declared(manifests):
            return False
        if not _all_cmd_wrappers_are_explicit_argv(manifests):
            return False
        if not _all_cwd_probes_are_generated_subdirs(manifests):
            return False
        if not _all_failure_recovery_gates_not_applicable(report):
            return False
    return bool(reports)


def _all_repo_runtime_metadata_matches_report(report: dict[str, Any]) -> bool:
    runtime = report.get("runtime_interpreter")
    metadata_rows = _repo_policy_metadata(report)
    return bool(runtime) and bool(metadata_rows) and all(
        metadata.get("runtime_python") == runtime for metadata in metadata_rows
    )


def audit_phase2cm_cross_runtime_shell_environment_matrix(
    *,
    runtime_report_jsons: list[str | Path],
    package_build_report_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    reports = [_read_json(path) for path in runtime_report_jsons]
    build = _read_json(package_build_report_json)
    runtimes = _runtime_paths(reports)
    versions = _runtime_versions(reports)
    implementations = [
        str(report.get("runtime_environment", {}).get("implementation", ""))
        for report in reports
    ]
    reference = reports[0] if reports else {}
    reference_manifest_signature = _manifest_signature(reference) if reports else []
    reference_metrics_signature = _metrics_signature(reference) if reports else {}
    training_identity = str(build.get("structured_runtime_cortex_python_identity", ""))

    checks = {
        "minimum_three_runtime_reports_met": len(reports) >= 3,
        "all_reports_are_phase2cl_family": _all_reports_phase2cl_family(reports),
        "package_build_passed": build.get("passed") is True,
        "package_training_python_identity_recorded": bool(training_identity),
        "minimum_three_distinct_runtime_paths_met": len(set(runtimes)) >= 3,
        "minimum_two_distinct_python_versions_met": len(set(versions)) >= 2,
        "all_runtime_environments_are_cpython": bool(implementations)
        and all(implementation == "CPython" for implementation in implementations),
        "all_runtime_environment_executables_match_reports": (
            _all_runtime_environment_executables_match_reports(reports)
        ),
        "all_runtime_reports_passed": bool(reports)
        and all(_top_level_success(report) for report in reports),
        "all_runtime_reports_share_seed": bool(reports)
        and len({report.get("seed") for report in reports}) == 1,
        "all_runtime_reports_share_phase2cl_manifest_matrix": (
            bool(reference_manifest_signature)
            and all(
                _manifest_signature(report) == reference_manifest_signature
                for report in reports
            )
        ),
        "all_runtime_reports_share_core_metrics": bool(reference_metrics_signature)
        and all(
            _metrics_signature(report) == reference_metrics_signature
            for report in reports
        ),
        "all_reports_used_structured_runtime_only_package_view": (
            _all_reports_structured_runtime_only_package_view(reports)
        ),
        "all_repositories_used_package_internal_runtime_cortex": bool(reports)
        and all(_all_repo_package_internal_runtime(report) for report in reports),
        "all_repositories_used_command_prefix_identity_mapping": bool(reports)
        and all(_all_repo_mapping_scope(report) for report in reports),
        "all_repo_runtime_metadata_matches_report": bool(reports)
        and all(_all_repo_runtime_metadata_matches_report(report) for report in reports),
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
        "all_report_manifests_pass_phase2cl_shape": (
            _all_report_manifests_pass_phase2cl_shape(reports)
        ),
        "bounded_claim_true_only_for_all_reports": bool(reports)
        and all(
            report.get(
                "ready_for_bounded_runtime_environment_shell_perturbation_claim"
            )
            is True
            and report.get("ready_for_general_shell_autonomy_claim") is False
            and report.get("ready_for_open_ended_native_perception_claim") is False
            and report.get("ready_for_production_autonomy_claim") is False
            and report.get("ready_for_epoch_making_architecture_claim") is False
            for report in reports
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cm_cross_runtime_shell_environment_matrix",
        "passed": passed,
        "ready_for_bounded_cross_runtime_shell_environment_claim": passed,
        "ready_for_general_shell_autonomy_claim": False,
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
            "perturbation_ids": list(PERTURBATION_IDS),
            "perturbation_counts_per_runtime": reference.get("metrics", {}).get(
                "perturbation_counts"
            ),
            "episodes_per_runtime": reference.get("metrics", {}).get("episodes"),
            "executed_actions_per_runtime": reference.get("metrics", {}).get(
                "executed_actions"
            ),
            "task_completion_success_rate_per_runtime": reference.get(
                "metrics", {}
            ).get("task_completion_success_rate"),
            "python_identity_mapping_scope": MAPPING_SCOPE,
        },
        "supported_claims": [
            (
                "bounded package-internal structured-runtime cortex completed the same "
                "repository-disjoint environment-overlay, cwd, and explicit cmd.exe argv "
                "wrapper perturbation matrix across three recorded Python executable "
                "paths and two recorded CPython versions using command-executable-prefix "
                "identity mapping, without free-form shell execution or unrelated expert "
                "loading"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "free-form shell autonomy",
            "arbitrary shell/environment generalization",
            "general runtime invariance",
            "non-CPython runtime invariance",
            "operating-system invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cn_runtime_environment_negative_controls"
            if passed
            else "repair_phase2cm_cross_runtime_shell_environment_matrix"
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
    parser = argparse.ArgumentParser(
        description="Audit the Phase2CM cross-runtime shell/environment matrix."
    )
    parser.add_argument("--runtime-report-json", action="append", required=True)
    parser.add_argument("--package-build-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2cm_cross_runtime_shell_environment_matrix(
        runtime_report_jsons=args.runtime_report_json,
        package_build_report_json=args.package_build_report_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
