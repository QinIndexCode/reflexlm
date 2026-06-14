from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _repo_key(report: dict[str, Any]) -> str:
    return str(report.get("repository_id", ""))


def _repo_matrix_signature(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for repo in sorted(report.get("repository_reports", []), key=_repo_key):
        rows.append(
            {
                "repository_id": repo.get("repository_id"),
                "origin": repo.get("provenance", {}).get("origin"),
                "head": repo.get("provenance", {}).get("head"),
                "recipe_ids": list(repo.get("recipe_ids", [])),
                "contract_signatures": list(repo.get("contract_signatures", [])),
                "episodes": repo.get("metrics", {}).get("episodes"),
            }
        )
    return rows


def _repo_policy_metadata(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        repo.get("policy_configuration", {}).get("policy_metadata", {})
        for repo in report.get("repository_reports", [])
    ]


def _all_repo_checks(report: dict[str, Any], check_name: str) -> bool:
    repos = report.get("repository_reports", [])
    return bool(repos) and all(
        repo.get("checks", {}).get(check_name) is True for repo in repos
    )


def _all_repo_package_internal_runtime(report: dict[str, Any]) -> bool:
    metadata_rows = _repo_policy_metadata(report)
    return bool(metadata_rows) and all(
        metadata.get("package_internal_expert") is True
        and metadata.get("expert_name") == "structured_runtime_cortex"
        for metadata in metadata_rows
    )


def _all_repo_canonicalization(report: dict[str, Any], expected: bool) -> bool:
    metadata_rows = _repo_policy_metadata(report)
    return bool(metadata_rows) and all(
        metadata.get("python_identity_canonicalization") is expected
        for metadata in metadata_rows
    )


def _all_repo_runtime_matches_report(report: dict[str, Any]) -> bool:
    runtime = report.get("runtime_interpreter")
    metadata_rows = _repo_policy_metadata(report)
    return bool(runtime) and bool(metadata_rows) and all(
        metadata.get("runtime_python") == runtime for metadata in metadata_rows
    )


def _top_level_success(report: dict[str, Any]) -> bool:
    checks = report.get("checks", {})
    metrics = report.get("metrics", {})
    return (
        report.get("passed") is True
        and checks.get("all_repository_runtime_suites_passed") is True
        and checks.get("all_repository_actions_were_allowlisted") is True
        and checks.get("all_repository_task_completion_predicates_satisfied") is True
        and checks.get("all_repositories_used_package_internal_runtime_cortex") is True
        and metrics.get("task_completion_success_rate") == 1.0
    )


def audit_phase2cj_runtime_interpreter_invariance(
    *,
    failed_before_report_json: str | Path,
    canonical_report_json: str | Path,
    alternate_report_json: str | Path,
    package_build_report_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    failed_before = _read_json(failed_before_report_json)
    canonical = _read_json(canonical_report_json)
    alternate = _read_json(alternate_report_json)
    build = _read_json(package_build_report_json)

    canonical_matrix = _repo_matrix_signature(canonical)
    alternate_matrix = _repo_matrix_signature(alternate)
    training_python_identity = build.get("structured_runtime_cortex_python_identity")
    canonical_runtime = canonical.get("runtime_interpreter")
    alternate_runtime = alternate.get("runtime_interpreter")

    checks = {
        "pre_fix_alternate_runtime_negative_control_failed": (
            failed_before.get("passed") is False
            and failed_before.get("metrics", {}).get("task_completion_success_rate", 1.0)
            < 1.0
        ),
        "package_build_passed": build.get("passed") is True,
        "package_records_structured_runtime_python_identity": bool(
            training_python_identity
        ),
        "canonical_runtime_matches_training_identity": (
            canonical_runtime == training_python_identity
        ),
        "alternate_runtime_differs_from_training_identity": (
            bool(alternate_runtime)
            and bool(training_python_identity)
            and alternate_runtime != training_python_identity
        ),
        "canonical_report_passed": _top_level_success(canonical),
        "alternate_report_passed": _top_level_success(alternate),
        "canonical_and_alternate_seeds_match": (
            canonical.get("seed") == alternate.get("seed")
        ),
        "canonical_and_alternate_metrics_match": (
            canonical.get("metrics", {}).get("repositories")
            == alternate.get("metrics", {}).get("repositories")
            and canonical.get("metrics", {}).get("generated_episode_templates")
            == alternate.get("metrics", {}).get("generated_episode_templates")
            and canonical.get("metrics", {}).get("episodes")
            == alternate.get("metrics", {}).get("episodes")
            and canonical.get("metrics", {}).get("task_completion_successes")
            == alternate.get("metrics", {}).get("task_completion_successes")
        ),
        "canonical_and_alternate_task_matrix_match": (
            bool(canonical_matrix) and canonical_matrix == alternate_matrix
        ),
        "canonical_reports_package_internal_runtime_cortex": (
            _all_repo_package_internal_runtime(canonical)
        ),
        "alternate_reports_package_internal_runtime_cortex": (
            _all_repo_package_internal_runtime(alternate)
        ),
        "canonical_reports_no_python_identity_canonicalization": (
            _all_repo_canonicalization(canonical, False)
        ),
        "alternate_reports_python_identity_canonicalization": (
            _all_repo_canonicalization(alternate, True)
        ),
        "canonical_repo_runtime_metadata_matches_report": (
            _all_repo_runtime_matches_report(canonical)
        ),
        "alternate_repo_runtime_metadata_matches_report": (
            _all_repo_runtime_matches_report(alternate)
        ),
        "canonical_repo_actions_allowlisted": _all_repo_checks(
            canonical, "all_model_selected_actions_were_allowlisted"
        ),
        "alternate_repo_actions_allowlisted": _all_repo_checks(
            alternate, "all_model_selected_actions_were_allowlisted"
        ),
        "canonical_repo_completion_predicates_satisfied": _all_repo_checks(
            canonical, "all_task_completion_predicates_satisfied"
        ),
        "alternate_repo_completion_predicates_satisfied": _all_repo_checks(
            alternate, "all_task_completion_predicates_satisfied"
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cj_runtime_interpreter_invariance_audit",
        "passed": passed,
        "ready_for_bounded_runtime_interpreter_invariance_claim": passed,
        "ready_for_general_runtime_interpreter_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "pre_fix_completion_success_rate": failed_before.get("metrics", {}).get(
                "task_completion_success_rate"
            ),
            "canonical_runtime_interpreter": canonical_runtime,
            "alternate_runtime_interpreter": alternate_runtime,
            "structured_runtime_cortex_python_identity": training_python_identity,
            "repositories": canonical.get("metrics", {}).get("repositories"),
            "episodes_per_runtime": canonical.get("metrics", {}).get("episodes"),
            "executed_actions_per_runtime": canonical.get("metrics", {}).get(
                "executed_actions"
            ),
            "task_completion_success_rate_per_runtime": canonical.get(
                "metrics", {}
            ).get("task_completion_success_rate"),
            "task_matrix_repository_count": len(canonical_matrix),
        },
        "supported_claims": [
            (
                "bounded package-internal structured-runtime cortex completed the same "
                "generated task matrix under the recorded training Python interpreter "
                "and one alternate Python executable path by mapping runtime interpreter "
                "identity to the package training-time identity before neural policy "
                "inference, then executing the selected action with the active runtime"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "general interpreter-version invariance",
            "operating-system invariance",
            "shell/environment invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2ck_cross_runtime_path_and_python_version_matrix"
            if passed
            else "repair_phase2cj_runtime_interpreter_invariance"
        ),
        "evidence": {
            "failed_before_report_json": str(failed_before_report_json),
            "canonical_report_json": str(canonical_report_json),
            "alternate_report_json": str(alternate_report_json),
            "package_build_report_json": str(package_build_report_json),
        },
    }
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit bounded runtime-interpreter invariance for Phase2CJ."
    )
    parser.add_argument("--failed-before-report-json", required=True)
    parser.add_argument("--canonical-report-json", required=True)
    parser.add_argument("--alternate-report-json", required=True)
    parser.add_argument("--package-build-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2cj_runtime_interpreter_invariance(
        failed_before_report_json=args.failed_before_report_json,
        canonical_report_json=args.canonical_report_json,
        alternate_report_json=args.alternate_report_json,
        package_build_report_json=args.package_build_report_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
