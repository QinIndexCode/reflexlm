import json
from pathlib import Path

from reflexlm.cli.audit_phase2cj_runtime_interpreter_invariance import (
    audit_phase2cj_runtime_interpreter_invariance,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _repo_report(*, runtime: str, canonicalized: bool, passed: bool = True) -> dict:
    return {
        "repository_id": "repo_a",
        "provenance": {"origin": "https://example.test/repo-a.git", "head": "abc123"},
        "recipe_ids": ["failure_dual_observe_recover", "timeout_stderr_recovery"],
        "contract_signatures": [
            "permissions=[RUN_COMMAND,DONE] completion=[RUN_COMMAND]",
            "permissions=[RUN_COMMAND,READ_STDERR,DONE] completion=[RUN_COMMAND,READ_STDERR]",
        ],
        "passed": passed,
        "checks": {
            "all_model_selected_actions_were_allowlisted": passed,
            "all_task_completion_predicates_satisfied": passed,
        },
        "metrics": {
            "episodes": 4,
            "executed_actions": 16,
            "task_completion_successes": 4 if passed else 3,
            "task_completion_success_rate": 1.0 if passed else 0.75,
        },
        "policy_configuration": {
            "policy_metadata": {
                "package_internal_expert": True,
                "expert_name": "structured_runtime_cortex",
                "runtime_python": runtime,
                "python_identity_canonicalization": canonicalized,
            }
        },
    }


def _runtime_report(*, runtime: str, canonicalized: bool, passed: bool = True) -> dict:
    return {
        "artifact_family": "phase2ci_unified_package_open_task_family_repo_runtime",
        "passed": passed,
        "seed": 20260608,
        "runtime_interpreter": runtime,
        "checks": {
            "all_repository_runtime_suites_passed": passed,
            "all_repository_actions_were_allowlisted": passed,
            "all_repository_task_completion_predicates_satisfied": passed,
            "all_repositories_used_package_internal_runtime_cortex": passed,
        },
        "metrics": {
            "repositories": 1,
            "generated_episode_templates": 2,
            "episodes": 4,
            "executed_actions": 16,
            "task_completion_successes": 4 if passed else 3,
            "task_completion_success_rate": 1.0 if passed else 0.75,
        },
        "repository_reports": [
            _repo_report(runtime=runtime, canonicalized=canonicalized, passed=passed)
        ],
    }


def test_phase2cj_audit_accepts_negative_control_and_two_runtime_passes(
    tmp_path: Path,
) -> None:
    canonical = "C:\\Python313\\python.exe"
    alternate = "D:\\repo\\.venv312\\Scripts\\python.exe"
    report = audit_phase2cj_runtime_interpreter_invariance(
        failed_before_report_json=_write(
            tmp_path / "failed-before.json",
            _runtime_report(runtime=alternate, canonicalized=False, passed=False),
        ),
        canonical_report_json=_write(
            tmp_path / "canonical.json",
            _runtime_report(runtime=canonical, canonicalized=False, passed=True),
        ),
        alternate_report_json=_write(
            tmp_path / "alternate.json",
            _runtime_report(runtime=alternate, canonicalized=True, passed=True),
        ),
        package_build_report_json=_write(
            tmp_path / "build.json",
            {
                "passed": True,
                "structured_runtime_cortex_python_identity": canonical,
            },
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_runtime_interpreter_invariance_claim"] is True
    assert report["ready_for_general_runtime_interpreter_invariance_claim"] is False
    assert report["next_required_experiment"] == (
        "phase2ck_cross_runtime_path_and_python_version_matrix"
    )


def test_phase2cj_audit_rejects_without_alternate_canonicalization(
    tmp_path: Path,
) -> None:
    canonical = "C:\\Python313\\python.exe"
    alternate = "D:\\repo\\.venv312\\Scripts\\python.exe"
    report = audit_phase2cj_runtime_interpreter_invariance(
        failed_before_report_json=_write(
            tmp_path / "failed-before.json",
            _runtime_report(runtime=alternate, canonicalized=False, passed=False),
        ),
        canonical_report_json=_write(
            tmp_path / "canonical.json",
            _runtime_report(runtime=canonical, canonicalized=False, passed=True),
        ),
        alternate_report_json=_write(
            tmp_path / "alternate.json",
            _runtime_report(runtime=alternate, canonicalized=False, passed=True),
        ),
        package_build_report_json=_write(
            tmp_path / "build.json",
            {
                "passed": True,
                "structured_runtime_cortex_python_identity": canonical,
            },
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["checks"]["alternate_reports_python_identity_canonicalization"] is False
