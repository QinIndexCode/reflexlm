import json
from pathlib import Path

from reflexlm.cli.audit_phase2ck_cross_runtime_matrix import (
    MAPPING_SCOPE,
    audit_phase2ck_cross_runtime_matrix,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _runtime_report(*, runtime: str, version: str, training_identity: str) -> dict:
    canonicalized = runtime.lower() != training_identity.lower()
    repo = {
        "repository_id": "repo_a",
        "provenance": {"origin": "https://example.test/repo-a.git", "head": "abc"},
        "recipe_ids": ["timeout_stderr_recovery"],
        "contract_signatures": ["permissions=[RUN_COMMAND] completion=[RUN_COMMAND]"],
        "checks": {
            "all_model_selected_actions_were_allowlisted": True,
            "all_task_completion_predicates_satisfied": True,
        },
        "metrics": {"episodes": 2},
        "policy_configuration": {
            "policy_metadata": {
                "package_internal_expert": True,
                "expert_name": "structured_runtime_cortex",
                "python_identity": training_identity,
                "runtime_python": runtime,
                "python_identity_canonicalization": canonicalized,
                "python_identity_mapping_scope": MAPPING_SCOPE,
            }
        },
    }
    return {
        "passed": True,
        "seed": 7,
        "runtime_interpreter": runtime,
        "runtime_environment": {
            "implementation": "CPython",
            "version": version,
            "executable": runtime,
        },
        "timeout_recovery_command_timeout_seconds": 0.5,
        "checks": {
            "all_repository_runtime_suites_passed": True,
            "all_repository_actions_were_allowlisted": True,
            "all_repository_task_completion_predicates_satisfied": True,
            "all_repositories_used_package_internal_runtime_cortex": True,
        },
        "metrics": {
            "repositories": 1,
            "generated_episode_templates": 1,
            "episodes": 2,
            "executed_actions": 8,
            "task_completion_successes": 2,
            "task_completion_success_rate": 1.0,
        },
        "package_metadata": {
            "native_head_policy_loaded": False,
            "verification_cortex_loaded": False,
            "structured_runtime_cortex_packaged": True,
        },
        "repository_reports": [repo],
    }


def test_phase2ck_audit_accepts_three_paths_and_two_versions(tmp_path: Path) -> None:
    training = "C:\\Python313\\python.exe"
    reports = [
        _write(tmp_path / "a.json", _runtime_report(runtime=training, version="3.13.2", training_identity=training)),
        _write(tmp_path / "b.json", _runtime_report(runtime="D:\\venv\\python.exe", version="3.12.10", training_identity=training)),
        _write(tmp_path / "c.json", _runtime_report(runtime="D:\\alias\\python.exe", version="3.12.10", training_identity=training)),
    ]

    report = audit_phase2ck_cross_runtime_matrix(
        runtime_report_jsons=reports,
        package_build_report_json=_write(
            tmp_path / "build.json",
            {
                "passed": True,
                "structured_runtime_cortex_python_identity": training,
            },
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["distinct_runtime_paths"] == 3
    assert report["metrics"]["distinct_python_versions"] == 2
    assert report["ready_for_general_runtime_invariance_claim"] is False


def test_phase2ck_audit_rejects_missing_runtime_version_diversity(tmp_path: Path) -> None:
    training = "C:\\Python313\\python.exe"
    reports = [
        _write(
            tmp_path / f"{index}.json",
            _runtime_report(
                runtime=f"D:\\runtime-{index}\\python.exe",
                version="3.13.2",
                training_identity=training,
            ),
        )
        for index in range(3)
    ]

    report = audit_phase2ck_cross_runtime_matrix(
        runtime_report_jsons=reports,
        package_build_report_json=_write(
            tmp_path / "build.json",
            {
                "passed": True,
                "structured_runtime_cortex_python_identity": training,
            },
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["checks"]["minimum_two_distinct_python_versions_met"] is False
