import json
from pathlib import Path

from reflexlm.cli.audit_phase2ck_cross_runtime_matrix import MAPPING_SCOPE
from reflexlm.cli.audit_phase2cm_cross_runtime_shell_environment_matrix import (
    audit_phase2cm_cross_runtime_shell_environment_matrix,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_step(
    *,
    perturbation_id: str,
    repository_id: str,
    token: str,
) -> dict:
    if perturbation_id == "direct_env_overlay":
        return {
            "action_type": "RUN_COMMAND",
            "argv": ["<PYTHON>", "-c", "print('direct')"],
            "env": {"PHASE2CL_ENV_TOKEN": token},
        }
    if perturbation_id == "cwd_subdir_probe":
        return {
            "action_type": "RUN_COMMAND",
            "argv": ["<PYTHON>", "-c", "print('cwd')"],
            "cwd": f".reflexlm_runtime_probe/phase2cl-{repository_id}-{token}",
        }
    return {
        "action_type": "RUN_COMMAND",
        "argv": ["cmd.exe", "/d", "/c", "<PYTHON>", "-c", "print('cmd')"],
        "env": {"PHASE2CL_ENV_TOKEN": token},
    }


def _manifest(repository_id: str) -> dict:
    perturbation_ids = [
        "direct_env_overlay",
        "cwd_subdir_probe",
        "cmd_wrapper_env_overlay",
    ]
    episodes = []
    for perturbation_id in perturbation_ids:
        token = f"{repository_id}-{perturbation_id}"
        run_step = _run_step(
            perturbation_id=perturbation_id,
            repository_id=repository_id,
            token=token,
        )
        episodes.append(
            {
                "episode_id": f"{repository_id}-{perturbation_id}",
                "permissions": [
                    run_step,
                    {"action_type": "READ_STDOUT"},
                    {"action_type": "DONE"},
                ],
                "completion_requirements": [
                    run_step,
                    {"action_type": "READ_STDOUT"},
                ],
                "requires_failure": False,
                "generator": {
                    "phase": "phase2cl",
                    "perturbation_id": perturbation_id,
                    "payload_token": token,
                },
            }
        )
    return {
        "workspace_root": f"D:/external/{repository_id}",
        "generated_by": {
            "phase": "phase2cl",
            "repository_id": repository_id,
            "perturbation_ids": perturbation_ids,
        },
        "episodes": episodes,
    }


def _runtime_report(
    tmp_path: Path,
    *,
    runtime: str,
    version: str,
    report_name: str,
) -> dict:
    repos = []
    for index in range(3):
        repository_id = f"repo_{index}"
        manifest_path = _write(
            tmp_path / report_name / "manifests" / f"{repository_id}.json",
            _manifest(repository_id),
        )
        repos.append(
            {
                "repository_id": repository_id,
                "generated_manifest_json": str(manifest_path),
                "provenance": {
                    "origin": f"https://example.test/{repository_id}.git",
                    "head": "abc123",
                },
                "perturbation_ids": [
                    "direct_env_overlay",
                    "cwd_subdir_probe",
                    "cmd_wrapper_env_overlay",
                ],
                "contract_signatures": [
                    "permissions=[RUN_COMMAND,READ_STDOUT,DONE] completion=[RUN_COMMAND,READ_STDOUT]"
                ],
                "passed": True,
                "checks": {
                    "all_model_selected_actions_were_allowlisted": True,
                    "all_task_completion_predicates_satisfied": True,
                    "failure_recovery_success_rate_meets_gate": True,
                },
                "metrics": {
                    "episodes": 3,
                    "executed_actions": 9,
                    "task_completion_successes": 3,
                    "failure_episodes": 0,
                    "failure_recovery_gate_applicable": False,
                },
                "policy_configuration": {
                    "policy_metadata": {
                        "package_internal_expert": True,
                        "expert_name": "structured_runtime_cortex",
                        "runtime_python": runtime,
                        "python_identity_mapping_scope": MAPPING_SCOPE,
                    }
                },
            }
        )
    return {
        "artifact_family": "phase2cl_runtime_environment_and_shell_perturbation_matrix",
        "passed": True,
        "seed": 20260608,
        "runtime_interpreter": runtime,
        "runtime_environment": {
            "implementation": "CPython",
            "version": version,
            "executable": runtime,
        },
        "ready_for_bounded_runtime_environment_shell_perturbation_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": {
            "all_repository_runtime_suites_passed": True,
            "all_repository_actions_were_allowlisted": True,
            "all_repository_task_completion_predicates_satisfied": True,
            "all_repositories_used_package_internal_runtime_cortex": True,
        },
        "metrics": {
            "repositories": 3,
            "perturbation_counts": {
                "direct_env_overlay": 3,
                "cwd_subdir_probe": 3,
                "cmd_wrapper_env_overlay": 3,
            },
            "episodes": 9,
            "executed_actions": 27,
            "task_completion_successes": 9,
            "task_completion_success_rate": 1.0,
        },
        "package_metadata": {
            "structured_runtime_cortex_packaged": True,
            "native_head_policy_loaded": False,
            "verification_cortex_loaded": False,
        },
        "repository_reports": repos,
    }


def test_phase2cm_audit_accepts_three_runtime_shell_env_matrix(
    tmp_path: Path,
) -> None:
    runtime_reports = [
        _write(
            tmp_path / "py313.json",
            _runtime_report(
                tmp_path,
                runtime="C:\\Python313\\python.exe",
                version="3.13.2",
                report_name="py313",
            ),
        ),
        _write(
            tmp_path / "venv312.json",
            _runtime_report(
                tmp_path,
                runtime="D:\\repo\\.venv312\\Scripts\\python.exe",
                version="3.12.10",
                report_name="venv312",
            ),
        ),
        _write(
            tmp_path / "alias312.json",
            _runtime_report(
                tmp_path,
                runtime="D:\\alias\\Scripts\\python.exe",
                version="3.12.10",
                report_name="alias312",
            ),
        ),
    ]

    report = audit_phase2cm_cross_runtime_shell_environment_matrix(
        runtime_report_jsons=runtime_reports,
        package_build_report_json=_write(
            tmp_path / "build.json",
            {
                "passed": True,
                "structured_runtime_cortex_python_identity": (
                    "C:\\Python313\\python.exe"
                ),
            },
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["distinct_runtime_paths"] == 3
    assert report["metrics"]["distinct_python_versions"] == 2
    assert report["ready_for_general_shell_autonomy_claim"] is False


def test_phase2cm_audit_rejects_missing_python_version_diversity(
    tmp_path: Path,
) -> None:
    runtime_reports = [
        _write(
            tmp_path / f"runtime-{index}.json",
            _runtime_report(
                tmp_path,
                runtime=f"D:\\runtime-{index}\\python.exe",
                version="3.13.2",
                report_name=f"runtime-{index}",
            ),
        )
        for index in range(3)
    ]

    report = audit_phase2cm_cross_runtime_shell_environment_matrix(
        runtime_report_jsons=runtime_reports,
        package_build_report_json=_write(
            tmp_path / "build.json",
            {
                "passed": True,
                "structured_runtime_cortex_python_identity": (
                    "C:\\Python313\\python.exe"
                ),
            },
        ),
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["checks"]["minimum_two_distinct_python_versions_met"] is False
