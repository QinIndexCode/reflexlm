import json
from pathlib import Path

from reflexlm.cli.audit_phase2cl_runtime_environment_shell_matrix import (
    audit_phase2cl_runtime_environment_shell_matrix,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _episode(
    *,
    repository_id: str,
    perturbation_id: str,
    token: str,
    run_step: dict,
) -> dict:
    return {
        "episode_id": f"{repository_id}-{perturbation_id}-{token}",
        "permissions": [
            run_step,
            {"action_type": "READ_STDOUT"},
            {"action_type": "DONE"},
        ],
        "completion_requirements": [run_step, {"action_type": "READ_STDOUT"}],
        "requires_failure": False,
        "generator": {
            "phase": "phase2cl",
            "perturbation_id": perturbation_id,
            "payload_token": token,
        },
    }


def _manifest(repository_id: str, *, cmd_shell_true: bool = False) -> dict:
    direct_token = f"{repository_id}-direct"
    cwd_token = f"{repository_id}-cwd"
    cmd_token = f"{repository_id}-cmd"
    cmd_step = {
        "action_type": "RUN_COMMAND",
        "argv": ["cmd.exe", "/d", "/c", "<PYTHON>", "-c", "print('x')"],
        "env": {"PHASE2CL_ENV_TOKEN": cmd_token},
    }
    if cmd_shell_true:
        cmd_step["shell"] = True
    return {
        "workspace_root": f"D:/external/{repository_id}",
        "generated_by": {
            "phase": "phase2cl",
            "repository_id": repository_id,
            "perturbation_ids": [
                "direct_env_overlay",
                "cwd_subdir_probe",
                "cmd_wrapper_env_overlay",
            ],
        },
        "episodes": [
            _episode(
                repository_id=repository_id,
                perturbation_id="direct_env_overlay",
                token=direct_token,
                run_step={
                    "action_type": "RUN_COMMAND",
                    "argv": ["<PYTHON>", "-c", "print('x')"],
                    "env": {"PHASE2CL_ENV_TOKEN": direct_token},
                },
            ),
            _episode(
                repository_id=repository_id,
                perturbation_id="cwd_subdir_probe",
                token=cwd_token,
                run_step={
                    "action_type": "RUN_COMMAND",
                    "argv": ["<PYTHON>", "-c", "print('x')"],
                    "cwd": f".reflexlm_runtime_probe/phase2cl-{repository_id}-{cwd_token}",
                },
            ),
            _episode(
                repository_id=repository_id,
                perturbation_id="cmd_wrapper_env_overlay",
                token=cmd_token,
                run_step=cmd_step,
            ),
        ],
    }


def _repo_report(repository_id: str, manifest_path: Path) -> dict:
    return {
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
            }
        },
    }


def _runtime_report(tmp_path: Path, *, cmd_shell_true: bool = False) -> dict:
    repos = []
    for index in range(3):
        repository_id = f"repo_{index}"
        manifest_path = _write(
            tmp_path / "manifests" / f"{repository_id}.json",
            _manifest(repository_id, cmd_shell_true=cmd_shell_true and index == 0),
        )
        repos.append(_repo_report(repository_id, manifest_path))
    return {
        "artifact_family": "phase2cl_runtime_environment_and_shell_perturbation_matrix",
        "passed": True,
        "ready_for_bounded_runtime_environment_shell_perturbation_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "runtime_interpreter": "C:\\Python313\\python.exe",
        "runtime_environment": {"version": "3.13.2"},
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
            "task_completion_success_rate": 1.0,
        },
        "package_metadata": {
            "structured_runtime_cortex_packaged": True,
            "native_head_policy_loaded": False,
            "verification_cortex_loaded": False,
        },
        "repository_reports": repos,
    }


def test_phase2cl_audit_accepts_contract_bounded_runtime_env_shell_matrix(
    tmp_path: Path,
) -> None:
    runtime_report = _write(tmp_path / "runtime.json", _runtime_report(tmp_path))

    report = audit_phase2cl_runtime_environment_shell_matrix(
        runtime_report_json=runtime_report,
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["repositories"] == 3
    assert report["ready_for_general_shell_autonomy_claim"] is False


def test_phase2cl_audit_rejects_shell_true_cmd_wrapper(tmp_path: Path) -> None:
    runtime_report = _write(
        tmp_path / "runtime.json",
        _runtime_report(tmp_path, cmd_shell_true=True),
    )

    report = audit_phase2cl_runtime_environment_shell_matrix(
        runtime_report_json=runtime_report,
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["checks"]["all_cmd_wrappers_are_explicit_argv_not_shell_true"] is False
