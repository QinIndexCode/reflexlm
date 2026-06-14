import json
from pathlib import Path

from reflexlm.cli.audit_phase2ck_cross_runtime_matrix import MAPPING_SCOPE
from reflexlm.cli.audit_phase2cn_runtime_environment_negative_controls import (
    audit_phase2cn_runtime_environment_negative_controls,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_step(repository_id: str, perturbation_id: str, token: str) -> dict:
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
        run_step = _run_step(repository_id, perturbation_id, token)
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
        "generated_by": {"repository_id": repository_id},
        "episodes": episodes,
    }


def _runtime_report(
    *,
    tmp_path: Path,
    runtime: str,
    version: str,
    label: str,
) -> dict:
    repos = []
    for index in range(3):
        repository_id = f"repo_{index}"
        manifest_path = _write(
            tmp_path / label / "manifests" / f"{repository_id}.json",
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


def test_phase2cn_negative_controls_reject_runtime_environment_overclaims(
    tmp_path: Path,
) -> None:
    runtime_reports = [
        _write(
            tmp_path / "py313.json",
            _runtime_report(
                tmp_path=tmp_path,
                runtime="C:\\Python313\\python.exe",
                version="3.13.2",
                label="py313",
            ),
        ),
        _write(
            tmp_path / "venv312.json",
            _runtime_report(
                tmp_path=tmp_path,
                runtime="D:\\repo\\.venv312\\Scripts\\python.exe",
                version="3.12.10",
                label="venv312",
            ),
        ),
        _write(
            tmp_path / "alias312.json",
            _runtime_report(
                tmp_path=tmp_path,
                runtime="D:\\alias\\Scripts\\python.exe",
                version="3.12.10",
                label="alias312",
            ),
        ),
    ]
    package_build = _write(
        tmp_path / "build.json",
        {
            "passed": True,
            "structured_runtime_cortex_python_identity": "C:\\Python313\\python.exe",
        },
    )
    phase2cm = _write(
        tmp_path / "phase2cm.json",
        {
            "passed": True,
            "evidence": {
                "runtime_report_jsons": [str(path) for path in runtime_reports],
                "package_build_report_json": str(package_build),
            },
        },
    )

    report = audit_phase2cn_runtime_environment_negative_controls(
        phase2cm_report_json=phase2cm,
        output_dir=tmp_path / "controls",
        output_report_json=tmp_path / "phase2cn.json",
    )

    assert report["passed"] is True
    assert report["checks"]["positive_control_still_passes"] is True
    assert report["checks"]["all_negative_controls_failed"] is True
    assert report["metrics"]["negative_control_count"] >= 6
    assert all(
        row["expected_failed_checks_observed"]
        for row in report["control_results"]
    )
