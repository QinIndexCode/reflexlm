import json
from pathlib import Path

from reflexlm.cli.audit_phase2ck_cross_runtime_matrix import MAPPING_SCOPE
from reflexlm.cli.audit_phase2cq_cross_runtime_stress_negative_controls import (
    audit_phase2cq_cross_runtime_stress_negative_controls,
)
from reflexlm.cli.run_phase2co_environment_stress_with_failure_recovery import (
    STRESS_IDS,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_steps(repository_id: str, stress_id: str, token: str) -> tuple[dict, dict]:
    if stress_id == "missing_env_then_overlay_recover":
        return (
            {
                "action_type": "RUN_COMMAND",
                "argv": ["<PYTHON>", "-c", "raise SystemExit(31)"],
                "expected_exit_code": 31,
            },
            {
                "action_type": "RUN_COMMAND",
                "argv": ["<PYTHON>", "-c", "print('ok')"],
                "env": {"PHASE2CO_ENV_TOKEN": token},
                "expected_exit_code": 0,
            },
        )
    if stress_id == "wrong_cwd_then_subdir_recover":
        return (
            {
                "action_type": "RUN_COMMAND",
                "argv": ["<PYTHON>", "-c", "raise SystemExit(32)"],
                "expected_exit_code": 32,
            },
            {
                "action_type": "RUN_COMMAND",
                "argv": ["<PYTHON>", "-c", "print('ok')"],
                "cwd": f".reflexlm_runtime_probe/phase2co-{repository_id}-{token}",
                "expected_exit_code": 0,
            },
        )
    return (
        {
            "action_type": "RUN_COMMAND",
            "argv": ["cmd.exe", "/d", "/c", "<PYTHON>", "-c", "raise SystemExit(33)"],
            "expected_exit_code": 33,
        },
        {
            "action_type": "RUN_COMMAND",
            "argv": ["cmd.exe", "/d", "/c", "<PYTHON>", "-c", "print('ok')"],
            "expected_exit_code": 0,
        },
    )


def _manifest(repository_id: str) -> dict:
    episodes = []
    for stress_id in STRESS_IDS:
        token = f"{repository_id}-{stress_id}"
        fail_step, recover_step = _run_steps(repository_id, stress_id, token)
        episodes.append(
            {
                "episode_id": f"{repository_id}-{stress_id}",
                "permissions": [
                    fail_step,
                    {"action_type": "READ_STDERR"},
                    recover_step,
                    {"action_type": "READ_STDOUT"},
                    {"action_type": "DONE"},
                ],
                "completion_requirements": [
                    fail_step,
                    {"action_type": "READ_STDERR"},
                    recover_step,
                    {"action_type": "READ_STDOUT"},
                ],
                "requires_failure": True,
                "generator": {
                    "phase": "phase2co",
                    "stress_id": stress_id,
                    "payload_token": token,
                },
            }
        )
    return {
        "workspace_root": f"D:/external/{repository_id}",
        "generated_by": {"repository_id": repository_id},
        "episodes": episodes,
    }


def _subreport() -> dict:
    return {
        "episode_reports": [
            {
                "requires_failure": True,
                "observed_failure": True,
                "observed_recovery_after_failure": True,
                "recovery_success": True,
                "task_completion_success": True,
                "unexpected_outcomes": 0,
                "selected_done": True,
            }
            for _ in STRESS_IDS
        ]
    }


def _runtime_report(
    tmp_path: Path,
    *,
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
        subreport_path = _write(
            tmp_path / label / "subreports" / f"{repository_id}.json",
            _subreport(),
        )
        repos.append(
            {
                "repository_id": repository_id,
                "generated_manifest_json": str(manifest_path),
                "report_json": str(subreport_path),
                "provenance": {
                    "origin": f"https://example.test/{repository_id}.git",
                    "head": "abc123",
                },
                "stress_ids": list(STRESS_IDS),
                "contract_signatures": [
                    "permissions=[RUN_COMMAND,READ_STDERR,RUN_COMMAND,READ_STDOUT,DONE] completion=[RUN_COMMAND,READ_STDERR,RUN_COMMAND,READ_STDOUT]"
                ],
                "checks": {
                    "all_model_selected_actions_were_allowlisted": True,
                    "all_task_completion_predicates_satisfied": True,
                },
                "metrics": {
                    "episodes": 3,
                    "failure_recovery_gate_applicable": True,
                    "failure_recovery_success_rate": 1.0,
                },
                "failure_recovery_metrics": {
                    "failure_episodes": 3,
                    "observed_failures": 3,
                    "observed_recoveries_after_failure": 3,
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
        "artifact_family": "phase2co_environment_stress_with_failure_recovery",
        "passed": True,
        "seed": 20260608,
        "runtime_interpreter": runtime,
        "runtime_environment": {
            "implementation": "CPython",
            "version": version,
            "executable": runtime,
        },
        "ready_for_bounded_environment_stress_failure_recovery_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
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
            "stress_counts": {stress_id: 3 for stress_id in STRESS_IDS},
            "episodes": 9,
            "executed_actions": 45,
            "task_completion_successes": 9,
            "task_completion_success_rate": 1.0,
            "failure_episodes": 9,
            "observed_failures": 9,
            "observed_recoveries_after_failure": 9,
            "failure_recovery_success_rate": 1.0,
        },
        "package_metadata": {
            "structured_runtime_cortex_packaged": True,
            "native_head_policy_loaded": False,
            "verification_cortex_loaded": False,
        },
        "repository_reports": repos,
    }


def test_phase2cq_negative_controls_reject_cross_runtime_stress_overclaims(
    tmp_path: Path,
) -> None:
    runtime_specs = [
        ("C:\\Python313\\python.exe", "3.13.2", "py313"),
        ("D:\\repo\\.venv312\\Scripts\\python.exe", "3.12.10", "venv312"),
        ("D:\\alias\\Scripts\\python.exe", "3.12.10", "alias312"),
    ]
    runtime_reports = [
        _write(
            tmp_path / f"{label}.json",
            _runtime_report(
                tmp_path,
                runtime=runtime,
                version=version,
                label=label,
            ),
        )
        for runtime, version, label in runtime_specs
    ]
    package_build = _write(
        tmp_path / "build.json",
        {
            "passed": True,
            "structured_runtime_cortex_python_identity": "C:\\Python313\\python.exe",
        },
    )
    phase2cp = _write(
        tmp_path / "phase2cp.json",
        {
            "passed": True,
            "evidence": {
                "runtime_report_jsons": [str(path) for path in runtime_reports],
                "package_build_report_json": str(package_build),
            },
        },
    )

    report = audit_phase2cq_cross_runtime_stress_negative_controls(
        phase2cp_report_json=phase2cp,
        output_dir=tmp_path / "controls",
        output_report_json=tmp_path / "phase2cq.json",
    )

    assert report["passed"] is True
    assert report["checks"]["positive_control_still_passes"] is True
    assert report["checks"]["all_negative_controls_failed"] is True
    assert report["metrics"]["negative_control_count"] >= 8
    assert all(
        row["expected_failed_checks_observed"]
        for row in report["control_results"]
    )
