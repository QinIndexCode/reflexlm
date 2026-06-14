import json
from pathlib import Path

from reflexlm.cli.audit_phase2co_environment_stress_recovery import (
    audit_phase2co_environment_stress_recovery,
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


def _subreport(*, observed_recovery: bool = True) -> dict:
    return {
        "episode_reports": [
            {
                "requires_failure": True,
                "observed_failure": True,
                "observed_recovery_after_failure": observed_recovery,
                "recovery_success": observed_recovery,
                "task_completion_success": observed_recovery,
                "unexpected_outcomes": 0,
                "selected_done": True,
            }
            for _ in STRESS_IDS
        ]
    }


def _runtime_report(tmp_path: Path, *, observed_recovery: bool = True) -> dict:
    repos = []
    for index in range(3):
        repository_id = f"repo_{index}"
        manifest_path = _write(
            tmp_path / "manifests" / f"{repository_id}.json",
            _manifest(repository_id),
        )
        subreport_path = _write(
            tmp_path / "subreports" / f"{repository_id}.json",
            _subreport(observed_recovery=observed_recovery or index != 0),
        )
        repos.append(
            {
                "repository_id": repository_id,
                "generated_manifest_json": str(manifest_path),
                "report_json": str(subreport_path),
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
                    }
                },
            }
        )
    return {
        "artifact_family": "phase2co_environment_stress_with_failure_recovery",
        "passed": True,
        "ready_for_bounded_environment_stress_failure_recovery_claim": True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "runtime_interpreter": "C:\\Python313\\python.exe",
        "runtime_environment": {"version": "3.13.2"},
        "metrics": {
            "repositories": 3,
            "stress_counts": {stress_id: 3 for stress_id in STRESS_IDS},
            "episodes": 9,
            "executed_actions": 45,
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


def test_phase2co_audit_accepts_episode_level_failure_recovery(tmp_path: Path) -> None:
    runtime_report = _write(tmp_path / "runtime.json", _runtime_report(tmp_path))

    report = audit_phase2co_environment_stress_recovery(
        runtime_report_json=runtime_report,
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["failure_recovery_success_rate"] == 1.0
    assert report["ready_for_general_shell_autonomy_claim"] is False


def test_phase2co_audit_rejects_missing_observed_recovery(tmp_path: Path) -> None:
    runtime_report = _write(
        tmp_path / "runtime.json",
        _runtime_report(tmp_path, observed_recovery=False),
    )

    report = audit_phase2co_environment_stress_recovery(
        runtime_report_json=runtime_report,
        output_report_json=tmp_path / "audit.json",
    )

    assert report["passed"] is False
    assert report["checks"]["all_subreport_episodes_observed_failure_recovery"] is False
