import json
from pathlib import Path

from reflexlm.cli.audit_phase2ck_cross_runtime_matrix import MAPPING_SCOPE
import reflexlm.cli.audit_phase2cs_fresh_runtime_execution_repetition_stability as phase2cs
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
    output_dir: Path,
    *,
    runtime: str,
    version: str,
    drift: bool = False,
) -> dict:
    repos = []
    manifest_dir = output_dir / "generated_manifests"
    for index in range(3):
        repository_id = f"repo_{index}"
        manifest_path = _write(
            manifest_dir / f"{repository_id}.json",
            _manifest(repository_id),
        )
        subreport_path = _write(
            output_dir / repository_id / "report.json",
            _subreport(),
        )
        observed_recoveries = 2 if drift and index == 0 else 3
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
                    "failure_recovery_success_rate": observed_recoveries / 3,
                },
                "failure_recovery_metrics": {
                    "failure_episodes": 3,
                    "observed_failures": 3,
                    "observed_recoveries_after_failure": observed_recoveries,
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
    observed_total = 8 if drift else 9
    return {
        "artifact_family": "phase2co_environment_stress_with_failure_recovery",
        "passed": not drift,
        "seed": 20260608,
        "runtime_interpreter": runtime,
        "runtime_environment": {
            "implementation": "CPython",
            "version": version,
            "executable": runtime,
        },
        "ready_for_bounded_environment_stress_failure_recovery_claim": not drift,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "generated_manifest_dir": str(manifest_dir),
        "stress_ids": list(STRESS_IDS),
        "generated_contract_signatures": [
            "permissions=[RUN_COMMAND,READ_STDERR,RUN_COMMAND,READ_STDOUT,DONE] completion=[RUN_COMMAND,READ_STDERR,RUN_COMMAND,READ_STDOUT]"
        ],
        "checks": {
            "all_repository_runtime_suites_passed": not drift,
            "all_repository_actions_were_allowlisted": True,
            "all_repository_task_completion_predicates_satisfied": True,
            "all_repositories_used_package_internal_runtime_cortex": True,
        },
        "metrics": {
            "repositories": 3,
            "stress_counts": {stress_id: 3 for stress_id in STRESS_IDS},
            "episodes": 9,
            "executed_actions": 45,
            "task_completion_successes": observed_total,
            "task_completion_success_rate": observed_total / 9,
            "failure_episodes": 9,
            "observed_failures": 9,
            "observed_recoveries_after_failure": observed_total,
            "failure_recovery_success_rate": observed_total / 9,
        },
        "package_metadata": {
            "structured_runtime_cortex_packaged": True,
            "native_head_policy_loaded": False,
            "verification_cortex_loaded": False,
        },
        "repository_reports": repos,
    }


def _phase2cp_fixture(tmp_path: Path) -> Path:
    runtime_specs = [
        ("C:\\Python313\\python.exe", "3.13.2"),
        ("D:\\repo\\.venv312\\Scripts\\python.exe", "3.12.10"),
        ("D:\\alias\\Scripts\\python.exe", "3.12.10"),
    ]
    build = _write(
        tmp_path / "build.json",
        {
            "passed": True,
            "package_path": str(tmp_path / "package"),
            "structured_runtime_cortex_python_identity": "C:\\Python313\\python.exe",
        },
    )
    return _write(
        tmp_path / "phase2cp.json",
        {
            "passed": True,
            "metrics": {
                "runtime_paths": [runtime for runtime, _ in runtime_specs],
                "python_versions": [version for _, version in runtime_specs],
            },
            "evidence": {
                "runtime_report_jsons": [],
                "package_build_report_json": str(build),
            },
        },
    )


def _fake_runner_factory(*, drift: bool = False):
    calls = {"count": 0}

    def _fake_runner(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        runtime = kwargs["runtime_interpreter"]
        version = "3.13.2" if "Python313" in runtime else "3.12.10"
        should_drift = drift and calls["count"] == 4
        _write(
            Path(kwargs["output_report_json"]),
            _runtime_report(
                output_dir,
                runtime=runtime,
                version=version,
                drift=should_drift,
            ),
        )
        calls["count"] += 1
        return {
            "command": [runtime, "-m", "fake"],
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "",
        }

    return _fake_runner


def test_phase2cs_accepts_fresh_execution_repetition_stability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        phase2cs,
        "_run_phase2co_subprocess",
        _fake_runner_factory(),
    )

    report = phase2cs.audit_phase2cs_fresh_runtime_execution_repetition_stability(
        phase2cp_report_json=_phase2cp_fixture(tmp_path),
        suite_json=_write(tmp_path / "suite.json", {"seed": 20260608}),
        output_dir=tmp_path / "fresh",
        output_report_json=tmp_path / "phase2cs.json",
        repetition_count=2,
        cwd=tmp_path,
    )

    assert report["passed"] is True
    assert report["metrics"]["fresh_runtime_execution_count"] == 6
    assert report["metrics"]["matrix_signature_mismatch_count"] == 0
    assert report["metrics"]["runtime_signature_mismatch_count"] == 0
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_phase2cs_rejects_fresh_execution_metric_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        phase2cs,
        "_run_phase2co_subprocess",
        _fake_runner_factory(drift=True),
    )

    report = phase2cs.audit_phase2cs_fresh_runtime_execution_repetition_stability(
        phase2cp_report_json=_phase2cp_fixture(tmp_path),
        suite_json=_write(tmp_path / "suite.json", {"seed": 20260608}),
        output_dir=tmp_path / "fresh",
        output_report_json=tmp_path / "phase2cs.json",
        repetition_count=2,
        cwd=tmp_path,
    )

    assert report["passed"] is False
    assert report["checks"]["all_repetition_matrix_audits_passed"] is False
    assert report["checks"]["all_runtime_signatures_match_first_repetition"] is False
    assert report["metrics"]["runtime_signature_mismatch_count"] == 1
