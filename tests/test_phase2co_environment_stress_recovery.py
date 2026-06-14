import json
from pathlib import Path

import reflexlm.cli.run_phase2co_environment_stress_with_failure_recovery as phase2co
from reflexlm.cli.run_phase2co_environment_stress_with_failure_recovery import (
    STRESS_IDS,
    _generate_manifest_for_repository,
    run_phase2co_environment_stress_with_failure_recovery,
)


def test_phase2co_manifest_contains_bounded_failure_recovery_stressors(
    tmp_path: Path,
) -> None:
    manifest = _generate_manifest_for_repository(
        suite_seed=20260608,
        repository={"repository_id": "repo_a", "workspace_root": str(tmp_path)},
    )

    stressors = {episode["generator"]["stress_id"] for episode in manifest["episodes"]}
    assert stressors == set(STRESS_IDS)
    for episode in manifest["episodes"]:
        assert "expected_sequence" not in episode
        assert "steps" not in episode
        assert episode["requires_failure"] is True
        assert episode["permissions"]
        assert episode["completion_requirements"]
        run_steps = [
            step
            for step in episode["completion_requirements"]
            if step["action_type"] == "RUN_COMMAND"
        ]
        assert len(run_steps) == 2
        assert run_steps[0]["expected_exit_code"] != 0
        assert run_steps[1]["expected_exit_code"] == 0
    cmd_episode = next(
        episode
        for episode in manifest["episodes"]
        if episode["generator"]["stress_id"] == "cmd_wrapper_failure_then_recover"
    )
    assert cmd_episode["permissions"][0]["argv"][:3] == ["cmd.exe", "/d", "/c"]


def test_phase2co_runner_uses_lazy_package_runtime_and_requires_failure_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite = {
        "source_repository_root": str(tmp_path),
        "seed": 20260608,
        "minimum_repository_count": 1,
        "repositories": [
            {"repository_id": "repo_a", "workspace_root": str(tmp_path)},
        ],
    }
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")

    class _FakePackage:
        def __init__(self, *_args, **kwargs) -> None:
            assert kwargs["load_native_head_policy"] is False
            assert kwargs["load_verification_cortex"] is False

        def create_structured_runtime_policy(self):
            return object()

        def metadata(self):
            return {
                "structured_runtime_cortex_packaged": True,
                "native_head_policy_loaded": False,
                "verification_cortex_loaded": False,
            }

    def _fake_run(**_kwargs):
        return {
            "passed": True,
            "checks": {
                "all_model_selected_actions_were_allowlisted": True,
                "all_task_completion_predicates_satisfied": True,
                "failure_recovery_success_rate_meets_gate": True,
            },
            "metrics": {
                "episodes": len(STRESS_IDS),
                "executed_actions": len(STRESS_IDS) * 5,
                "task_completion_successes": len(STRESS_IDS),
                "failure_recovery_gate_applicable": True,
            },
            "episode_reports": [
                {
                    "requires_failure": True,
                    "observed_failure": True,
                    "observed_recovery_after_failure": True,
                    "recovery_success": True,
                }
                for _ in STRESS_IDS
            ],
            "policy_configuration": {
                "policy_metadata": {
                    "package_internal_expert": True,
                    "expert_name": "structured_runtime_cortex",
                }
            },
        }

    monkeypatch.setattr(phase2co, "NativeNervousPolicyPackage", _FakePackage)
    monkeypatch.setattr(phase2co, "run_phase2bn_model_selected_sealed_runtime", _fake_run)
    monkeypatch.setattr(
        phase2co,
        "_git_repository_provenance",
        lambda root: {"git_root": str(root), "origin": str(root), "head": "abc"},
    )
    monkeypatch.setattr(
        phase2co,
        "_repo_identity_checks",
        lambda **_kwargs: {
            "minimum_independent_repository_count_met": True,
            "all_git_roots_are_distinct": True,
            "all_git_roots_differ_from_source_repository": True,
            "all_origins_are_distinct": True,
            "all_heads_are_recorded": True,
        },
    )

    report = run_phase2co_environment_stress_with_failure_recovery(
        package_path=tmp_path / "package",
        suite_json=suite_path,
        output_dir=tmp_path / "out",
        output_report_json=tmp_path / "report.json",
    )

    assert report["passed"] is True
    assert report["checks"]["all_stress_families_present"] is True
    assert report["checks"]["all_episodes_required_failure"] is True
    assert report["checks"]["all_failure_episodes_observed_recovery_after_failure"] is True
    assert report["ready_for_general_shell_autonomy_claim"] is False
