import json
from pathlib import Path

import reflexlm.cli.run_phase2cl_runtime_environment_and_shell_perturbation_matrix as phase2cl
from reflexlm.cli.run_phase2cl_runtime_environment_and_shell_perturbation_matrix import (
    PERTURBATION_IDS,
    _generate_manifest_for_repository,
    run_phase2cl_runtime_environment_and_shell_perturbation_matrix,
)


def test_phase2cl_manifest_contains_bounded_env_cwd_and_cmd_perturbations(
    tmp_path: Path,
) -> None:
    manifest = _generate_manifest_for_repository(
        suite_seed=20260608,
        repository={"repository_id": "repo_a", "workspace_root": str(tmp_path)},
    )

    perturbations = {
        episode["generator"]["perturbation_id"] for episode in manifest["episodes"]
    }
    assert perturbations == set(PERTURBATION_IDS)
    for episode in manifest["episodes"]:
        assert "expected_sequence" not in episode
        assert "steps" not in episode
        assert episode["permissions"]
        assert episode["completion_requirements"]
    cmd_episode = next(
        episode
        for episode in manifest["episodes"]
        if episode["generator"]["perturbation_id"] == "cmd_wrapper_env_overlay"
    )
    assert cmd_episode["permissions"][0]["argv"][:3] == ["cmd.exe", "/d", "/c"]
    assert cmd_episode["permissions"][0]["env"]["PHASE2CL_ENV_TOKEN"]


def test_phase2cl_runner_uses_lazy_package_runtime_policy(tmp_path: Path, monkeypatch) -> None:
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
            },
            "metrics": {
                "episodes": len(PERTURBATION_IDS),
                "executed_actions": len(PERTURBATION_IDS) * 3,
                "task_completion_successes": len(PERTURBATION_IDS),
            },
            "policy_configuration": {
                "policy_metadata": {
                    "package_internal_expert": True,
                    "expert_name": "structured_runtime_cortex",
                }
            },
        }

    monkeypatch.setattr(phase2cl, "NativeNervousPolicyPackage", _FakePackage)
    monkeypatch.setattr(phase2cl, "run_phase2bn_model_selected_sealed_runtime", _fake_run)
    monkeypatch.setattr(
        phase2cl,
        "_git_repository_provenance",
        lambda root: {"git_root": str(root), "origin": str(root), "head": "abc"},
    )
    monkeypatch.setattr(
        phase2cl,
        "_repo_identity_checks",
        lambda **_kwargs: {
            "minimum_independent_repository_count_met": True,
            "all_git_roots_are_distinct": True,
            "all_git_roots_differ_from_source_repository": True,
            "all_origins_are_distinct": True,
            "all_heads_are_recorded": True,
        },
    )

    report = run_phase2cl_runtime_environment_and_shell_perturbation_matrix(
        package_path=tmp_path / "package",
        suite_json=suite_path,
        output_dir=tmp_path / "out",
        output_report_json=tmp_path / "report.json",
    )

    assert report["passed"] is True
    assert report["checks"]["all_perturbation_families_present"] is True
    assert report["ready_for_general_shell_autonomy_claim"] is False
