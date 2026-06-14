from reflexlm.cli.run_phase2bq_open_task_family_repo_runtime import (
    RECIPE_IDS,
    _contract_signature,
    _generate_manifest_for_repository,
    _payload_token,
)


def test_phase2bq_payload_tokens_are_deterministic_and_repo_specific() -> None:
    first = _payload_token(
        seed=20260608,
        repository_id="repo-a",
        recipe_id="file_terminal_fusion",
        index=0,
    )
    second = _payload_token(
        seed=20260608,
        repository_id="repo-a",
        recipe_id="file_terminal_fusion",
        index=0,
    )
    other_repo = _payload_token(
        seed=20260608,
        repository_id="repo-b",
        recipe_id="file_terminal_fusion",
        index=0,
    )

    assert first == second
    assert first != other_repo


def test_phase2bq_generator_uses_contracts_not_expected_step_sequences() -> None:
    manifest = _generate_manifest_for_repository(
        suite_seed=20260608,
        repository={
            "repository_id": "repo-a",
            "workspace_root": "D:/example/repo-a",
        },
        recipes_per_repository=len(RECIPE_IDS),
        repetitions_per_episode=2,
        timeout_recovery_command_timeout_seconds=0.5,
    )

    assert len(manifest["episodes"]) == len(RECIPE_IDS)
    for episode in manifest["episodes"]:
        assert "steps" not in episode
        assert "expected_sequence" not in episode
        assert episode["permissions"]
        assert episode["completion_requirements"]
        assert _contract_signature(episode)
    timeout_episode = next(
        episode
        for episode in manifest["episodes"]
        if episode["generator"]["recipe_id"] == "timeout_stderr_recovery"
    )
    assert timeout_episode["completion_requirements"][0]["timeout_seconds"] == 0.5
    assert manifest["generated_by"]["timeout_recovery_command_timeout_seconds"] == 0.5
