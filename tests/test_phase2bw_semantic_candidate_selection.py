from reflexlm.cli.run_phase2bw_semantic_candidate_selection import _semantic_manifest


def test_phase2bw_manifest_requires_semantic_nonfirst_recovery_without_sequence(
    tmp_path,
) -> None:
    manifest = _semantic_manifest(
        workspace_root=tmp_path,
        repository_id="test_repo",
    )

    assert len(manifest["episodes"]) == 3
    for episode in manifest["episodes"]:
        contract = episode["semantic_candidate_contract"]
        assert contract["correct_candidate_is_first"] is False
        assert "expected_sequence" not in episode
        assert "steps" not in episode
        assert episode["initial_state"]["process"]["exit_code"] == 1
        assert episode["initial_state"]["terminal"]["stderr_unread"] is True
        assert episode["requires_failure"] is True


def test_phase2bw_completion_requires_only_semantically_correct_command(tmp_path) -> None:
    manifest = _semantic_manifest(
        workspace_root=tmp_path,
        repository_id="test_repo",
    )

    for episode in manifest["episodes"]:
        completion_commands = [
            row for row in episode["completion_requirements"]
            if row["action_type"] == "RUN_COMMAND"
        ]
        permission_commands = [
            row for row in episode["permissions"] if row["action_type"] == "RUN_COMMAND"
        ]
        assert len(completion_commands) == 1
        assert len(permission_commands) == 3
        assert completion_commands[0] != permission_commands[0]


def test_phase2bw_uses_same_candidate_identities_across_failure_kinds(tmp_path) -> None:
    manifest = _semantic_manifest(
        workspace_root=tmp_path,
        repository_id="test_repo",
    )

    candidate_sets = [
        {
            tuple(row["argv"])
            for row in episode["permissions"]
            if row["action_type"] == "RUN_COMMAND"
        }
        for episode in manifest["episodes"]
    ]

    assert all(candidate_set == candidate_sets[0] for candidate_set in candidate_sets[1:])
