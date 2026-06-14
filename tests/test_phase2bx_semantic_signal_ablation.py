from reflexlm.cli.run_phase2bx_semantic_signal_ablation import _manifest


def test_phase2bx_manifests_keep_semantic_signal_only_in_initial_receptor_state(
    tmp_path,
) -> None:
    visible = _manifest(workspace_root=tmp_path, repository_id="repo", erased=False)
    erased = _manifest(workspace_root=tmp_path, repository_id="repo", erased=True)

    assert len(visible["episodes"]) == len(erased["episodes"])
    for visible_episode, erased_episode in zip(visible["episodes"], erased["episodes"]):
        assert "expected_sequence" not in visible_episode
        assert "steps" not in visible_episode
        assert visible_episode["semantic_ablation_contract"]["correct_candidate_is_first"] is False
        assert erased_episode["semantic_ablation_contract"]["semantic_signal_erased"] is True
        assert visible_episode["completion_requirements"] == erased_episode["completion_requirements"]
        assert visible_episode["permissions"] == erased_episode["permissions"]
        assert (
            visible_episode["initial_state"]["terminal"]["stderr_delta"]
            != erased_episode["initial_state"]["terminal"]["stderr_delta"]
        )


def test_phase2bx_contains_compositional_recovery_candidates(tmp_path) -> None:
    manifest = _manifest(workspace_root=tmp_path, repository_id="repo", erased=False)
    composition = [
        episode
        for episode in manifest["episodes"]
        if episode["semantic_ablation_contract"]["correct_candidate"]
        in {"dependency_path", "port_permission"}
    ]

    assert len(composition) == 2
    for episode in composition:
        correct_commands = [
            row for row in episode["completion_requirements"]
            if row["action_type"] == "RUN_COMMAND"
        ]
        assert len(correct_commands) == 1
        assert " and " in " ".join(correct_commands[0]["argv"]).lower()
