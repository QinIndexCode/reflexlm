from reflexlm.cli.run_phase2ca_no_lexical_overlap_transfer import (
    NO_LEXICAL_OVERLAP_COMMANDS,
    NO_LEXICAL_OVERLAP_RECIPES,
    _lexical_overlap,
    _manifest,
)


def test_phase2ca_correct_pairs_have_zero_normalized_lexical_overlap() -> None:
    for recipe in NO_LEXICAL_OVERLAP_RECIPES.values():
        correct_command = NO_LEXICAL_OVERLAP_COMMANDS[str(recipe["correct"])]
        assert _lexical_overlap(str(recipe["failure"]), correct_command) == []


def test_phase2ca_manifest_preserves_erasure_control_and_bounded_actions(tmp_path) -> None:
    visible = _manifest(workspace_root=tmp_path, repository_id="repo", erased=False)
    erased = _manifest(workspace_root=tmp_path, repository_id="repo", erased=True)

    assert len(visible["episodes"]) == len(NO_LEXICAL_OVERLAP_RECIPES)
    for left, right in zip(visible["episodes"], erased["episodes"]):
        contract = left["no_lexical_overlap_contract"]
        assert contract["correct_candidate_is_first"] is False
        assert contract["failure_correct_command_overlap"] == []
        assert "expected_sequence" not in left and "steps" not in left
        assert left["permissions"] == right["permissions"]
        assert left["completion_requirements"] == right["completion_requirements"]
        assert (
            left["initial_state"]["terminal"]["stderr_delta"]
            != right["initial_state"]["terminal"]["stderr_delta"]
        )
