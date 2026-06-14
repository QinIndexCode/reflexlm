from reflexlm.cli.run_phase2by_learned_semantic_affordance import _training_groups
from reflexlm.cli.run_phase2bz_open_vocabulary_transfer import (
    OPEN_VOCABULARY_COMMANDS,
    OPEN_VOCABULARY_RECIPES,
    _manifest,
)


def test_phase2bz_open_vocabulary_concepts_are_absent_from_training_groups() -> None:
    groups = _training_groups()

    assert all(concept not in groups for concept in OPEN_VOCABULARY_COMMANDS)


def test_phase2bz_manifest_has_nonfirst_unseen_candidates_and_erasure_control(
    tmp_path,
) -> None:
    visible = _manifest(workspace_root=tmp_path, repository_id="repo", erased=False)
    erased = _manifest(workspace_root=tmp_path, repository_id="repo", erased=True)

    assert len(visible["episodes"]) == len(OPEN_VOCABULARY_RECIPES)
    for left, right in zip(visible["episodes"], erased["episodes"]):
        assert left["open_vocabulary_contract"]["correct_candidate_is_first"] is False
        assert "expected_sequence" not in left and "steps" not in left
        assert left["permissions"] == right["permissions"]
        assert left["completion_requirements"] == right["completion_requirements"]
        assert (
            left["initial_state"]["terminal"]["stderr_delta"]
            != right["initial_state"]["terminal"]["stderr_delta"]
        )
