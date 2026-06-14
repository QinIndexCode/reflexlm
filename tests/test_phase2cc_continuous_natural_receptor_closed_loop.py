from pathlib import Path

from reflexlm.cli.run_phase2cc_continuous_natural_receptor_closed_loop import (
    _run_suite,
    _transition_rows,
)
from reflexlm.cli.run_phase2cb_natural_failure_no_lexical_overlap_transfer import (
    NATURAL_INTENT_COMMANDS,
)
from reflexlm.models.semantic_matcher import _command_semantic_text


def _row(task_id: str, operation: str, failure: str) -> dict:
    return {
        "task_id": task_id,
        "source_kind": "public_repo",
        "repo_origin": "https://example.test/repo.git",
        "repo_commit": "abc123",
        "learned_patch_descriptor_target": {"operation": operation},
        "runtime_visible_contract": {"no_gold_hint": True},
        "runtime_visible_evidence": {
            "changed_files": [f"src/{task_id}.py"],
            "pytest_before_patch": {"stdout_excerpt": f"E   {failure}\n"},
        },
    }


class _KeywordMatcher:
    def score_texts(self, observation: str, commands: list[str]) -> list[float]:
        if "NameError" in observation:
            target = NATURAL_INTENT_COMMANDS["insert_import"]
        elif "AssertionError" in observation:
            target = NATURAL_INTENT_COMMANDS["replace_literal"]
        elif "AttributeError" in observation:
            target = NATURAL_INTENT_COMMANDS["replace_attribute"]
        else:
            target = "withheld"
        return [
            1.0 if _command_semantic_text(command) == target else 0.0
            for command in commands
        ]


def test_phase2cc_transition_rows_pair_different_operations() -> None:
    rows = [
        _row("attr", "replace_attribute", "AttributeError: missing method"),
        _row("import", "insert_import", "NameError: name 'os' is not defined"),
        _row("literal", "replace_literal", "AssertionError: assert 89 == 88"),
    ]

    transitions = _transition_rows(rows)

    assert len(transitions) == 3
    assert all(
        left["learned_patch_descriptor_target"]["operation"]
        != right["learned_patch_descriptor_target"]["operation"]
        for left, right in transitions
    )


def test_phase2cc_visible_closed_loop_uses_second_receptor(tmp_path: Path) -> None:
    rows = [
        _row("attr", "replace_attribute", "AttributeError: missing method"),
        _row("import", "insert_import", "NameError: name 'os' is not defined"),
        _row("literal", "replace_literal", "AssertionError: assert 89 == 88"),
    ]

    report = _run_suite(
        transitions=[(rows[0], rows[1])],
        all_rows=rows,
        suite_id="visible",
        matcher=_KeywordMatcher(),
        workspace_root=tmp_path,
        output_jsonl=tmp_path / "visible.jsonl",
    )
    frozen = _run_suite(
        transitions=[(rows[0], rows[1])],
        all_rows=rows,
        suite_id="frozen_first_receptor",
        matcher=_KeywordMatcher(),
        workspace_root=tmp_path,
        output_jsonl=tmp_path / "frozen.jsonl",
    )

    assert report["stage1_accuracy"] == 1.0
    assert report["stage2_accuracy"] == 1.0
    assert report["task_completion_rate"] == 1.0
    assert report["action_switch_rate"] == 1.0
    assert report["all_correct_candidates_are_nonfirst"] is True
    assert report["runtime_transitions"] == 6
    assert frozen["stage2_accuracy"] == 0.0
