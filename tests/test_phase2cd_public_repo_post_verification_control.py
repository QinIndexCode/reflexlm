from pathlib import Path

from reflexlm.cli.run_phase2cd_public_repo_post_verification_control import (
    CONTINUE_INTENT,
    FINISH_INTENT,
    _run_suite,
    _split_repo_disjoint,
)
from reflexlm.models.semantic_matcher import _command_semantic_text


class _VerificationMatcher:
    def score_state(self, state):
        text = " ".join(state.runtime_evidence.terminal_observations)
        target = FINISH_INTENT if "2 passed" in text else CONTINUE_INTENT
        return [
            1.0 if _command_semantic_text(command) == target else 0.0
            for command in state.goal.command_allowlist
        ]


def _row(repo: str, task_id: str) -> dict:
    return {
        "repo_origin": repo,
        "task": {"task_id": task_id},
        "execution": {
            "selection_policy": "package_loaded_native_head",
            "success": True,
        },
        "pre": {
            "exit_code": 1,
            "stdout": "FAILED test_example.py::test_case\nAssertionError",
            "stderr": "",
            "target": "test_example.py",
        },
        "post": {
            "exit_code": 0,
            "stdout": "2 passed in 0.02s",
            "stderr": "",
            "target": "test_example.py",
        },
    }


def test_phase2cd_split_is_repo_disjoint() -> None:
    rows = [_row(f"https://example.test/repo{i}.git", f"task{i}") for i in range(4)]

    train, holdout = _split_repo_disjoint(rows)

    assert train
    assert holdout
    assert {row["repo_origin"] for row in train}.isdisjoint(
        {row["repo_origin"] for row in holdout}
    )


def test_phase2cd_visible_post_feedback_switches_to_finish(tmp_path: Path) -> None:
    rows = [_row("https://example.test/repo.git", "task0")]

    visible = _run_suite(
        rows=rows,
        suite_id="visible",
        matcher=_VerificationMatcher(),
        workspace_root=tmp_path,
    )
    frozen = _run_suite(
        rows=rows,
        suite_id="frozen_pre",
        matcher=_VerificationMatcher(),
        workspace_root=tmp_path,
    )

    assert visible["pre_continue_accuracy"] == 1.0
    assert visible["post_finish_accuracy"] == 1.0
    assert visible["control_action_change_rate"] == 1.0
    assert visible["row_reports"][0]["selected_pre_intent"] == CONTINUE_INTENT
    assert visible["row_reports"][0]["selected_post_intent"] == FINISH_INTENT
    assert frozen["post_finish_accuracy"] == 0.0
