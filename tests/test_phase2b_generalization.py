import json
from pathlib import Path

from reflexlm.cli.analyze_phase2b_generalization import _split_index, _sft_row_signature


def _row(
    *,
    episode_id: str,
    task_type: str,
    prompt: str,
    target: str,
) -> dict[str, object]:
    return {
        "example_id": f"{episode_id}-0",
        "episode_id": episode_id,
        "task_type": task_type,
        "user_prompt": prompt,
        "target_text": target,
        "action_type": "DONE",
        "command": None,
        "file_target": None,
    }


def test_sft_signature_catches_prompt_target_overlap() -> None:
    first = _sft_row_signature(
        _row(
            episode_id="a-00001",
            task_type="a",
            prompt="same prompt",
            target='{"action":"DONE"}',
        )
    )
    second = _sft_row_signature(
        _row(
            episode_id="b-00001",
            task_type="b",
            prompt="same prompt",
            target='{"action":"DONE"}',
        )
    )

    assert first["prompt"] == second["prompt"]
    assert first["prompt_target"] == second["prompt_target"]


def test_split_index_detects_hidden_markers() -> None:
    index = _split_index(
        [
            _row(
                episode_id="a-00001",
                task_type="a",
                prompt="visible field recovery_hint should not be present",
                target='{"action":"DONE"}',
            )
        ],
        metadata_by_episode={},
    )

    assert index["hidden_leakage_hits"] == [
        {"example_id": "a-00001-0", "marker": "recovery_hint"}
    ]


def test_split_index_uses_scenario_metadata(tmp_path: Path) -> None:
    metadata = {
        "a-00001": {"task_type": "a", "scenario_template": "s1"},
        "a-00002": {"task_type": "a", "scenario_template": "s2"},
    }
    index = _split_index(
        [
            _row(
                episode_id="a-00001",
                task_type="a",
                prompt="prompt 1",
                target='{"action":"DONE"}',
            ),
            _row(
                episode_id="a-00002",
                task_type="a",
                prompt="prompt 2",
                target='{"action":"WAIT"}',
            ),
        ],
        metadata_by_episode=metadata,
    )

    assert index["scenario_count"] == 2
    assert index["scenarios"] == {"a::s1", "a::s2"}
