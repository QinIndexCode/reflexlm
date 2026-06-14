import json
from pathlib import Path

import pytest

from reflexlm.cli.collect_phase2bk_runtime_world_model_trajectories import (
    collect_phase2bk_runtime_world_model_trajectories,
)
from reflexlm.data.jsonl import read_jsonl
from reflexlm.schema import ActionType, SourceType


def test_phase2bk_collector_records_real_workspace_confined_transitions(
    tmp_path: Path,
) -> None:
    watched = tmp_path / "watched.txt"
    watched.write_text("runtime observation", encoding="utf-8")
    manifest = {
        "workspace_root": str(tmp_path),
        "tasks": [
            {
                "episode_id": "run-command",
                "action_type": "RUN_COMMAND",
                "argv": ["<PYTHON>", "-c", "print('observed subprocess')"],
                "watched_paths": ["watched.txt"],
            },
            {
                "episode_id": "read-file",
                "action_type": "READ_FILE",
                "file_target": "watched.txt",
                "watched_paths": ["watched.txt"],
            },
            {
                "episode_id": "wait",
                "action_type": "WAIT",
                "wait_ms": 1,
                "watched_paths": ["watched.txt"],
            },
            {
                "episode_id": "refresh",
                "action_type": "REFRESH_STATE",
                "watched_paths": ["watched.txt"],
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = collect_phase2bk_runtime_world_model_trajectories(
        manifest_json=manifest_path,
        output_jsonl=tmp_path / "trajectories.jsonl",
        output_report_json=tmp_path / "report.json",
    )
    records = read_jsonl(tmp_path / "trajectories.jsonl")

    assert report["passed"] is True
    assert report["rows"] == 4
    assert report["shell_false_rows"] == 4
    assert {record.action.type for record in records if record.action} == {
        ActionType.RUN_COMMAND,
        ActionType.READ_FILE,
        ActionType.WAIT,
        ActionType.REFRESH_STATE,
    }
    assert all(record.source == SourceType.RUNTIME_OBSERVATION for record in records)
    assert "observed subprocess" in records[0].next_state.terminal.stdout_delta


def test_phase2bk_collector_repeats_real_execution_as_distinct_episodes(
    tmp_path: Path,
) -> None:
    watched = tmp_path / "watched.txt"
    watched.write_text("runtime observation", encoding="utf-8")
    manifest = {
        "workspace_root": str(tmp_path),
        "repetitions_per_task": 3,
        "tasks": [
            {
                "episode_id": "refresh",
                "action_type": "REFRESH_STATE",
                "watched_paths": ["watched.txt"],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = collect_phase2bk_runtime_world_model_trajectories(
        manifest_json=manifest_path,
        output_jsonl=tmp_path / "trajectories.jsonl",
        output_report_json=tmp_path / "report.json",
    )
    records = read_jsonl(tmp_path / "trajectories.jsonl")

    assert report["rows"] == 3
    assert report["manifest_tasks"] == 1
    assert report["repetitions_per_task"] == 3
    assert len({record.episode_id for record in records}) == 3
    assert [row["repetition"] for row in report["evidence_rows"]] == [0, 1, 2]


def test_phase2bk_collector_records_continuous_failure_feedback_episode(
    tmp_path: Path,
) -> None:
    watched = tmp_path / "watched.txt"
    watched.write_text("runtime observation", encoding="utf-8")
    manifest = {
        "workspace_root": str(tmp_path),
        "episodes": [
            {
                "episode_id": "continuous-failure",
                "description": "observe failing command then read stderr feedback",
                "watched_paths": ["watched.txt"],
                "steps": [
                    {
                        "action_type": "RUN_COMMAND",
                        "argv": [
                            "<PYTHON>",
                            "-c",
                            "import sys; sys.stderr.write('observed failure\\n'); raise SystemExit(7)",
                        ],
                        "expected_exit_code": 7,
                        "reward": 1.0,
                    },
                    {
                        "action_type": "READ_STDERR",
                        "reward": 1.0,
                    },
                    {
                        "action_type": "DONE",
                        "reward": 1.0,
                    },
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = collect_phase2bk_runtime_world_model_trajectories(
        manifest_json=manifest_path,
        output_jsonl=tmp_path / "trajectories.jsonl",
        output_report_json=tmp_path / "report.json",
    )
    records = read_jsonl(tmp_path / "trajectories.jsonl")

    assert report["rows"] == 3
    assert report["continuous_episode_rows"] == 2
    assert report["nonzero_exit_rows"] == 1
    assert report["expected_outcome_matched_rows"] == 3
    assert [record.t for record in records] == [0, 1, 2]
    assert records[0].done is False
    assert records[-1].done is True
    assert records[1].state == records[0].next_state
    assert "observed failure" in records[1].state.terminal.stderr_delta
    assert records[1].action is not None
    assert records[1].action.type == ActionType.READ_STDERR
    assert (
        "observed failure"
        in records[1].next_state.runtime_evidence.terminal_observations[0]
    )


def test_phase2bk_collector_preserves_pending_file_change_until_read(
    tmp_path: Path,
) -> None:
    watched = tmp_path / "watched.txt"
    watched.write_text("before", encoding="utf-8")
    manifest = {
        "workspace_root": str(tmp_path),
        "episodes": [
            {
                "episode_id": "file-change-consumption",
                "description": "retain a detected file change until the file is read",
                "task_type": "external_file_change_reflex",
                "watched_paths": ["watched.txt"],
                "steps": [
                    {
                        "action_type": "RUN_COMMAND",
                        "argv": [
                            "<PYTHON>",
                            "-c",
                            "from pathlib import Path; Path('watched.txt').write_text('after', encoding='utf-8')",
                        ],
                    },
                    {"action_type": "REFRESH_STATE"},
                    {"action_type": "READ_FILE", "file_target": "watched.txt"},
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    collect_phase2bk_runtime_world_model_trajectories(
        manifest_json=manifest_path,
        output_jsonl=tmp_path / "trajectories.jsonl",
        output_report_json=tmp_path / "report.json",
    )
    records = read_jsonl(tmp_path / "trajectories.jsonl")

    assert records[0].next_state.filesystem.dirty_files == ["watched.txt"]
    assert records[1].next_state.filesystem.dirty_files == ["watched.txt"]
    assert records[2].next_state.filesystem.dirty_files == []


def test_phase2bk_collector_preserves_unread_terminal_channels_until_consumed(
    tmp_path: Path,
) -> None:
    manifest = {
        "workspace_root": str(tmp_path),
        "episodes": [
            {
                "episode_id": "terminal-channel-consumption",
                "description": "retain unread stdout and stderr across refresh",
                "task_type": "common_error_recovery_routine",
                "steps": [
                    {
                        "action_type": "RUN_COMMAND",
                        "argv": [
                            "<PYTHON>",
                            "-c",
                            "import sys; sys.stderr.write('diagnostic'); print('output')",
                        ],
                    },
                    {"action_type": "REFRESH_STATE"},
                    {"action_type": "READ_STDERR"},
                    {"action_type": "READ_STDOUT"},
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = collect_phase2bk_runtime_world_model_trajectories(
        manifest_json=manifest_path,
        output_jsonl=tmp_path / "trajectories.jsonl",
        output_report_json=tmp_path / "report.json",
    )
    records = read_jsonl(tmp_path / "trajectories.jsonl")

    assert report["expected_outcome_matched_rows"] == 4
    assert records[0].next_state.terminal.stdout_unread is True
    assert records[0].next_state.terminal.stderr_unread is True
    assert records[1].next_state.terminal.stdout_unread is True
    assert records[1].next_state.terminal.stderr_unread is True
    assert records[2].next_state.terminal.stdout_unread is True
    assert records[2].next_state.terminal.stderr_unread is False
    assert records[3].next_state.terminal.stdout_unread is False


def test_phase2bk_collector_records_expected_timeout_episode(tmp_path: Path) -> None:
    manifest = {
        "workspace_root": str(tmp_path),
        "episodes": [
            {
                "episode_id": "timeout",
                "steps": [
                    {
                        "action_type": "RUN_COMMAND",
                        "argv": ["<PYTHON>", "-c", "import time; time.sleep(2)"],
                        "timeout_seconds": 0.1,
                        "expected_exit_code": 124,
                        "expected_timed_out": True,
                        "reward": 1.0,
                    },
                    {
                        "action_type": "DONE",
                        "reward": 1.0,
                    },
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = collect_phase2bk_runtime_world_model_trajectories(
        manifest_json=manifest_path,
        output_jsonl=tmp_path / "trajectories.jsonl",
        output_report_json=tmp_path / "report.json",
        timeout_seconds=1.0,
    )
    records = read_jsonl(tmp_path / "trajectories.jsonl")

    assert report["timed_out_rows"] == 1
    assert report["expected_outcome_matched_rows"] == 2
    assert records[0].next_state.process.exit_code == 124
    assert records[0].next_state.process.interrupted is True


def test_phase2bk_collector_rejects_path_escape(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "workspace_root": str(tmp_path),
                "tasks": [
                    {
                        "episode_id": "escape",
                        "action_type": "READ_FILE",
                        "file_target": "../outside.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes workspace root"):
        collect_phase2bk_runtime_world_model_trajectories(
            manifest_json=manifest_path,
            output_jsonl=tmp_path / "trajectories.jsonl",
            output_report_json=tmp_path / "report.json",
        )
