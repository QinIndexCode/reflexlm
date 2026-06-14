from pathlib import Path

from reflexlm.cli.collect_phase2bk_runtime_world_model_trajectories import _execute_task


def test_execute_task_applies_bounded_env_override(tmp_path: Path) -> None:
    record, evidence = _execute_task(
        {
            "episode_id": "env-override",
            "action_type": "RUN_COMMAND",
            "argv": [
                "<PYTHON>",
                "-c",
                "import os; print(os.environ['PHASE2CL_ENV_TOKEN'])",
            ],
            "env": {"PHASE2CL_ENV_TOKEN": "token-123"},
        },
        workspace_root=tmp_path,
        timeout_seconds=5.0,
    )

    assert record.next_state.terminal.stdout_delta.strip() == "token-123"
    assert evidence["env_override_keys"] == ["PHASE2CL_ENV_TOKEN"]
