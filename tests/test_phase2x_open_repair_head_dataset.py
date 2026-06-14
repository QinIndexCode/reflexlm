import json
from pathlib import Path

from reflexlm.cli.build_phase2x_open_repair_head_dataset import (
    build_phase2x_open_repair_head_dataset,
    phase2x_task_to_control_episode_rows,
    phase2x_task_to_head_row,
)


def _task(**overrides):
    payload = {
        "task_id": "phase2x:train:00001",
        "split": "train",
        "task_family": "open_ended_repair",
        "repo_origin": "https://github.com/example/project.git",
        "repo_commit": "a" * 40,
        "task_spec_sha256": "b" * 64,
        "problem_statement": "Repair a public repo task without slot hints.",
        "difficulty_axes": ["dependency_or_environment_issue"],
        "requires_patch": True,
        "evaluation_command": "python -m pytest -q tests/test_public_case.py --maxfail=1",
        "rollback_command": "git checkout -- .",
        "allowed_write_scope": "src/package/module.py",
        "baseline_budget": {
            "max_commands": 30,
            "max_wall_clock_seconds": 1800,
        },
        "sealed_feedback_used": False,
    }
    payload.update(overrides)
    return payload


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    return path


def _trace(trace_id: str = "trace-1") -> dict:
    return {
        "trace_id": trace_id,
        "runtime_visible_evidence": {
            "pytest_before_patch": {
                "exit_code": 1,
                "stdout_excerpt": "F\n>       assert 2 == 1\nE       assert 2 == 1\n",
                "stderr_excerpt": "",
            },
            "changed_files": ["src/package/module.py"],
            "watched_files": ["tests/test_public_case.py"],
        },
    }


def test_phase2x_task_to_head_row_adds_open_repair_control_labels() -> None:
    row = phase2x_task_to_head_row(_task())

    assert row["prompt_style"] == "phase2x_open_repair_initial_control_head_v1"
    assert row["patch_proposal_label"] == 1
    assert row["test_selection_slot"] == 0
    assert row["rollback_safety_label"] == 1
    assert row["bounded_edit_scope_label"] == 1
    assert row["stop_condition_label"] == 0
    assert row["progress_monitor_label"] == 0
    assert row["verification_state_label"] == 0
    assert "candidate_0" not in row["state_prompt"].lower()
    assert row["open_repair_control_label_scope"] == "initial_state_only"


def test_phase2x_head_dataset_manifest_marks_initial_control_boundary(tmp_path: Path) -> None:
    train = _write_jsonl(tmp_path / "train.tasks.jsonl", [_task()])
    val = _write_jsonl(tmp_path / "val.tasks.jsonl", [_task(task_id="phase2x:val:00001", split="val")])

    manifest = build_phase2x_open_repair_head_dataset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=val,
        output_dir=tmp_path / "head",
    )

    assert manifest["dataset_family"] == "phase2x_open_repair_head_dataset"
    assert manifest["claim_boundary"] == "initial_control_head_rows_not_full_repair_execution"
    assert manifest["full_repair_control_training_ready"] is False
    assert (tmp_path / "head" / "train.jsonl").exists()
    assert manifest["splits"]["train"]["rows"] == 1


def test_phase2x_task_to_scripted_control_episode_rows_have_label_diversity() -> None:
    rows = phase2x_task_to_control_episode_rows(_task())

    assert [row["control_stage"] for row in rows] == [
        "pre_patch",
        "post_patch_pre_test",
        "post_test_pass",
        "post_test_fail",
    ]
    assert sorted({row["patch_proposal_label"] for row in rows}) == [0, 1]
    assert sorted({row["rollback_safety_label"] for row in rows}) == [0, 1]
    assert sorted({row["stop_condition_label"] for row in rows}) == [0, 1]
    assert sorted({row["progress_monitor_label"] for row in rows}) == [0, 1, 2]
    assert sorted({row["verification_state_label"] for row in rows}) == [0, 1, 2]
    assert {row["open_repair_control_label_scope"] for row in rows} == {
        "scripted_full_control_episode"
    }


def test_phase2x_head_dataset_scripted_full_control_is_training_ready_but_not_result_evidence(tmp_path: Path) -> None:
    train = _write_jsonl(tmp_path / "train.tasks.jsonl", [_task()])
    val = _write_jsonl(tmp_path / "val.tasks.jsonl", [_task(task_id="phase2x:val:00001", split="val")])

    manifest = build_phase2x_open_repair_head_dataset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=val,
        output_dir=tmp_path / "head",
        episode_control_mode="scripted_full_control",
    )

    assert manifest["full_repair_control_training_ready"] is True
    assert manifest["claim_boundary"] == "scripted_full_control_training_not_real_repair_execution"
    assert manifest["splits"]["train"]["rows"] == 4
    assert manifest["splits"]["train"]["label_diversity"]["progress_monitor_label"] == [0, 1, 2]


def test_phase2x_head_dataset_runtime_aligned_uses_receptor_state_prompt(tmp_path: Path) -> None:
    train_task = _task(source_trace_id="trace-1")
    val_task = _task(task_id="phase2x:val:00001", split="val", source_trace_id="trace-1")
    train = _write_jsonl(tmp_path / "train.tasks.jsonl", [train_task])
    val = _write_jsonl(tmp_path / "val.tasks.jsonl", [val_task])
    traces = _write_jsonl(tmp_path / "traces.jsonl", [_trace()])

    manifest = build_phase2x_open_repair_head_dataset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=val,
        output_dir=tmp_path / "head",
        episode_control_mode="runtime_aligned_scripted_control",
        source_traces_jsonl=traces,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "head" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert manifest["runtime_aligned_state_prompt"] is True
    assert manifest["claim_boundary"] == "runtime_aligned_scripted_control_training_not_real_repair_execution"
    assert rows[0]["prompt_style"] == "phase2x_open_repair_runtime_aligned_control_head_v1"
    assert "Visible transition summary:" in rows[0]["state_prompt"]
    assert "assert 2 == 1" in rows[0]["state_prompt"]
    assert rows[0]["patch_proposal_label"] == 1
