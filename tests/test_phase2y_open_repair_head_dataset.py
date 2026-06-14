import json
from pathlib import Path

from reflexlm.cli.build_phase2y_open_repair_head_dataset import (
    build_phase2y_open_repair_head_dataset,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _task(index: int, mode: str) -> dict:
    commands = [f"python -m pytest -q tests/test_{index}.py --maxfail=1"]
    if mode == "multi_test_selection":
        commands.append("python -m pytest -q tests --maxfail=2")
    return {
        "task_id": f"phase2y:train:{index:05d}",
        "repo_origin": "https://github.com/example/repo.git",
        "repo_commit": "a" * 40,
        "task_spec_sha256": "b" * 64,
        "repair_mode": mode,
        "requires_patch": mode != "no_edit_control",
        "patch_type": "nonliteral_symbolic" if mode == "nonliteral_symbolic_patch" else "bounded_runtime_patch",
        "evaluation_commands": commands,
        "rollback_command": "git checkout -- .",
        "allowed_write_scope": "src/module.py",
        "difficulty_axes": [mode],
        "expected_policy": {
            "patch_proposal": 0 if mode == "no_edit_control" else 1,
            "bounded_edit_scope": 1,
            "rollback_safety": 1 if mode == "rollback_required" else 0,
            "stop_condition": 1 if mode == "no_edit_control" else 0,
        },
        "sealed_feedback_used": False,
    }


def _tasks() -> list[dict]:
    modes = [
        "nonliteral_symbolic_patch",
        "multi_test_selection",
        "rollback_required",
        "no_edit_control",
    ]
    return [_task(index, modes[index % len(modes)]) for index in range(16)]


def test_phase2y_head_dataset_preserves_control_labels(tmp_path: Path) -> None:
    train = _write_jsonl(tmp_path / "train.tasks.jsonl", _tasks())
    val = _write_jsonl(tmp_path / "val.tasks.jsonl", _tasks())

    manifest = build_phase2y_open_repair_head_dataset(
        train_tasks_jsonl=train,
        val_tasks_jsonl=val,
        output_dir=tmp_path / "head",
        manifest_json=tmp_path / "head_manifest.json",
    )

    assert manifest["passed"] is True
    assert manifest["full_repair_control_training_ready"] is True
    rows = [
        json.loads(line)
        for line in (tmp_path / "head" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    no_edit = next(row for row in rows if row["source_task_manifest"]["repair_mode"] == "no_edit_control")
    multi = next(row for row in rows if row["source_task_manifest"]["repair_mode"] == "multi_test_selection")
    rollback = next(row for row in rows if row["source_task_manifest"]["repair_mode"] == "rollback_required")
    nonliteral = next(row for row in rows if row["source_task_manifest"]["repair_mode"] == "nonliteral_symbolic_patch")
    assert no_edit["patch_proposal_label"] == 0
    assert no_edit["stop_condition_label"] == 1
    assert multi["test_selection_slot"] == 1
    assert rollback["rollback_safety_label"] == 1
    assert nonliteral["patch_proposal_label"] == 1
    assert "sealed" not in nonliteral["state_prompt"].lower()
