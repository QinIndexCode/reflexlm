from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.llm.head_dataset import build_phase2c_head_state_prompt_from_state
from reflexlm.schema import (
    FileSystemState,
    GoalSpec,
    ProcessState,
    ProcessStatus,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _commands(task: dict[str, Any]) -> list[str]:
    values = task.get("evaluation_commands")
    commands = [str(item) for item in values] if isinstance(values, list) else []
    rollback = str(task.get("rollback_command") or "").strip()
    if rollback and rollback not in commands:
        commands.append(rollback)
    return commands or ["python -m pytest -q --maxfail=1"]


def _mode_label(task: dict[str, Any], key: str, default: int) -> int:
    policy = task.get("expected_policy")
    if isinstance(policy, dict) and key in policy:
        return int(policy[key])
    return default


def _state_prompt(task: dict[str, Any], commands: list[str]) -> str:
    mode = str(task.get("repair_mode") or "")
    exit_code = 0 if mode == "no_edit_control" else 1
    stdout = (
        "1 passed\n"
        if mode == "no_edit_control"
        else (
            "F\n"
            f"Phase2Y repair pressure mode: {mode}\n"
            "The selected repair policy must use the command allowlist and bounded write scope.\n"
        )
    )
    write_scope = str(task.get("allowed_write_scope") or "")
    state = SystemStateFrame(
        time=TimeState(tick=1, runtime_ms=0),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description=(
                f"Phase2Y {mode} open-repair pressure task. "
                "Use runtime-visible evidence only."
            ),
            command_allowlist=commands,
            watched_paths=[path for path in [write_scope, task.get("materialized_test_target")] if path],
            success_criteria=[
                "selected_tests_pass",
                "bounded_write_scope_respected",
                "open_repair_control_outputs_respected",
            ],
            safety_notes=["do_not_use_oracle_patch_diff", "do_not_use_evaluation_only_feedback"],
        ),
        process=ProcessState(status=ProcessStatus.EXITED, exit_code=exit_code),
        terminal=TerminalState(
            stdout_delta=stdout,
            stderr_delta="",
            stdout_lines=len(stdout.splitlines()),
            stderr_lines=0,
            last_command=commands[0] if commands else "",
        ),
        filesystem=FileSystemState(
            watched_paths=[path for path in [write_scope, task.get("materialized_test_target")] if path],
            changed_paths=[write_scope] if write_scope else [],
            dirty_files=[write_scope] if mode != "no_edit_control" and write_scope else [],
        ),
    )
    return build_phase2c_head_state_prompt_from_state(state)


def _row(task: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    commands = _commands(task)
    mode = str(task.get("repair_mode") or "")
    test_slot = 1 if mode == "multi_test_selection" and len(commands) > 1 else 0
    stop = _mode_label(task, "stop_condition", 1 if mode == "no_edit_control" else 0)
    patch = _mode_label(task, "patch_proposal", 0 if mode == "no_edit_control" else 1)
    rollback = _mode_label(task, "rollback_safety", 1 if mode == "rollback_required" else 0)
    verification = 2 if mode == "no_edit_control" else 0
    progress = {
        "nonliteral_symbolic_patch": 0,
        "multi_test_selection": 1,
        "rollback_required": 2,
        "no_edit_control": 2,
    }.get(mode, 0)
    return {
        "example_id": f"{task.get('task_id')}:{split}:phase2y_control",
        "episode_id": task.get("task_id"),
        "prompt_style": "phase2y_open_repair_runtime_aligned_pressure_control_head_v1",
        "head_scope": "debug_cortex_open_repair_control",
        "task_type": "test_failure_reflex",
        "action_type": "RUN_COMMAND",
        "action_index": 4,
        "internal_target": "ESCALATE_TO_DEBUG_CORTEX",
        "internal_target_index": 1,
        "route_name": "debug_cortex",
        "route_index": 1,
        "command_intent": "test_rerun",
        "command": commands[test_slot],
        "command_slot": test_slot,
        "candidate_commands": commands,
        "candidate_files": [str(task.get("allowed_write_scope") or "")],
        "file_target": None,
        "file_slot": -100,
        "legal_action_mask": {
            "WAIT": 0,
            "READ_STDOUT": 0,
            "READ_STDERR": 0,
            "READ_FILE": 0,
            "RUN_COMMAND": 1,
            "STOP_PROCESS": 0,
            "ASK_USER": 0,
            "REFRESH_STATE": 0,
            "BLOCK": 0,
            "DONE": 1 if stop else 0,
        },
        "nsi_reference": {
            "reflex_action": "RUN_COMMAND",
            "route_name": "debug_cortex",
            "receptor_failure_signal": "phase2y_open_repair_pressure",
            "debug_action_stage": mode,
            "confidence": 0.85,
            "risk": 0.2 if mode != "rollback_required" else 0.5,
            "salience": 0.8,
            "prediction_error": 0.25,
        },
        "confidence_target": 0.85,
        "inhibition_target": 0.0,
        "salience_target": 0.8,
        "risk_target": 0.2 if mode != "rollback_required" else 0.5,
        "prediction_error_target": 0.25,
        "patch_proposal_label": patch,
        "test_selection_slot": test_slot,
        "rollback_safety_label": rollback,
        "stop_condition_label": stop,
        "bounded_edit_scope_label": 1,
        "progress_monitor_label": progress,
        "verification_state_label": verification,
        "open_repair_control_label_scope": "phase2y_materialized_pressure_task",
        "runtime_overrides": [
            "debug_cortex_escalation",
            "phase2y_open_repair_pressure_control",
            "no_json_motor_output",
        ],
        "source_task_manifest": {
            "task_id": task.get("task_id"),
            "task_spec_sha256": task.get("task_spec_sha256"),
            "repo_origin": task.get("repo_origin"),
            "repo_commit": task.get("repo_commit"),
            "sealed_feedback_used": False,
            "repair_mode": mode,
        },
        "state_prompt": _state_prompt(task, commands),
        "t": index,
    }


def _mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        mode = str((row.get("source_task_manifest") or {}).get("repair_mode") or "")
        counts[mode] = counts.get(mode, 0) + 1
    return dict(sorted(counts.items()))


def build_phase2y_open_repair_head_dataset(
    *,
    train_tasks_jsonl: str | Path,
    val_tasks_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
) -> dict[str, Any]:
    train_tasks = _read_jsonl(train_tasks_jsonl)
    val_tasks = _read_jsonl(val_tasks_jsonl)
    train_rows = [_row(task, split="train", index=index) for index, task in enumerate(train_tasks)]
    val_rows = [_row(task, split="val", index=index) for index, task in enumerate(val_tasks)]
    output = Path(output_dir)
    _write_jsonl(output / "train.jsonl", train_rows)
    _write_jsonl(output / "val.jsonl", val_rows)
    manifest = {
        "dataset_family": "phase2y_open_repair_head_dataset",
        "passed": bool(train_rows and val_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "mode_counts": {
            "train": _mode_counts(train_rows),
            "val": _mode_counts(val_rows),
        },
        "full_repair_control_training_ready": True,
        "open_repair_control_label_scope": "phase2y_materialized_pressure_task",
        "effective_split_hashes": {
            "phase2y_open_repair_train": _rows_sha256(train_rows),
            "phase2y_open_repair_val": _rows_sha256(val_rows),
        },
        "claim_boundary": "phase2y_head_training_rows_not_execution_result",
        "outputs": {
            "train_jsonl": str(output / "train.jsonl"),
            "val_jsonl": str(output / "val.jsonl"),
        },
        "inputs": {
            "train_tasks_jsonl": str(Path(train_tasks_jsonl)),
            "val_tasks_jsonl": str(Path(val_tasks_jsonl)),
        },
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2Y open-repair head dataset.")
    parser.add_argument("--train-tasks-jsonl", required=True)
    parser.add_argument("--val-tasks-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    args = parser.parse_args()
    report = build_phase2y_open_repair_head_dataset(
        train_tasks_jsonl=args.train_tasks_jsonl,
        val_tasks_jsonl=args.val_tasks_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
