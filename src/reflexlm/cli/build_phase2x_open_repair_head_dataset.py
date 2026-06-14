from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from reflexlm.llm.candidate_features import command_intent_for_text
from reflexlm.llm.head_dataset import build_phase2c_head_state_prompt_from_state
from reflexlm.models.features import ACTION_ORDER, ROUTE_ORDER
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER
from reflexlm.schema import (
    ActionType,
    FileSystemState,
    GoalSpec,
    InternalTarget,
    ProcessState,
    ProcessStatus,
    RouteName,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(value: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(value, list):
        for row in value:
            digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()
    digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _candidate_commands(task: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    evaluation = str(task.get("evaluation_command") or "").strip()
    rollback = str(task.get("rollback_command") or "").strip()
    if evaluation:
        commands.append(evaluation)
    if rollback and rollback not in commands:
        commands.append(rollback)
    return commands


def _state_prompt(task: dict[str, Any], commands: list[str], *, control_stage: str = "initial") -> str:
    axes = task.get("difficulty_axes") if isinstance(task.get("difficulty_axes"), list) else []
    budget = task.get("baseline_budget") if isinstance(task.get("baseline_budget"), dict) else {}
    lines = [
        "Phase2X open-repair native-head state input.",
        "This row trains explicit repair-control heads, not free-form JSON or shell autonomy.",
        "Use only public task manifest fields and runtime-visible task constraints.",
        "",
        "Task:",
        str(task.get("problem_statement") or ""),
        "",
        "Repository:",
        f"origin={task.get('repo_origin')}",
        f"commit={task.get('repo_commit')}",
        "",
        "Repair constraints:",
        f"control_stage={control_stage}",
        f"requires_patch={bool(task.get('requires_patch'))}",
        f"allowed_write_scope={task.get('allowed_write_scope')}",
        f"evaluation_command={task.get('evaluation_command')}",
        f"rollback_command={task.get('rollback_command')}",
        f"difficulty_axes={','.join(str(axis) for axis in axes)}",
        f"baseline_max_commands={budget.get('max_commands')}",
        f"baseline_max_wall_clock_seconds={budget.get('max_wall_clock_seconds')}",
        "",
        "Candidate commands:",
    ]
    lines.extend(f"- {command}" for command in commands)
    lines.extend(
        [
            "",
            "Open-repair head semantics:",
            "- patch_proposal predicts whether a bounded patch proposal is required.",
            "- test_selection chooses the verification command slot.",
            "- rollback_safety predicts whether rollback support is required before editing.",
            "- progress_monitor, verification_state, and stop_condition describe the current control state.",
            "- Scripted control rows train state transitions only; they do not prove real repair execution without later row-level result artifacts.",
        ]
    )
    return "\n".join(lines)


def _runtime_state_for_trace(
    task: dict[str, Any],
    trace: dict[str, Any] | None,
    *,
    control_stage: str,
) -> SystemStateFrame | None:
    if not trace:
        return None
    evidence = trace.get("runtime_visible_evidence")
    if not isinstance(evidence, dict):
        return None
    before = evidence.get("pytest_before_patch") if isinstance(evidence.get("pytest_before_patch"), dict) else {}
    changed_files = [str(item) for item in evidence.get("changed_files", []) if item]
    watched_files = [str(item) for item in evidence.get("watched_files", []) if item]
    evaluation = str(task.get("evaluation_command") or "").strip()
    if control_stage == "post_test_pass":
        stdout = "1 passed\n"
        exit_code = 0
        last_command = evaluation
    elif control_stage in {"pre_patch", "post_test_fail"}:
        stdout = str(before.get("stdout_excerpt") or "")
        exit_code = int(before.get("exit_code") if before.get("exit_code") is not None else 1)
        last_command = evaluation
    else:
        stdout = (
            str(before.get("stdout_excerpt") or "")
            + "\nsource inspected; bounded patch prepared; rerun the targeted failing test."
        )
        exit_code = int(before.get("exit_code") if before.get("exit_code") is not None else 1)
        last_command = ""
    return SystemStateFrame(
        time=TimeState(tick=0, runtime_ms=0),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description=str(task.get("problem_statement") or ""),
            command_allowlist=[command for command in _candidate_commands(task) if command],
            watched_paths=[*watched_files, *changed_files],
            success_criteria=["post_patch_test_exit_code_zero", "bounded_write_scope_respected"],
            safety_notes=["do_not_use_oracle_patch_diff", "do_not_use_sealed_feedback"],
        ),
        process=ProcessState(status=ProcessStatus.EXITED, exit_code=exit_code),
        terminal=TerminalState(
            stdout_delta=stdout,
            stderr_delta=str(before.get("stderr_excerpt") or ""),
            stdout_lines=len(stdout.splitlines()),
            stderr_lines=len(str(before.get("stderr_excerpt") or "").splitlines()),
            last_command=last_command,
        ),
        filesystem=FileSystemState(
            watched_paths=[*watched_files, *changed_files],
            changed_paths=changed_files,
            dirty_files=changed_files if control_stage in {"post_patch_pre_test", "post_test_fail"} else [],
        ),
    )


def _base_head_row(task: dict[str, Any], *, control_stage: str) -> dict[str, Any]:
    commands = _candidate_commands(task)
    evaluation = str(task.get("evaluation_command") or "").strip()
    if not commands or evaluation not in commands:
        raise ValueError(f"missing evaluation command for task {task.get('task_id')}")
    action = ActionType.RUN_COMMAND
    target = InternalTarget.ESCALATE_TO_DEBUG_CORTEX
    route = RouteName.DEBUG
    command_slot = commands.index(evaluation)
    requires_patch = bool(task.get("requires_patch"))
    rollback_required = bool(str(task.get("rollback_command") or "").strip())
    return {
        "example_id": str(task.get("task_id")),
        "episode_id": str(task.get("task_id")),
        "t": 0,
        "task_type": TaskType.TEST_FAILURE.value,
        "prompt_style": "phase2x_open_repair_initial_control_head_v1",
        "state_prompt": _state_prompt(task, commands, control_stage=control_stage),
        "head_scope": "debug_cortex_open_repair_control",
        "internal_target": target.value,
        "internal_target_index": INTERNAL_TARGET_ORDER.index(target),
        "route_name": route.value,
        "route_index": ROUTE_ORDER.index(route),
        "action_type": action.value,
        "action_index": ACTION_ORDER.index(action),
        "command_intent": command_intent_for_text(evaluation),
        "command": evaluation,
        "file_target": None,
        "command_slot": command_slot,
        "file_slot": -100,
        "confidence_target": 0.85,
        "inhibition_target": 0.0,
        "salience_target": 0.8,
        "risk_target": 0.2,
        "urgency_target": 0.6,
        "prediction_error_target": 0.25,
        "legal_action_mask": {item.value: int(item == ActionType.RUN_COMMAND) for item in ACTION_ORDER},
        "candidate_commands": commands,
        "candidate_files": [],
        "nsi_reference": {
            "salience": 0.8,
            "risk": 0.2,
            "prediction_error": 0.25,
            "confidence": 0.85,
            "reflex_action": action.value,
            "route_name": route.value,
            "receptor_failure_signal": "source_inspected",
            "debug_action_stage": "source_inspected",
        },
        "runtime_overrides": [
            "debug_cortex_escalation",
            "phase2x_open_repair_initial_control",
            "no_json_motor_output",
        ],
        "patch_proposal_label": int(requires_patch),
        "test_selection_slot": command_slot,
        "rollback_safety_label": int(rollback_required),
        "stop_condition_label": 0,
        "bounded_edit_scope_label": int(
            str(task.get("allowed_write_scope") or "").strip() not in {"", "workspace"}
        ),
        "progress_monitor_label": 0,
        "verification_state_label": 0,
        "open_repair_control_label_scope": "initial_state_only",
        "control_stage": control_stage,
        "source_task_manifest": {
            "task_id": task.get("task_id"),
            "task_spec_sha256": task.get("task_spec_sha256"),
            "repo_origin": task.get("repo_origin"),
            "repo_commit": task.get("repo_commit"),
            "sealed_feedback_used": task.get("sealed_feedback_used") is True,
        },
    }


def _base_head_row_from_runtime_trace(
    task: dict[str, Any],
    trace: dict[str, Any] | None,
    *,
    control_stage: str,
) -> dict[str, Any]:
    row = _base_head_row(task, control_stage=control_stage)
    state = _runtime_state_for_trace(task, trace, control_stage=control_stage)
    if state is None:
        return row
    row["prompt_style"] = "phase2x_open_repair_runtime_aligned_control_head_v1"
    row["state_prompt"] = build_phase2c_head_state_prompt_from_state(state)
    row["candidate_commands"] = state.goal.command_allowlist[:4]
    row["candidate_files"] = state.filesystem.dirty_files + [
        path
        for path in state.filesystem.changed_paths + state.filesystem.watched_paths
        if path not in state.filesystem.dirty_files
    ]
    row["runtime_overrides"] = [
        "debug_cortex_escalation",
        "phase2x_open_repair_runtime_aligned_control",
        "no_json_motor_output",
    ]
    row["nsi_reference"] = {
        **(row.get("nsi_reference") if isinstance(row.get("nsi_reference"), dict) else {}),
        "receptor_failure_signal": "source_inspected",
        "debug_action_stage": "source_inspected",
    }
    return row


def phase2x_task_to_head_row(task: dict[str, Any]) -> dict[str, Any]:
    return _base_head_row(task, control_stage="initial")


def phase2x_task_to_control_episode_rows(
    task: dict[str, Any],
    *,
    trace: dict[str, Any] | None = None,
    runtime_aligned: bool = False,
) -> list[dict[str, Any]]:
    """Create generic open-repair control-state rows from task metadata.

    These rows are scripted control supervision, not observed repair outcomes.
    They are intentionally based on task-manifest capabilities only: patch
    required, verification command, rollback availability, and bounded scope.
    """

    stage_labels = [
        {
            "stage": "pre_patch",
            "patch_proposal_label": 1,
            "rollback_safety_label": 1,
            "stop_condition_label": 0,
            "progress_monitor_label": 0,
            "verification_state_label": 0,
        },
        {
            "stage": "post_patch_pre_test",
            "patch_proposal_label": 0,
            "rollback_safety_label": 1,
            "stop_condition_label": 0,
            "progress_monitor_label": 1,
            "verification_state_label": 0,
        },
        {
            "stage": "post_test_pass",
            "patch_proposal_label": 0,
            "rollback_safety_label": 0,
            "stop_condition_label": 1,
            "progress_monitor_label": 2,
            "verification_state_label": 1,
        },
        {
            "stage": "post_test_fail",
            "patch_proposal_label": 1,
            "rollback_safety_label": 1,
            "stop_condition_label": 0,
            "progress_monitor_label": 2,
            "verification_state_label": 2,
        },
    ]
    rows: list[dict[str, Any]] = []
    for index, labels in enumerate(stage_labels):
        stage = str(labels["stage"])
        row = (
            _base_head_row_from_runtime_trace(task, trace, control_stage=stage)
            if runtime_aligned
            else _base_head_row(task, control_stage=stage)
        )
        row["example_id"] = f"{task.get('task_id')}:{stage}"
        row["t"] = index
        row["patch_proposal_label"] = int(labels["patch_proposal_label"])
        row["rollback_safety_label"] = int(labels["rollback_safety_label"])
        row["stop_condition_label"] = int(labels["stop_condition_label"])
        row["progress_monitor_label"] = int(labels["progress_monitor_label"])
        row["verification_state_label"] = int(labels["verification_state_label"])
        row["open_repair_control_label_scope"] = "scripted_full_control_episode"
        rows.append(row)
    return rows


def _label_diversity(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    fields = (
        "patch_proposal_label",
        "test_selection_slot",
        "rollback_safety_label",
        "stop_condition_label",
        "bounded_edit_scope_label",
        "progress_monitor_label",
        "verification_state_label",
    )
    return {
        field: sorted({int(row[field]) for row in rows if field in row})
        for field in fields
    }


def build_phase2x_open_repair_head_dataset(
    *,
    train_tasks_jsonl: str | Path,
    val_tasks_jsonl: str | Path,
    output_dir: str | Path,
    holdout_tasks_jsonl: str | Path | None = None,
    episode_control_mode: str = "initial",
    source_traces_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    if episode_control_mode not in {
        "initial",
        "scripted_full_control",
        "runtime_aligned_scripted_control",
    }:
        raise ValueError(
            "episode_control_mode must be 'initial', 'scripted_full_control', or "
            "'runtime_aligned_scripted_control'"
        )
    output = Path(output_dir)
    train_tasks = _read_jsonl(train_tasks_jsonl)
    val_tasks = _read_jsonl(val_tasks_jsonl)
    holdout_tasks = _read_jsonl(holdout_tasks_jsonl) if holdout_tasks_jsonl else []
    traces = {
        str(row.get("trace_id")): row
        for row in (_read_jsonl(source_traces_jsonl) if source_traces_jsonl else [])
    }
    runtime_aligned = episode_control_mode == "runtime_aligned_scripted_control"
    if episode_control_mode in {"scripted_full_control", "runtime_aligned_scripted_control"}:
        train_rows = [
            item
            for row in train_tasks
            for item in phase2x_task_to_control_episode_rows(
                row,
                trace=traces.get(str(row.get("source_trace_id"))),
                runtime_aligned=runtime_aligned,
            )
        ]
        val_rows = [
            item
            for row in val_tasks
            for item in phase2x_task_to_control_episode_rows(
                row,
                trace=traces.get(str(row.get("source_trace_id"))),
                runtime_aligned=runtime_aligned,
            )
        ]
        holdout_rows = [
            item
            for row in holdout_tasks
            for item in phase2x_task_to_control_episode_rows(
                row,
                trace=traces.get(str(row.get("source_trace_id"))),
                runtime_aligned=runtime_aligned,
            )
        ]
    else:
        train_rows = [phase2x_task_to_head_row(row) for row in train_tasks]
        val_rows = [phase2x_task_to_head_row(row) for row in val_tasks]
        holdout_rows = [phase2x_task_to_head_row(row) for row in holdout_tasks]
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "train.jsonl", train_rows)
    _write_jsonl(output / "val.jsonl", val_rows)
    if holdout_tasks_jsonl:
        _write_jsonl(output / "holdout.jsonl", holdout_rows)
    manifest = {
        "dataset_family": "phase2x_open_repair_head_dataset",
        "json_text_target": False,
        "sealed_feedback_used": False,
        "claim_boundary": (
            "runtime_aligned_scripted_control_training_not_real_repair_execution"
            if episode_control_mode == "runtime_aligned_scripted_control"
            else
            "scripted_full_control_training_not_real_repair_execution"
            if episode_control_mode == "scripted_full_control"
            else "initial_control_head_rows_not_full_repair_execution"
        ),
        "episode_control_mode": episode_control_mode,
        "open_repair_control_label_scope": (
            "runtime_aligned_scripted_control_episode"
            if runtime_aligned
            else "scripted_full_control_episode"
            if episode_control_mode == "scripted_full_control"
            else "initial_state_only"
        ),
        "full_repair_control_training_ready": episode_control_mode
        in {"scripted_full_control", "runtime_aligned_scripted_control"},
        "runtime_aligned_state_prompt": runtime_aligned,
        "source_traces_jsonl": str(Path(source_traces_jsonl)) if source_traces_jsonl else None,
        "splits": {
            "train": {
                "source_jsonl": str(Path(train_tasks_jsonl)),
                "path": str(output / "train.jsonl"),
                "source_rows": len(train_tasks),
                "rows": len(train_rows),
                "source_sha256": _sha256(train_tasks),
                "sha256": _sha256(train_rows),
                "label_diversity": _label_diversity(train_rows),
            },
            "val": {
                "source_jsonl": str(Path(val_tasks_jsonl)),
                "path": str(output / "val.jsonl"),
                "source_rows": len(val_tasks),
                "rows": len(val_rows),
                "source_sha256": _sha256(val_tasks),
                "sha256": _sha256(val_rows),
                "label_diversity": _label_diversity(val_rows),
            },
        },
    }
    if holdout_tasks_jsonl:
        manifest["splits"]["holdout"] = {
            "source_jsonl": str(Path(holdout_tasks_jsonl)),
            "path": str(output / "holdout.jsonl"),
            "source_rows": len(holdout_tasks),
            "rows": len(holdout_rows),
            "source_sha256": _sha256(holdout_tasks),
            "sha256": _sha256(holdout_rows),
            "label_diversity": _label_diversity(holdout_rows),
        }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2X open-repair native-head rows from task manifests.")
    parser.add_argument("--train-tasks-jsonl", required=True)
    parser.add_argument("--val-tasks-jsonl", required=True)
    parser.add_argument("--holdout-tasks-jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--episode-control-mode",
        choices=["initial", "scripted_full_control", "runtime_aligned_scripted_control"],
        default="initial",
    )
    parser.add_argument("--source-traces-jsonl")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    manifest = build_phase2x_open_repair_head_dataset(
        train_tasks_jsonl=args.train_tasks_jsonl,
        val_tasks_jsonl=args.val_tasks_jsonl,
        holdout_tasks_jsonl=args.holdout_tasks_jsonl,
        output_dir=args.output_dir,
        episode_control_mode=args.episode_control_mode,
        source_traces_jsonl=args.source_traces_jsonl,
    )
    if args.output_json:
        _write_json(args.output_json, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
