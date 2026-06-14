from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from reflexlm.data.jsonl import write_jsonl
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    FileSystemState,
    GoalSpec,
    ProcessState,
    ProcessStatus,
    RuntimeEvidenceState,
    SourceType,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
    TrajectoryRecord,
    validate_command_against_goal,
)


SUPPORTED_ACTIONS = {
    ActionType.RUN_COMMAND,
    ActionType.READ_STDOUT,
    ActionType.READ_STDERR,
    ActionType.READ_FILE,
    ActionType.WAIT,
    ActionType.REFRESH_STATE,
    ActionType.DONE,
}


def _resolve_within(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes workspace root: {resolved}") from error
    return resolved


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _snapshot(paths: list[Path], root: Path) -> dict[str, tuple[int, int, str]]:
    snapshot: dict[str, tuple[int, int, str]] = {}
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        stat = path.stat()
        snapshot[_relative(path, root)] = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return snapshot


def _changed_paths(
    before: dict[str, tuple[int, int, str]],
    after: dict[str, tuple[int, int, str]],
) -> list[str]:
    return sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )


def _bounded_text(text: str, limit: int = 4000) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized if len(normalized) <= limit else normalized[:limit] + "\n<TRUNCATED>"


def _command_text(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv)


def _resolved_argv(task: dict[str, Any]) -> list[str]:
    return [
        sys.executable if str(part) == "<PYTHON>" else str(part)
        for part in task.get("argv", [])
    ]


def _goal_for_steps(
    payload: dict[str, Any],
    *,
    steps: list[dict[str, Any]],
    workspace_root: Path,
) -> GoalSpec:
    watched_rel: list[str] = []
    for path in payload.get("watched_paths", []):
        relative = _relative(_resolve_within(workspace_root, str(path)), workspace_root)
        if relative not in watched_rel:
            watched_rel.append(relative)
    command_allowlist: list[str] = []
    for step in steps:
        for path in step.get("watched_paths", []):
            relative = _relative(_resolve_within(workspace_root, str(path)), workspace_root)
            if relative not in watched_rel:
                watched_rel.append(relative)
        argv = _resolved_argv(step)
        if argv:
            command = _command_text(argv)
            if command not in command_allowlist:
                command_allowlist.append(command)
    return GoalSpec(
        task_type=TaskType(str(payload.get("task_type", TaskType.ROUTINE_RECOVERY.value))),
        description=str(
            payload.get("description", "Observe a bounded real runtime transition.")
        ),
        command_allowlist=command_allowlist,
        watched_paths=watched_rel,
        success_criteria=["collect_real_runtime_transition"],
        safety_notes=["workspace_confined", "fixed_action_space", "shell_false"],
    )


def _initial_state(
    *,
    goal: GoalSpec,
    watched_paths: list[str],
    runtime_evidence: RuntimeEvidenceState,
) -> SystemStateFrame:
    return SystemStateFrame(
        time=TimeState(tick=0),
        goal=goal,
        process=ProcessState(status=ProcessStatus.EXITED, exit_code=0),
        terminal=TerminalState(prompt_visible=True),
        filesystem=FileSystemState(watched_paths=watched_paths),
        runtime_evidence=runtime_evidence,
    )


def _execute_task(
    task: dict[str, Any],
    *,
    workspace_root: Path,
    timeout_seconds: float,
    state: SystemStateFrame | None = None,
    goal: GoalSpec | None = None,
    t: int = 0,
    done: bool = True,
) -> tuple[TrajectoryRecord, dict[str, Any]]:
    action_type = ActionType(str(task["action_type"]))
    if action_type not in SUPPORTED_ACTIONS:
        raise ValueError(f"unsupported runtime collection action: {action_type.value}")
    cwd = _resolve_within(workspace_root, str(task.get("cwd", ".")))
    if not cwd.is_dir():
        raise ValueError(f"task cwd is not a directory: {cwd}")
    watched_source = task.get("watched_paths", goal.watched_paths if goal else [])
    watched = [_resolve_within(workspace_root, str(path)) for path in watched_source]
    watched_rel = [_relative(path, workspace_root) for path in watched]
    before = _snapshot(watched, workspace_root)
    argv = _resolved_argv(task)
    command = _command_text(argv) if argv else None
    file_target = None
    if action_type == ActionType.READ_FILE:
        file_path = _resolve_within(workspace_root, str(task["file_target"]))
        if not file_path.is_file():
            raise ValueError(f"READ_FILE target is not a file: {file_path}")
        file_target = _relative(file_path, workspace_root)
    goal = goal or _goal_for_steps(task, steps=[task], workspace_root=workspace_root)
    runtime_evidence = RuntimeEvidenceState(
        source="phase2bk_real_runtime_collector",
        version="phase2bk.runtime_world_model.v2",
        watched_files=watched_rel,
    )
    state = state or _initial_state(
        goal=goal, watched_paths=watched_rel, runtime_evidence=runtime_evidence
    )
    action = ActionDecision(
        type=action_type,
        command=command if action_type == ActionType.RUN_COMMAND else None,
        file_target=file_target,
        reason="phase2bk_manifest_bounded_runtime_action",
        confidence=1.0,
    )
    validate_command_against_goal(action, goal)
    started = time.perf_counter()
    stdout = ""
    stderr = ""
    exit_code = 0
    timed_out = False
    if action_type == ActionType.RUN_COMMAND:
        if not argv:
            raise ValueError("RUN_COMMAND requires non-empty argv")
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env_overlay = task.get("env", {})
        if env_overlay is None:
            env_overlay = {}
        if not isinstance(env_overlay, dict):
            raise ValueError("RUN_COMMAND env override must be an object")
        for key, value in env_overlay.items():
            key_text = str(key)
            if not key_text or "=" in key_text:
                raise ValueError(f"invalid environment variable name: {key_text!r}")
            env[key_text] = str(value)
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=min(float(task.get("timeout_seconds", timeout_seconds)), timeout_seconds),
                shell=False,
                env=env,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            exit_code = int(completed.returncode)
        except subprocess.TimeoutExpired as error:
            timed_out = True
            exit_code = 124
            stdout = str(error.stdout or "")
            stderr = str(error.stderr or "")
    elif action_type == ActionType.READ_FILE:
        stdout = _resolve_within(workspace_root, str(task["file_target"])).read_text(
            encoding="utf-8",
            errors="replace",
        )
    elif action_type == ActionType.READ_STDOUT:
        stdout = state.terminal.stdout_delta
    elif action_type == ActionType.READ_STDERR:
        stderr = state.terminal.stderr_delta
    elif action_type == ActionType.WAIT:
        wait_ms = max(0, min(int(task.get("wait_ms", 5)), 1000))
        time.sleep(wait_ms / 1000.0)
        stdout = f"waited_ms={wait_ms}"
    elif action_type == ActionType.REFRESH_STATE:
        stdout = "workspace_state_refreshed"
    elif action_type == ActionType.DONE:
        stdout = "bounded_runtime_episode_done"
    duration_ms = max(1, int((time.perf_counter() - started) * 1000))
    after = _snapshot(watched, workspace_root)
    changed = _changed_paths(before, after)
    pending_dirty_files = list(
        dict.fromkeys(state.filesystem.dirty_files + state.filesystem.changed_paths + changed)
    )
    if action_type == ActionType.READ_FILE and file_target is not None:
        pending_dirty_files = [
            path for path in pending_dirty_files if path != file_target
        ]
    process = (
        ProcessState(
            status=ProcessStatus.EXITED,
            exit_code=exit_code,
            runtime_ms=duration_ms,
            interrupted=timed_out,
        )
        if action_type == ActionType.RUN_COMMAND
        else state.process.model_copy()
    )
    observed_stdout = bool(state.terminal.stdout_unread and state.terminal.stdout_delta)
    observed_stderr = bool(state.terminal.stderr_unread and state.terminal.stderr_delta)
    terminal_observations = list(state.runtime_evidence.terminal_observations)
    if action_type == ActionType.READ_STDOUT and observed_stdout:
        terminal_observations.append(_bounded_text(state.terminal.stdout_delta))
    if action_type == ActionType.READ_STDERR and observed_stderr:
        terminal_observations.append(_bounded_text(state.terminal.stderr_delta))
    terminal_observations = terminal_observations[-8:]
    has_pending_terminal_observation = (
        state.terminal.stdout_unread or state.terminal.stderr_unread
    )
    terminal = (
        TerminalState(
            stdout_delta=_bounded_text(stdout),
            stderr_delta=_bounded_text(stderr),
            stdout_unread=bool(stdout),
            stderr_unread=bool(stderr),
            stdout_lines=len(stdout.splitlines()),
            stderr_lines=len(stderr.splitlines()),
            prompt_visible=True,
            last_output_channel="stderr" if stderr else "stdout",
            last_command=command if action_type == ActionType.RUN_COMMAND else state.terminal.last_command,
        )
        if action_type == ActionType.RUN_COMMAND
        else (
            state.terminal.model_copy(
                update={
                    "stdout_delta": "",
                    "stdout_unread": False,
                    "stdout_lines": 0,
                    "last_output_channel": (
                        "stderr" if state.terminal.stderr_unread else None
                    ),
                }
            )
            if action_type == ActionType.READ_STDOUT
            else (
                state.terminal.model_copy(
                    update={
                        "stderr_delta": "",
                        "stderr_unread": False,
                        "stderr_lines": 0,
                        "last_output_channel": (
                            "stdout" if state.terminal.stdout_unread else None
                        ),
                    }
                )
                if action_type == ActionType.READ_STDERR
                else (
                    state.terminal.model_copy()
                    if has_pending_terminal_observation
                    or action_type == ActionType.DONE
                    else TerminalState(
                        stdout_delta=_bounded_text(stdout),
                        stderr_delta=_bounded_text(stderr),
                        stdout_lines=len(stdout.splitlines()),
                        stderr_lines=len(stderr.splitlines()),
                        prompt_visible=True,
                        last_output_channel="stderr" if stderr else "stdout",
                        last_command=state.terminal.last_command,
                    )
                )
            )
        )
    )
    next_state = SystemStateFrame(
        time=TimeState(
            tick=state.time.tick + 1,
            runtime_ms=duration_ms,
            wall_clock_ms=state.time.wall_clock_ms + duration_ms,
            since_last_output_ms=0,
            since_last_state_change_ms=0 if changed else duration_ms,
        ),
        goal=goal,
        process=process,
        terminal=terminal,
        filesystem=FileSystemState(
            watched_paths=watched_rel,
            changed_paths=changed,
            dirty_files=pending_dirty_files,
            external_change_detected=bool(changed),
        ),
        runtime_evidence=runtime_evidence.model_copy(
            update={
                "changed_files": pending_dirty_files,
                "terminal_observations": terminal_observations,
            }
        ),
    )
    record = TrajectoryRecord(
        episode_id=str(task["episode_id"]),
        t=t,
        goal=goal,
        state=state,
        action=action,
        next_state=next_state,
        reward=float(task.get("reward", 1.0 if exit_code == 0 and not timed_out else 0.0)),
        done=done,
        source=SourceType.RUNTIME_OBSERVATION,
    )
    evidence = {
        "episode_id": record.episode_id,
        "action_type": action_type.value,
        "cwd": _relative(cwd, workspace_root),
        "argv": argv,
        "shell": False,
        "env_override_keys": sorted(str(key) for key in task.get("env", {}) or {}),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "watched_paths": watched_rel,
        "changed_paths": changed,
        "t": t,
        "done": done,
        "state_chain_tick": state.time.tick,
        "next_state_chain_tick": next_state.time.tick,
        "expected_exit_code": task.get("expected_exit_code"),
        "expected_timed_out": task.get("expected_timed_out"),
        "observation_available": (
            observed_stdout
            if action_type == ActionType.READ_STDOUT
            else (
                observed_stderr
                if action_type == ActionType.READ_STDERR
                else action_type != ActionType.READ_FILE or file_target is not None
            )
        ),
    }
    evidence["expected_outcome_matched"] = (
        (task.get("expected_exit_code") is None or exit_code == int(task["expected_exit_code"]))
        and (
            task.get("expected_timed_out") is None
            or timed_out is bool(task["expected_timed_out"])
        )
        and evidence["observation_available"]
    )
    return record, evidence


def _execute_episode(
    episode: dict[str, Any],
    *,
    episode_id: str,
    workspace_root: Path,
    timeout_seconds: float,
) -> tuple[list[TrajectoryRecord], list[dict[str, Any]]]:
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("episode steps must be a non-empty list")
    if not all(isinstance(step, dict) for step in steps):
        raise ValueError("each episode step must be an object")
    goal = _goal_for_steps(episode, steps=steps, workspace_root=workspace_root)
    runtime_evidence = RuntimeEvidenceState(
        source="phase2bk_real_runtime_collector",
        version="phase2bk.runtime_world_model.v2",
        watched_files=goal.watched_paths,
    )
    state = _initial_state(
        goal=goal,
        watched_paths=goal.watched_paths,
        runtime_evidence=runtime_evidence,
    )
    records: list[TrajectoryRecord] = []
    evidence_rows: list[dict[str, Any]] = []
    for t, step in enumerate(steps):
        task = dict(step)
        task["episode_id"] = episode_id
        record, evidence = _execute_task(
            task,
            workspace_root=workspace_root,
            timeout_seconds=timeout_seconds,
            state=state,
            goal=goal,
            t=t,
            done=t == len(steps) - 1,
        )
        records.append(record)
        evidence_rows.append(evidence)
        state = record.next_state
    return records, evidence_rows


def collect_phase2bk_runtime_world_model_trajectories(
    *,
    manifest_json: str | Path,
    output_jsonl: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    manifest_path = Path(manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    workspace_root = Path(str(manifest["workspace_root"])).resolve()
    if not workspace_root.is_dir():
        raise ValueError(f"workspace_root is not a directory: {workspace_root}")
    tasks = manifest.get("tasks", [])
    episodes = manifest.get("episodes", [])
    if not isinstance(tasks, list) or not isinstance(episodes, list) or not (tasks or episodes):
        raise ValueError("manifest must contain non-empty tasks or episodes")
    repetitions_per_task = int(manifest.get("repetitions_per_task", 1))
    if repetitions_per_task < 1:
        raise ValueError("repetitions_per_task must be at least 1")
    records: list[TrajectoryRecord] = []
    evidence_rows: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("each manifest task must be an object")
        task_repetitions = int(task.get("repetitions", repetitions_per_task))
        if task_repetitions < 1:
            raise ValueError("task repetitions must be at least 1")
        base_episode_id = str(task["episode_id"])
        for repetition in range(task_repetitions):
            repeated_task = dict(task)
            repeated_task["episode_id"] = (
                base_episode_id
                if task_repetitions == 1
                else f"{base_episode_id}-rep-{repetition:03d}"
            )
            record, evidence = _execute_task(
                repeated_task,
                workspace_root=workspace_root,
                timeout_seconds=timeout_seconds,
            )
            evidence["base_episode_id"] = base_episode_id
            evidence["repetition"] = repetition
            records.append(record)
            evidence_rows.append(evidence)
    repetitions_per_episode = int(manifest.get("repetitions_per_episode", 1))
    if repetitions_per_episode < 1:
        raise ValueError("repetitions_per_episode must be at least 1")
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("each manifest episode must be an object")
        episode_repetitions = int(episode.get("repetitions", repetitions_per_episode))
        if episode_repetitions < 1:
            raise ValueError("episode repetitions must be at least 1")
        base_episode_id = str(episode["episode_id"])
        for repetition in range(episode_repetitions):
            episode_id = (
                base_episode_id
                if episode_repetitions == 1
                else f"{base_episode_id}-rep-{repetition:03d}"
            )
            episode_records, episode_evidence = _execute_episode(
                episode,
                episode_id=episode_id,
                workspace_root=workspace_root,
                timeout_seconds=timeout_seconds,
            )
            for evidence in episode_evidence:
                evidence["base_episode_id"] = base_episode_id
                evidence["repetition"] = repetition
            records.extend(episode_records)
            evidence_rows.extend(episode_evidence)
    write_jsonl(Path(output_jsonl), records)
    report = {
        "artifact_family": "phase2bk_real_runtime_world_model_trajectories",
        "passed": True,
        "schema_version": "phase2bk.runtime_world_model.v2",
        "workspace_root": str(workspace_root),
        "manifest_tasks": len(tasks),
        "manifest_episodes": len(episodes),
        "repetitions_per_task": repetitions_per_task,
        "rows": len(records),
        "runtime_observation_rows": sum(
            record.source == SourceType.RUNTIME_OBSERVATION for record in records
        ),
        "shell_false_rows": sum(row["shell"] is False for row in evidence_rows),
        "successful_rows": sum(record.reward > 0.0 for record in records),
        "continuous_episode_rows": sum(record.t > 0 for record in records),
        "timed_out_rows": sum(row["timed_out"] for row in evidence_rows),
        "nonzero_exit_rows": sum(row["exit_code"] != 0 for row in evidence_rows),
        "expected_outcome_matched_rows": sum(
            row["expected_outcome_matched"] for row in evidence_rows
        ),
        "action_counts": {
            action.value: sum(record.action is not None and record.action.type == action for record in records)
            for action in sorted(SUPPORTED_ACTIONS, key=lambda item: item.value)
        },
        "trajectory_jsonl": str(Path(output_jsonl)),
        "manifest_json": str(manifest_path),
        "evidence_rows": evidence_rows,
        "ready_for_bounded_real_runtime_transition_claim": True,
        "ready_for_open_ended_runtime_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
    }
    output_report = Path(output_report_json)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect workspace-confined real runtime trajectories for NSI world-model training."
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()
    report = collect_phase2bk_runtime_world_model_trajectories(
        manifest_json=args.manifest_json,
        output_jsonl=args.output_jsonl,
        output_report_json=args.output_report_json,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
