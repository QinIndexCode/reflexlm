from __future__ import annotations

import os
import time
from pathlib import Path

import psutil

from reflexlm.schema import (
    FileSystemState,
    ProcessState,
    ProcessStatus,
    TerminalState,
    TimeState,
)


class ProcessReceptor:
    def snapshot(self, pid: int | None) -> ProcessState:
        if pid is None:
            return ProcessState(status=ProcessStatus.EXITED)
        try:
            proc = psutil.Process(pid)
        except psutil.Error:
            return ProcessState(pid=pid, status=ProcessStatus.EXITED)

        try:
            with proc.oneshot():
                status_map = {
                    psutil.STATUS_RUNNING: ProcessStatus.RUNNING,
                    psutil.STATUS_SLEEPING: ProcessStatus.SLEEPING,
                    psutil.STATUS_DISK_SLEEP: ProcessStatus.BLOCKED,
                    psutil.STATUS_STOPPED: ProcessStatus.BLOCKED,
                    psutil.STATUS_TRACING_STOP: ProcessStatus.BLOCKED,
                    psutil.STATUS_ZOMBIE: ProcessStatus.EXITED,
                    psutil.STATUS_DEAD: ProcessStatus.EXITED,
                }
                raw_status = proc.status()
                status = status_map.get(raw_status, ProcessStatus.RUNNING)
                create_time = proc.create_time()
                runtime_ms = int((time.time() - create_time) * 1000)
                cpu_percent = proc.cpu_percent(interval=0.0)
                memory_mb = proc.memory_info().rss / (1024 * 1024)
                return ProcessState(
                    pid=proc.pid,
                    parent_pid=proc.ppid(),
                    status=status,
                    exit_code=proc.returncode() if hasattr(proc, "returncode") else None,
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    runtime_ms=runtime_ms,
                    waiting_for_input=False,
                    interrupted=False,
                    has_children=bool(proc.children(recursive=False)),
                    resource_alert=cpu_percent > 95.0,
                )
        except psutil.Error:
            return ProcessState(pid=pid, status=ProcessStatus.EXITED)


class TerminalReceptor:
    INPUT_MARKERS = ("enter ", "input", "password", "y/n", "choice:")

    def snapshot(
        self,
        stdout_delta: str = "",
        stderr_delta: str = "",
        *,
        prompt_visible: bool = False,
        last_command: str | None = None,
    ) -> TerminalState:
        combined = f"{stdout_delta}\n{stderr_delta}".lower()
        input_requested = any(marker in combined for marker in self.INPUT_MARKERS)
        stdout_lines = len([line for line in stdout_delta.splitlines() if line.strip()])
        stderr_lines = len([line for line in stderr_delta.splitlines() if line.strip()])
        last_output_channel = None
        if stderr_delta.strip():
            last_output_channel = "stderr"
        elif stdout_delta.strip():
            last_output_channel = "stdout"
        return TerminalState(
            stdout_delta=stdout_delta,
            stderr_delta=stderr_delta,
            stdout_lines=stdout_lines,
            stderr_lines=stderr_lines,
            prompt_visible=prompt_visible,
            input_requested=input_requested,
            last_output_channel=last_output_channel,
            last_command=last_command,
        )


class FileSystemReceptor:
    def snapshot(
        self,
        watched_paths: list[str],
        previous_mtimes: dict[str, float] | None = None,
    ) -> tuple[FileSystemState, dict[str, float]]:
        has_baseline = previous_mtimes is not None
        previous_mtimes = previous_mtimes or {}
        changed_paths: list[str] = []
        next_mtimes: dict[str, float] = {}
        for path_text in watched_paths:
            path = Path(path_text)
            if not path.exists():
                continue
            if path.is_file():
                candidates = [path]
            else:
                candidates = [candidate for candidate in path.rglob("*") if candidate.is_file()]
            for candidate in candidates:
                try:
                    mtime = candidate.stat().st_mtime
                except OSError:
                    continue
                key = os.fspath(candidate)
                next_mtimes[key] = mtime
                if key in previous_mtimes and previous_mtimes[key] != mtime:
                    changed_paths.append(key)
                elif has_baseline and key not in previous_mtimes:
                    changed_paths.append(key)
        if has_baseline:
            changed_paths.extend(
                key for key in previous_mtimes if key not in next_mtimes
            )
        state = FileSystemState(
            watched_paths=watched_paths,
            changed_paths=sorted(changed_paths),
            dirty_files=sorted(changed_paths),
            external_change_detected=bool(changed_paths),
            stale_cache_detected=bool(changed_paths),
            conflict_detected=False,
        )
        return state, next_mtimes


class TimeReceptor:
    def snapshot(
        self,
        *,
        tick: int,
        start_ms: int,
        last_output_ms: int,
        last_change_ms: int,
    ) -> TimeState:
        now_ms = int(time.time() * 1000)
        return TimeState(
            tick=tick,
            runtime_ms=max(now_ms - start_ms, 0),
            wall_clock_ms=now_ms,
            since_last_output_ms=max(now_ms - last_output_ms, 0),
            since_last_state_change_ms=max(now_ms - last_change_ms, 0),
        )
