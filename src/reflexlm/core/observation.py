from __future__ import annotations

import time
from dataclasses import dataclass, field

from reflexlm.core.dataset import (
    DEFAULT_MAX_TEXT_TOKENS,
    DEFAULT_VOCAB_SIZE,
    observation_from_state,
)
from reflexlm.core.schema import ComputerObservation
from reflexlm.models.features import StateVectorizer
from reflexlm.runtime.receptors import (
    FileSystemReceptor,
    ProcessReceptor,
    TerminalReceptor,
    TimeReceptor,
)
from reflexlm.schema import (
    FileSystemState,
    GoalSpec,
    RuntimeEvidenceState,
    SafetyState,
    SourceType,
    SystemStateFrame,
    UserState,
)


@dataclass(slots=True)
class ReflexCoreObservationContext:
    """Bounded live receptor bridge for ReflexCore V0.

    The context records only terminal/process/filesystem/time state for the
    configured goal. Commands still come from the goal allowlist, and filesystem
    visibility is limited to goal.watched_paths.
    """

    goal: GoalSpec
    vocab_size: int = DEFAULT_VOCAB_SIZE
    max_text_tokens: int = DEFAULT_MAX_TEXT_TOKENS
    vectorizer: StateVectorizer = field(default_factory=StateVectorizer)
    process_receptor: ProcessReceptor = field(default_factory=ProcessReceptor)
    terminal_receptor: TerminalReceptor = field(default_factory=TerminalReceptor)
    filesystem_receptor: FileSystemReceptor = field(default_factory=FileSystemReceptor)
    time_receptor: TimeReceptor = field(default_factory=TimeReceptor)
    _tick: int = field(default=0, init=False)
    _start_ms: int = field(default=0, init=False)
    _last_output_ms: int = field(default=0, init=False)
    _last_change_ms: int = field(default=0, init=False)
    _previous_mtimes: dict[str, float] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        now_ms = int(time.time() * 1000)
        self._tick = 0
        self._start_ms = now_ms
        self._last_output_ms = now_ms
        self._last_change_ms = now_ms

    def observe_state(
        self,
        *,
        pid: int | None = None,
        stdout_delta: str = "",
        stderr_delta: str = "",
        prompt_visible: bool = False,
        last_command: str | None = None,
        safety: SafetyState | None = None,
        user: UserState | None = None,
    ) -> SystemStateFrame:
        """Return a bounded state frame from live receptors."""

        now_ms = int(time.time() * 1000)
        terminal = self.terminal_receptor.snapshot(
            stdout_delta=stdout_delta,
            stderr_delta=stderr_delta,
            prompt_visible=prompt_visible,
            last_command=last_command,
        ).model_copy(
            update={
                "stdout_unread": bool(stdout_delta),
                "stderr_unread": bool(stderr_delta),
            }
        )
        filesystem, next_mtimes = self.filesystem_receptor.snapshot(
            list(self.goal.watched_paths),
            previous_mtimes=self._previous_mtimes,
        )
        process = self.process_receptor.snapshot(pid)
        if stdout_delta or stderr_delta:
            self._last_output_ms = now_ms
        if filesystem.external_change_detected:
            self._last_change_ms = now_ms
        time_state = self.time_receptor.snapshot(
            tick=self._tick,
            start_ms=self._start_ms,
            last_output_ms=self._last_output_ms,
            last_change_ms=self._last_change_ms,
        )
        self._tick += 1
        self._previous_mtimes = next_mtimes
        return SystemStateFrame(
            time=time_state,
            goal=self.goal,
            process=process,
            terminal=terminal,
            filesystem=filesystem,
            user=user or UserState(),
            safety=safety or SafetyState(),
            runtime_evidence=_runtime_evidence_from_receptors(
                filesystem=filesystem,
                stdout_delta=stdout_delta,
                stderr_delta=stderr_delta,
            ),
        )

    def observe(
        self,
        *,
        pid: int | None = None,
        stdout_delta: str = "",
        stderr_delta: str = "",
        prompt_visible: bool = False,
        last_command: str | None = None,
        safety: SafetyState | None = None,
        user: UserState | None = None,
    ) -> ComputerObservation:
        """Return a vectorized ReflexCore observation from live receptors."""

        state = self.observe_state(
            pid=pid,
            stdout_delta=stdout_delta,
            stderr_delta=stderr_delta,
            prompt_visible=prompt_visible,
            last_command=last_command,
            safety=safety,
            user=user,
        )
        return observation_from_state(
            state,
            vectorizer=self.vectorizer,
            vocab_size=self.vocab_size,
            max_text_tokens=self.max_text_tokens,
        )


def _runtime_evidence_from_receptors(
    *,
    filesystem: FileSystemState,
    stdout_delta: str,
    stderr_delta: str,
) -> RuntimeEvidenceState:
    terminal_observations = [
        text
        for text in (stdout_delta, stderr_delta)
        if text.strip()
    ]
    return RuntimeEvidenceState(
        source=SourceType.RUNTIME_OBSERVATION.value,
        version="reflexcore_v0_live_observation",
        changed_files=list(filesystem.changed_paths),
        watched_files=list(filesystem.watched_paths),
        terminal_observations=terminal_observations,
    )
