from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
import math
from pathlib import Path

import torch

from reflexlm.core.dataset import observation_from_state
from reflexlm.core.motor import decode_reflexcore_motor
from reflexlm.core.model import ReflexCoreV0
from reflexlm.core.observation import ReflexCoreObservationContext
from reflexlm.core.schema import action_to_index
from reflexlm.runtime.safety import SafetyDecision, SafetyLayer
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    FileSystemState,
    GoalSpec,
    ProcessState,
    ProcessStatus,
    SystemStateFrame,
    TerminalState,
    TimeState,
    UserState,
)


def _unique_paths(paths: list[str]) -> list[str]:
    unique: list[str] = []
    for path in paths:
        if path and path not in unique:
            unique.append(path)
    return unique


def _sandbox_relative_path(path: str, sandbox_root: Path) -> str:
    if not path:
        return path
    try:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.resolve().relative_to(sandbox_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path
    return path.replace("\\", "/")


@dataclass(slots=True)
class ReflexCoreSandboxConfig:
    sandbox_root: Path
    allowed_commands: tuple[str, ...] = ()
    max_steps: int = 8
    allow_process_execution: bool = False
    command_observe_timeout_s: float = 1.0
    wait_observe_timeout_s: float = 0.25
    resource_alert_on_timeout: bool = False


@dataclass(slots=True)
class ReflexCoreStepResult:
    state: SystemStateFrame
    proposed_action: ActionDecision
    safety_decision: SafetyDecision
    stdout: str = ""
    stderr: str = ""
    done: bool = False
    predicted_next_state: list[float] | None = None
    model_prediction_error: float | None = None
    observed_prediction_error: float | None = None


@dataclass(slots=True)
class ReflexCoreProposal:
    safety_decision: SafetyDecision
    hidden: torch.Tensor | None
    predicted_next_state: list[float] | None = None
    model_prediction_error: float | None = None


@dataclass(slots=True)
class ReflexCoreLiveLoopResult:
    initial_state: SystemStateFrame
    trace: list[ReflexCoreStepResult]
    final_state: SystemStateFrame


class ReflexCoreSandboxRunner:
    """Local V0 loop for terminal/file sandbox tasks only."""

    def __init__(
        self,
        config: ReflexCoreSandboxConfig,
        *,
        safety_layer: SafetyLayer | None = None,
    ) -> None:
        self.config = config
        self.safety_layer = safety_layer or SafetyLayer()
        self.config.sandbox_root.mkdir(parents=True, exist_ok=True)
        self._active_process: subprocess.Popen[str] | None = None
        self._active_command: str | None = None

    def initial_state(self, goal: GoalSpec) -> SystemStateFrame:
        goal = goal.model_copy(update={"command_allowlist": list(self.config.allowed_commands)})
        return SystemStateFrame(
            time=TimeState(tick=0, wall_clock_ms=int(time.time() * 1000)),
            goal=goal,
            process=ProcessState(status=ProcessStatus.EXITED),
            terminal=TerminalState(prompt_visible=True),
            filesystem=FileSystemState(watched_paths=[str(self.config.sandbox_root)]),
            user=UserState(),
        )

    def live_observation_context(
        self,
        goal: GoalSpec,
        *,
        vocab_size: int,
        max_text_tokens: int = 128,
    ) -> ReflexCoreObservationContext:
        return ReflexCoreObservationContext(
            goal=self._sandbox_goal(goal),
            vocab_size=vocab_size,
            max_text_tokens=max_text_tokens,
        )

    def _sandbox_goal(self, goal: GoalSpec) -> GoalSpec:
        watched_paths = list(dict.fromkeys([*goal.watched_paths, str(self.config.sandbox_root)]))
        return goal.model_copy(
            update={
                "command_allowlist": list(self.config.allowed_commands),
                "watched_paths": watched_paths,
            }
        )

    def propose(self, model: ReflexCoreV0, state: SystemStateFrame) -> SafetyDecision:
        return self.propose_with_state(model, state).safety_decision

    def propose_with_state(
        self,
        model: ReflexCoreV0,
        state: SystemStateFrame,
        *,
        hidden: torch.Tensor | None = None,
    ) -> ReflexCoreProposal:
        observation = observation_from_state(
            state,
            vocab_size=model.config.vocab_size,
        )
        vector = torch.tensor(observation.vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        text = torch.tensor(observation.text_tokens, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            outputs = model(vector, text, hidden=hidden)
        decoded = decode_reflexcore_motor(outputs, state)
        next_hidden = outputs["hidden"]
        safety_decision = self.safety_layer.enforce(decoded.action, state.goal, state)
        action_for_prediction = safety_decision.action or ActionDecision(
            type=ActionType.BLOCK,
            reason=safety_decision.reason,
            confidence=decoded.action.confidence,
        )
        prediction_outputs = self._predict_next_for_action(
            model,
            vector,
            text,
            hidden=hidden,
            action=action_for_prediction,
        )
        return ReflexCoreProposal(
            safety_decision=safety_decision,
            hidden=next_hidden if isinstance(next_hidden, torch.Tensor) else None,
            predicted_next_state=prediction_outputs[0],
            model_prediction_error=prediction_outputs[1],
        )

    def _predict_next_for_action(
        self,
        model: ReflexCoreV0,
        vector: torch.Tensor,
        text: torch.Tensor,
        *,
        hidden: torch.Tensor | None,
        action: ActionDecision,
    ) -> tuple[list[float] | None, float | None]:
        action_indices = torch.tensor(
            [[action_to_index(action.type)]],
            dtype=torch.long,
            device=vector.device,
        )
        with torch.no_grad():
            outputs = model(
                vector,
                text,
                hidden=hidden,
                action_indices=action_indices,
            )
        next_state = outputs.get("next_state")
        prediction_error = outputs.get("prediction_error")
        predicted_next_state = (
            [float(item) for item in next_state[0, -1].detach().cpu().tolist()]
            if isinstance(next_state, torch.Tensor)
            else None
        )
        predicted_error = (
            float(prediction_error[0, -1].detach().cpu().reshape(-1)[0].item())
            if isinstance(prediction_error, torch.Tensor)
            else None
        )
        return predicted_next_state, predicted_error

    def run_model_loop(
        self,
        model: ReflexCoreV0,
        initial_state: SystemStateFrame,
    ) -> list[ReflexCoreStepResult]:
        state = initial_state
        hidden: torch.Tensor | None = None
        trace: list[ReflexCoreStepResult] = []
        for _step in range(self.config.max_steps):
            proposal = self.propose_with_state(model, state, hidden=hidden)
            hidden = proposal.hidden
            action = proposal.safety_decision.action
            if action is None:
                action = ActionDecision(
                    type=ActionType.BLOCK,
                    reason=proposal.safety_decision.reason,
                    confidence=1.0,
                )
            result = self.step(state, action)
            predicted_result = self.attach_prediction(result, proposal)
            trace.append(predicted_result)
            state = predicted_result.state
            if predicted_result.done:
                break
        return trace

    def run_model_live_observation_loop(
        self,
        model: ReflexCoreV0,
        goal: GoalSpec,
        *,
        max_text_tokens: int = 128,
    ) -> ReflexCoreLiveLoopResult:
        """Run observe -> propose -> act -> observe with live bounded receptors."""

        context = self.live_observation_context(
            goal,
            vocab_size=model.config.vocab_size,
            max_text_tokens=max_text_tokens,
        )
        state = context.observe_state(prompt_visible=True)
        initial_state = state
        hidden: torch.Tensor | None = None
        trace: list[ReflexCoreStepResult] = []
        for _step in range(self.config.max_steps):
            proposal = self.propose_with_state(model, state, hidden=hidden)
            hidden = proposal.hidden
            action = proposal.safety_decision.action
            if action is None:
                action = ActionDecision(
                    type=ActionType.BLOCK,
                    reason=proposal.safety_decision.reason,
                    confidence=1.0,
                )
            result = self.step(state, action)
            predicted_result = self.attach_prediction(result, proposal)
            observed_result = self.reobserve_step_result(context, predicted_result)
            trace.append(observed_result)
            state = observed_result.state
            if observed_result.done:
                break
        return ReflexCoreLiveLoopResult(
            initial_state=initial_state,
            trace=trace,
            final_state=state,
        )

    def reobserve_step_result(
        self,
        context: ReflexCoreObservationContext,
        result: ReflexCoreStepResult,
    ) -> ReflexCoreStepResult:
        action = result.safety_decision.action or result.proposed_action
        observed_state = context.observe_state(
            pid=self._active_process.pid if self._active_process is not None else None,
            stdout_delta=result.stdout,
            stderr_delta=result.stderr,
            prompt_visible=result.state.terminal.prompt_visible,
            last_command=action.command or action.type.value,
            safety=result.state.safety,
            user=result.state.user,
        )
        process = self._merge_live_process_state(
            observed_state.process,
            result.state.process,
        )
        filesystem = self._merge_live_filesystem_state(
            observed_state.filesystem,
            result.state.filesystem,
            consumed_file_target=(
                action.file_target if action.type == ActionType.READ_FILE else None
            ),
            consumed_refresh=action.type == ActionType.REFRESH_STATE,
        )
        reobserved_state = observed_state.model_copy(
            update={"process": process, "filesystem": filesystem}
        )
        if action.type in {ActionType.READ_STDOUT, ActionType.READ_STDERR}:
            terminal_update: dict[str, bool] = {}
            if action.type == ActionType.READ_STDOUT:
                terminal_update["stdout_unread"] = False
            if action.type == ActionType.READ_STDERR:
                terminal_update["stderr_unread"] = False
            reobserved_state = reobserved_state.model_copy(
                update={
                    "terminal": reobserved_state.terminal.model_copy(
                        update=terminal_update
                    )
                }
            )
        observed_prediction_error = self._observed_prediction_error(
            result.predicted_next_state,
            reobserved_state,
            context,
        )
        prediction_update: dict[str, float | None] = {
            "model_prediction_error": result.model_prediction_error,
            "observed_prediction_error": observed_prediction_error,
            "prediction_error_delta": None,
        }
        if (
            result.model_prediction_error is not None
            and observed_prediction_error is not None
        ):
            prediction_update["prediction_error_delta"] = (
                observed_prediction_error - result.model_prediction_error
            )
        reobserved_state = reobserved_state.model_copy(
            update={
                "runtime_evidence": reobserved_state.runtime_evidence.model_copy(
                    update={
                        **prediction_update,
                        "changed_files": list(reobserved_state.filesystem.changed_paths),
                        "watched_files": list(reobserved_state.filesystem.watched_paths),
                    }
                )
            }
        )
        return ReflexCoreStepResult(
            state=reobserved_state,
            proposed_action=result.proposed_action,
            safety_decision=result.safety_decision,
            stdout=result.stdout,
            stderr=result.stderr,
            done=result.done,
            predicted_next_state=result.predicted_next_state,
            model_prediction_error=result.model_prediction_error,
            observed_prediction_error=observed_prediction_error,
        )

    def _merge_live_filesystem_state(
        self,
        observed: FileSystemState,
        predicted: FileSystemState,
        *,
        consumed_file_target: str | None = None,
        consumed_refresh: bool = False,
    ) -> FileSystemState:
        """Merge external receptor changes with the runner's unread-file memory.

        A live filesystem receptor reports files whose mtimes changed since the
        previous snapshot. It cannot know which changed files the motor loop has
        already consumed. The runner state does know that, so preserve remaining
        dirty/changed paths while still adding newly observed external changes.
        """

        consumed = (
            _sandbox_relative_path(consumed_file_target, self.config.sandbox_root)
            if consumed_file_target
            else None
        )
        changed_paths = _unique_paths(
            [
                path
                for path in (
                    _sandbox_relative_path(path, self.config.sandbox_root)
                    for path in list(predicted.changed_paths) + list(observed.changed_paths)
                )
                if path != consumed
            ]
        )
        dirty_files = _unique_paths(
            [
                path
                for path in (
                    _sandbox_relative_path(path, self.config.sandbox_root)
                    for path in list(predicted.dirty_files) + list(observed.dirty_files)
                )
                if path != consumed
            ]
        )
        watched_paths = _unique_paths(
            list(observed.watched_paths) + list(predicted.watched_paths)
        )
        return observed.model_copy(
            update={
                "watched_paths": watched_paths,
                "changed_paths": changed_paths,
                "dirty_files": dirty_files,
                "external_change_detected": False
                if consumed_refresh
                else bool(
                    observed.external_change_detected
                    or predicted.external_change_detected
                ),
                "stale_cache_detected": False
                if consumed_refresh
                else bool(
                    observed.stale_cache_detected or predicted.stale_cache_detected
                ),
                "conflict_detected": bool(
                    observed.conflict_detected or predicted.conflict_detected
                ),
            }
        )

    def attach_prediction(
        self,
        result: ReflexCoreStepResult,
        proposal: ReflexCoreProposal,
    ) -> ReflexCoreStepResult:
        return ReflexCoreStepResult(
            state=result.state,
            proposed_action=result.proposed_action,
            safety_decision=result.safety_decision,
            stdout=result.stdout,
            stderr=result.stderr,
            done=result.done,
            predicted_next_state=proposal.predicted_next_state,
            model_prediction_error=proposal.model_prediction_error,
            observed_prediction_error=result.observed_prediction_error,
        )

    def _observed_prediction_error(
        self,
        predicted_next_state: list[float] | None,
        observed_state: SystemStateFrame,
        context: ReflexCoreObservationContext,
    ) -> float | None:
        if predicted_next_state is None:
            return None
        observed = observation_from_state(
            observed_state,
            vectorizer=context.vectorizer,
            vocab_size=context.vocab_size,
            max_text_tokens=context.max_text_tokens,
        ).vector
        if len(observed) != len(predicted_next_state):
            return None
        squared = [
            (float(predicted) - float(actual)) ** 2
            for predicted, actual in zip(predicted_next_state, observed)
        ]
        return math.sqrt(sum(squared)) / max(math.sqrt(float(len(squared))), 1.0)

    def _merge_live_process_state(
        self,
        observed: ProcessState,
        execution: ProcessState,
    ) -> ProcessState:
        if self._active_process is None:
            return execution
        return observed.model_copy(
            update={
                "runtime_ms": max(observed.runtime_ms, execution.runtime_ms),
                "last_output_ms": max(observed.last_output_ms, execution.last_output_ms),
                "waiting_for_input": observed.waiting_for_input
                or execution.waiting_for_input,
                "interrupted": execution.interrupted,
                "resource_alert": observed.resource_alert or execution.resource_alert,
            }
        )

    def step(
        self,
        state: SystemStateFrame,
        action: ActionDecision,
    ) -> ReflexCoreStepResult:
        safety = self.safety_layer.enforce(action, state.goal, state)
        if not safety.allowed or safety.action is None:
            next_state = self._advance_terminal_state(
                state,
                stderr=safety.reason,
                action=ActionDecision(
                    type=ActionType.BLOCK,
                    reason=safety.reason,
                    confidence=action.confidence,
                ),
            )
            return ReflexCoreStepResult(
                state=next_state,
                proposed_action=action,
                safety_decision=safety,
                stderr=safety.reason,
            )
        safe_action = safety.action
        if safe_action.type == ActionType.RUN_COMMAND:
            return self._run_command(state, safe_action, safety)
        if safe_action.type == ActionType.READ_FILE:
            return self._read_file(state, safe_action, safety)
        if safe_action.type == ActionType.READ_STDERR:
            return self._terminal_read(
                state,
                safe_action,
                safety,
                stderr=state.terminal.stderr_delta,
            )
        if safe_action.type == ActionType.READ_STDOUT:
            return self._terminal_read(
                state,
                safe_action,
                safety,
                stdout=state.terminal.stdout_delta,
            )
        if safe_action.type == ActionType.REFRESH_STATE:
            return self._refresh(state, safe_action, safety)
        if safe_action.type in {ActionType.WAIT, ActionType.ASK_USER}:
            return self._wait(state, safe_action, safety)
        if safe_action.type == ActionType.STOP_PROCESS:
            return self._stop_process(state, safe_action, safety)
        if safe_action.type == ActionType.DONE:
            return ReflexCoreStepResult(
                state=self._advance_terminal_state(state, action=safe_action),
                proposed_action=action,
                safety_decision=safety,
                done=True,
            )
        return ReflexCoreStepResult(
            state=self._advance_terminal_state(state, action=safe_action),
            proposed_action=action,
            safety_decision=safety,
        )

    def _run_command(
        self,
        state: SystemStateFrame,
        action: ActionDecision,
        safety: SafetyDecision,
    ) -> ReflexCoreStepResult:
        command = action.command or ""
        if not self.config.allow_process_execution:
            next_state = self._advance_terminal_state(
                state,
                stderr="sandbox execution disabled; command serialized but not executed",
                action=action,
            )
            return ReflexCoreStepResult(
                state=next_state,
                proposed_action=action,
                safety_decision=SafetyDecision(
                    allowed=False,
                    action=ActionDecision(
                        type=ActionType.BLOCK,
                        reason="sandbox_execution_disabled",
                        confidence=action.confidence,
                    ),
                    reason="sandbox_execution_disabled",
                ),
                stderr="sandbox_execution_disabled",
            )
        if command not in self.config.allowed_commands:
            raise RuntimeError("safety layer failed to enforce command allowlist")
        try:
            command_args = shlex.split(command, posix=True)
        except ValueError as exc:
            next_state = self._advance_terminal_state(
                state,
                stderr=f"invalid allowlisted command syntax: {exc}",
                action=action,
            )
            return ReflexCoreStepResult(
                state=next_state,
                proposed_action=action,
                safety_decision=safety,
                stderr=next_state.terminal.stderr_delta,
            )
        if not command_args:
            next_state = self._advance_terminal_state(
                state,
                stderr="empty allowlisted command",
                action=action,
            )
            return ReflexCoreStepResult(
                state=next_state,
                proposed_action=action,
                safety_decision=safety,
                stderr=next_state.terminal.stderr_delta,
            )
        executable = command_args[0]
        if shutil.which(executable) is None:
            next_state = self._advance_terminal_state(
                state,
                stderr=f"command executable not found: {executable}",
                action=action,
            )
            return ReflexCoreStepResult(
                state=next_state,
                proposed_action=action,
                safety_decision=safety,
                stderr=next_state.terminal.stderr_delta,
            )
        process = subprocess.Popen(
            command_args,
            cwd=self.config.sandbox_root,
            shell=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = process.communicate(
                timeout=self.config.command_observe_timeout_s
            )
        except subprocess.TimeoutExpired:
            self._active_process = process
            self._active_command = command
            next_state = state.model_copy(
                update={
                    "time": state.time.model_copy(
                        update={
                            "tick": state.time.tick + 1,
                            "since_last_output_ms": state.time.since_last_output_ms
                            + int(self.config.command_observe_timeout_s * 1000),
                            "since_last_state_change_ms": 0,
                        }
                    ),
                    "process": ProcessState(
                        status=ProcessStatus.RUNNING,
                        runtime_ms=(
                            60_000
                            if self.config.resource_alert_on_timeout
                            else int(self.config.command_observe_timeout_s * 1000)
                        ),
                        last_output_ms=(
                            30_000 if self.config.resource_alert_on_timeout else 0
                        ),
                        cpu_percent=99.0 if self.config.resource_alert_on_timeout else 0.0,
                        resource_alert=self.config.resource_alert_on_timeout,
                    ),
                    "terminal": TerminalState(
                        prompt_visible=False,
                        last_command=command,
                    ),
                }
            )
            return ReflexCoreStepResult(
                state=next_state,
                proposed_action=action,
                safety_decision=safety,
            )
        completed_returncode = process.returncode
        next_state = self._advance_terminal_state(
            state,
            stdout=stdout,
            stderr=stderr,
            action=action,
            exit_code=completed_returncode,
        )
        return ReflexCoreStepResult(
            state=next_state,
            proposed_action=action,
            safety_decision=safety,
            stdout=stdout,
            stderr=stderr,
            done=completed_returncode == 0,
        )

    def _read_file(
        self,
        state: SystemStateFrame,
        action: ActionDecision,
        safety: SafetyDecision,
    ) -> ReflexCoreStepResult:
        file_target = action.file_target or ""
        target = (self.config.sandbox_root / file_target).resolve()
        root = self.config.sandbox_root.resolve()
        if not str(target).startswith(str(root)):
            stderr = "read target escapes sandbox"
        elif target.exists() and target.is_file():
            stderr = ""
            stdout = target.read_text(encoding="utf-8", errors="replace")[:4096]
            next_state = self._advance_terminal_state(
                state,
                stdout=stdout,
                action=action,
            )
            remaining_changed = [
                path for path in state.filesystem.changed_paths if path != file_target
            ]
            remaining_dirty = [
                path for path in state.filesystem.dirty_files if path != file_target
            ]
            next_state = next_state.model_copy(
                update={
                    "filesystem": state.filesystem.model_copy(
                        update={
                            "changed_paths": remaining_changed,
                            "dirty_files": remaining_dirty,
                            "external_change_detected": False,
                            "stale_cache_detected": False,
                            "conflict_detected": (
                                state.filesystem.conflict_detected
                                and bool(remaining_changed)
                            ),
                        }
                    )
                }
            )
            return ReflexCoreStepResult(
                state=next_state,
                proposed_action=action,
                safety_decision=safety,
                stdout=stdout,
            )
        else:
            stderr = "read target missing"
        next_state = self._advance_terminal_state(state, stderr=stderr, action=action)
        return ReflexCoreStepResult(
            state=next_state,
            proposed_action=action,
            safety_decision=safety,
            stderr=stderr,
        )

    def _terminal_read(
        self,
        state: SystemStateFrame,
        action: ActionDecision,
        safety: SafetyDecision,
        *,
        stdout: str = "",
        stderr: str = "",
    ) -> ReflexCoreStepResult:
        next_state = self._advance_terminal_state(
            state,
            stdout=stdout,
            stderr=stderr,
            action=action,
        )
        terminal_update: dict[str, bool] = {}
        if action.type == ActionType.READ_STDOUT:
            terminal_update["stdout_unread"] = False
        if action.type == ActionType.READ_STDERR:
            terminal_update["stderr_unread"] = False
        if terminal_update:
            next_state = next_state.model_copy(
                update={
                    "terminal": next_state.terminal.model_copy(update=terminal_update)
                }
            )
        return ReflexCoreStepResult(
            state=next_state,
            proposed_action=action,
            safety_decision=safety,
            stdout=stdout,
            stderr=stderr,
        )

    def _refresh(
        self,
        state: SystemStateFrame,
        action: ActionDecision,
        safety: SafetyDecision,
    ) -> ReflexCoreStepResult:
        watched = [str(self.config.sandbox_root)]
        changed = sorted(
            str(path.relative_to(self.config.sandbox_root))
            for path in self.config.sandbox_root.rglob("*")
            if path.is_file()
        )[:8]
        next_state = state.model_copy(
            update={
                "time": state.time.model_copy(update={"tick": state.time.tick + 1}),
                "filesystem": state.filesystem.model_copy(
                    update={
                        "watched_paths": watched,
                        "changed_paths": changed,
                        "dirty_files": changed,
                        "external_change_detected": False,
                        "stale_cache_detected": False,
                    }
                ),
                "terminal": state.terminal.model_copy(update={"last_command": action.type.value}),
            }
        )
        return ReflexCoreStepResult(
            state=next_state,
            proposed_action=action,
            safety_decision=safety,
        )

    def _wait(
        self,
        state: SystemStateFrame,
        action: ActionDecision,
        safety: SafetyDecision,
    ) -> ReflexCoreStepResult:
        if self._active_process is not None:
            return self._wait_for_active_process(state, action, safety)
        next_state = state.model_copy(
            update={
                "time": state.time.model_copy(
                    update={
                        "tick": state.time.tick + 1,
                        "since_last_output_ms": state.time.since_last_output_ms + 100,
                    }
                ),
                "terminal": state.terminal.model_copy(update={"last_command": action.type.value}),
            }
        )
        return ReflexCoreStepResult(
            state=next_state,
            proposed_action=action,
            safety_decision=safety,
        )

    def _wait_for_active_process(
        self,
        state: SystemStateFrame,
        action: ActionDecision,
        safety: SafetyDecision,
    ) -> ReflexCoreStepResult:
        process = self._active_process
        if process is None:
            return self._wait(state, action, safety)
        wait_ms = int(self.config.wait_observe_timeout_s * 1000)
        try:
            stdout, stderr = process.communicate(timeout=self.config.wait_observe_timeout_s)
        except subprocess.TimeoutExpired:
            next_state = state.model_copy(
                update={
                    "time": state.time.model_copy(
                        update={
                            "tick": state.time.tick + 1,
                            "since_last_output_ms": state.time.since_last_output_ms + wait_ms,
                            "since_last_state_change_ms": 0,
                        }
                    ),
                    "process": state.process.model_copy(
                        update={
                            "status": ProcessStatus.RUNNING,
                            "runtime_ms": state.process.runtime_ms + wait_ms,
                        }
                    ),
                    "terminal": TerminalState(
                        prompt_visible=False,
                        last_command=action.type.value,
                    ),
                }
            )
            return ReflexCoreStepResult(
                state=next_state,
                proposed_action=action,
                safety_decision=safety,
            )
        self._active_process = None
        self._active_command = None
        next_state = self._advance_terminal_state(
            state,
            stdout=stdout,
            stderr=stderr,
            action=action,
            exit_code=process.returncode,
        )
        return ReflexCoreStepResult(
            state=next_state,
            proposed_action=action,
            safety_decision=safety,
            stdout=stdout,
            stderr=stderr,
            done=process.returncode == 0,
        )

    def _stop_process(
        self,
        state: SystemStateFrame,
        action: ActionDecision,
        safety: SafetyDecision,
    ) -> ReflexCoreStepResult:
        if self._active_process is not None:
            return self._stop_active_process(state, action, safety)
        next_state = state.model_copy(
            update={
                "time": state.time.model_copy(
                    update={
                        "tick": state.time.tick + 1,
                        "since_last_output_ms": 0,
                        "since_last_state_change_ms": 0,
                    }
                ),
                "process": state.process.model_copy(
                    update={
                        "status": ProcessStatus.EXITED,
                        "exit_code": -9,
                        "interrupted": True,
                        "resource_alert": False,
                    }
                ),
                "terminal": TerminalState(
                    stderr_delta="stopped hung sandbox process",
                    stderr_unread=True,
                    prompt_visible=True,
                    last_output_channel="stderr",
                    last_command=action.type.value,
                ),
            }
        )
        return ReflexCoreStepResult(
            state=next_state,
            proposed_action=action,
            safety_decision=safety,
            stderr="stopped hung sandbox process",
        )

    def _stop_active_process(
        self,
        state: SystemStateFrame,
        action: ActionDecision,
        safety: SafetyDecision,
    ) -> ReflexCoreStepResult:
        process = self._active_process
        if process is None:
            return self._stop_process(state, action, safety)
        stdout = ""
        stderr = ""
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=1.0)
        self._active_process = None
        self._active_command = None
        next_state = state.model_copy(
            update={
                "time": state.time.model_copy(
                    update={
                        "tick": state.time.tick + 1,
                        "since_last_output_ms": 0,
                        "since_last_state_change_ms": 0,
                    }
                ),
                "process": state.process.model_copy(
                    update={
                        "status": ProcessStatus.EXITED,
                        "exit_code": process.returncode,
                        "interrupted": True,
                        "resource_alert": False,
                    }
                ),
                "terminal": TerminalState(
                    stdout_delta=stdout,
                    stderr_delta=stderr or "stopped active sandbox process",
                    stdout_unread=bool(stdout),
                    stderr_unread=bool(stderr or "stopped active sandbox process"),
                    stdout_lines=len([line for line in stdout.splitlines() if line.strip()]),
                    stderr_lines=len(
                        [
                            line
                            for line in (stderr or "stopped active sandbox process").splitlines()
                            if line.strip()
                        ]
                    ),
                    prompt_visible=True,
                    last_output_channel="stderr",
                    last_command=action.type.value,
                ),
            }
        )
        return ReflexCoreStepResult(
            state=next_state,
            proposed_action=action,
            safety_decision=safety,
            stdout=stdout,
            stderr=stderr or "stopped active sandbox process",
        )

    def _advance_terminal_state(
        self,
        state: SystemStateFrame,
        *,
        stdout: str = "",
        stderr: str = "",
        action: ActionDecision,
        exit_code: int | None = None,
    ) -> SystemStateFrame:
        next_time = state.time.model_copy(
            update={
                "tick": state.time.tick + 1,
                "since_last_output_ms": 0 if stdout or stderr else state.time.since_last_output_ms,
                "since_last_state_change_ms": 0,
            }
        )
        next_process = state.process.model_copy(
            update={
                "status": ProcessStatus.EXITED,
                "exit_code": exit_code,
            }
        )
        next_terminal = TerminalState(
            stdout_delta=stdout,
            stderr_delta=stderr,
            stdout_unread=bool(stdout),
            stderr_unread=bool(stderr),
            stdout_lines=len([line for line in stdout.splitlines() if line.strip()]),
            stderr_lines=len([line for line in stderr.splitlines() if line.strip()]),
            prompt_visible=True,
            input_requested=False,
            last_output_channel="stderr" if stderr else ("stdout" if stdout else None),
            last_command=action.command or action.type.value,
        )
        return state.model_copy(
            update={
                "time": next_time,
                "process": next_process,
                "terminal": next_terminal,
            }
        )
