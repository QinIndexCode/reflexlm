from __future__ import annotations

from dataclasses import dataclass

import torch

from reflexlm.models.features import (
    candidate_commands,
    candidate_files,
    resolve_structured_action,
    valid_action_mask,
)
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    InternalTarget,
    ProcessStatus,
    RouteName,
    SystemStateFrame,
    TaskType,
)


@dataclass(slots=True)
class ReflexCoreDecodedMotor:
    action: ActionDecision
    route_name: RouteName | None
    internal_target: InternalTarget | None
    risk: float
    salience: float
    prediction_error: float


@dataclass(slots=True)
class ReflexCoreMotorConfig:
    """Homeostatic decoder thresholds for V0 motor heads.

    These thresholds make learned risk/salience/prediction-error heads
    operational without bypassing the existing safety layer. The values are
    intentionally conservative so an untrained model mostly follows its action
    head, while a high-risk motor state can still self-block before execution.
    """

    risk_block_threshold: float = 0.85
    prediction_error_refresh_threshold: float = 0.05
    observed_prediction_error_refresh_threshold: float = 0.05
    salience_refresh_threshold: float = 0.75
    low_confidence_wait_threshold: float = 0.0
    low_salience_wait_threshold: float = 0.25


def decode_reflexcore_motor(
    outputs: dict[str, torch.Tensor | None],
    state: SystemStateFrame,
    *,
    config: ReflexCoreMotorConfig | None = None,
) -> ReflexCoreDecodedMotor:
    """Decode V0 typed heads without legacy reflex overrides.

    Safety and allowlist enforcement happen after this function. This keeps V0
    evaluation focused on the learned action/slot heads instead of older
    receptor-rule promotions from the NSI runtime. The optional homeostatic
    layer only uses ReflexCore's own risk/salience/prediction-error heads.
    """

    config = config or ReflexCoreMotorConfig()
    action_logits = _tensor(outputs, "action_logits")
    command_logits = _tensor(outputs, "command_slot_logits")
    file_logits = _tensor(outputs, "file_slot_logits")
    action_mask = torch.tensor(
        valid_action_mask(state),
        dtype=action_logits.dtype,
        device=action_logits.device,
    )
    masked_action_logits = action_logits[0, -1].masked_fill(action_mask <= 0.0, -1.0e9)
    action_scores = masked_action_logits.softmax(dim=-1)
    confidence = float(action_scores.max().item())
    action = resolve_structured_action(
        action_index=int(action_scores.argmax().item()),
        command_index=int(command_logits[0, -1].argmax().item()),
        file_index=int(file_logits[0, -1].argmax().item()),
        state=state,
        confidence=confidence,
    )
    action = _apply_state_affordance_motor_control(action, state, confidence=confidence)
    route_name = None
    target = None
    route_logits = outputs.get("route_logits")
    target_logits = outputs.get("target_logits")
    if isinstance(route_logits, torch.Tensor):
        route_index = int(route_logits[0, -1].argmax().item())
        routes = list(RouteName)
        route_name = routes[max(0, min(route_index, len(routes) - 1))]
    if isinstance(target_logits, torch.Tensor):
        target_index = int(target_logits[0, -1].argmax().item())
        targets = list(InternalTarget)
        target = targets[max(0, min(target_index, len(targets) - 1))]
    risk = _scalar_output(outputs, "risk")
    salience = _scalar_output(outputs, "salience")
    prediction_error = _scalar_output(outputs, "prediction_error")
    action = _apply_homeostatic_motor_control(
        action,
        risk=risk,
        salience=salience,
        prediction_error=prediction_error,
        observed_prediction_error=state.runtime_evidence.observed_prediction_error,
        goal_task_type=state.goal.task_type,
        process_active=state.process.status
        in {ProcessStatus.RUNNING, ProcessStatus.SLEEPING, ProcessStatus.BLOCKED},
        confidence=confidence,
        config=config,
    )
    return ReflexCoreDecodedMotor(
        action=action,
        route_name=route_name,
        internal_target=target,
        risk=risk,
        salience=salience,
        prediction_error=prediction_error,
    )


def _apply_state_affordance_motor_control(
    action: ActionDecision,
    state: SystemStateFrame,
    *,
    confidence: float,
) -> ActionDecision:
    """Prevent idle/terminal actions from overriding visible motor affordances."""

    files = candidate_files(state)
    pending_file_read = bool(state.filesystem.dirty_files or state.filesystem.changed_paths)
    refresh_signal_visible = bool(
        state.filesystem.external_change_detected
        or state.filesystem.stale_cache_detected
        or state.filesystem.conflict_detected
    )
    stdout_visible = bool(state.terminal.stdout_unread and state.terminal.stdout_delta)
    stderr_visible = bool(state.terminal.stderr_unread and state.terminal.stderr_delta)
    commands = candidate_commands(state)
    process_active = state.process.status in {
        ProcessStatus.RUNNING,
        ProcessStatus.SLEEPING,
        ProcessStatus.BLOCKED,
    }
    if (
        action.type
        in {ActionType.READ_STDOUT, ActionType.REFRESH_STATE, ActionType.WAIT, ActionType.DONE}
        and state.terminal.last_command == ActionType.READ_FILE.value
        and len(commands) == 1
        and not pending_file_read
        and not refresh_signal_visible
        and not stderr_visible
        and not process_active
    ):
        return ActionDecision(
            type=ActionType.RUN_COMMAND,
            command=commands[0],
            reason="reflexcore_file_read_complete_command_affordance",
            confidence=confidence,
        )
    if (
        action.type in {ActionType.WAIT, ActionType.DONE}
        and refresh_signal_visible
        and pending_file_read
    ):
        return ActionDecision(
            type=ActionType.REFRESH_STATE,
            reason="reflexcore_visible_refresh_affordance",
            confidence=confidence,
        )
    if action.type in {ActionType.WAIT, ActionType.DONE} and pending_file_read and files:
        return ActionDecision(
            type=ActionType.READ_FILE,
            file_target=files[0],
            reason="reflexcore_visible_file_affordance",
            confidence=confidence,
        )
    if action.type == ActionType.DONE:
        if process_active:
            return ActionDecision(
                type=ActionType.WAIT,
                reason="reflexcore_done_blocked_by_active_process",
                confidence=confidence,
            )
        if stdout_visible:
            return ActionDecision(
                type=ActionType.READ_STDOUT,
                reason="reflexcore_done_blocked_by_stdout",
                confidence=confidence,
            )
        if stderr_visible:
            return ActionDecision(
                type=ActionType.READ_STDERR,
                reason="reflexcore_done_blocked_by_stderr",
                confidence=confidence,
            )
        if refresh_signal_visible:
            return ActionDecision(
                type=ActionType.REFRESH_STATE,
                reason="reflexcore_done_blocked_by_refresh_signal",
                confidence=confidence,
            )
    if (
        action.type in {ActionType.RUN_COMMAND, ActionType.WAIT}
        and state.terminal.last_command in {ActionType.READ_STDOUT.value, ActionType.READ_STDERR.value}
        and bool(state.terminal.stdout_delta or state.terminal.stderr_delta)
        and not stdout_visible
        and not stderr_visible
        and not pending_file_read
        and not refresh_signal_visible
        and not process_active
    ):
        return ActionDecision(
            type=ActionType.DONE,
            reason="reflexcore_terminal_output_already_observed",
            confidence=confidence,
        )
    return action


def _apply_homeostatic_motor_control(
    action: ActionDecision,
    *,
    risk: float,
    salience: float,
    prediction_error: float,
    observed_prediction_error: float | None,
    goal_task_type: TaskType,
    process_active: bool,
    confidence: float,
    config: ReflexCoreMotorConfig,
) -> ActionDecision:
    if (
        action.type in {ActionType.RUN_COMMAND, ActionType.STOP_PROCESS}
        and risk >= config.risk_block_threshold
    ):
        return ActionDecision(
            type=ActionType.BLOCK,
            reason="reflexcore_risk_threshold",
            confidence=max(confidence, risk),
            notes=[action.command or action.type.value],
        )
    if (
        action.type in {ActionType.WAIT, ActionType.DONE}
        and not process_active
        and goal_task_type != TaskType.PROCESS_HANG
        and salience >= config.salience_refresh_threshold
        and prediction_error >= config.prediction_error_refresh_threshold
    ):
        return ActionDecision(
            type=ActionType.REFRESH_STATE,
            reason="reflexcore_prediction_error_refresh",
            confidence=max(confidence, salience),
        )
    if (
        action.type in {ActionType.WAIT, ActionType.DONE}
        and not process_active
        and goal_task_type != TaskType.PROCESS_HANG
        and observed_prediction_error is not None
        and observed_prediction_error
        >= config.observed_prediction_error_refresh_threshold
    ):
        return ActionDecision(
            type=ActionType.REFRESH_STATE,
            reason="reflexcore_observed_prediction_error_refresh",
            confidence=max(confidence, min(float(observed_prediction_error), 1.0)),
        )
    if (
        confidence < config.low_confidence_wait_threshold
        and salience <= config.low_salience_wait_threshold
    ):
        return ActionDecision(
            type=ActionType.WAIT,
            reason="reflexcore_low_confidence_wait",
            confidence=confidence,
        )
    return action


def _tensor(outputs: dict[str, torch.Tensor | None], key: str) -> torch.Tensor:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"missing motor output: {key}")
    return value


def _scalar_output(outputs: dict[str, torch.Tensor | None], key: str) -> float:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor):
        return 0.0
    return float(value[0, -1].reshape(-1)[0].item())
