from __future__ import annotations

from dataclasses import dataclass
from reflexlm.models.features import (
    ACTION_ORDER,
    ROUTE_ORDER,
    candidate_commands,
    candidate_files,
)
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    InternalTarget,
    RouteName,
    SystemStateFrame,
    TaskType,
    validate_command_against_goal,
)


INTERNAL_TARGET_ORDER = list(InternalTarget)
BOUNDED_DEBUG_CORTEX_ACTIONS = {
    ActionType.WAIT,
    ActionType.READ_STDOUT,
    ActionType.READ_STDERR,
    ActionType.READ_FILE,
    ActionType.RUN_COMMAND,
    ActionType.REFRESH_STATE,
    ActionType.DONE,
}
PERSISTENT_FAILURE_RECOVERY_PROMOTION_ACTIONS = {
    ActionType.RUN_COMMAND,
    ActionType.WAIT,
    ActionType.DONE,
}


@dataclass(slots=True)
class SynapticMotorPlan:
    """Internal action-head result before runtime motor serialization."""

    internal_target: InternalTarget
    action_type: ActionType | None
    route_name: RouteName
    command_slot: int | None = None
    file_slot: int | None = None
    confidence: float = 1.0
    inhibited: bool = False
    reason: str | None = None
    raw_internal_target: InternalTarget | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "internal_target": self.internal_target.value,
            "action_type": self.action_type.value if self.action_type else None,
            "route_name": self.route_name.value,
            "command_slot": self.command_slot,
            "file_slot": self.file_slot,
            "confidence": round(self.confidence, 6),
            "inhibited": self.inhibited,
            "reason": self.reason,
            "raw_internal_target": (
                self.raw_internal_target.value if self.raw_internal_target else None
            ),
        }


def clamp_confidence(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def authorize_bounded_debug_cortex_action(
    action: ActionDecision,
    state: SystemStateFrame,
    *,
    confidence_threshold: float = 0.80,
) -> ActionDecision | None:
    """Authorize a learned recovery candidate without weakening hard safety gates."""
    if state.safety.dangerous_command_detected:
        return None
    if action.confidence < confidence_threshold:
        return None
    if action.type not in BOUNDED_DEBUG_CORTEX_ACTIONS:
        return None
    if action.type == ActionType.RUN_COMMAND:
        try:
            validate_command_against_goal(action, state.goal)
        except ValueError:
            return None
    if action.type == ActionType.READ_FILE and action.file_target not in candidate_files(state):
        return None
    return action


def persistent_failure_recovery_required(state: SystemStateFrame) -> bool:
    """Return whether a visible failed process still has an untried recovery."""
    commands = candidate_commands(state)
    return (
        state.process.exit_code not in (None, 0)
        and not state.terminal.stdout_unread
        and not state.terminal.stderr_unread
        and any(command != state.terminal.last_command for command in commands)
    )


def persistent_failure_recovery_should_promote(
    state: SystemStateFrame,
    proposed_action: ActionType,
) -> bool:
    """Wake an untried recovery only after observation/refresh actions are exhausted."""
    return (
        proposed_action in PERSISTENT_FAILURE_RECOVERY_PROMOTION_ACTIONS
        and persistent_failure_recovery_required(state)
    )


def authorize_persistent_failure_recovery(
    *,
    state: SystemStateFrame,
    command_slot: int,
    confidence: float,
    confidence_threshold: float = 0.65,
) -> ActionDecision | None:
    """Promote the learned command slot only when it differs from the failed command."""
    if not persistent_failure_recovery_required(state):
        return None
    commands = candidate_commands(state)
    if not commands:
        return None
    selected = commands[max(0, min(command_slot, len(commands) - 1))]
    if selected == state.terminal.last_command:
        return None
    return authorize_bounded_debug_cortex_action(
        ActionDecision(
            type=ActionType.RUN_COMMAND,
            command=selected,
            reason="persistent_failure_recovery_constraint",
            confidence=confidence,
        ),
        state,
        confidence_threshold=confidence_threshold,
    )


def internal_target_for_state(state: SystemStateFrame) -> InternalTarget:
    if state.safety.dangerous_command_detected:
        return InternalTarget.INHIBIT
    if state.goal.task_type == TaskType.TEST_FAILURE:
        return InternalTarget.ESCALATE_TO_DEBUG_CORTEX
    return InternalTarget.REFLEX_MOTOR


def route_for_internal_target(
    state: SystemStateFrame,
    target: InternalTarget,
) -> RouteName:
    if target == InternalTarget.ESCALATE_TO_DEBUG_CORTEX:
        return RouteName.DEBUG
    if state.safety.dangerous_command_detected:
        return RouteName.SAFETY
    if state.filesystem.external_change_detected or state.filesystem.stale_cache_detected:
        return RouteName.FILE
    if state.goal.task_type == TaskType.FILE_CHANGE:
        return RouteName.FILE
    if state.goal.task_type == TaskType.PROCESS_HANG:
        return RouteName.TERMINAL
    return RouteName.PLANNER


def plan_from_head_indices(
    *,
    state: SystemStateFrame,
    action_index: int,
    route_index: int,
    target_index: int | None = None,
    command_slot: int,
    file_slot: int,
    confidence: float,
    inhibition_score: float = 0.0,
) -> SynapticMotorPlan:
    raw_target = (
        INTERNAL_TARGET_ORDER[max(0, min(target_index, len(INTERNAL_TARGET_ORDER) - 1))]
        if target_index is not None
        else None
    )
    target = internal_target_for_state(state)
    if target == InternalTarget.INHIBIT:
        return SynapticMotorPlan(
            internal_target=InternalTarget.INHIBIT,
            action_type=ActionType.BLOCK,
            route_name=RouteName.SAFETY,
            confidence=clamp_confidence(max(confidence, inhibition_score)),
            inhibited=True,
            reason="safety_inhibition",
            raw_internal_target=raw_target,
        )
    if state.filesystem.external_change_detected or state.filesystem.stale_cache_detected:
        return SynapticMotorPlan(
            internal_target=InternalTarget.REFLEX_MOTOR,
            action_type=ActionType.REFRESH_STATE,
            route_name=RouteName.FILE,
            confidence=clamp_confidence(confidence),
            reason="stale_state_refresh_receptor",
            raw_internal_target=raw_target,
        )
    if state.goal.task_type == TaskType.FILE_CHANGE and state.filesystem.dirty_files:
        return SynapticMotorPlan(
            internal_target=InternalTarget.REFLEX_MOTOR,
            action_type=ActionType.READ_FILE,
            route_name=RouteName.FILE,
            file_slot=0,
            confidence=clamp_confidence(confidence),
            reason="pending_file_read_receptor",
            raw_internal_target=raw_target,
        )
    if state.terminal.stderr_unread:
        return SynapticMotorPlan(
            internal_target=InternalTarget.REFLEX_MOTOR,
            action_type=ActionType.READ_STDERR,
            route_name=RouteName.TERMINAL,
            confidence=clamp_confidence(confidence),
            reason="pending_stderr_receptor",
            raw_internal_target=raw_target,
        )
    if state.terminal.stdout_unread:
        return SynapticMotorPlan(
            internal_target=InternalTarget.REFLEX_MOTOR,
            action_type=ActionType.READ_STDOUT,
            route_name=RouteName.TERMINAL,
            confidence=clamp_confidence(confidence),
            reason="pending_stdout_receptor",
            raw_internal_target=raw_target,
        )
    if target == InternalTarget.ESCALATE_TO_DEBUG_CORTEX:
        return SynapticMotorPlan(
            internal_target=target,
            action_type=None,
            route_name=RouteName.DEBUG,
            confidence=clamp_confidence(confidence),
            reason="semantic_debug_requires_cortex",
            raw_internal_target=raw_target,
        )
    action_type = ACTION_ORDER[max(0, min(action_index, len(ACTION_ORDER) - 1))]
    route_name = ROUTE_ORDER[max(0, min(route_index, len(ROUTE_ORDER) - 1))]
    return SynapticMotorPlan(
        internal_target=InternalTarget.REFLEX_MOTOR,
        action_type=action_type,
        route_name=route_name,
        command_slot=max(0, command_slot),
        file_slot=max(0, file_slot),
        confidence=clamp_confidence(confidence),
        raw_internal_target=raw_target,
    )


def serialize_motor_action(
    plan: SynapticMotorPlan,
    state: SystemStateFrame,
    *,
    cortex_action: ActionDecision | None = None,
) -> ActionDecision:
    """Materialize one bounded motor action from internal heads.

    Escalation is not serialized as an external action. A Debug/Semantic Cortex
    must provide its own head-selected action; otherwise runtime emits a safe
    wait marker rather than asking the LLM to produce JSON text.
    """

    if plan.internal_target in {
        InternalTarget.ESCALATE_TO_DEBUG_CORTEX,
        InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX,
    }:
        if cortex_action is not None:
            return cortex_action
        return ActionDecision(
            type=ActionType.WAIT,
            reason=(
                plan.reason
                if str(plan.reason or "").startswith("homeostatic_")
                else plan.internal_target.value.lower()
            ),
            confidence=plan.confidence,
        )
    action_type = plan.action_type or ActionType.WAIT
    command = None
    file_target = None
    if action_type == ActionType.RUN_COMMAND:
        commands = candidate_commands(state)
        if not commands:
            return ActionDecision(
                type=ActionType.WAIT,
                reason="no_command_slot_available",
                confidence=plan.confidence,
            )
        command_index = plan.command_slot or 0
        command = commands[max(0, min(command_index, len(commands) - 1))]
    if action_type == ActionType.READ_FILE:
        files = candidate_files(state)
        if not files:
            return ActionDecision(
                type=ActionType.WAIT,
                reason="no_file_slot_available",
                confidence=plan.confidence,
            )
        file_index = plan.file_slot or 0
        file_target = files[max(0, min(file_index, len(files) - 1))]
    return ActionDecision(
        type=action_type,
        command=command,
        file_target=file_target,
        reason=plan.reason,
        confidence=plan.confidence,
    )
