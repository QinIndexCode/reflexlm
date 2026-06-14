from __future__ import annotations

from dataclasses import dataclass

from reflexlm.schema import ActionDecision, ActionType, GoalSpec, SystemStateFrame

DANGEROUS_COMMAND_PATTERNS = (
    "rm -rf",
    "del /f /s /q",
    "format ",
    "mkfs",
    "shutdown",
    "reboot",
)


def is_dangerous_command(command: str | None) -> bool:
    if not command:
        return False
    lowered = command.lower()
    return any(pattern in lowered for pattern in DANGEROUS_COMMAND_PATTERNS)


@dataclass(slots=True)
class SafetyDecision:
    allowed: bool
    action: ActionDecision
    reason: str


class SafetyLayer:
    """Phase 1 safety layer that blocks dangerous actions before execution."""

    def enforce(
        self,
        proposed: ActionDecision,
        goal: GoalSpec,
        state: SystemStateFrame,
    ) -> SafetyDecision:
        if proposed.type == ActionType.RUN_COMMAND and proposed.command:
            if proposed.command not in goal.command_allowlist:
                blocked = ActionDecision(
                    type=ActionType.BLOCK,
                    reason="command_not_allowlisted",
                    confidence=1.0,
                    notes=[proposed.command],
                )
                return SafetyDecision(False, blocked, "command_not_allowlisted")
            if is_dangerous_command(proposed.command):
                blocked = ActionDecision(
                    type=ActionType.BLOCK,
                    reason="dangerous_command_detected",
                    confidence=1.0,
                    notes=[proposed.command],
                )
                return SafetyDecision(False, blocked, "dangerous_command_detected")
        if state.user.manual_input_active and proposed.type not in (
            ActionType.WAIT,
            ActionType.ASK_USER,
            ActionType.BLOCK,
        ):
            held = ActionDecision(
                type=ActionType.WAIT,
                reason="manual_input_active",
                confidence=1.0,
            )
            return SafetyDecision(False, held, "manual_input_active")
        return SafetyDecision(True, proposed, "allowed")

