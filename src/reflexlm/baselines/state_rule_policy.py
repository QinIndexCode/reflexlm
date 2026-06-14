from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reflexlm.models.features import candidate_commands, candidate_files
from reflexlm.schema import ActionDecision, ActionType, SystemStateFrame


@dataclass(slots=True)
class StateRulePolicyStats:
    token_cost: int = 0
    model_calls: int = 0
    parse_failures: int = 0
    retries: int = 0


class BoundedStateRulePolicy:
    """Strong state-only rule baseline without manifest sequence access."""

    def __init__(self, *, policy_label: str = "bounded_state_rule") -> None:
        self.policy_label = policy_label
        self.stats = StateRulePolicyStats()
        self.last_call: dict[str, Any] = {}

    def reset(self) -> None:
        self.stats = StateRulePolicyStats()
        self.last_call = {}

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "bounded_state_rule",
            "policy_label": self.policy_label,
            "uses_manifest_action_sequence": False,
            "uses_model": False,
        }

    def act(self, state: SystemStateFrame) -> ActionDecision:
        action = self._select_action(state)
        self.last_call = {
            "action_source": "visible_persistent_state_rules",
            "selected_reason": action.reason,
        }
        return action

    def _select_action(self, state: SystemStateFrame) -> ActionDecision:
        if state.filesystem.external_change_detected or state.filesystem.stale_cache_detected:
            return ActionDecision(
                type=ActionType.REFRESH_STATE,
                reason="rule_refresh_changed_state",
                confidence=1.0,
            )
        files = candidate_files(state)
        if state.filesystem.dirty_files and files:
            return ActionDecision(
                type=ActionType.READ_FILE,
                file_target=files[0],
                reason="rule_read_dirty_file",
                confidence=1.0,
            )
        if state.terminal.stderr_unread:
            return ActionDecision(
                type=ActionType.READ_STDERR,
                reason="rule_read_unread_stderr",
                confidence=1.0,
            )
        if state.terminal.stdout_unread:
            return ActionDecision(
                type=ActionType.READ_STDOUT,
                reason="rule_read_unread_stdout",
                confidence=1.0,
            )
        commands = candidate_commands(state)
        if state.process.exit_code not in (None, 0):
            recovery = next(
                (command for command in commands if command != state.terminal.last_command),
                None,
            )
            if recovery is not None:
                return ActionDecision(
                    type=ActionType.RUN_COMMAND,
                    command=recovery,
                    reason="rule_run_distinct_allowlisted_recovery",
                    confidence=1.0,
                )
        if state.terminal.last_command is None and commands:
            return ActionDecision(
                type=ActionType.RUN_COMMAND,
                command=commands[0],
                reason="rule_run_initial_allowlisted_command",
                confidence=1.0,
            )
        return ActionDecision(
            type=ActionType.DONE,
            reason="rule_no_pending_visible_state",
            confidence=1.0,
        )
