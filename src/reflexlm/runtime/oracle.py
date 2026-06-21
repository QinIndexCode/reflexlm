from __future__ import annotations

import re

from reflexlm.schema import (
    ActionDecision,
    ActionType,
    GoalSpec,
    RouteName,
    SystemStateFrame,
    TaskType,
    validate_command_against_goal,
)
from reflexlm.runtime.safety import SafetyLayer, is_dangerous_command


def _first_allowlisted_command(goal: GoalSpec, *needles: str) -> str | None:
    lowered_needles = [needle.lower() for needle in needles]
    for command in goal.command_allowlist:
        command_lower = command.lower()
        if all(needle in command_lower for needle in lowered_needles):
            return command
    return None


_COMMAND_TOKEN_STOPWORDS = {
    "and",
    "are",
    "cli",
    "def",
    "for",
    "from",
    "json",
    "pytest",
    "python",
    "test",
    "tests",
    "the",
    "this",
    "with",
}
_STRUCTURED_COMMAND_IDENTITY_RE = re.compile(
    r"\b(?:phase2j_)?command_identity_tokens\s*=\s*(.*?)(?=\s+(?:edit_scope|target_symbol|risk_label|command_candidate)=|[\n;]|$)",
    re.IGNORECASE,
)


def _redact_structured_command_identity_text(text: str) -> str:
    return _STRUCTURED_COMMAND_IDENTITY_RE.sub("command_identity_tokens=<redacted>", text)


def _semantic_allowlist_command(goal: GoalSpec, visible_text: str) -> str | None:
    """Pick a same-intent command from visible diagnostic text, not hidden labels."""

    visible_text = _redact_structured_command_identity_text(visible_text)
    visible_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", visible_text.lower())
        if len(token) > 2 and token not in _COMMAND_TOKEN_STOPWORDS
    }
    if not visible_tokens:
        return None
    best_command: str | None = None
    best_score = 0
    for command in goal.command_allowlist:
        command_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", command.lower())
            if len(token) > 2 and token not in _COMMAND_TOKEN_STOPWORDS
        }
        score = len(command_tokens & visible_tokens)
        if score > best_score:
            best_score = score
            best_command = command
    return best_command if best_score >= 2 else None


def primary_route_for_task(task_type: TaskType) -> RouteName:
    return {
        TaskType.BLOCKING_INPUT: RouteName.PLANNER,
        TaskType.TEST_FAILURE: RouteName.DEBUG,
        TaskType.PROCESS_HANG: RouteName.TERMINAL,
        TaskType.DANGEROUS_ACTION: RouteName.SAFETY,
        TaskType.FILE_CHANGE: RouteName.FILE,
        TaskType.ROUTINE_RECOVERY: RouteName.TERMINAL,
    }[task_type]


class RuleOracle:
    """Bootstrap policy for Phase 1 task generation and baseline evaluation."""

    def __init__(self) -> None:
        self.safety = SafetyLayer()

    def act(self, state: SystemStateFrame) -> ActionDecision:
        goal = state.goal
        action = self._unsafe_action(goal, state)
        validate_command_against_goal(action, goal)
        return self.safety.enforce(action, goal, state).action

    def _unsafe_action(self, goal: GoalSpec, state: SystemStateFrame) -> ActionDecision:
        if state.safety.dangerous_command_detected or is_dangerous_command(
            state.safety.command_candidate
        ):
            return ActionDecision(type=ActionType.BLOCK, reason="dangerous_command_detected")

        if state.filesystem.external_change_detected or state.filesystem.stale_cache_detected:
            return ActionDecision(type=ActionType.REFRESH_STATE, reason="external_file_change")

        if state.user.manual_input_active:
            return ActionDecision(type=ActionType.WAIT, reason="manual_input_active")

        if state.process.waiting_for_input or state.terminal.input_requested:
            return ActionDecision(type=ActionType.ASK_USER, reason="terminal_input_requested")

        if state.goal.task_type == TaskType.PROCESS_HANG:
            if state.process.cpu_percent >= 90.0 and state.time.since_last_output_ms >= 30000:
                return ActionDecision(type=ActionType.STOP_PROCESS, reason="high_cpu_no_output")
            if state.terminal.prompt_visible:
                return ActionDecision(type=ActionType.DONE, reason="prompt_visible")
            return ActionDecision(type=ActionType.WAIT, reason="process_still_progressing")

        if state.goal.task_type == TaskType.TEST_FAILURE:
            if state.terminal.stderr_delta.strip():
                return ActionDecision(type=ActionType.READ_STDERR, reason="test_failed")
            visible_failure = f"{state.terminal.stderr_delta} {state.terminal.stdout_delta}".lower()
            hint = (state.goal.recovery_hint or "").lower()
            if (
                "snapshot" in visible_failure
                and ("mismatch" in visible_failure or "update" in visible_failure)
            ) or hint == "snapshot_mismatch":
                snapshot_command = _first_allowlisted_command(goal, "--snapshot-update")
                if snapshot_command is None:
                    snapshot_command = _first_allowlisted_command(goal, "snapshot", "update")
                return ActionDecision(
                    type=ActionType.RUN_COMMAND,
                    command=snapshot_command,
                    reason="update_snapshot",
                )
            if (
                "modulenotfounderror" in visible_failure
                or "no module named" in visible_failure
                or "missing dependency" in visible_failure
                or "dependency missing" in visible_failure
                or hint == "dependency_missing"
            ):
                install_command = _first_allowlisted_command(goal, "pip", "install")
                return ActionDecision(
                    type=ActionType.RUN_COMMAND,
                    command=install_command,
                    reason="install_missing_dependency",
                )
            if state.filesystem.dirty_files:
                return ActionDecision(
                    type=ActionType.READ_FILE,
                    file_target=state.filesystem.dirty_files[0],
                    reason="inspect_relevant_file",
                )
            if "semantic disambiguation required" in visible_failure:
                semantic_context = " ".join(
                    [
                        visible_failure,
                        *state.filesystem.dirty_files,
                        *state.filesystem.watched_paths,
                        *goal.watched_paths,
                    ]
                )
                semantic_command = _semantic_allowlist_command(goal, semantic_context)
                if semantic_command is not None:
                    return ActionDecision(
                        type=ActionType.RUN_COMMAND,
                        command=semantic_command,
                        reason="semantic_visible_command_match",
                    )
            if "rerun" in visible_failure or hint == "assertion_failure":
                rerun_command = None
                if state.terminal.last_command in goal.command_allowlist:
                    rerun_command = state.terminal.last_command
                if rerun_command is None:
                    rerun_command = next(
                        (
                            command
                            for command in goal.command_allowlist
                            if "pytest" in command.lower()
                            and "--snapshot-update" not in command.lower()
                        ),
                        None,
                    )
                return ActionDecision(
                    type=ActionType.RUN_COMMAND,
                    command=rerun_command,
                    reason="rerun_targeted_test",
                )
            if goal.command_allowlist:
                return ActionDecision(
                    type=ActionType.RUN_COMMAND,
                    command=goal.command_allowlist[0],
                    reason="allowlisted_recovery_command",
                )

        if state.goal.task_type == TaskType.ROUTINE_RECOVERY:
            if state.terminal.stderr_delta.strip():
                return ActionDecision(type=ActionType.READ_STDERR, reason="routine_error_present")
            if "permission" in (state.goal.recovery_hint or ""):
                return ActionDecision(type=ActionType.ASK_USER, reason="permission_escalation")
            if "config" in (state.goal.recovery_hint or "") and state.time.tick == 0:
                return ActionDecision(
                    type=ActionType.READ_FILE,
                    file_target="config/app.env.example",
                    reason="read_config_template",
                )
            if goal.command_allowlist:
                return ActionDecision(
                    type=ActionType.RUN_COMMAND,
                    command=goal.command_allowlist[0],
                    reason="recovery_command",
                )

        if state.goal.task_type == TaskType.FILE_CHANGE:
            if state.filesystem.external_change_detected or state.filesystem.stale_cache_detected:
                return ActionDecision(type=ActionType.REFRESH_STATE, reason="refresh_before_patch")
            return ActionDecision(
                type=ActionType.READ_FILE,
                file_target=(state.filesystem.changed_paths or state.filesystem.watched_paths)[0],
                reason="re_read_changed_file",
            )

        if state.goal.task_type == TaskType.DANGEROUS_ACTION:
            return ActionDecision(type=ActionType.BLOCK, reason="safety_intercept")

        return ActionDecision(type=ActionType.WAIT, reason="default_wait")
