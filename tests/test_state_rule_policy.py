from reflexlm.baselines.state_rule_policy import BoundedStateRulePolicy
from reflexlm.schema import (
    ActionType,
    FileSystemState,
    GoalSpec,
    ProcessState,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


def test_bounded_state_rule_uses_distinct_allowlisted_recovery() -> None:
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="recover visible failure",
            command_allowlist=["python fail.py", "python recover.py"],
        ),
        process=ProcessState(exit_code=17),
        terminal=TerminalState(last_command="python fail.py"),
        filesystem=FileSystemState(),
    )

    action = BoundedStateRulePolicy().act(state)

    assert action.type == ActionType.RUN_COMMAND
    assert action.command == "python recover.py"


def test_bounded_state_rule_prioritizes_unread_terminal_channels() -> None:
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="inspect visible failure",
            command_allowlist=["python fail.py", "python recover.py"],
        ),
        process=ProcessState(exit_code=17),
        terminal=TerminalState(
            last_command="python fail.py",
            stdout_unread=True,
            stderr_unread=True,
        ),
        filesystem=FileSystemState(),
    )

    action = BoundedStateRulePolicy().act(state)

    assert action.type == ActionType.READ_STDERR
