from reflexlm.eval import EventGatedSequencePolicy
from reflexlm.models.features import command_failure_match_scores
from reflexlm.schema import (
    ActionType,
    FileSystemState,
    GoalSpec,
    ProcessState,
    RuntimeEvidenceState,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


def _policy() -> EventGatedSequencePolicy:
    policy = EventGatedSequencePolicy.__new__(EventGatedSequencePolicy)
    return policy


def test_event_gate_reads_unread_stderr_without_neural_call() -> None:
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(task_type=TaskType.TEST_FAILURE, description="debug"),
        process=ProcessState(exit_code=17),
        terminal=TerminalState(stderr_unread=True),
        filesystem=FileSystemState(),
    )

    action = _policy()._deterministic_receptor_action(state)

    assert action is not None
    assert action.type == ActionType.READ_STDERR


def test_event_gate_leaves_failure_recovery_to_neural_policy() -> None:
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="recover",
            command_allowlist=["python fail.py", "python recover.py"],
        ),
        process=ProcessState(exit_code=17),
        terminal=TerminalState(last_command="python fail.py"),
        filesystem=FileSystemState(),
    )

    assert _policy()._deterministic_receptor_action(state) is None


def test_failure_match_scores_prefer_assertion_pytest_over_snapshot_update() -> None:
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="AssertionError: assertion failure in bounded runtime",
            command_allowlist=[
                "python -c \"print('snapshot update recovery')\"",
                "python -c \"print('pip install dependency recovery')\"",
                "python -c \"print('pytest assertion recovery')\"",
            ],
        ),
        process=ProcessState(exit_code=1),
        terminal=TerminalState(),
        filesystem=FileSystemState(),
    )

    assert command_failure_match_scores(state) == [0.0, 0.0, 1.0]


def test_failure_match_scores_use_runtime_terminal_observation_memory() -> None:
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="recover",
            command_allowlist=[
                "python -c \"print('repair file permissions')\"",
                "python -c \"print('release occupied TCP port')\"",
            ],
        ),
        process=ProcessState(exit_code=1),
        terminal=TerminalState(),
        filesystem=FileSystemState(),
        runtime_evidence=RuntimeEvidenceState(
            terminal_observations=[
                "Address already in use on TCP port 8123",
            ],
        ),
    )

    assert command_failure_match_scores(state) == [0.0, 1.0]
