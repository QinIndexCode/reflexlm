from reflexlm.models.features import StateVectorizer
from reflexlm.schema import (
    FileSystemState,
    GoalSpec,
    ProcessState,
    ProcessStatus,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


def test_state_vectorizer_handles_optional_last_command_with_candidate_slots() -> None:
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="Select a bounded candidate.",
            command_allowlist=[
                "python -m pytest -q tests/test_a.py",
                "python -m pytest -q tests/test_b.py",
            ],
            watched_paths=["src/a.py", "src/b.py"],
        ),
        process=ProcessState(status=ProcessStatus.EXITED, exit_code=1),
        terminal=TerminalState(
            stderr_delta="AssertionError: missing symbol",
            last_command=None,
        ),
        filesystem=FileSystemState(
            watched_paths=["src/a.py", "src/b.py"],
            changed_paths=["src/a.py"],
            dirty_files=["src/a.py"],
        ),
    )

    vector = StateVectorizer().vectorize_state(state)

    assert vector.shape[0] > 0
