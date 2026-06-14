from reflexlm.schema import (
    ActionDecision,
    ActionType,
    FileSystemState,
    GoalSpec,
    ProcessState,
    ProcessStatus,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
    TrajectoryRecord,
    UserState,
    validate_command_against_goal,
)


def make_state(goal: GoalSpec) -> SystemStateFrame:
    return SystemStateFrame(
        time=TimeState(tick=0),
        goal=goal,
        process=ProcessState(status=ProcessStatus.RUNNING),
        terminal=TerminalState(),
        filesystem=FileSystemState(),
        user=UserState(),
    )


def test_allowlisted_run_command_validates() -> None:
    goal = GoalSpec(
        task_type=TaskType.TEST_FAILURE,
        description="test",
        command_allowlist=["python -m pytest -q tests/test_auth.py"],
    )
    action = ActionDecision(
        type=ActionType.RUN_COMMAND,
        command="python -m pytest -q tests/test_auth.py",
    )
    validate_command_against_goal(action, goal)


def test_run_command_without_payload_is_rejected() -> None:
    try:
        ActionDecision(type=ActionType.RUN_COMMAND)
    except ValueError as exc:
        assert "command payload" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("RUN_COMMAND without payload should fail")


def test_trajectory_goal_consistency() -> None:
    goal = GoalSpec(task_type=TaskType.FILE_CHANGE, description="refresh stale file")
    state = make_state(goal)
    record = TrajectoryRecord(
        episode_id="e1",
        t=0,
        goal=goal,
        state=state,
        action=ActionDecision(type=ActionType.REFRESH_STATE),
        next_state=state,
        reward=1.0,
        done=True,
        source="synthetic",
    )
    assert record.goal.task_type == TaskType.FILE_CHANGE

