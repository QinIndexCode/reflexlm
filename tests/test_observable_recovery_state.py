from reflexlm.data.tasks import build_env
from reflexlm.schema import ActionType, TaskType


def test_routine_recovery_preserves_observable_error_after_read_stderr() -> None:
    env = build_env(TaskType.ROUTINE_RECOVERY, 3, profile="wide_ood")
    state = env.reset()

    assert state.goal.recovery_hint == "permission_denied"
    assert "PermissionError" in state.terminal.stderr_delta

    next_state, _reward, done, _info = env.step(env.oracle_action(state))

    assert done is False
    assert next_state.terminal.stderr_delta == ""
    assert "Parsed recovery error: PermissionError" in next_state.terminal.stdout_delta
    assert env.oracle_action(next_state).type == ActionType.ASK_USER
