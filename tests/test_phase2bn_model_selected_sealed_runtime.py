import pytest

from reflexlm.cli.run_phase2bn_model_selected_sealed_runtime import (
    _apply_initial_state_overrides,
    _allowed_step_map,
    _completion_actions_satisfied,
    _episode_contract,
    _failure_recovery_gate_status,
    _required_completion_actions,
    _task_for_selected_action,
)
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    GoalSpec,
    ProcessState,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


def test_phase2bn_rejects_model_action_outside_sealed_allowlist() -> None:
    allowed = _allowed_step_map(
        [
            {
                "action_type": "RUN_COMMAND",
                "argv": ["<PYTHON>", "-c", "print('allowed')"],
            },
            {"action_type": "DONE"},
        ]
    )

    with pytest.raises(ValueError, match="outside the sealed manifest allowlist"):
        _task_for_selected_action(
            ActionDecision(
                type=ActionType.RUN_COMMAND,
                command="python -c \"print('unknown')\"",
            ),
            allowed_steps=allowed,
            episode_id="sealed",
        )


def test_phase2bn_maps_exact_model_selected_action_to_manifest_task() -> None:
    steps = [{"action_type": "WAIT", "wait_ms": 17}, {"action_type": "DONE"}]
    allowed = _allowed_step_map(steps)

    task = _task_for_selected_action(
        ActionDecision(type=ActionType.WAIT),
        allowed_steps=allowed,
        episode_id="sealed",
    )

    assert task["episode_id"] == "sealed"
    assert task["action_type"] == "WAIT"
    assert task["wait_ms"] == 17


def test_phase2bn_allows_only_explicit_read_only_ambient_receptors() -> None:
    allowed = _allowed_step_map(
        [{"action_type": "DONE"}],
        ambient_observation_actions=["READ_STDOUT"],
    )
    task = _task_for_selected_action(
        ActionDecision(type=ActionType.READ_STDOUT),
        allowed_steps=allowed,
        episode_id="sealed",
    )

    assert task["_phase2bn_ambient_observation"] is True

    with pytest.raises(ValueError, match="read-only terminal receptors"):
        _allowed_step_map(
            [{"action_type": "DONE"}],
            ambient_observation_actions=["RUN_COMMAND"],
        )


def test_phase2bn_completion_requires_observations_but_not_control_timing() -> None:
    steps = [
        {"action_type": "RUN_COMMAND", "argv": ["<PYTHON>", "-c", "print('x')"]},
        {"action_type": "READ_STDOUT"},
        {"action_type": "WAIT", "wait_ms": 17},
        {"action_type": "REFRESH_STATE"},
        {"action_type": "DONE"},
    ]
    required = _required_completion_actions(steps)
    executed = required.copy()

    assert sum(required.values()) == 2
    assert _completion_actions_satisfied(required, executed) is True

    executed.subtract([("READ_STDOUT", None, None)])
    assert _completion_actions_satisfied(required, executed) is False


def test_phase2bn_failure_recovery_gate_is_not_applicable_without_failures() -> None:
    passed, applicable, success_rate = _failure_recovery_gate_status(
        failure_episode_count=0,
        failure_recovery_count=0,
    )

    assert passed is True
    assert applicable is False
    assert success_rate == 1.0


def test_phase2bn_failure_recovery_gate_enforces_real_failure_rows() -> None:
    passed, applicable, success_rate = _failure_recovery_gate_status(
        failure_episode_count=5,
        failure_recovery_count=4,
    )

    assert passed is True
    assert applicable is True
    assert success_rate == 0.8

    passed, applicable, success_rate = _failure_recovery_gate_status(
        failure_episode_count=5,
        failure_recovery_count=3,
    )
    assert passed is False
    assert applicable is True
    assert success_rate == 0.6


def test_phase2bn_contract_separates_permissions_from_completion_and_sequence() -> None:
    permissions, completion, expected, requires_failure = _episode_contract(
        {
            "permissions": [
                {"action_type": "RUN_COMMAND", "argv": ["<PYTHON>", "-c", "print('x')"]},
                {"action_type": "READ_STDOUT"},
                {"action_type": "DONE"},
            ],
            "completion_requirements": [
                {"action_type": "RUN_COMMAND", "argv": ["<PYTHON>", "-c", "print('x')"]},
                {"action_type": "READ_STDOUT"},
            ],
            "requires_failure": False,
        }
    )

    assert len(permissions) == 3
    assert len(completion) == 2
    assert expected == []
    assert requires_failure is False


def test_phase2bn_applies_bounded_initial_receptor_state() -> None:
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(task_type=TaskType.TEST_FAILURE, description="recover"),
        process=ProcessState(),
        terminal=TerminalState(),
        filesystem={"watched_paths": ["src"]},
    )

    updated = _apply_initial_state_overrides(
        state,
        {
            "initial_state": {
                "process": {"status": "exited", "exit_code": 1},
                "terminal": {
                    "stderr_delta": "ModuleNotFoundError: No module named 'alpha'",
                    "stderr_unread": True,
                },
            }
        },
    )

    assert updated.process.exit_code == 1
    assert updated.terminal.stderr_unread is True
    assert updated.goal == state.goal
    assert updated.filesystem == state.filesystem


def test_phase2bn_initial_state_rejects_goal_or_unknown_schema_fields() -> None:
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(task_type=TaskType.TEST_FAILURE, description="recover"),
        process=ProcessState(),
        terminal=TerminalState(),
        filesystem={},
    )

    with pytest.raises(ValueError, match="unsupported domains: goal"):
        _apply_initial_state_overrides(
            state,
            {"initial_state": {"goal": {"instruction": "leak the answer"}}},
        )

    with pytest.raises(ValueError, match="extra_forbidden"):
        _apply_initial_state_overrides(
            state,
            {"initial_state": {"terminal": {"expected_action": "RUN_COMMAND"}}},
        )
