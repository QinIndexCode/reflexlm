from reflexlm.data.tasks import build_env
from reflexlm.eval import PolicyStats
from reflexlm.llm.native_head_policy import NativeHeadPolicy
from reflexlm.schema import ActionDecision, ActionType, InternalTarget, TaskType


class _DummyNsiPolicy:
    def __init__(self) -> None:
        self.last_call = {
            "salience": 0.8,
            "risk": 0.1,
            "prediction_error": 0.6,
            "confidence": 0.9,
        }

    def act(self, _state):
        return ActionDecision(type=ActionType.WAIT, reason="dummy_low_level")

    def reset(self) -> None:
        self.last_call = {}


def _policy_shell() -> NativeHeadPolicy:
    policy = NativeHeadPolicy.__new__(NativeHeadPolicy)
    policy.stats = PolicyStats()
    policy.last_call = {}
    policy.nsi_policy = _DummyNsiPolicy()
    policy._debug_continuation = None
    policy._last_cache_reset_reason = None
    policy.zero_nsi_latent = False
    policy.enable_debug_continuation = True
    policy.continuation_control = "normal"
    policy.enable_native_head_calls = True
    return policy


def test_phase2f_debug_receptor_reads_stderr_before_qwen_call() -> None:
    state = build_env(TaskType.TEST_FAILURE, 0, profile="phase2f_latent_sensitive").reset()
    policy = _policy_shell()

    action = policy.act(state)

    assert action.type == ActionType.READ_STDERR
    assert action.reason == "debug_receptor_read_stderr"
    assert policy.stats.model_calls == 0
    assert policy.last_call["action_source"] == "low_level_debug_receptor"
    assert policy.last_call["qwen_called"] is False


def test_phase2f_debug_receptor_seeds_continuation_cache() -> None:
    env = build_env(TaskType.TEST_FAILURE, 0, profile="phase2i_semantic_val")
    state = env.reset()
    policy = _policy_shell()

    action = policy.act(state)

    assert action.type == ActionType.READ_STDERR
    assert policy.stats.model_calls == 0
    assert policy.last_call["debug_continuation_cached"] is True
    assert policy._debug_continuation is not None
    assert policy._debug_continuation["next"] == "read_source_file"

    env.step(action)
    source_state = env.state
    assert source_state is not None
    continuation_action = policy.act(source_state)
    assert continuation_action.type == ActionType.READ_FILE
    assert continuation_action.file_target == source_state.goal.watched_paths[0]
    assert policy.stats.model_calls == 0
    assert policy.last_call["cache_hit"] is True


def test_phase2f_continuation_cache_completes_source_inspection_without_qwen_call() -> None:
    env = build_env(TaskType.TEST_FAILURE, 2, profile="debug_ood_v2")
    state = env.reset()
    _, _, _, _ = env.step(ActionDecision(type=ActionType.READ_STDERR))
    source_state = env.state
    assert source_state is not None

    policy = _policy_shell()
    policy._debug_continuation = {
        "version": "phase2f_debug_continuation_v1",
        "key": policy._debug_cache_key(source_state),
        "next": "read_source_file",
    }
    first_action = policy.act(source_state)
    assert first_action.type == ActionType.READ_FILE
    assert policy.stats.model_calls == 0
    assert policy.last_call["cache_hit"] is True
    assert policy.last_call["qwen_called"] is False

    env.step(first_action)
    rerun_state = env.state
    assert rerun_state is not None
    second_action = policy.act(rerun_state)
    assert second_action.type == ActionType.RUN_COMMAND
    assert second_action.command == rerun_state.terminal.last_command
    assert policy.stats.model_calls == 0
    assert policy.last_call["cache_hit"] is True
    assert policy.last_call["qwen_called"] is False


def test_phase2k_continuation_cache_uses_prior_command_after_last_command_is_cleared() -> None:
    env = build_env(TaskType.TEST_FAILURE, 0, profile="phase2k_continuation_pressure_val")
    state = env.reset()
    policy = _policy_shell()

    first = policy.act(state)
    assert first.type == ActionType.READ_STDERR
    assert policy._debug_continuation is not None
    prior_command = policy._debug_continuation["prior_command"]

    env.step(first)
    source_state = env.state
    assert source_state is not None
    second = policy.act(source_state)
    assert second.type == ActionType.READ_FILE
    env.step(second)
    command_state = env.state
    assert command_state is not None
    assert command_state.terminal.last_command == ""

    third = policy.act(command_state)
    assert third.type == ActionType.RUN_COMMAND
    assert third.command == prior_command
    assert policy.stats.model_calls == 0
    assert policy.last_call["cache_hit"] is True


def test_phase2f_continuation_cache_invalidates_on_visible_stale_state() -> None:
    env = build_env(TaskType.TEST_FAILURE, 2, profile="debug_ood_v2")
    state = env.reset()
    stale_state = state.model_copy(
        update={
            "filesystem": state.filesystem.model_copy(
                update={"stale_cache_detected": True}
            )
        }
    )
    policy = _policy_shell()
    policy._debug_continuation = {
        "version": "phase2f_debug_continuation_v1",
        "key": policy._debug_cache_key(stale_state),
        "next": "run_last_command",
    }

    action, plan, reason = policy._debug_continuation_action(
        stale_state,
        InternalTarget.ESCALATE_TO_DEBUG_CORTEX,
    )

    assert action is None
    assert plan is None
    assert reason == "visible_safety_or_stale_state"
    assert policy._debug_continuation is None


def test_phase2f_native_head_only_disables_continuation_cache() -> None:
    env = build_env(TaskType.TEST_FAILURE, 2, profile="debug_ood_v2")
    state = env.reset()
    _, _, _, _ = env.step(ActionDecision(type=ActionType.READ_STDERR))
    source_state = env.state
    assert source_state is not None

    policy = _policy_shell()
    policy.enable_debug_continuation = False
    policy._debug_continuation = {
        "version": "phase2f_debug_continuation_v1",
        "key": policy._debug_cache_key(source_state),
        "next": "read_source_file",
    }

    action, plan, reason = policy._debug_continuation_action(
        source_state,
        InternalTarget.ESCALATE_TO_DEBUG_CORTEX,
    )

    assert action is None
    assert plan is None
    assert reason is None


def test_phase2l_wrong_cache_control_swaps_prior_command_without_gold_label() -> None:
    env = build_env(TaskType.TEST_FAILURE, 0, profile="phase2l_counterfactual_continuation_val")
    state = env.reset()
    policy = _policy_shell()
    policy.continuation_control = "wrong_cache"

    first = policy.act(state)
    assert first.type == ActionType.READ_STDERR
    assert policy._debug_continuation is not None
    assert policy._debug_continuation["wrong_cache_injected"] is True
    assert policy._debug_continuation["prior_command"] != state.terminal.last_command

    env.step(first)
    source_state = env.state
    assert source_state is not None
    second = policy.act(source_state)
    assert second.type == ActionType.READ_FILE
    env.step(second)
    command_state = env.state
    assert command_state is not None

    third = policy.act(command_state)
    assert third.type == ActionType.RUN_COMMAND
    assert third.command == policy._counterfactual_prior_command(state)
    assert policy.last_call["continuation_control"] == "wrong_cache"
    assert policy.last_call["wrong_cache_injected"] is True


def test_phase2l_cache_erased_control_blocks_debug_continuation() -> None:
    env = build_env(TaskType.TEST_FAILURE, 0, profile="phase2l_counterfactual_continuation_val")
    state = env.reset()
    policy = _policy_shell()
    policy.continuation_control = "cache_erased"
    policy.enable_debug_continuation = False

    assert policy._build_debug_continuation(
        state,
        ActionDecision(type=ActionType.READ_STDERR),
    ) is None


def test_phase2f_continuation_only_uses_visible_receptor_signal_without_qwen() -> None:
    env = build_env(TaskType.TEST_FAILURE, 0, profile="debug_ood_v2")
    state = env.reset()
    _, _, _, _ = env.step(ActionDecision(type=ActionType.READ_STDERR))
    failure_state = env.state
    assert failure_state is not None

    policy = _policy_shell()
    policy.enable_native_head_calls = False
    action, plan, reason = policy._continuation_only_action(
        failure_state,
        InternalTarget.ESCALATE_TO_DEBUG_CORTEX,
    )

    assert reason is None
    assert plan is not None
    assert action is not None
    assert action.type == ActionType.RUN_COMMAND
    assert action.command is not None
    assert "--snapshot-update" in action.command
    assert policy.stats.model_calls == 0


def test_phase2g_semantic_required_invalidates_last_command_continuation() -> None:
    env = build_env(TaskType.TEST_FAILURE, 2, profile="external_trace_v2_semantic_required")
    state = env.reset()
    _, _, _, _ = env.step(ActionDecision(type=ActionType.READ_STDERR))
    source_state = env.state
    assert source_state is not None
    _, _, _, _ = env.step(ActionDecision(type=ActionType.READ_FILE, file_target=source_state.goal.watched_paths[0]))
    semantic_state = env.state
    assert semantic_state is not None

    policy = _policy_shell()
    policy._debug_continuation = {
        "version": "phase2f_debug_continuation_v1",
        "key": policy._debug_cache_key(semantic_state),
        "next": "run_last_command",
    }
    action, plan, reason = policy._debug_continuation_action(
        semantic_state,
        InternalTarget.ESCALATE_TO_DEBUG_CORTEX,
    )

    assert action is None
    assert plan is None
    assert reason == "semantic_command_ambiguity"


def test_phase2g_continuation_only_takes_wrong_last_command_on_semantic_required() -> None:
    env = build_env(TaskType.TEST_FAILURE, 2, profile="external_trace_v2_semantic_required")
    state = env.reset()
    _, _, _, _ = env.step(ActionDecision(type=ActionType.READ_STDERR))
    source_state = env.state
    assert source_state is not None
    _, _, _, _ = env.step(ActionDecision(type=ActionType.READ_FILE, file_target=source_state.goal.watched_paths[0]))
    semantic_state = env.state
    assert semantic_state is not None

    policy = _policy_shell()
    policy.enable_native_head_calls = False
    action, plan, reason = policy._continuation_only_action(
        semantic_state,
        InternalTarget.ESCALATE_TO_DEBUG_CORTEX,
    )
    correct = env.oracle_action(semantic_state)

    assert reason is None
    assert plan is not None
    assert action is not None
    assert action.type == ActionType.RUN_COMMAND
    assert action.command == semantic_state.terminal.last_command
    assert action.command != correct.command
