from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from reflexlm.core.dataset import observation_from_state
from reflexlm.core.evaluation import prompt_only_heuristic
from reflexlm.core.model import ReflexCoreV0
from reflexlm.core.motor import decode_reflexcore_motor
from reflexlm.data.tasks import build_env
from reflexlm.runtime.oracle import RuleOracle
from reflexlm.runtime.safety import SafetyLayer
from reflexlm.schema import ActionDecision, ActionType, SystemStateFrame, TaskType

PolicyFn = Callable[[SystemStateFrame], ActionDecision]


@dataclass(slots=True)
class ClosedLoopEpisodeResult:
    episode_id: str
    task_type: str
    success: bool
    steps: int
    reward: float
    action_correct: int
    action_total: int
    dangerous_intercepted: bool
    stale_state_prevented: bool


def evaluate_reflexcore_closed_loop(
    model: ReflexCoreV0,
    *,
    profile: str = "default",
    episodes_per_task: int = 3,
    device: str = "cpu",
    max_steps: int | None = None,
) -> dict[str, object]:
    model.to(device)
    model.eval()
    results: list[ClosedLoopEpisodeResult] = []
    for task_type in TaskType:
        for episode_index in range(episodes_per_task):
            env = build_env(task_type, episode_index, profile=profile)
            if max_steps is not None:
                env.max_steps = max_steps
            results.append(_run_model_episode(model, env, device=device))
    return summarize_closed_loop_results(results)


def evaluate_closed_loop_baselines(
    *,
    profile: str = "default",
    episodes_per_task: int = 3,
    max_steps: int | None = None,
) -> dict[str, dict[str, object]]:
    oracle = RuleOracle()
    policies: dict[str, PolicyFn] = {
        "rule_oracle": oracle.act,
        "prompt_only_heuristic": prompt_only_heuristic,
        "static_wait": lambda _state: ActionDecision(type=ActionType.WAIT, reason="static_wait"),
    }
    return {
        name: evaluate_policy_closed_loop(
            policy,
            profile=profile,
            episodes_per_task=episodes_per_task,
            max_steps=max_steps,
        )
        for name, policy in policies.items()
    }


def evaluate_policy_closed_loop(
    policy: PolicyFn,
    *,
    profile: str = "default",
    episodes_per_task: int = 3,
    max_steps: int | None = None,
) -> dict[str, object]:
    results: list[ClosedLoopEpisodeResult] = []
    for task_type in TaskType:
        for episode_index in range(episodes_per_task):
            env = build_env(task_type, episode_index, profile=profile)
            if max_steps is not None:
                env.max_steps = max_steps
            results.append(_run_policy_episode(policy, env))
    return summarize_closed_loop_results(results)


def closed_loop_acceptance_against_baselines(
    model_summary: dict[str, object],
    baseline_summaries: dict[str, dict[str, object]],
    *,
    required_baselines: list[str],
) -> dict[str, object]:
    model_success = model_summary.get("success_rate")
    result: dict[str, object] = {"required": required_baselines, "passed": True, "details": {}}
    details: dict[str, object] = {}
    for baseline_name in required_baselines:
        baseline = baseline_summaries.get(baseline_name)
        if not isinstance(model_success, float) or baseline is None:
            baseline_success = None
            passed = False
        else:
            baseline_success = baseline.get("success_rate")
            passed = isinstance(baseline_success, float) and model_success > baseline_success
        details[baseline_name] = {
            "passed": passed,
            "model_success_rate": model_success,
            "baseline_success_rate": baseline_success,
        }
        result["passed"] = bool(result["passed"]) and passed
    result["details"] = details
    return result


def summarize_closed_loop_results(
    results: list[ClosedLoopEpisodeResult],
) -> dict[str, object]:
    task_groups: dict[str, list[ClosedLoopEpisodeResult]] = {}
    for result in results:
        task_groups.setdefault(result.task_type, []).append(result)
    episode_count = len(results)
    reward_sum = sum(result.reward for result in results)
    action_correct = sum(result.action_correct for result in results)
    action_total = sum(result.action_total for result in results)
    return {
        "episode_count": episode_count,
        "success_count": sum(1 for result in results if result.success),
        "success_rate": _ratio(sum(1 for result in results if result.success), episode_count),
        "avg_reward": reward_sum / max(episode_count, 1),
        "avg_steps": sum(result.steps for result in results) / max(episode_count, 1),
        "action_accuracy": _ratio(action_correct, action_total),
        "dangerous_intercept_rate": _ratio(
            sum(1 for result in results if result.dangerous_intercepted),
            sum(1 for result in results if result.task_type == TaskType.DANGEROUS_ACTION.value),
        ),
        "stale_state_prevention_rate": _ratio(
            sum(1 for result in results if result.stale_state_prevented),
            sum(1 for result in results if result.task_type == TaskType.FILE_CHANGE.value),
        ),
        "by_task": {
            task_type: {
                "episodes": len(task_results),
                "success_rate": _ratio(
                    sum(1 for result in task_results if result.success),
                    len(task_results),
                ),
                "action_accuracy": _ratio(
                    sum(result.action_correct for result in task_results),
                    sum(result.action_total for result in task_results),
                ),
            }
            for task_type, task_results in sorted(task_groups.items())
        },
        "episodes": [
            {
                "episode_id": result.episode_id,
                "task_type": result.task_type,
                "success": result.success,
                "steps": result.steps,
                "reward": result.reward,
                "action_accuracy": _ratio(result.action_correct, result.action_total),
                "dangerous_intercepted": result.dangerous_intercepted,
                "stale_state_prevented": result.stale_state_prevented,
            }
            for result in results
        ],
    }


def _run_model_episode(
    model: ReflexCoreV0,
    env: object,
    *,
    device: str,
) -> ClosedLoopEpisodeResult:
    safety = SafetyLayer()
    state = env.reset()
    hidden: torch.Tensor | None = None
    done = False
    step_count = 0
    reward_sum = 0.0
    action_correct = 0
    action_total = 0
    task_completed = False
    dangerous_intercepted = False
    stale_state_prevented = False
    with torch.no_grad():
        while not done and step_count < env.max_steps:
            observation = observation_from_state(state, vocab_size=model.config.vocab_size)
            vector = (
                torch.tensor(observation.vector, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            text = (
                torch.tensor(observation.text_tokens, dtype=torch.long, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            outputs = model(vector, text, hidden=hidden)
            hidden_value = outputs.get("hidden")
            hidden = hidden_value if isinstance(hidden_value, torch.Tensor) else None
            decoded = decode_reflexcore_motor(outputs, state)
            action = safety.enforce(decoded.action, state.goal, state).action
            state, reward, done, info = env.step(action)
            reward_sum += reward
            action_total += 1
            action_correct += int(_action_matches(action, info.correct_action))
            task_completed = task_completed or info.task_completed
            dangerous_intercepted = dangerous_intercepted or info.dangerous_intercepted
            stale_state_prevented = stale_state_prevented or info.stale_state_prevented
            step_count += 1
    return ClosedLoopEpisodeResult(
        episode_id=env.episode_id,
        task_type=env.task_type.value,
        success=task_completed,
        steps=step_count,
        reward=reward_sum,
        action_correct=action_correct,
        action_total=action_total,
        dangerous_intercepted=dangerous_intercepted,
        stale_state_prevented=stale_state_prevented,
    )


def _run_policy_episode(policy: PolicyFn, env: object) -> ClosedLoopEpisodeResult:
    state = env.reset()
    done = False
    step_count = 0
    reward_sum = 0.0
    action_correct = 0
    action_total = 0
    task_completed = False
    dangerous_intercepted = False
    stale_state_prevented = False
    while not done and step_count < env.max_steps:
        action = policy(state)
        state, reward, done, info = env.step(action)
        reward_sum += reward
        action_total += 1
        action_correct += int(_action_matches(action, info.correct_action))
        task_completed = task_completed or info.task_completed
        dangerous_intercepted = dangerous_intercepted or info.dangerous_intercepted
        stale_state_prevented = stale_state_prevented or info.stale_state_prevented
        step_count += 1
    return ClosedLoopEpisodeResult(
        episode_id=env.episode_id,
        task_type=env.task_type.value,
        success=task_completed,
        steps=step_count,
        reward=reward_sum,
        action_correct=action_correct,
        action_total=action_total,
        dangerous_intercepted=dangerous_intercepted,
        stale_state_prevented=stale_state_prevented,
    )


def _action_matches(predicted: ActionDecision, target: ActionDecision) -> bool:
    if predicted.type != target.type:
        return False
    if target.command is not None and predicted.command != target.command:
        return False
    if target.file_target is not None and predicted.file_target != target.file_target:
        return False
    return True


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
