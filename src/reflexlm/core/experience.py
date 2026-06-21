from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reflexlm.core.dataset import (
    assert_no_hidden_oracle_fields,
    observation_from_state,
    write_reflexcore_jsonl,
)
from reflexlm.core.runner import ReflexCoreStepResult
from reflexlm.core.schema import MotorAction, ReflexCoreTrainingExample, dataset_hash
from reflexlm.schema import ActionDecision, ActionType, SourceType, SystemStateFrame


@dataclass(slots=True)
class ReflexCoreExperienceSummary:
    path: str | None
    episode_id: str
    example_count: int
    dataset_hash: str
    source: str
    live_observation: bool = False
    runtime_observation_examples: int = 0
    changed_file_observations: int = 0
    terminal_observation_examples: int = 0
    post_safety_actions: bool = True
    observed_prediction_error_examples: int = 0
    observed_prediction_error_mean: float | None = None
    observed_prediction_error_max: float | None = None
    model_prediction_error_mean: float | None = None
    free_shell_generation: bool = False
    gui_or_vision: bool = False


def examples_from_step_trace(
    *,
    initial_state: SystemStateFrame,
    trace: list[ReflexCoreStepResult],
    episode_id: str,
    vocab_size: int = 4096,
    max_text_tokens: int = 128,
    source: SourceType = SourceType.MODEL,
) -> list[ReflexCoreTrainingExample]:
    """Convert a bounded model rollout into self-supervised training examples.

    The recorded action is the post-safety typed motor action, not the raw model
    proposal. This preserves the existing allowlist/safety boundary while still
    letting ReflexCore learn from its own observe-act-observe transitions.
    """

    state = initial_state
    examples: list[ReflexCoreTrainingExample] = []
    for step_index, result in enumerate(trace):
        action = _recorded_action(result)
        example = ReflexCoreTrainingExample(
            episode_id=episode_id,
            t=step_index,
            observation=observation_from_state(
                state,
                vocab_size=vocab_size,
                max_text_tokens=max_text_tokens,
            ),
            action=MotorAction.from_decision(action),
            next_observation=observation_from_state(
                result.state,
                vocab_size=vocab_size,
                max_text_tokens=max_text_tokens,
            ),
            reward=1.0 if result.done else 0.0,
            done=result.done,
            source=source.value,
        )
        assert_no_hidden_oracle_fields(example)
        examples.append(example)
        state = result.state
    return examples


def write_experience_jsonl(
    path: Path,
    *,
    initial_state: SystemStateFrame,
    trace: list[ReflexCoreStepResult],
    episode_id: str,
    vocab_size: int = 4096,
    max_text_tokens: int = 128,
) -> ReflexCoreExperienceSummary:
    examples = examples_from_step_trace(
        initial_state=initial_state,
        trace=trace,
        episode_id=episode_id,
        vocab_size=vocab_size,
        max_text_tokens=max_text_tokens,
    )
    write_reflexcore_jsonl(path, examples)
    return _experience_summary(
        path=str(path),
        episode_id=episode_id,
        examples=examples,
        trace=trace,
        source=SourceType.MODEL,
    )


def summarize_experience(
    *,
    initial_state: SystemStateFrame,
    trace: list[ReflexCoreStepResult],
    episode_id: str,
    vocab_size: int = 4096,
    max_text_tokens: int = 128,
) -> ReflexCoreExperienceSummary:
    examples = examples_from_step_trace(
        initial_state=initial_state,
        trace=trace,
        episode_id=episode_id,
        vocab_size=vocab_size,
        max_text_tokens=max_text_tokens,
    )
    return _experience_summary(
        path=None,
        episode_id=episode_id,
        examples=examples,
        trace=trace,
        source=SourceType.MODEL,
    )


def _recorded_action(result: ReflexCoreStepResult) -> ActionDecision:
    action = result.safety_decision.action
    if action is not None:
        return action
    return ActionDecision(
        type=ActionType.BLOCK,
        reason=result.safety_decision.reason,
        confidence=1.0,
    )


def _experience_summary(
    *,
    path: str | None,
    episode_id: str,
    examples: list[ReflexCoreTrainingExample],
    trace: list[ReflexCoreStepResult],
    source: SourceType,
) -> ReflexCoreExperienceSummary:
    runtime_examples = [
        example
        for example in examples
        if (
            example.observation.runtime_evidence.source
            == SourceType.RUNTIME_OBSERVATION.value
            or example.next_observation.runtime_evidence.source
            == SourceType.RUNTIME_OBSERVATION.value
        )
    ]
    changed_file_observations = sum(
        1
        for example in examples
        if (
            example.observation.runtime_evidence.changed_files
            or example.next_observation.runtime_evidence.changed_files
            or example.observation.filesystem.changed_paths
            or example.next_observation.filesystem.changed_paths
        )
    )
    terminal_observation_examples = sum(
        1
        for example in examples
        if (
            example.observation.runtime_evidence.terminal_observations
            or example.next_observation.runtime_evidence.terminal_observations
            or example.observation.terminal.stdout_delta
            or example.observation.terminal.stderr_delta
            or example.next_observation.terminal.stdout_delta
            or example.next_observation.terminal.stderr_delta
        )
    )
    observed_errors = [
        float(result.observed_prediction_error)
        for result in trace
        if result.observed_prediction_error is not None
    ]
    model_errors = [
        float(result.model_prediction_error)
        for result in trace
        if result.model_prediction_error is not None
    ]
    return ReflexCoreExperienceSummary(
        path=path,
        episode_id=episode_id,
        example_count=len(examples),
        dataset_hash=dataset_hash(examples),
        source=source.value,
        live_observation=bool(runtime_examples),
        runtime_observation_examples=len(runtime_examples),
        changed_file_observations=changed_file_observations,
        terminal_observation_examples=terminal_observation_examples,
        observed_prediction_error_examples=len(observed_errors),
        observed_prediction_error_mean=(
            sum(observed_errors) / len(observed_errors) if observed_errors else None
        ),
        observed_prediction_error_max=max(observed_errors) if observed_errors else None,
        model_prediction_error_mean=(
            sum(model_errors) / len(model_errors) if model_errors else None
        ),
    )
