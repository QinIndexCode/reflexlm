from __future__ import annotations

import json
import math
import itertools
import time
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from reflexlm.data.jsonl import read_jsonl
from reflexlm.data.tasks import build_env_from_episode_id, parse_episode_id
from reflexlm.models.features import (
    ROUTE_ORDER,
    StateVectorizer,
    action_to_index,
    candidate_commands,
    candidate_files,
    command_failure_match_scores,
    prompt_token_estimate,
    resolve_structured_action,
    valid_action_mask,
)
from reflexlm.reporting import METRIC_ORDER, per_task_rows, summarize_rows
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    RouteName,
    SystemStateFrame,
    TaskType,
    dump_jsonable,
)
from reflexlm.runtime.nervous_system import (
    INTERNAL_TARGET_ORDER,
    InternalTarget,
    authorize_bounded_debug_cortex_action,
    authorize_persistent_failure_recovery,
    persistent_failure_recovery_should_promote,
    plan_from_head_indices,
    serialize_motor_action,
)
from reflexlm.runtime.homeostasis import (
    HomeostaticControlConfig,
    HomeostaticSynapticController,
)


class Policy(Protocol):
    def act(self, state: SystemStateFrame) -> ActionDecision:
        raise NotImplementedError


@dataclass(slots=True)
class PolicyStats:
    token_cost: int = 0
    model_calls: int = 0
    parse_failures: int = 0
    retries: int = 0


class RuleOraclePolicyAdapter:
    def __init__(self) -> None:
        from reflexlm.runtime.oracle import RuleOracle

        self.oracle = RuleOracle()
        self.stats = PolicyStats()
        self.last_call: dict[str, Any] = {}

    def reset(self) -> None:
        self.stats = PolicyStats()
        self.last_call = {}

    def metadata(self) -> dict[str, Any]:
        return {"policy_family": "rule_oracle", "policy_label": "rule_oracle"}

    def act(self, state: SystemStateFrame) -> ActionDecision:
        self.stats.model_calls += 1
        action = self.oracle.act(state)
        self.last_call = {"action_source": "rule_oracle"}
        return action


class SequenceModelPolicy:
    def __init__(
        self,
        model: torch.nn.Module,
        vectorizer: StateVectorizer,
        *,
        policy_label: str,
        use_legal_action_mask: bool = False,
        training_summary: dict[str, Any] | None = None,
        authorize_bounded_debug_cortex_recovery: bool = False,
        use_synaptic_motor_plan: bool = True,
        command_permutation_ensemble: bool = False,
        semantic_command_prior_weight: float = 0.0,
        semantic_command_scorer: Any | None = None,
        enable_homeostatic_control: bool = True,
        homeostatic_config: HomeostaticControlConfig | None = None,
        enable_online_homeostatic_adaptation: bool | None = None,
        enable_cross_episode_homeostatic_memory: bool | None = None,
        homeostatic_persistence_scope: str | None = None,
    ) -> None:
        self.model = model.eval()
        self.vectorizer = vectorizer
        self.policy_label = policy_label
        self.use_legal_action_mask = use_legal_action_mask
        self.training_summary = training_summary or {}
        self.authorize_bounded_debug_cortex_recovery = authorize_bounded_debug_cortex_recovery
        self.use_synaptic_motor_plan = use_synaptic_motor_plan
        self.command_permutation_ensemble = command_permutation_ensemble
        self.semantic_command_prior_weight = max(0.0, float(semantic_command_prior_weight))
        self.semantic_command_scorer = semantic_command_scorer
        self.enable_homeostatic_control = enable_homeostatic_control
        self.homeostatic_persistence_scope = homeostatic_persistence_scope
        resolved_homeostatic_config = homeostatic_config or HomeostaticControlConfig()
        calibration = self.training_summary.get("prediction_error_calibration")
        if homeostatic_config is None and isinstance(calibration, dict):
            calibrated_threshold = calibration.get("threshold")
            if calibrated_threshold is not None and 0.0 < float(calibrated_threshold) <= 1.0:
                resolved_homeostatic_config.surprise_wake_threshold = float(
                    calibrated_threshold
                )
        if enable_online_homeostatic_adaptation is not None:
            resolved_homeostatic_config.online_failure_sensitivity_enabled = (
                enable_online_homeostatic_adaptation
            )
        if enable_cross_episode_homeostatic_memory is not None:
            resolved_homeostatic_config.preserve_adaptive_threshold_across_reset = (
                enable_cross_episode_homeostatic_memory
            )
        self.homeostatic_controller = HomeostaticSynapticController(
            resolved_homeostatic_config
        )
        self.device = next(self.model.parameters()).device
        self.hidden: torch.Tensor | None = None
        self.predicted_next_state: torch.Tensor | None = None
        self.stats = PolicyStats()
        self.last_call: dict[str, Any] = {}

    def reset(self) -> None:
        self.hidden = None
        self.predicted_next_state = None
        self.stats = PolicyStats()
        self.last_call = {}
        self.homeostatic_controller.reset()

    def save_homeostatic_state(
        self,
        path: str | Path,
        *,
        authenticity_key: str | bytes | None = None,
    ) -> dict[str, object]:
        return self.homeostatic_controller.save_persistent_state(
            path,
            persistence_scope=self.homeostatic_persistence_scope,
            authenticity_key=authenticity_key,
        )

    def load_homeostatic_state(
        self,
        path: str | Path,
        *,
        authenticity_key: str | bytes | None = None,
    ) -> dict[str, object]:
        self.hidden = None
        self.predicted_next_state = None
        self.stats = PolicyStats()
        self.last_call = {}
        return self.homeostatic_controller.load_persistent_state_file(
            path,
            persistence_scope=self.homeostatic_persistence_scope,
            authenticity_key=authenticity_key,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "sequence_model",
            "policy_label": self.policy_label,
            "vectorizer": asdict(self.vectorizer),
            "use_legal_action_mask": self.use_legal_action_mask,
            "training_summary": self.training_summary,
            "authorize_bounded_debug_cortex_recovery": self.authorize_bounded_debug_cortex_recovery,
            "use_synaptic_motor_plan": self.use_synaptic_motor_plan,
            "command_permutation_ensemble": self.command_permutation_ensemble,
            "semantic_command_prior_weight": self.semantic_command_prior_weight,
            "semantic_command_scorer": (
                self.semantic_command_scorer.metadata()
                if callable(getattr(self.semantic_command_scorer, "metadata", None))
                else None
            ),
            "homeostatic_control_enabled": self.enable_homeostatic_control,
            "homeostatic_persistence_scope_fingerprint": (
                self.homeostatic_controller.persistent_state_scope_fingerprint(
                    self.homeostatic_persistence_scope
                )
            ),
            "homeostatic_control": self.homeostatic_controller.snapshot(),
        }

    def act(self, state: SystemStateFrame) -> ActionDecision:
        vector = torch.tensor(
            self.vectorizer.vectorize_state(state),
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, -1)
        observed_temporal_prediction_error_raw: float | None = None
        observed_temporal_prediction_error: float | None = None
        observed_full_state_novelty_raw: float | None = None
        observed_full_state_novelty: float | None = None
        if self.predicted_next_state is not None:
            previous_prediction = self.predicted_next_state.to(
                device=vector.device,
                dtype=vector.dtype,
            )
            if previous_prediction.shape == vector[0, 0].shape:
                delta = previous_prediction - vector[0, 0]
                observed_full_state_novelty_raw = float(
                    torch.linalg.vector_norm(delta).item()
                    / math.sqrt(max(int(delta.numel()), 1))
                )
                observed_full_state_novelty = max(
                    0.0,
                    min(observed_full_state_novelty_raw, 1.0),
                )
                world_model_mask = torch.tensor(
                    self.vectorizer.world_model_target_mask(),
                    dtype=delta.dtype,
                    device=delta.device,
                )
                controllable_delta = delta * world_model_mask
                observed_temporal_prediction_error_raw = float(
                    torch.linalg.vector_norm(controllable_delta).item()
                    / math.sqrt(max(int(world_model_mask.sum().item()), 1))
                )
                observed_temporal_prediction_error = max(
                    0.0,
                    min(observed_temporal_prediction_error_raw, 1.0),
                )
        prior_hidden = self.hidden
        with torch.inference_mode():
            outputs = self.model(vector, hidden=prior_hidden)
        self.hidden = outputs.get("hidden")
        action_logits = outputs["action_logits"][0, 0]
        if self.use_legal_action_mask:
            mask = torch.tensor(
                valid_action_mask(state),
                dtype=torch.bool,
                device=action_logits.device,
            )
            action_logits = action_logits.masked_fill(~mask, -1.0e4)
        action_index = int(action_logits.argmax().item())
        target_index = int(outputs["target_logits"][0, 0].argmax().item()) if "target_logits" in outputs else 0
        command_index = int(outputs["command_slot_logits"][0, 0].argmax().item())
        command_ensemble_scores: list[float] = []
        semantic_command_prior_scores: list[float] = []
        command_ensemble_extra_calls = 0
        if self.command_permutation_ensemble:
            (
                command_index,
                command_ensemble_scores,
                semantic_command_prior_scores,
                command_ensemble_extra_calls,
            ) = self._permutation_equivariant_command_index(
                state=state,
                prior_hidden=prior_hidden,
                base_outputs=outputs,
            )
        file_index = int(outputs["file_slot_logits"][0, 0].argmax().item())
        confidence = float(torch.softmax(action_logits, dim=-1).max().item())
        route_index = int(outputs["route_logits"][0, 0].argmax().item()) if "route_logits" in outputs else 0
        route_probs = (
            torch.softmax(outputs["route_logits"][0, 0], dim=-1).tolist()
            if "route_logits" in outputs
            else []
        )
        salience = float(torch.sigmoid(outputs["salience"][0, 0]).item()) if "salience" in outputs else 0.0
        risk = float(outputs["risk"][0, 0].item()) if "risk" in outputs else 0.0
        model_prediction_error_raw = (
            float(outputs["prediction_error"][0, 0].item()) if "prediction_error" in outputs else 0.0
        )
        model_prediction_error = max(0.0, min(model_prediction_error_raw, 1.0))
        if observed_temporal_prediction_error is None:
            prediction_error = model_prediction_error
            prediction_error_source = "model_head"
        elif observed_temporal_prediction_error >= model_prediction_error:
            prediction_error = observed_temporal_prediction_error
            prediction_error_source = "observed_temporal_next_state"
        else:
            prediction_error = model_prediction_error
            prediction_error_source = "model_head"
        self.stats.model_calls += 1 + command_ensemble_extra_calls
        self.stats.token_cost += prompt_token_estimate(state) * (
            1 + command_ensemble_extra_calls
        )
        plan = plan_from_head_indices(
            state=state,
            action_index=action_index,
            route_index=route_index,
            target_index=target_index,
            command_slot=command_index,
            file_slot=file_index,
            confidence=confidence,
            inhibition_score=risk,
        )
        raw_motor_candidate = resolve_structured_action(
            action_index,
            command_index,
            file_index,
            state,
            confidence,
        )
        receptor_priority_reasons = {
            "stale_state_refresh_receptor",
            "pending_file_read_receptor",
            "pending_stderr_receptor",
            "pending_stdout_receptor",
        }
        homeostatic_decision = self.homeostatic_controller.observe(
            proposed_action=raw_motor_candidate.type,
            salience=salience,
            risk=risk,
            prediction_error=prediction_error,
            temporal_observation_available=observed_temporal_prediction_error is not None,
            hard_dangerous=state.safety.dangerous_command_detected,
            receptor_priority=plan.reason in receptor_priority_reasons,
            failure_visible=(
                state.process.exit_code not in (None, 0)
                or state.process.interrupted
                or state.process.resource_alert
            ),
        )
        if self.enable_homeostatic_control:
            if homeostatic_decision.inhibited:
                plan.internal_target = InternalTarget.INHIBIT
                plan.action_type = ActionType.BLOCK
                plan.route_name = RouteName.SAFETY
                plan.inhibited = True
                plan.reason = homeostatic_decision.reason
            elif (
                homeostatic_decision.target
                == InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX
                and plan.internal_target == InternalTarget.REFLEX_MOTOR
            ):
                plan.internal_target = InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX
                plan.action_type = None
                plan.reason = homeostatic_decision.reason
            elif (
                homeostatic_decision.habituated
                and plan.internal_target == InternalTarget.REFLEX_MOTOR
            ):
                plan.action_type = ActionType.WAIT
                plan.reason = homeostatic_decision.reason
        cortex_action = None
        persistent_recovery_constraint_applied = False
        if self.authorize_bounded_debug_cortex_recovery:
            if persistent_failure_recovery_should_promote(
                state,
                raw_motor_candidate.type,
            ):
                persistent_recovery_constraint_applied = True
                cortex_action = authorize_persistent_failure_recovery(
                    state=state,
                    command_slot=command_index,
                    confidence=confidence,
                )
                if cortex_action is not None:
                    plan.action_type = ActionType.RUN_COMMAND
                    plan.command_slot = command_index
            elif plan.internal_target in {
                InternalTarget.ESCALATE_TO_DEBUG_CORTEX,
                InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX,
            }:
                cortex_action = authorize_bounded_debug_cortex_action(
                    raw_motor_candidate,
                    state,
                )
        action = (
            serialize_motor_action(plan, state, cortex_action=cortex_action)
            if self.use_synaptic_motor_plan
            else raw_motor_candidate
        )
        next_state = outputs.get("next_state")
        next_state_action_source = "model_forward_action_distribution"
        predict_next_state = getattr(self.model, "predict_next_state", None)
        if callable(predict_next_state) and "memory" in outputs:
            with torch.inference_mode():
                next_state = predict_next_state(
                    outputs["memory"],
                    vector,
                    action_indices=torch.tensor(
                        [[action_to_index(action.type)]],
                        dtype=torch.long,
                        device=self.device,
                    ),
                )
            next_state_action_source = "executed_motor_action"
        self.predicted_next_state = (
            next_state[0, 0].detach().to("cpu") if next_state is not None else None
        )
        self.last_call = {
            "action_index": action_index,
            "target_index": target_index,
            "raw_internal_target": INTERNAL_TARGET_ORDER[target_index].value,
            "command_index": command_index,
            "command_permutation_ensemble_applied": self.command_permutation_ensemble
            and len(candidate_commands(state)) > 1,
            "command_ensemble_scores": command_ensemble_scores,
            "semantic_command_prior_scores": semantic_command_prior_scores,
            "semantic_command_prior_weight": self.semantic_command_prior_weight,
            "command_ensemble_extra_calls": command_ensemble_extra_calls,
            "file_index": file_index,
            "confidence": confidence,
            "route_index": route_index,
            "route_name": str(ROUTE_ORDER[route_index].value) if "route_index" in locals() else None,
            "route_probs": route_probs,
            "salience": salience,
            "risk": risk,
            "prediction_error": prediction_error,
            "prediction_error_source": prediction_error_source,
            "model_prediction_error": model_prediction_error,
            "model_prediction_error_raw": model_prediction_error_raw,
            "observed_temporal_prediction_error": observed_temporal_prediction_error,
            "observed_temporal_prediction_error_raw": observed_temporal_prediction_error_raw,
            "observed_full_state_novelty": observed_full_state_novelty,
            "observed_full_state_novelty_raw": observed_full_state_novelty_raw,
            "temporal_observation_available": observed_temporal_prediction_error is not None,
            "next_state_prediction_available": self.predicted_next_state is not None,
            "next_state_action_source": next_state_action_source,
            "raw_motor_candidate": raw_motor_candidate.model_dump(mode="json"),
            "bounded_debug_cortex_action_authorized": cortex_action is not None,
            "persistent_recovery_constraint_applied": persistent_recovery_constraint_applied,
            "synaptic_motor_plan_applied": self.use_synaptic_motor_plan,
            "synaptic_plan": plan.to_dict(),
            "homeostatic_control_enabled": self.enable_homeostatic_control,
            "homeostatic_decision": homeostatic_decision.to_dict(),
            "homeostatic_state": self.homeostatic_controller.snapshot(),
        }
        return action

    def _permutation_equivariant_command_index(
        self,
        *,
        state: SystemStateFrame,
        prior_hidden: torch.Tensor | None,
        base_outputs: dict[str, torch.Tensor],
    ) -> tuple[int, list[float], list[float], int]:
        commands = candidate_commands(state)
        if len(commands) <= 1:
            return 0, [1.0] if commands else [], [], 0
        scores = [0.0 for _ in commands]
        permutations = list(itertools.permutations(range(len(commands))))
        for permutation_index, permutation in enumerate(permutations):
            if permutation_index == 0 and permutation == tuple(range(len(commands))):
                outputs = base_outputs
            else:
                permuted_goal = state.goal.model_copy(
                    update={
                        "command_allowlist": [
                            commands[original_index] for original_index in permutation
                        ]
                    }
                )
                permuted_state = state.model_copy(update={"goal": permuted_goal})
                permuted_vector = torch.tensor(
                    self.vectorizer.vectorize_state(permuted_state),
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, 1, -1)
                with torch.inference_mode():
                    outputs = self.model(permuted_vector, hidden=prior_hidden)
            probabilities = torch.softmax(
                outputs["command_slot_logits"][0, 0],
                dim=-1,
            )
            for position, original_index in enumerate(permutation):
                scores[original_index] += float(probabilities[position].item())
        semantic_scores = (
            list(self.semantic_command_scorer.score_state(state))
            if self.semantic_command_scorer is not None
            else command_failure_match_scores(state)
        )
        if len(semantic_scores) != len(commands):
            raise ValueError("semantic command scorer returned the wrong number of scores")
        adjusted_scores = list(scores)
        if self.semantic_command_prior_weight > 0.0:
            for index, semantic_score in enumerate(semantic_scores):
                adjusted_scores[index] += self.semantic_command_prior_weight * semantic_score
        selected = max(range(len(adjusted_scores)), key=adjusted_scores.__getitem__)
        return selected, adjusted_scores, semantic_scores, len(permutations) - 1


class EventGatedSequencePolicy:
    """Skip neural forwards for deterministic visible receptor transitions."""

    def __init__(
        self,
        model: torch.nn.Module,
        vectorizer: StateVectorizer,
        *,
        policy_label: str,
        training_summary: dict[str, Any] | None = None,
        authorize_bounded_debug_cortex_recovery: bool = True,
        use_synaptic_motor_plan: bool = True,
        command_permutation_ensemble: bool = False,
        semantic_command_prior_weight: float = 0.0,
        semantic_command_scorer: Any | None = None,
    ) -> None:
        self.policy_label = policy_label
        self.neural_policy = SequenceModelPolicy(
            model,
            vectorizer,
            policy_label=f"{policy_label}.neural",
            training_summary=training_summary,
            authorize_bounded_debug_cortex_recovery=authorize_bounded_debug_cortex_recovery,
            use_synaptic_motor_plan=use_synaptic_motor_plan,
            command_permutation_ensemble=command_permutation_ensemble,
            semantic_command_prior_weight=semantic_command_prior_weight,
            semantic_command_scorer=semantic_command_scorer,
        )
        self.authorize_bounded_debug_cortex_recovery = authorize_bounded_debug_cortex_recovery
        self.use_synaptic_motor_plan = use_synaptic_motor_plan
        self.command_permutation_ensemble = command_permutation_ensemble
        self.semantic_command_prior_weight = max(0.0, float(semantic_command_prior_weight))
        self.semantic_command_scorer = semantic_command_scorer
        self.stats = PolicyStats()
        self.last_call: dict[str, Any] = {}

    def reset(self) -> None:
        self.neural_policy.reset()
        self.stats = PolicyStats()
        self.last_call = {}

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "event_gated_sequence_model",
            "policy_label": self.policy_label,
            "neural_policy": self.neural_policy.metadata(),
            "gated_actions": [
                ActionType.REFRESH_STATE.value,
                ActionType.READ_FILE.value,
                ActionType.READ_STDERR.value,
                ActionType.READ_STDOUT.value,
                ActionType.DONE.value,
            ],
            "command_permutation_ensemble": self.command_permutation_ensemble,
            "semantic_command_prior_weight": self.semantic_command_prior_weight,
            "semantic_command_scorer": (
                self.semantic_command_scorer.metadata()
                if callable(getattr(self.semantic_command_scorer, "metadata", None))
                else None
            ),
        }

    def act(self, state: SystemStateFrame) -> ActionDecision:
        deterministic = self._deterministic_receptor_action(state)
        if deterministic is not None:
            self.last_call = {
                "event_gate": "deterministic_visible_receptor",
                "qwen_called": False,
                "action_source": "event_gate",
                "reason": deterministic.reason,
            }
            return deterministic
        before = _policy_stats(self.neural_policy)
        before_token_cost = before.token_cost
        before_model_calls = before.model_calls
        before_parse_failures = before.parse_failures
        before_retries = before.retries
        action = self.neural_policy.act(state)
        after = _policy_stats(self.neural_policy)
        self.stats.token_cost += after.token_cost - before_token_cost
        self.stats.model_calls += after.model_calls - before_model_calls
        self.stats.parse_failures += after.parse_failures - before_parse_failures
        self.stats.retries += after.retries - before_retries
        self.last_call = {
            **dict(self.neural_policy.last_call),
            "event_gate": "neural_command_or_ambiguous_state",
            "qwen_called": True,
        }
        return action

    def _deterministic_receptor_action(
        self,
        state: SystemStateFrame,
    ) -> ActionDecision | None:
        if state.filesystem.external_change_detected or state.filesystem.stale_cache_detected:
            return ActionDecision(
                type=ActionType.REFRESH_STATE,
                reason="event_gate_refresh_changed_state",
                confidence=1.0,
            )
        file_candidates = candidate_files(state)
        if state.goal.task_type == TaskType.FILE_CHANGE and state.filesystem.dirty_files and file_candidates:
            return ActionDecision(
                type=ActionType.READ_FILE,
                file_target=file_candidates[0],
                reason="event_gate_read_dirty_file",
                confidence=1.0,
            )
        if state.terminal.stderr_unread:
            return ActionDecision(
                type=ActionType.READ_STDERR,
                reason="event_gate_read_unread_stderr",
                confidence=1.0,
            )
        if state.terminal.stdout_unread:
            return ActionDecision(
                type=ActionType.READ_STDOUT,
                reason="event_gate_read_unread_stdout",
                confidence=1.0,
            )
        if (
            state.terminal.last_command is not None
            and state.process.exit_code == 0
            and not state.filesystem.dirty_files
            and not state.filesystem.external_change_detected
            and not state.filesystem.stale_cache_detected
        ):
            return ActionDecision(
                type=ActionType.DONE,
                reason="event_gate_success_no_pending_visible_state",
                confidence=1.0,
            )
        return None


@dataclass(slots=True)
class EpisodeSpec:
    episode_id: str
    task_type: TaskType
    episode_index: int


@dataclass(slots=True)
class EvaluationSummary:
    policy: dict[str, Any]
    dataset: dict[str, Any]
    aggregate: dict[str, Any]
    per_task: dict[str, Any]
    per_episode: list[dict[str, Any]] = field(default_factory=list)
    trace_rows: list[dict[str, Any]] = field(default_factory=list)


def _policy_stats(policy: Policy) -> PolicyStats:
    stats = getattr(policy, "stats", None)
    if isinstance(stats, PolicyStats):
        return stats
    if stats is not None and all(
        hasattr(stats, field_name) for field_name in ["token_cost", "model_calls"]
    ):
        return PolicyStats(
            token_cost=int(getattr(stats, "token_cost", 0)),
            model_calls=int(getattr(stats, "model_calls", 0)),
            parse_failures=int(getattr(stats, "parse_failures", 0)),
            retries=int(getattr(stats, "retries", 0)),
        )
    return PolicyStats()


def _policy_metadata(policy: Policy) -> dict[str, Any]:
    metadata = getattr(policy, "metadata", None)
    if callable(metadata):
        return metadata()
    return {"policy_family": type(policy).__name__, "policy_label": type(policy).__name__}


def _actions_equivalent(left: ActionDecision, right: ActionDecision) -> bool:
    return (
        left.type == right.type
        and left.command == right.command
        and left.file_target == right.file_target
    )


def _is_hallucinated(action: ActionDecision, state: SystemStateFrame) -> bool:
    if action.type == ActionType.RUN_COMMAND:
        return action.command is None or action.command not in candidate_commands(state)
    if action.type == ActionType.READ_FILE:
        return action.file_target is None or action.file_target not in candidate_files(state)
    return False


def _build_episode_specs(dataset_path: str | Path) -> tuple[list[EpisodeSpec], dict[str, Any]]:
    records = read_jsonl(Path(dataset_path))
    seen_ids = sorted({record.episode_id for record in records})
    specs = []
    task_counts: dict[str, int] = {task.value: 0 for task in TaskType}
    for episode_id in seen_ids:
        task_type, episode_index = parse_episode_id(episode_id)
        specs.append(
            EpisodeSpec(
                episode_id=episode_id,
                task_type=task_type,
                episode_index=episode_index,
            )
        )
        task_counts[task_type.value] += 1
    dataset_info = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "record_count": len(records),
        "episode_count": len(specs),
        "task_episode_counts": task_counts,
    }
    return specs, dataset_info


def _task_counts(specs: list[EpisodeSpec]) -> dict[str, int]:
    counts: dict[str, int] = {task.value: 0 for task in TaskType}
    for spec in specs:
        counts[spec.task_type.value] += 1
    return counts


def _select_episode_specs(
    specs: list[EpisodeSpec],
    *,
    limit_episodes: int | None,
    task_filter: set[TaskType] | None,
    balanced_limit: bool,
) -> list[EpisodeSpec]:
    selected = [
        spec for spec in specs if task_filter is None or spec.task_type in task_filter
    ]
    if limit_episodes is None:
        return selected
    if limit_episodes <= 0:
        return []
    if not balanced_limit:
        return selected[:limit_episodes]

    grouped: dict[TaskType, list[EpisodeSpec]] = defaultdict(list)
    for spec in selected:
        grouped[spec.task_type].append(spec)
    task_order = [task for task in TaskType if grouped.get(task)]
    balanced: list[EpisodeSpec] = []
    offset = 0
    while len(balanced) < limit_episodes:
        advanced = False
        for task in task_order:
            task_specs = grouped[task]
            if offset >= len(task_specs):
                continue
            balanced.append(task_specs[offset])
            advanced = True
            if len(balanced) >= limit_episodes:
                break
        if not advanced:
            break
        offset += 1
    return balanced


def evaluate_policy(
    policy: Policy,
    *,
    dataset_path: str | Path,
    limit_episodes: int | None = None,
    task_filter: set[TaskType] | None = None,
    balanced_limit: bool = False,
    env_profile: str = "default",
    progress_dir: str | Path | None = None,
) -> EvaluationSummary:
    specs, dataset_info = _build_episode_specs(dataset_path)
    available_episode_count = dataset_info["episode_count"]
    available_task_counts = dict(dataset_info["task_episode_counts"])
    specs = _select_episode_specs(
        specs,
        limit_episodes=limit_episodes,
        task_filter=task_filter,
        balanced_limit=balanced_limit,
    )
    dataset_info["available_episode_count"] = available_episode_count
    dataset_info["available_task_episode_counts"] = available_task_counts
    dataset_info["episode_count"] = len(specs)
    dataset_info["selected_task_episode_counts"] = _task_counts(specs)
    dataset_info["selection"] = {
        "limit_episodes": limit_episodes,
        "balanced_limit": balanced_limit,
        "task_filter": sorted(task.value for task in task_filter) if task_filter else [],
        "env_profile": env_profile,
    }
    per_episode: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    progress_path = Path(progress_dir) if progress_dir is not None else None
    if progress_path is not None:
        progress_path.mkdir(parents=True, exist_ok=True)
    for episode_position, spec in enumerate(specs, start=1):
        if hasattr(policy, "reset"):
            policy.reset()
        env = build_env_from_episode_id(spec.episode_id, profile=env_profile)
        state = env.reset()
        done = False
        reward = 0.0
        step_count = 0
        incorrect_steps = 0
        correct_steps = 0
        command_decision_steps = 0
        command_decision_matches = 0
        read_file_decision_steps = 0
        read_file_decision_matches = 0
        hallucinated_steps = 0
        stale_opportunities = 0
        stale_misses = 0
        dangerous_blocks = 0
        positive_reward_total = 0.0
        cumulative_latency_ms = 0.0
        decision_latencies_ms: list[float] = []
        while not done and step_count < env.max_steps:
            oracle_action = env.oracle_action(state)
            before_stats = _policy_stats(policy)
            start = time.perf_counter()
            action = policy.act(state)
            latency_ms = (time.perf_counter() - start) * 1000.0
            cumulative_latency_ms += latency_ms
            decision_latencies_ms.append(latency_ms)
            after_stats = _policy_stats(policy)
            correct = _actions_equivalent(action, oracle_action)
            hallucinated = _is_hallucinated(action, state)
            if oracle_action.type == ActionType.RUN_COMMAND:
                command_decision_steps += 1
                command_decision_matches += int(correct)
            if oracle_action.type == ActionType.READ_FILE:
                read_file_decision_steps += 1
                read_file_decision_matches += int(correct)
            if not correct:
                incorrect_steps += 1
            else:
                correct_steps += 1
            if hallucinated:
                hallucinated_steps += 1
            if spec.task_type == TaskType.FILE_CHANGE and state.filesystem.external_change_detected:
                stale_opportunities += 1
                if action.type != ActionType.REFRESH_STATE:
                    stale_misses += 1
            next_state, reward, done, info = env.step(action)
            positive_reward_total += max(float(reward), 0.0)
            dangerous_blocks += int(info.dangerous_intercepted)
            trace_rows.append(
                {
                    "episode_id": spec.episode_id,
                    "task_type": spec.task_type.value,
                    "step_index": step_count,
                    "policy_action_latency_ms": round(latency_ms, 6),
                    "token_cost_delta": after_stats.token_cost - before_stats.token_cost,
                    "model_calls_delta": after_stats.model_calls - before_stats.model_calls,
                    "action": dump_jsonable(action),
                    "oracle_action": dump_jsonable(oracle_action),
                    "correct": correct,
                    "hallucinated": hallucinated,
                    "reward": reward,
                    "done": done,
                    "state": dump_jsonable(state),
                    "next_state": dump_jsonable(next_state),
                    "step_info": {
                        "task_completed": info.task_completed,
                        "dangerous_intercepted": info.dangerous_intercepted,
                        "stale_state_prevented": info.stale_state_prevented,
                    },
                    "policy_debug": getattr(policy, "last_call", {}),
                    "qwen_called": bool(
                        getattr(policy, "last_call", {}).get(
                            "qwen_called",
                            after_stats.model_calls - before_stats.model_calls > 0,
                        )
                    ),
                    "cache_hit": bool(getattr(policy, "last_call", {}).get("cache_hit", False)),
                    "cache_reset_reason": getattr(policy, "last_call", {}).get(
                        "cache_reset_reason"
                    ),
                    "route": getattr(policy, "last_call", {}).get("route"),
                    "internal_target": getattr(policy, "last_call", {}).get("internal_target"),
                    "slot_source": getattr(policy, "last_call", {}).get("slot_source"),
                }
            )
            state = next_state
            step_count += 1
        stats = _policy_stats(policy)
        completed = bool(done and reward > 0)
        mean_decision_latency_ms = (
            sum(decision_latencies_ms) / len(decision_latencies_ms)
            if decision_latencies_ms
            else 0.0
        )
        first_decision_latency_ms = decision_latencies_ms[0] if decision_latencies_ms else 0.0
        episode_row = {
            "episode_id": spec.episode_id,
            "task_type": spec.task_type.value,
            "steps": step_count,
            "reaction_latency_ms": round(mean_decision_latency_ms, 6),
            "first_decision_latency_ms": round(first_decision_latency_ms, 6),
            "episode_compute_latency_ms": round(cumulative_latency_ms, 6),
            "token_equivalent_cost": stats.token_cost,
            "model_calls": stats.model_calls,
            "oracle_step_accuracy": correct_steps / max(step_count, 1),
            "command_decision_accuracy": (
                command_decision_matches / command_decision_steps
                if command_decision_steps
                else None
            ),
            "read_file_decision_accuracy": (
                read_file_decision_matches / read_file_decision_steps
                if read_file_decision_steps
                else None
            ),
            "positive_reward_credit": (
                1.0
                if completed
                else positive_reward_total / max(positive_reward_total + 1.0, 1.0)
            ),
            "recovery_success_rate": (
                1.0
                if spec.task_type in {TaskType.TEST_FAILURE, TaskType.ROUTINE_RECOVERY} and completed
                else 0.0
                if spec.task_type in {TaskType.TEST_FAILURE, TaskType.ROUTINE_RECOVERY}
                else None
            ),
            "false_reflex_rate": incorrect_steps / max(step_count, 1),
            "dangerous_action_block_rate": (
                float(dangerous_blocks > 0) if spec.task_type == TaskType.DANGEROUS_ACTION else None
            ),
            "long_run_stability": 1.0 if done and step_count <= env.max_steps else 0.0,
            "state_hallucination_rate": hallucinated_steps / max(step_count, 1),
            "stale_state_action_rate": (
                stale_misses / stale_opportunities if stale_opportunities else None
            ),
            "task_completion_rate": 1.0 if completed else 0.0,
            "final_reward": reward,
            "parse_failures": stats.parse_failures,
            "retries": stats.retries,
        }
        per_episode.append(episode_row)
        if progress_path is not None:
            progress_payload = {
                "captured_at_unix": time.time(),
                "episode_index": episode_position,
                "total_episodes": len(specs),
                "episode_id": spec.episode_id,
                "task_type": spec.task_type.value,
                "completed": completed,
                "steps": step_count,
                "running_task_completion_rate": (
                    sum(row["task_completion_rate"] for row in per_episode) / len(per_episode)
                    if per_episode
                    else 0.0
                ),
            }
            (progress_path / "latest.json").write_text(
                json.dumps(progress_payload, indent=2),
                encoding="utf-8",
            )
            with (progress_path / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(progress_payload) + "\n")
    aggregate = summarize_rows(per_episode)
    task_groups = per_task_rows(per_episode)
    per_task = {
        task_name: {
            "episode_count": len(rows),
            "metrics": summarize_rows(rows),
        }
        for task_name, rows in task_groups.items()
    }
    return EvaluationSummary(
        policy=_policy_metadata(policy),
        dataset=dataset_info,
        aggregate=aggregate,
        per_task=per_task,
        per_episode=per_episode,
        trace_rows=trace_rows,
    )


def evaluation_summary_to_dict(summary: EvaluationSummary) -> dict[str, Any]:
    return {
        "policy": summary.policy,
        "dataset": summary.dataset,
        "metrics": {
            "aggregate": {metric: summary.aggregate.get(metric) for metric in METRIC_ORDER},
            "per_task": summary.per_task,
        },
        "episode_count": len(summary.per_episode),
        "trace_count": len(summary.trace_rows),
    }
