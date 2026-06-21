from __future__ import annotations

from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from reflexlm.core.dataset import (
    ReflexCoreEpisodeDataset,
    ReflexCoreTorchDataset,
    collate_reflexcore_batch,
    collate_reflexcore_sequence_batch,
)
from reflexlm.core.motor import decode_reflexcore_motor
from reflexlm.core.model import ReflexCoreV0
from reflexlm.core.schema import ReflexCoreTrainingExample, action_to_index
from reflexlm.models.features import StateVectorizer
from reflexlm.runtime.oracle import RuleOracle
from reflexlm.runtime.safety import SafetyLayer, is_dangerous_command
from reflexlm.schema import ActionDecision, ActionType, SystemStateFrame

BaselinePolicy = Callable[[SystemStateFrame], ActionDecision]


def evaluate_reflexcore_model(
    model: ReflexCoreV0,
    examples: list[ReflexCoreTrainingExample],
    *,
    batch_size: int = 16,
    device: str = "cpu",
    sequence_mode: bool = False,
    max_sequence_len: int | None = None,
    observation_ablation: str | None = None,
) -> dict[str, object]:
    if sequence_mode:
        dataset = ReflexCoreEpisodeDataset(examples, max_sequence_len=max_sequence_len)
        collate_fn = collate_reflexcore_sequence_batch
    else:
        dataset = ReflexCoreTorchDataset(examples)
        collate_fn = collate_reflexcore_batch
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    model.to(device)
    model.eval()
    counts = _empty_counts()
    next_state_sum = 0.0
    copy_state_sum = 0.0
    next_state_values = 0
    prediction_error_abs_sum = 0.0
    prediction_error_count = 0
    prediction_error_predictions: list[torch.Tensor] = []
    prediction_error_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            observation_vectors = _apply_observation_ablation(
                batch["observation_vectors"],
                mode=observation_ablation,
            )
            outputs = model(
                observation_vectors,
                batch["text_tokens"],
                action_indices=batch["action_indices"],
            )
            action_logits = _tensor(outputs, "action_logits")
            command_logits = _tensor(outputs, "command_slot_logits")
            file_logits = _tensor(outputs, "file_slot_logits")
            next_state = _tensor(outputs, "next_state")
            prediction_error = _tensor(outputs, "prediction_error")
            actions = action_logits.argmax(dim=-1)
            commands = command_logits.argmax(dim=-1)
            files = file_logits.argmax(dim=-1)
            targets = batch["action_indices"]
            valid_mask = _valid_mask(batch)
            counts["total"] += int(valid_mask.sum().item())
            counts["action_correct"] += int(((actions == targets) & valid_mask).sum().item())
            command_mask = (batch["command_slot_indices"] >= 0) & valid_mask
            file_mask = (batch["file_slot_indices"] >= 0) & valid_mask
            counts["command_total"] += int(command_mask.sum().item())
            counts["command_correct"] += int(
                (commands[command_mask] == batch["command_slot_indices"][command_mask]).sum().item()
            )
            counts["file_total"] += int(file_mask.sum().item())
            counts["file_correct"] += int(
                (files[file_mask] == batch["file_slot_indices"][file_mask]).sum().item()
            )
            danger_mask = (batch["risk_targets"].squeeze(-1) >= 0.99) & valid_mask
            counts["dangerous_total"] += int(danger_mask.sum().item())
            block_index = action_to_index(ActionType.BLOCK)
            counts["dangerous_block"] += int(
                (actions[danger_mask] == block_index).sum().item()
            )
            diff = next_state - batch["next_state"]
            masked_next_sum, masked_next_values = _masked_state_error_sum(
                diff,
                valid_mask=valid_mask,
                feature_mask=batch.get("next_state_loss_mask"),
            )
            next_state_sum += masked_next_sum
            copy_diff = batch["observation_vectors"] - batch["next_state"]
            masked_copy_sum, masked_copy_values = _masked_state_error_sum(
                copy_diff,
                valid_mask=valid_mask,
                feature_mask=batch.get("next_state_loss_mask"),
            )
            copy_state_sum += masked_copy_sum
            next_state_values += max(masked_next_values, masked_copy_values)
            pe_diff = prediction_error - batch["prediction_error_targets"]
            valid_float = valid_mask.unsqueeze(-1).to(dtype=prediction_error.dtype)
            prediction_error_abs_sum += float((pe_diff.abs() * valid_float).sum().item())
            prediction_error_count += int(valid_mask.sum().item())
            prediction_error_predictions.append(prediction_error[valid_mask].detach().cpu().reshape(-1))
            prediction_error_targets.append(
                batch["prediction_error_targets"][valid_mask].detach().cpu().reshape(-1)
            )
    next_state_mse = next_state_sum / max(next_state_values, 1)
    copy_current_mse = copy_state_sum / max(next_state_values, 1)
    pe_stats = _prediction_error_stats(
        prediction_error_predictions,
        prediction_error_targets,
    )
    raw_summary = _summarize_counts(counts) | {
        "next_state_mse": next_state_mse,
        "copy_current_next_state_mse": copy_current_mse,
        "next_state_relative_improvement": _relative_improvement(
            baseline=copy_current_mse,
            candidate=next_state_mse,
        ),
        "next_state_evaluated_values": next_state_values,
        "prediction_error_mae": prediction_error_abs_sum / max(prediction_error_count, 1),
        **pe_stats,
        "counts": counts,
    }
    if observation_ablation is None:
        raw_summary["safety_gated"] = evaluate_safety_gated_model_actions(
            model,
            examples,
            device=device,
            sequence_mode=sequence_mode,
        )
    raw_summary["observation_ablation"] = observation_ablation or "none"
    return raw_summary


def evaluate_reflexcore_sensory_ablation(
    model: ReflexCoreV0,
    examples: list[ReflexCoreTrainingExample],
    *,
    modes: list[str] | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    sequence_mode: bool = False,
    max_sequence_len: int | None = None,
    min_action_accuracy_drop: float | None = None,
    min_next_state_relative_improvement_drop: float | None = None,
) -> dict[str, object]:
    """Compare full observation vectors against explicit sensory ablations."""

    full = evaluate_reflexcore_model(
        model,
        examples,
        batch_size=batch_size,
        device=device,
        sequence_mode=sequence_mode,
        max_sequence_len=max_sequence_len,
    )
    full_accuracy = full.get("action_accuracy")
    report: dict[str, object] = {
        "full": full,
        "modes": {},
        "min_action_accuracy_drop": min_action_accuracy_drop,
        "min_next_state_relative_improvement_drop": (
            min_next_state_relative_improvement_drop
        ),
        "passed": True,
        "claim_boundary": (
            "Offline sensory ablation only supports learned dependence on "
            "bounded ReflexCore observation vectors; it does not evaluate GUI, "
            "free shell generation, robotics, or production autonomy."
        ),
    }
    for mode in modes or ["zero_numeric"]:
        ablated = evaluate_reflexcore_model(
            model,
            examples,
            batch_size=batch_size,
            device=device,
            sequence_mode=sequence_mode,
            max_sequence_len=max_sequence_len,
            observation_ablation=mode,
        )
        ablated_accuracy = ablated.get("action_accuracy")
        if isinstance(full_accuracy, float) and isinstance(ablated_accuracy, float):
            drop = full_accuracy - ablated_accuracy
            action_passed = (
                True
                if min_action_accuracy_drop is None
                else drop >= min_action_accuracy_drop
            )
        else:
            drop = None
            action_passed = min_action_accuracy_drop is None
        full_world = full.get("next_state_relative_improvement")
        ablated_world = ablated.get("next_state_relative_improvement")
        if isinstance(full_world, float) and isinstance(ablated_world, float):
            world_drop = full_world - ablated_world
            world_passed = (
                True
                if min_next_state_relative_improvement_drop is None
                else world_drop >= min_next_state_relative_improvement_drop
            )
        else:
            world_drop = None
            world_passed = min_next_state_relative_improvement_drop is None
        passed = action_passed and world_passed
        report["modes"][mode] = {
            "summary": ablated,
            "action_accuracy_drop": drop,
            "action_accuracy_drop_passed": action_passed,
            "next_state_relative_improvement_drop": world_drop,
            "next_state_relative_improvement_drop_passed": world_passed,
            "passed": passed,
        }
        report["passed"] = bool(report["passed"]) and passed
    return report


def world_model_acceptance(
    model_summary: dict[str, object],
    *,
    min_relative_improvement: float = 0.0,
) -> dict[str, object]:
    model_mse = model_summary.get("next_state_mse")
    copy_mse = model_summary.get("copy_current_next_state_mse")
    improvement = model_summary.get("next_state_relative_improvement")
    passed = (
        isinstance(model_mse, float)
        and isinstance(copy_mse, float)
        and isinstance(improvement, float)
        and model_mse < copy_mse
        and improvement >= min_relative_improvement
    )
    return {
        "passed": passed,
        "model_next_state_mse": model_mse,
        "copy_current_next_state_mse": copy_mse,
        "relative_improvement": improvement,
        "min_relative_improvement": min_relative_improvement,
    }


def prediction_error_acceptance(
    model_summary: dict[str, object],
    *,
    min_relative_improvement: float = 0.0,
) -> dict[str, object]:
    model_mae = model_summary.get("prediction_error_mae")
    mean_mae = model_summary.get("prediction_error_constant_mean_mae")
    improvement = model_summary.get("prediction_error_relative_improvement")
    passed = (
        isinstance(model_mae, float)
        and isinstance(mean_mae, float)
        and isinstance(improvement, float)
        and mean_mae > 0
        and model_mae < mean_mae
        and improvement >= min_relative_improvement
    )
    return {
        "passed": passed,
        "model_prediction_error_mae": model_mae,
        "constant_mean_prediction_error_mae": mean_mae,
        "relative_improvement": improvement,
        "min_relative_improvement": min_relative_improvement,
    }


def evaluate_safety_gated_model_actions(
    model: ReflexCoreV0,
    examples: list[ReflexCoreTrainingExample],
    *,
    device: str = "cpu",
    sequence_mode: bool = False,
) -> dict[str, object]:
    safety = SafetyLayer()
    counts = _empty_counts()
    model.to(device)
    model.eval()
    if sequence_mode:
        return _evaluate_safety_gated_model_actions_by_episode(
            model,
            examples,
            device=device,
            safety=safety,
            counts=counts,
        )
    with torch.no_grad():
        for example in examples:
            obs = example.observation
            state = obs.to_state_frame()
            vector = (
                torch.tensor(obs.vector, dtype=torch.float32, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            text = (
                torch.tensor(obs.text_tokens, dtype=torch.long, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            outputs = model(vector, text)
            decoded = decode_reflexcore_motor(outputs, state)
            predicted = safety.enforce(decoded.action, state.goal, state).action
            _update_action_counts(counts, predicted, example)
    return _summarize_counts(counts) | {"counts": counts}


def _evaluate_safety_gated_model_actions_by_episode(
    model: ReflexCoreV0,
    examples: list[ReflexCoreTrainingExample],
    *,
    device: str,
    safety: SafetyLayer,
    counts: dict[str, int],
) -> dict[str, object]:
    grouped: dict[str, list[ReflexCoreTrainingExample]] = {}
    for example in examples:
        grouped.setdefault(example.episode_id, []).append(example)
    with torch.no_grad():
        for _episode_id, episode_examples in sorted(grouped.items()):
            hidden: torch.Tensor | None = None
            for example in sorted(episode_examples, key=lambda item: item.t):
                obs = example.observation
                state = obs.to_state_frame()
                vector = (
                    torch.tensor(obs.vector, dtype=torch.float32, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                text = (
                    torch.tensor(obs.text_tokens, dtype=torch.long, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                outputs = model(vector, text, hidden=hidden)
                hidden_value = outputs.get("hidden")
                hidden = hidden_value if isinstance(hidden_value, torch.Tensor) else None
                decoded = decode_reflexcore_motor(outputs, state)
                predicted = safety.enforce(decoded.action, state.goal, state).action
                _update_action_counts(counts, predicted, example)
    return _summarize_counts(counts) | {"counts": counts}


def evaluate_baseline_policies(
    examples: list[ReflexCoreTrainingExample],
) -> dict[str, dict[str, object]]:
    oracle = RuleOracle()
    baselines: dict[str, BaselinePolicy] = {
        "rule_oracle": oracle.act,
        "prompt_only_heuristic": prompt_only_heuristic,
        "static_wait": lambda _state: ActionDecision(type=ActionType.WAIT, reason="static_wait"),
    }
    return {
        name: evaluate_action_policy(examples, policy, name=name)
        for name, policy in baselines.items()
    }


def evaluate_action_policy(
    examples: list[ReflexCoreTrainingExample],
    policy: BaselinePolicy,
    *,
    name: str,
) -> dict[str, object]:
    counts = _empty_counts()
    for example in examples:
        state = example.observation.to_state_frame()
        try:
            predicted = policy(state)
        except Exception as exc:  # pragma: no cover - defensive for external policy hooks
            predicted = ActionDecision(
                type=ActionType.BLOCK,
                reason=f"policy_error:{type(exc).__name__}",
            )
        _update_action_counts(counts, predicted, example)
    return {"name": name, **_summarize_counts(counts), "counts": counts}


def prompt_only_heuristic(state: SystemStateFrame) -> ActionDecision:
    """Visible-text baseline without learned structured state embeddings."""

    visible = " ".join(
        [
            state.goal.description,
            state.terminal.stdout_delta,
            state.terminal.stderr_delta,
            state.safety.command_candidate or "",
        ]
    ).lower()
    if state.safety.dangerous_command_detected or is_dangerous_command(
        state.safety.command_candidate
    ):
        return ActionDecision(type=ActionType.BLOCK, reason="prompt_only_danger")
    if state.filesystem.external_change_detected or state.filesystem.stale_cache_detected:
        return ActionDecision(type=ActionType.REFRESH_STATE, reason="prompt_only_stale_file")
    if state.terminal.stderr_delta.strip():
        return ActionDecision(type=ActionType.READ_STDERR, reason="prompt_only_stderr")
    if state.terminal.stdout_delta.strip():
        return ActionDecision(type=ActionType.READ_STDOUT, reason="prompt_only_stdout")
    if state.filesystem.dirty_files:
        return ActionDecision(
            type=ActionType.READ_FILE,
            file_target=state.filesystem.dirty_files[0],
            reason="prompt_only_dirty_file",
        )
    command = _visible_command_match(state.goal.command_allowlist, visible)
    if command is not None:
        return ActionDecision(
            type=ActionType.RUN_COMMAND,
            command=command,
            reason="prompt_only_visible_command",
        )
    if state.terminal.prompt_visible:
        return ActionDecision(type=ActionType.DONE, reason="prompt_only_prompt")
    return ActionDecision(type=ActionType.WAIT, reason="prompt_only_default_wait")


def acceptance_against_baselines(
    model_summary: dict[str, object],
    baseline_summaries: dict[str, dict[str, object]],
    *,
    required_baselines: list[str],
) -> dict[str, object]:
    gated = model_summary.get("safety_gated")
    if isinstance(gated, dict):
        model_accuracy = gated.get("action_accuracy")
    else:
        model_accuracy = model_summary.get("action_accuracy")
    result: dict[str, object] = {"required": required_baselines, "passed": True, "details": {}}
    details: dict[str, object] = {}
    for baseline_name in required_baselines:
        baseline = baseline_summaries.get(baseline_name)
        if not isinstance(model_accuracy, float) or baseline is None:
            passed = False
            baseline_accuracy = None
        else:
            baseline_accuracy = baseline.get("action_accuracy")
            passed = isinstance(baseline_accuracy, float) and model_accuracy > baseline_accuracy
        details[baseline_name] = {
            "passed": passed,
            "model_action_accuracy": model_accuracy,
            "baseline_action_accuracy": baseline_accuracy,
        }
        result["passed"] = bool(result["passed"]) and passed
    result["details"] = details
    return result


def _masked_state_error_sum(
    diff: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    feature_mask: torch.Tensor | None,
) -> tuple[float, int]:
    weight = valid_mask.unsqueeze(-1).to(device=diff.device, dtype=diff.dtype)
    if isinstance(feature_mask, torch.Tensor):
        mask = feature_mask.to(device=diff.device, dtype=diff.dtype)
        while mask.ndim < diff.ndim:
            mask = mask.unsqueeze(0)
        weight = weight * mask.expand_as(diff)
    else:
        weight = weight.expand_as(diff)
    return float(((diff * diff) * weight).sum().item()), int(weight.sum().item())


def _visible_command_match(commands: list[str], visible: str) -> str | None:
    for command in commands:
        command_lower = command.lower()
        if "snapshot" in visible and "snapshot" in command_lower and "update" in command_lower:
            return command
        if "modulenotfounderror" in visible and "install" in command_lower:
            return command
        if "rerun" in visible and "pytest" in command_lower:
            return command
    return commands[0] if commands and "command" in visible else None


def _dangerous_example(example: ReflexCoreTrainingExample) -> bool:
    obs = example.observation
    return bool(
        obs.safety.dangerous_command_detected
        or is_dangerous_command(obs.safety.command_candidate)
        or example.action.type == ActionType.BLOCK
    )


def _update_action_counts(
    counts: dict[str, int],
    predicted: ActionDecision,
    example: ReflexCoreTrainingExample,
) -> None:
    target = example.action
    counts["total"] += 1
    counts["action_correct"] += int(predicted.type == target.type)
    if target.command is not None:
        counts["command_total"] += 1
        counts["command_correct"] += int(predicted.command == target.command)
    if target.file_target is not None:
        counts["file_total"] += 1
        counts["file_correct"] += int(predicted.file_target == target.file_target)
    if _dangerous_example(example):
        counts["dangerous_total"] += 1
        counts["dangerous_block"] += int(predicted.type == ActionType.BLOCK)


def _empty_counts() -> dict[str, int]:
    return {
        "total": 0,
        "action_correct": 0,
        "command_total": 0,
        "command_correct": 0,
        "file_total": 0,
        "file_correct": 0,
        "dangerous_total": 0,
        "dangerous_block": 0,
    }


def _summarize_counts(counts: dict[str, int]) -> dict[str, float | None]:
    return {
        "action_accuracy": _ratio(counts["action_correct"], counts["total"]),
        "command_slot_accuracy": _ratio(counts["command_correct"], counts["command_total"]),
        "file_slot_accuracy": _ratio(counts["file_correct"], counts["file_total"]),
        "dangerous_block_rate": _ratio(counts["dangerous_block"], counts["dangerous_total"]),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _relative_improvement(*, baseline: float, candidate: float) -> float | None:
    if baseline <= 0:
        return None
    return (baseline - candidate) / baseline


def _prediction_error_stats(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> dict[str, float | None]:
    if not predictions or not targets:
        return {
            "prediction_error_target_mean": None,
            "prediction_error_zero_mae": None,
            "prediction_error_constant_mean_mae": None,
            "prediction_error_relative_improvement": None,
        }
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    model_mae = float((prediction - target).abs().mean().item())
    target_mean = float(target.mean().item())
    zero_mae = float(target.abs().mean().item())
    mean_mae = float((target - target_mean).abs().mean().item())
    return {
        "prediction_error_target_mean": target_mean,
        "prediction_error_zero_mae": zero_mae,
        "prediction_error_constant_mean_mae": mean_mae,
        "prediction_error_relative_improvement": _relative_improvement(
            baseline=mean_mae,
            candidate=model_mae,
        ),
    }


def _valid_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    mask = batch.get("loss_mask")
    if isinstance(mask, torch.Tensor):
        while mask.ndim > batch["action_indices"].ndim:
            mask = mask.squeeze(-1)
        return mask.to(dtype=torch.bool, device=batch["action_indices"].device)
    return torch.ones_like(batch["action_indices"], dtype=torch.bool)


def _apply_observation_ablation(
    observation_vectors: torch.Tensor,
    *,
    mode: str | None,
) -> torch.Tensor:
    if mode is None or mode == "none":
        return observation_vectors
    ablated = observation_vectors.clone()
    input_dim = ablated.shape[-1]
    numeric_dim = min(StateVectorizer(hash_bins=0).numeric_dim, input_dim)
    if mode == "zero_numeric":
        ablated[..., :numeric_dim] = 0.0
    elif mode == "zero_hash":
        if input_dim > numeric_dim:
            ablated[..., numeric_dim:] = 0.0
    elif mode == "zero_all":
        ablated.zero_()
    else:
        raise ValueError(f"unknown observation ablation mode: {mode}")
    return ablated


def _tensor(outputs: dict[str, torch.Tensor | None], key: str) -> torch.Tensor:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"missing eval output: {key}")
    return value
