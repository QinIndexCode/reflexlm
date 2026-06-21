from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(slots=True)
class ReflexCoreLossWeights:
    text: float = 0.2
    action: float = 1.0
    command_slot: float = 0.3
    file_slot: float = 0.2
    risk: float = 0.3
    salience: float = 0.2
    prediction_error: float = 0.2
    next_state: float = 0.5


def compute_reflexcore_losses(
    outputs: dict[str, Tensor | None],
    batch: dict[str, Tensor],
    *,
    weights: ReflexCoreLossWeights | None = None,
) -> dict[str, Tensor]:
    weights = weights or ReflexCoreLossWeights()
    text_logits = _require_tensor(outputs, "text_logits")
    action_logits = _require_tensor(outputs, "action_logits")
    command_logits = _require_tensor(outputs, "command_slot_logits")
    file_logits = _require_tensor(outputs, "file_slot_logits")
    risk = _require_tensor(outputs, "risk")
    salience = _require_tensor(outputs, "salience")
    prediction_error = _require_tensor(outputs, "prediction_error")
    next_state = _require_tensor(outputs, "next_state")
    mask = batch.get("loss_mask")
    if isinstance(mask, Tensor):
        mask = mask.to(next_state.device)

    text_loss = _sequence_ce(text_logits, batch["text_targets"])
    action_loss = _sequence_ce(action_logits, batch["action_indices"])
    command_slot_loss = _sequence_ce(
        command_logits,
        batch["command_slot_indices"],
        ignore_index=-100,
    )
    file_slot_loss = _sequence_ce(
        file_logits,
        batch["file_slot_indices"],
        ignore_index=-100,
    )
    risk_loss = _masked_binary_cross_entropy(
        risk,
        batch["risk_targets"].to(risk.device),
        mask,
    )
    salience_loss = _masked_binary_cross_entropy(
        salience,
        batch["salience_targets"].to(salience.device),
        mask,
    )
    prediction_error_loss = _masked_mse(
        prediction_error,
        batch["prediction_error_targets"].to(prediction_error.device),
        mask,
    )
    next_state_feature_mask = batch.get("next_state_loss_mask")
    if isinstance(next_state_feature_mask, Tensor):
        next_state_feature_mask = next_state_feature_mask.to(next_state.device)
    next_state_loss = _masked_mse(
        next_state,
        batch["next_state"].to(next_state.device),
        mask,
        feature_mask=next_state_feature_mask,
    )
    total = (
        weights.text * text_loss
        + weights.action * action_loss
        + weights.command_slot * command_slot_loss
        + weights.file_slot * file_slot_loss
        + weights.risk * risk_loss
        + weights.salience * salience_loss
        + weights.prediction_error * prediction_error_loss
        + weights.next_state * next_state_loss
    )
    losses = {
        "loss": total,
        "text_loss": text_loss,
        "action_loss": action_loss,
        "command_slot_loss": command_slot_loss,
        "file_slot_loss": file_slot_loss,
        "risk_loss": risk_loss,
        "salience_loss": salience_loss,
        "prediction_error_loss": prediction_error_loss,
        "next_state_loss": next_state_loss,
    }
    for name, loss in losses.items():
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} is not finite")
    return losses


def compute_reflexcore_action_loss(
    outputs: dict[str, Tensor | None],
    batch: dict[str, Tensor],
) -> Tensor:
    """Action imitation loss for auxiliary sensory views."""

    return _sequence_ce(
        _require_tensor(outputs, "action_logits"),
        batch["action_indices"],
    )


def _require_tensor(outputs: dict[str, Tensor | None], key: str) -> Tensor:
    value = outputs.get(key)
    if not isinstance(value, Tensor):
        raise KeyError(f"missing tensor output: {key}")
    return value


def _sequence_ce(
    logits: Tensor,
    targets: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    logits = _ensure_sequence(logits)
    targets = _ensure_target_sequence(targets).to(logits.device)
    if ignore_index is not None and bool((targets == ignore_index).all().item()):
        return logits.sum() * 0.0
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def _masked_binary_cross_entropy(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None,
) -> Tensor:
    loss = F.binary_cross_entropy(prediction, target, reduction="none")
    return _masked_mean(loss, mask)


def _masked_mse(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None,
    *,
    feature_mask: Tensor | None = None,
) -> Tensor:
    loss = (prediction - target) ** 2
    return _masked_mean(loss, mask, feature_mask=feature_mask)


def _masked_mean(
    loss: Tensor,
    mask: Tensor | None,
    *,
    feature_mask: Tensor | None = None,
) -> Tensor:
    weight = torch.ones_like(loss)
    if mask is not None:
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(-1)
        weight = weight * mask.to(loss.device, dtype=loss.dtype).expand_as(loss)
    if feature_mask is not None:
        while feature_mask.ndim < loss.ndim:
            feature_mask = feature_mask.unsqueeze(0)
        weight = weight * feature_mask.to(loss.device, dtype=loss.dtype).expand_as(loss)
    return (loss * weight).sum() / weight.sum().clamp(min=1.0)


def _ensure_sequence(tensor: Tensor) -> Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    if tensor.ndim != 3:
        raise ValueError("expected logits with shape [batch, seq, classes]")
    return tensor


def _ensure_target_sequence(tensor: Tensor) -> Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(1)
    if tensor.ndim != 2:
        raise ValueError("expected targets with shape [batch, seq]")
    return tensor
