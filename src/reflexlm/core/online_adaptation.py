from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from reflexlm.core.dataset import (
    ReflexCoreEpisodeDataset,
    ReflexCoreTorchDataset,
    collate_reflexcore_batch,
    collate_reflexcore_sequence_batch,
    observation_from_state,
    read_reflexcore_jsonl,
)
from reflexlm.core.losses import ReflexCoreLossWeights, compute_reflexcore_losses
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
from reflexlm.core.motor import decode_reflexcore_motor
from reflexlm.core.schema import ReflexCoreTrainingExample, dataset_hash
from reflexlm.core.training import file_hash
from reflexlm.runtime.safety import SafetyLayer
from reflexlm.schema import ActionType


@dataclass(slots=True)
class ReflexCoreOnlineAdaptationConfig:
    checkpoint_path: Path
    experience_path: Path
    output_dir: Path
    retention_path: Path | None = None
    holdout_path: Path | None = None
    max_retention_loss_increase: float = 0.0
    max_holdout_loss_increase: float = 0.0
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 1e-4
    device: str = "cpu"
    sequence_mode: bool = True
    max_sequence_len: int | None = 8
    max_text_tokens: int = 128
    trainable_scope: str = "all"


def adapt_reflexcore_from_experience(
    config: ReflexCoreOnlineAdaptationConfig,
) -> dict[str, object]:
    """Fine-tune an existing ReflexCore checkpoint on post-safety experience.

    This is intentionally small and bounded: the input examples are already
    typed ReflexCore transitions, usually produced by `write_experience_jsonl`.
    The function does not execute actions or inspect the host outside the
    provided dataset/checkpoint paths.
    """

    if config.epochs <= 0:
        raise ValueError("epochs must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.max_retention_loss_increase < 0:
        raise ValueError("max_retention_loss_increase must be non-negative")
    if config.max_holdout_loss_increase < 0:
        raise ValueError("max_holdout_loss_increase must be non-negative")
    if config.trainable_scope not in {"all", "world_model_only"}:
        raise ValueError("trainable_scope must be 'all' or 'world_model_only'")

    examples = read_reflexcore_jsonl(config.experience_path)
    if not examples:
        raise ValueError("experience dataset must contain at least one example")
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
    model = ReflexCoreV0(ReflexCoreV0Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(config.device)
    model.to(device)
    trainable_counts = _set_trainable_scope(model, config.trainable_scope)
    dataset, collate_fn = _dataset_and_collate(
        examples,
        sequence_mode=config.sequence_mode,
        max_sequence_len=config.max_sequence_len,
        max_text_tokens=config.max_text_tokens,
    )
    if dataset.input_dim != model.config.input_dim:
        raise ValueError(
            f"experience input_dim {dataset.input_dim} does not match checkpoint "
            f"input_dim {model.config.input_dim}"
        )
    loader = _loader(dataset, collate_fn, batch_size=config.batch_size)
    retention_examples = (
        read_reflexcore_jsonl(config.retention_path)
        if config.retention_path is not None
        else []
    )
    retention_loader: DataLoader | None = None
    if retention_examples:
        retention_loader = _validated_loader(
            retention_examples,
            model_input_dim=model.config.input_dim,
            sequence_mode=config.sequence_mode,
            max_sequence_len=config.max_sequence_len,
            max_text_tokens=config.max_text_tokens,
            batch_size=config.batch_size,
        )
    holdout_examples = (
        read_reflexcore_jsonl(config.holdout_path)
        if config.holdout_path is not None
        else []
    )
    holdout_loader: DataLoader | None = None
    if holdout_examples:
        holdout_loader = _validated_loader(
            holdout_examples,
            model_input_dim=model.config.input_dim,
            sequence_mode=config.sequence_mode,
            max_sequence_len=config.max_sequence_len,
            max_text_tokens=config.max_text_tokens,
            batch_size=config.batch_size,
        )
    weights = ReflexCoreLossWeights()
    before_metrics = _mean_losses(model, loader, device=device, weights=weights)
    before = before_metrics["loss"]
    retention_before = (
        _mean_losses(model, retention_loader, device=device, weights=weights)["loss"]
        if retention_loader is not None
        else None
    )
    holdout_before = (
        _mean_losses(model, holdout_loader, device=device, weights=weights)["loss"]
        if holdout_loader is not None
        else None
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("trainable_scope selected no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["observation_vectors"],
                batch["text_tokens"],
                action_indices=batch["action_indices"],
            )
            losses = compute_reflexcore_losses(outputs, batch, weights=weights)
            losses["loss"].backward()
            optimizer.step()
            total += float(losses["loss"].detach().cpu().item())
            count += 1
        epoch_loss = total / max(count, 1)
        if not torch.isfinite(torch.tensor(epoch_loss)):
            raise FloatingPointError("online adaptation loss is not finite")
        history.append({"epoch": float(epoch), "loss": epoch_loss})
    after_metrics = _mean_losses(model, loader, device=device, weights=weights)
    after = after_metrics["loss"]
    retention_after = (
        _mean_losses(model, retention_loader, device=device, weights=weights)["loss"]
        if retention_loader is not None
        else None
    )
    holdout_after = (
        _mean_losses(model, holdout_loader, device=device, weights=weights)["loss"]
        if holdout_loader is not None
        else None
    )
    retention_gate = _retention_gate(
        before=retention_before,
        after=retention_after,
        max_loss_increase=config.max_retention_loss_increase,
    )
    holdout_gate = _loss_increase_gate(
        before=holdout_before,
        after=holdout_after,
        max_loss_increase=config.max_holdout_loss_increase,
    )
    loss_not_increased = after <= before
    retention_passed = retention_gate["passed"]
    holdout_passed = holdout_gate["passed"]
    accepted = (
        loss_not_increased
        and retention_passed is not False
        and holdout_passed is not False
    )
    rejected_reason = _rejected_reason(
        loss_not_increased=loss_not_increased,
        retention_passed=retention_passed,
        holdout_passed=holdout_passed,
    )
    pe_motor_probe = _prediction_error_motor_probe(
        before_model=ReflexCoreV0(ReflexCoreV0Config(**checkpoint["config"])),
        before_state_dict=checkpoint["model_state_dict"],
        after_model=model,
        examples=examples,
        device=device,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    adapted_checkpoint = config.output_dir / "reflexcore_v0_adapted.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "base_checkpoint": str(config.checkpoint_path),
            "experience_path": str(config.experience_path),
            "experience_hash": dataset_hash(examples),
            "retention_path": str(config.retention_path) if config.retention_path else None,
            "retention_hash": dataset_hash(retention_examples) if retention_examples else None,
            "holdout_path": str(config.holdout_path) if config.holdout_path else None,
            "holdout_hash": dataset_hash(holdout_examples) if holdout_examples else None,
            "history": history,
            "accepted": accepted,
            "rejected_reason": rejected_reason,
            "trainable_scope": config.trainable_scope,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "prediction_error_motor_probe": pe_motor_probe,
        },
        adapted_checkpoint,
    )
    live_prediction_errors = [
        float(example.next_observation.runtime_evidence.observed_prediction_error)
        for example in examples
        if example.next_observation.runtime_evidence.observed_prediction_error is not None
    ]
    report = {
        "config": _json_config(config),
        "experience_examples": len(examples),
        "experience_hash": dataset_hash(examples),
        "source_values": sorted({example.source for example in examples}),
        "live_prediction_error_examples": len(live_prediction_errors),
        "live_prediction_error_target_mean": (
            sum(live_prediction_errors) / len(live_prediction_errors)
            if live_prediction_errors
            else None
        ),
        "retention_examples": len(retention_examples),
        "retention_hash": dataset_hash(retention_examples) if retention_examples else None,
        "holdout_examples": len(holdout_examples),
        "holdout_hash": dataset_hash(holdout_examples) if holdout_examples else None,
        "before_loss": before,
        "after_loss": after,
        "loss_delta": before - after,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "prediction_error_loss_delta": (
            before_metrics["prediction_error_loss"]
            - after_metrics["prediction_error_loss"]
        ),
        "next_state_loss_delta": (
            before_metrics["next_state_loss"] - after_metrics["next_state_loss"]
        ),
        "prediction_error_motor_probe": pe_motor_probe,
        "loss_not_increased": loss_not_increased,
        "retention_before_loss": retention_before,
        "retention_after_loss": retention_after,
        "retention_loss_delta": (
            retention_before - retention_after
            if retention_before is not None and retention_after is not None
            else None
        ),
        "retention_gate": retention_gate,
        "holdout_before_loss": holdout_before,
        "holdout_after_loss": holdout_after,
        "holdout_loss_delta": (
            holdout_before - holdout_after
            if holdout_before is not None and holdout_after is not None
            else None
        ),
        "holdout_gate": holdout_gate,
        "accepted": accepted,
        "rejected_reason": rejected_reason,
        "base_checkpoint": str(config.checkpoint_path),
        "adapted_checkpoint": str(adapted_checkpoint),
        "adapted_model_hash": file_hash(adapted_checkpoint),
        "history": history,
        "trainable_scope": config.trainable_scope,
        "trainable_parameter_count": trainable_counts["trainable"],
        "frozen_parameter_count": trainable_counts["frozen"],
        "free_shell_generation": False,
        "gui_or_vision": False,
        "claim_boundary": (
            "Online adaptation only updates ReflexCore V0 from already recorded "
            "bounded terminal/process/filesystem/time experience examples."
        ),
    }
    (config.output_dir / "online_adaptation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def _prediction_error_motor_probe(
    *,
    before_model: ReflexCoreV0,
    before_state_dict: dict[str, torch.Tensor],
    after_model: ReflexCoreV0,
    examples: list[ReflexCoreTrainingExample],
    device: torch.device,
) -> dict[str, object]:
    live_examples = [
        example
        for example in examples
        if example.next_observation.runtime_evidence.observed_prediction_error is not None
    ]
    if not live_examples:
        return {
            "available": False,
            "examples": 0,
            "base_refresh_count": 0,
            "adapted_refresh_count": 0,
            "base_safe_refresh_count": 0,
            "adapted_safe_refresh_count": 0,
            "adapted_safety_allowed_count": 0,
            "refresh_gain": 0,
            "changed_to_refresh_count": 0,
            "changed_to_safe_refresh_count": 0,
            "mean_prediction_error_delta": None,
        }
    before_model.load_state_dict(before_state_dict)
    before_model.to(device)
    after_model.to(device)
    before_model.eval()
    after_model.eval()
    base_refresh_count = 0
    adapted_refresh_count = 0
    base_safe_refresh_count = 0
    adapted_safe_refresh_count = 0
    adapted_safety_allowed_count = 0
    pe_deltas: list[float] = []
    changed_to_refresh = 0
    changed_to_safe_refresh = 0
    with torch.no_grad():
        for example in live_examples:
            base_action, base_safe_action, base_allowed, base_pe = _decode_probe_action(
                before_model,
                example,
                device=device,
            )
            (
                adapted_action,
                adapted_safe_action,
                adapted_allowed,
                adapted_pe,
            ) = _decode_probe_action(after_model, example, device=device)
            base_refresh = base_action.type == ActionType.REFRESH_STATE
            adapted_refresh = adapted_action.type == ActionType.REFRESH_STATE
            base_safe_refresh = base_safe_action.type == ActionType.REFRESH_STATE
            adapted_safe_refresh = adapted_safe_action.type == ActionType.REFRESH_STATE
            base_refresh_count += int(base_refresh)
            adapted_refresh_count += int(adapted_refresh)
            base_safe_refresh_count += int(base_safe_refresh)
            adapted_safe_refresh_count += int(adapted_safe_refresh)
            adapted_safety_allowed_count += int(adapted_allowed)
            changed_to_refresh += int((not base_refresh) and adapted_refresh)
            changed_to_safe_refresh += int(
                (not base_safe_refresh) and adapted_safe_refresh and adapted_allowed
            )
            pe_deltas.append(adapted_pe - base_pe)
    return {
        "available": True,
        "examples": len(live_examples),
        "base_refresh_count": base_refresh_count,
        "adapted_refresh_count": adapted_refresh_count,
        "base_safe_refresh_count": base_safe_refresh_count,
        "adapted_safe_refresh_count": adapted_safe_refresh_count,
        "adapted_safety_allowed_count": adapted_safety_allowed_count,
        "refresh_gain": adapted_refresh_count - base_refresh_count,
        "changed_to_refresh_count": changed_to_refresh,
        "changed_to_safe_refresh_count": changed_to_safe_refresh,
        "mean_prediction_error_delta": sum(pe_deltas) / len(pe_deltas),
    }


def _decode_probe_action(
    model: ReflexCoreV0,
    example: ReflexCoreTrainingExample,
    *,
    device: torch.device,
):
    state = example.observation.to_state_frame()
    state = state.model_copy(
        update={
            "runtime_evidence": state.runtime_evidence.model_copy(
                update={
                    "model_prediction_error": None,
                    "observed_prediction_error": None,
                    "prediction_error_delta": None,
                }
            )
        }
    )
    probe_observation = observation_from_state(
        state,
        vocab_size=model.config.vocab_size,
        max_text_tokens=len(example.observation.text_tokens),
    )
    if len(probe_observation.vector) != model.config.input_dim:
        raise ValueError("probe observation vector dimension mismatch")
    vector = torch.tensor(
        probe_observation.vector,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).unsqueeze(0)
    text = torch.tensor(
        probe_observation.text_tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).unsqueeze(0)
    outputs = model(vector, text)
    decoded = decode_reflexcore_motor(outputs, state)
    safety = SafetyLayer().enforce(decoded.action, state.goal, state)
    return decoded.action, safety.action, safety.allowed, decoded.prediction_error


def _loader(dataset, collate_fn, *, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


def _set_trainable_scope(
    model: ReflexCoreV0,
    scope: str,
) -> dict[str, int]:
    if scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif scope == "world_model_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for module in (model.next_state_head, model.prediction_error_head):
            for parameter in module.parameters():
                parameter.requires_grad = True
    else:
        raise ValueError("unknown trainable scope")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    return {"trainable": trainable, "frozen": frozen}


def _validated_loader(
    examples: list,
    *,
    model_input_dim: int,
    sequence_mode: bool,
    max_sequence_len: int | None,
    max_text_tokens: int,
    batch_size: int,
) -> DataLoader:
    dataset, collate_fn = _dataset_and_collate(
        examples,
        sequence_mode=sequence_mode,
        max_sequence_len=max_sequence_len,
        max_text_tokens=max_text_tokens,
    )
    if dataset.input_dim != model_input_dim:
        raise ValueError(
            f"dataset input_dim {dataset.input_dim} does not match checkpoint "
            f"input_dim {model_input_dim}"
        )
    return _loader(dataset, collate_fn, batch_size=batch_size)


def _dataset_and_collate(
    examples: list,
    *,
    sequence_mode: bool,
    max_sequence_len: int | None,
    max_text_tokens: int,
):
    if sequence_mode:
        return (
            ReflexCoreEpisodeDataset(
                examples,
                max_text_tokens=max_text_tokens,
                max_sequence_len=max_sequence_len,
            ),
            collate_reflexcore_sequence_batch,
        )
    return (
        ReflexCoreTorchDataset(examples, max_text_tokens=max_text_tokens),
        collate_reflexcore_batch,
    )


def _mean_losses(
    model: ReflexCoreV0,
    loader: DataLoader,
    *,
    device: torch.device,
    weights: ReflexCoreLossWeights,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                batch["observation_vectors"],
                batch["text_tokens"],
                action_indices=batch["action_indices"],
            )
            losses = compute_reflexcore_losses(outputs, batch, weights=weights)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu().item())
            count += 1
    mean_losses = {
        key: value / max(count, 1)
        for key, value in totals.items()
    }
    if not mean_losses:
        raise ValueError("online adaptation loader produced no batches")
    for key, value in mean_losses.items():
        if not torch.isfinite(torch.tensor(value)):
            raise FloatingPointError(f"online adaptation mean {key} is not finite")
    return mean_losses


def _retention_gate(
    *,
    before: float | None,
    after: float | None,
    max_loss_increase: float,
) -> dict[str, object]:
    return _loss_increase_gate(
        before=before,
        after=after,
        max_loss_increase=max_loss_increase,
    )


def _loss_increase_gate(
    *,
    before: float | None,
    after: float | None,
    max_loss_increase: float,
) -> dict[str, object]:
    if before is None or after is None:
        return {
            "available": False,
            "passed": None,
            "max_loss_increase": max_loss_increase,
        }
    loss_increase = after - before
    return {
        "available": True,
        "passed": loss_increase <= max_loss_increase,
        "before_loss": before,
        "after_loss": after,
        "loss_increase": loss_increase,
        "max_loss_increase": max_loss_increase,
    }


def _rejected_reason(
    *,
    loss_not_increased: bool,
    retention_passed: object,
    holdout_passed: object,
) -> str | None:
    if not loss_not_increased:
        return "experience_loss_increased"
    if retention_passed is False:
        return "retention_loss_increased"
    if holdout_passed is False:
        return "holdout_loss_increased"
    return None


def _json_config(config: ReflexCoreOnlineAdaptationConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["checkpoint_path"] = str(config.checkpoint_path)
    payload["experience_path"] = str(config.experience_path)
    payload["retention_path"] = str(config.retention_path) if config.retention_path else None
    payload["holdout_path"] = str(config.holdout_path) if config.holdout_path else None
    payload["output_dir"] = str(config.output_dir)
    return payload
