from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from reflexlm.baselines.flat_model import FlatTextBaselineModel, FlatTextModelConfig
from reflexlm.data.jsonl import read_jsonl
from reflexlm.models.features import (
    StateVectorizer,
    ACTION_ORDER,
    HARD_TASK_TYPES,
    action_to_index,
    command_slot_target,
    delta_norm_target,
    file_slot_target,
    route_target,
    risk_target,
    salience_target,
    urgency_target,
    valid_action_mask,
)
from reflexlm.models.nsi_model import NSIModelConfig, NSIReflexModel
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER, internal_target_for_state
from reflexlm.schema import TrajectoryRecord


@dataclass(slots=True)
class TrainerConfig:
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-3
    device: str = "cpu"
    seed: int = 13
    action_class_weighting: str = "none"
    hard_task_sampling_multiplier: float = 1.0
    aux_loss_scale: float = 1.0
    use_legal_action_mask: bool = False
    route_conditioned_action: bool = False
    hidden_dim: int | None = None
    encoder_depth: int | None = None
    gru_layers: int | None = None
    dropout: float | None = None
    command_slot_loss_weight: float = 0.2
    file_slot_loss_weight: float = 0.2
    auxiliary_heads: bool = True
    hash_bins: int = 256
    include_route_features: bool = True
    include_task_features: bool = True
    include_failure_signal_features: bool = True
    include_slot_semantic_features: bool = True
    action_conditioned_world_model: bool = True
    residual_world_model: bool = False


class EpisodeSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, records: list[TrajectoryRecord], vectorizer: StateVectorizer) -> None:
        grouped: dict[str, list[TrajectoryRecord]] = {}
        for record in records:
            grouped.setdefault(record.episode_id, []).append(record)
        self.episodes = [sorted(items, key=lambda item: item.t) for items in grouped.values()]
        self.vectorizer = vectorizer

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode = self.episodes[index]
        state_vectors = np.stack(
            [self.vectorizer.vectorize_state(record.state) for record in episode]
        )
        next_vectors = np.stack(
            [self.vectorizer.vectorize_state(record.next_state) for record in episode]
        )
        payload = {
            "inputs": torch.tensor(state_vectors, dtype=torch.float32),
            "next_state": torch.tensor(next_vectors, dtype=torch.float32),
            "action": torch.tensor(
                [action_to_index(record.action.type) for record in episode],
                dtype=torch.long,
            ),
            "action_mask": torch.tensor(
                np.stack([valid_action_mask(record.state) for record in episode]),
                dtype=torch.float32,
            ),
            "command_slot": torch.tensor(
                [command_slot_target(record) for record in episode],
                dtype=torch.long,
            ),
            "file_slot": torch.tensor(
                [file_slot_target(record) for record in episode],
                dtype=torch.long,
            ),
            "route": torch.tensor([route_target(record) for record in episode], dtype=torch.long),
            "internal_target": torch.tensor(
                [
                    INTERNAL_TARGET_ORDER.index(internal_target_for_state(record.state))
                    for record in episode
                ],
                dtype=torch.long,
            ),
            "salience": torch.tensor(
                [salience_target(record) for record in episode], dtype=torch.float32
            ),
            "urgency": torch.tensor(
                [urgency_target(record) for record in episode], dtype=torch.float32
            ),
            "risk": torch.tensor([risk_target(record) for record in episode], dtype=torch.float32),
            "delta_norm": torch.tensor(
                [delta_norm_target(record, self.vectorizer) for record in episode],
                dtype=torch.float32,
            ),
            "mask": torch.ones(len(episode), dtype=torch.bool),
        }
        return payload


def collate_episode_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["inputs"].shape[0] for item in batch)
    collated: dict[str, list[torch.Tensor]] = {}
    for item in batch:
        pad = max_len - item["inputs"].shape[0]
        for key, tensor in item.items():
            if key not in collated:
                collated[key] = []
            if tensor.ndim == 1:
                pad_value = -100 if tensor.dtype == torch.long and key in {"command_slot", "file_slot"} else 0
                padded = torch.nn.functional.pad(tensor, (0, pad), value=pad_value)
            else:
                padded = torch.nn.functional.pad(tensor, (0, 0, 0, pad))
            collated[key].append(padded)
    return {key: torch.stack(values) for key, values in collated.items()}


def build_dataloader(
    dataset_path: str | Path,
    *,
    vectorizer: StateVectorizer,
    batch_size: int,
    shuffle: bool,
    seed: int,
    hard_task_sampling_multiplier: float = 1.0,
) -> DataLoader[dict[str, torch.Tensor]]:
    records = read_jsonl(Path(dataset_path))
    dataset = EpisodeSequenceDataset(records, vectorizer)
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if shuffle and hard_task_sampling_multiplier > 1.0:
        weights = []
        for episode in dataset.episodes:
            task_type = episode[0].goal.task_type
            weights.append(hard_task_sampling_multiplier if task_type in HARD_TASK_TYPES else 1.0)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        collate_fn=collate_episode_batch,
        generator=generator,
    )


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    return (loss * weights).sum() / weights.sum().clamp(min=1.0)


def _safe_slot_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    valid = targets != -100
    if not bool(valid.any()):
        return logits.new_tensor(0.0)
    loss = nn.functional.cross_entropy(logits[valid], targets[valid], reduction="mean")
    return loss


def _action_class_weights(
    dataloader: DataLoader[dict[str, torch.Tensor]],
    *,
    mode: str,
    device: str,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = torch.zeros(len(ACTION_ORDER), dtype=torch.float32)
    dataset = dataloader.dataset
    if not isinstance(dataset, EpisodeSequenceDataset):
        return None
    for episode in dataset.episodes:
        for record in episode:
            if record.action is not None:
                counts[action_to_index(record.action.type)] += 1.0
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / counts
    if mode == "inverse_sqrt":
        weights = torch.sqrt(weights)
    elif mode != "inverse":
        raise ValueError(f"Unsupported action_class_weighting mode: {mode}")
    weights = weights / weights.mean().clamp(min=1e-6)
    return weights.to(device)


def _train_epoch(
    model: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
    structured: bool,
    action_class_weights: torch.Tensor | None = None,
    aux_loss_scale: float = 1.0,
    use_legal_action_mask: bool = False,
    command_slot_loss_weight: float = 0.2,
    file_slot_loss_weight: float = 0.2,
    world_model_feature_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    model.train()
    action_loss_fn = nn.CrossEntropyLoss(weight=action_class_weights, reduction="none")
    route_loss_fn = nn.CrossEntropyLoss(reduction="none")
    target_loss_fn = nn.CrossEntropyLoss(reduction="none")
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")
    mse_loss = nn.MSELoss(reduction="none")
    totals: dict[str, float] = {"loss": 0.0}
    batches = 0
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = batch["inputs"].to(device)
        mask = batch["mask"].to(device)
        outputs = (
            model(inputs, action_indices=batch["action"].to(device))
            if structured
            else model(inputs)
        )
        action_logits = outputs["action_logits"]
        if use_legal_action_mask:
            action_mask = batch["action_mask"].to(device).bool()
            action_logits = action_logits.masked_fill(~action_mask, -1.0e4)
        action_loss = action_loss_fn(
            action_logits.transpose(1, 2), batch["action"].to(device)
        )
        loss = _masked_mean(action_loss, mask)
        command_loss = _safe_slot_loss(
            outputs["command_slot_logits"][mask],
            batch["command_slot"].to(device)[mask],
        )
        file_loss = _safe_slot_loss(
            outputs["file_slot_logits"][mask],
            batch["file_slot"].to(device)[mask],
        )
        loss = loss + command_slot_loss_weight * command_loss + file_slot_loss_weight * file_loss
        if "target_logits" in outputs:
            target_loss = target_loss_fn(
                outputs["target_logits"].transpose(1, 2),
                batch["internal_target"].to(device),
            )
            loss = loss + 0.2 * _masked_mean(target_loss, mask)
        if structured and aux_loss_scale > 0.0:
            route_loss = route_loss_fn(
                outputs["route_logits"].transpose(1, 2), batch["route"].to(device)
            )
            salience_loss = _masked_mean(
                bce_loss(outputs["salience"], batch["salience"].to(device)), mask
            )
            urgency_loss = _masked_mean(
                mse_loss(outputs["urgency"], batch["urgency"].to(device)), mask
            )
            risk_loss = _masked_mean(
                mse_loss(outputs["risk"], batch["risk"].to(device)), mask
            )
            delta_loss = _masked_mean(
                mse_loss(outputs["prediction_error"], batch["delta_norm"].to(device)), mask
            )
            next_state_feature_loss = mse_loss(
                outputs["next_state"], batch["next_state"].to(device)
            )
            if world_model_feature_mask is not None:
                feature_mask = world_model_feature_mask.to(device)
                next_state_row_loss = (
                    next_state_feature_loss * feature_mask
                ).sum(dim=-1) / feature_mask.sum().clamp(min=1.0)
            else:
                next_state_row_loss = next_state_feature_loss.mean(dim=-1)
            next_state_loss = _masked_mean(next_state_row_loss, mask)
            loss = loss + aux_loss_scale * (
                0.2 * _masked_mean(route_loss, mask)
                + 0.1 * salience_loss
                + 0.05 * urgency_loss
                + 0.05 * risk_loss
                + 0.05 * delta_loss
                + 0.1 * next_state_loss
            )
        loss.backward()
        optimizer.step()
        totals["loss"] += float(loss.item())
        batches += 1
    totals["loss"] /= max(batches, 1)
    return totals


def train_nsi_model(
    dataset_path: str | Path,
    *,
    trainer_config: TrainerConfig,
    smoke: bool = False,
) -> tuple[NSIReflexModel, StateVectorizer, dict[str, Any]]:
    torch.manual_seed(trainer_config.seed)
    np.random.seed(trainer_config.seed)
    vectorizer = StateVectorizer(
        structured=True,
        hash_bins=trainer_config.hash_bins,
        include_route_features=trainer_config.include_route_features,
        include_task_features=trainer_config.include_task_features,
        include_failure_signal_features=trainer_config.include_failure_signal_features,
        include_slot_semantic_features=trainer_config.include_slot_semantic_features,
    )
    dataloader = build_dataloader(
        dataset_path,
        vectorizer=vectorizer,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        seed=trainer_config.seed,
        hard_task_sampling_multiplier=trainer_config.hard_task_sampling_multiplier,
    )
    model_config = (
        NSIModelConfig.smoke(vectorizer.vector_dim)
        if smoke
        else NSIModelConfig(
            input_dim=vectorizer.vector_dim,
            route_conditioned_action=trainer_config.route_conditioned_action,
        )
    )
    model_config = replace(
        model_config,
        hidden_dim=(
            trainer_config.hidden_dim
            if trainer_config.hidden_dim is not None
            else model_config.hidden_dim
        ),
        encoder_depth=(
            trainer_config.encoder_depth
            if trainer_config.encoder_depth is not None
            else model_config.encoder_depth
        ),
        gru_layers=(
            trainer_config.gru_layers
            if trainer_config.gru_layers is not None
            else model_config.gru_layers
        ),
        dropout=model_config.dropout if trainer_config.dropout is None else trainer_config.dropout,
        auxiliary_heads=trainer_config.auxiliary_heads,
        action_conditioned_world_model=trainer_config.action_conditioned_world_model,
        residual_world_model=trainer_config.residual_world_model,
    )
    if smoke and trainer_config.route_conditioned_action:
        model_config.route_conditioned_action = True
    model = NSIReflexModel(model_config).to(trainer_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config.learning_rate)
    action_class_weights = _action_class_weights(
        dataloader,
        mode=trainer_config.action_class_weighting,
        device=trainer_config.device,
    )
    world_model_feature_mask = torch.tensor(
        vectorizer.world_model_target_mask(),
        dtype=torch.float32,
    )
    history = []
    for epoch in range(trainer_config.epochs):
        metrics = _train_epoch(
            model,
            dataloader,
            optimizer,
            device=trainer_config.device,
            structured=True,
            action_class_weights=action_class_weights,
            aux_loss_scale=trainer_config.aux_loss_scale,
            use_legal_action_mask=trainer_config.use_legal_action_mask,
            command_slot_loss_weight=trainer_config.command_slot_loss_weight,
            file_slot_loss_weight=trainer_config.file_slot_loss_weight,
            world_model_feature_mask=world_model_feature_mask,
        )
        metrics["epoch"] = epoch + 1
        history.append(metrics)
    summary = {
        "model_kind": "nsi",
        "parameter_count": model.parameter_count(),
        "model_config": asdict(model_config),
        "vectorizer": asdict(vectorizer),
        "trainer_config": asdict(trainer_config),
        "epochs": trainer_config.epochs,
        "history": history,
        "smoke": smoke,
        "dataset_path": str(dataset_path),
        "train_episode_count": len(dataloader.dataset),
    }
    return model, vectorizer, summary


def train_flat_text_baseline(
    dataset_path: str | Path,
    *,
    trainer_config: TrainerConfig,
    smoke: bool = False,
) -> tuple[FlatTextBaselineModel, StateVectorizer, dict[str, Any]]:
    torch.manual_seed(trainer_config.seed)
    np.random.seed(trainer_config.seed)
    vectorizer = StateVectorizer(
        structured=False,
        numeric_features=False,
        hash_bins=trainer_config.hash_bins,
    )
    dataloader = build_dataloader(
        dataset_path,
        vectorizer=vectorizer,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        seed=trainer_config.seed,
        hard_task_sampling_multiplier=trainer_config.hard_task_sampling_multiplier,
    )
    model_config = (
        FlatTextModelConfig.smoke(vectorizer.vector_dim)
        if smoke
        else FlatTextModelConfig(input_dim=vectorizer.vector_dim)
    )
    model_config = replace(
        model_config,
        hidden_dim=(
            trainer_config.hidden_dim
            if trainer_config.hidden_dim is not None
            else model_config.hidden_dim
        ),
        gru_layers=(
            trainer_config.gru_layers
            if trainer_config.gru_layers is not None
            else model_config.gru_layers
        ),
        dropout=model_config.dropout if trainer_config.dropout is None else trainer_config.dropout,
    )
    model = FlatTextBaselineModel(model_config).to(trainer_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config.learning_rate)
    action_class_weights = _action_class_weights(
        dataloader,
        mode=trainer_config.action_class_weighting,
        device=trainer_config.device,
    )
    history = []
    for epoch in range(trainer_config.epochs):
        metrics = _train_epoch(
            model,
            dataloader,
            optimizer,
            device=trainer_config.device,
            structured=False,
            action_class_weights=action_class_weights,
            aux_loss_scale=0.0,
            use_legal_action_mask=trainer_config.use_legal_action_mask,
            command_slot_loss_weight=trainer_config.command_slot_loss_weight,
            file_slot_loss_weight=trainer_config.file_slot_loss_weight,
        )
        metrics["epoch"] = epoch + 1
        history.append(metrics)
    summary = {
        "model_kind": "flat_text",
        "parameter_count": model.parameter_count(),
        "model_config": asdict(model_config),
        "vectorizer": asdict(vectorizer),
        "trainer_config": asdict(trainer_config),
        "epochs": trainer_config.epochs,
        "history": history,
        "smoke": smoke,
        "dataset_path": str(dataset_path),
        "train_episode_count": len(dataloader.dataset),
    }
    return model, vectorizer, summary


def save_model_checkpoint(
    model: torch.nn.Module,
    vectorizer: StateVectorizer,
    *,
    checkpoint_path: str | Path,
    model_kind: str,
    summary: dict[str, Any],
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "artifact_version": 1,
            "model_kind": model_kind,
            "model_config": summary["model_config"],
            "vectorizer": summary["vectorizer"],
            "training_summary": summary,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_model_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[torch.nn.Module, StateVectorizer, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model_kind = payload["model_kind"]
    vectorizer = StateVectorizer(**payload["vectorizer"])
    if model_kind == "nsi":
        model_config_payload = dict(payload["model_config"])
        model_config_payload.setdefault("action_conditioned_world_model", False)
        model_config_payload.setdefault("residual_world_model", False)
        model = NSIReflexModel(NSIModelConfig(**model_config_payload))
    elif model_kind == "flat_text":
        model = FlatTextBaselineModel(FlatTextModelConfig(**payload["model_config"]))
    else:
        raise ValueError(f"Unsupported checkpoint model kind: {model_kind}")
    load_result = model.load_state_dict(payload["model_state_dict"], strict=False)
    payload["checkpoint_load"] = {
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    model.to(device)
    model.eval()
    return model, vectorizer, payload
