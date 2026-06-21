from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from reflexlm.core.dataset import (
    ReflexCoreEpisodeDataset,
    ReflexCoreTorchDataset,
    collate_reflexcore_batch,
    collate_reflexcore_sequence_batch,
    read_reflexcore_jsonl,
)
from reflexlm.core.losses import (
    ReflexCoreLossWeights,
    compute_reflexcore_action_loss,
    compute_reflexcore_losses,
)
from reflexlm.core.model import ReflexCoreV0, ReflexCoreV0Config
from reflexlm.core.schema import dataset_hash
from reflexlm.models.features import StateVectorizer


def train_reflexcore_v0(
    *,
    dataset_path: Path,
    config_path: Path,
    output_dir: Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    device: str | None = None,
    seed: int | None = None,
    sequence_mode: bool | None = None,
    max_sequence_len: int | None = None,
) -> dict[str, object]:
    raw_config = load_reflexcore_yaml(config_path)
    train_config = dict(raw_config.get("training", {}))
    model_config = dict(raw_config.get("model", {}))
    dataset_config = dict(raw_config.get("dataset", {}))
    sensory_training_config = dict(raw_config.get("sensory_training", {}))
    loss_weight_config = dict(raw_config.get("loss_weights", {}))
    loss_weights = ReflexCoreLossWeights(**loss_weight_config)
    numeric_action_aux_weight = float(
        sensory_training_config.get("numeric_action_aux_weight", 0.0)
    )
    numeric_action_aux_zero_text = bool(
        sensory_training_config.get("numeric_action_aux_zero_text", True)
    )
    numeric_action_aux_zero_hash = bool(
        sensory_training_config.get("numeric_action_aux_zero_hash", True)
    )
    resolved_seed = seed if seed is not None else int(train_config.get("seed", 13))
    torch.manual_seed(resolved_seed)

    examples = read_reflexcore_jsonl(dataset_path)
    resolved_sequence_mode = (
        bool(sequence_mode)
        if sequence_mode is not None
        else bool(dataset_config.get("sequence_mode", False))
    )
    resolved_max_sequence_len = (
        max_sequence_len
        if max_sequence_len is not None
        else dataset_config.get("max_sequence_len")
    )
    max_text_tokens = int(dataset_config.get("max_text_tokens", 64))
    if resolved_sequence_mode:
        dataset = ReflexCoreEpisodeDataset(
            examples,
            max_text_tokens=max_text_tokens,
            max_sequence_len=(
                int(resolved_max_sequence_len)
                if resolved_max_sequence_len
                else None
            ),
        )
        collate_fn = collate_reflexcore_sequence_batch
    else:
        dataset = ReflexCoreTorchDataset(examples, max_text_tokens=max_text_tokens)
        collate_fn = collate_reflexcore_batch
    model_config["input_dim"] = dataset.input_dim
    config = ReflexCoreV0Config(**model_config)
    model = ReflexCoreV0(config)
    resolved_device = torch.device(device or train_config.get("device", "cpu"))
    model.to(resolved_device)
    resolved_epochs = epochs if epochs is not None else int(train_config.get("epochs", 1))
    resolved_batch_size = (
        batch_size if batch_size is not None else int(train_config.get("batch_size", 4))
    )
    resolved_learning_rate = (
        learning_rate
        if learning_rate is not None
        else float(train_config.get("learning_rate", 1e-3))
    )
    loader = DataLoader(
        dataset,
        batch_size=resolved_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=resolved_learning_rate)
    history: list[dict[str, float]] = []
    for epoch in range(resolved_epochs):
        model.train()
        totals: dict[str, float] = {}
        count = 0
        for batch in loader:
            batch = {key: value.to(resolved_device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["observation_vectors"],
                batch["text_tokens"],
                action_indices=batch["action_indices"],
            )
            losses = compute_reflexcore_losses(outputs, batch, weights=loss_weights)
            if numeric_action_aux_weight > 0.0:
                numeric_vectors = numeric_only_observation_vectors(
                    batch["observation_vectors"],
                    zero_hash=numeric_action_aux_zero_hash,
                )
                numeric_text_tokens = (
                    torch.zeros_like(batch["text_tokens"])
                    if numeric_action_aux_zero_text
                    else batch["text_tokens"]
                )
                numeric_outputs = model(
                    numeric_vectors,
                    numeric_text_tokens,
                    action_indices=batch["action_indices"],
                )
                numeric_action_aux_loss = compute_reflexcore_action_loss(
                    numeric_outputs,
                    batch,
                )
                losses["numeric_action_aux_loss"] = numeric_action_aux_loss
                losses["loss"] = (
                    losses["loss"]
                    + numeric_action_aux_weight * numeric_action_aux_loss
                )
            losses["loss"].backward()
            optimizer.step()
            count += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu().item())
        history.append({name: value / max(count, 1) for name, value in sorted(totals.items())})
        history[-1]["epoch"] = float(epoch)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "reflexcore_v0.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "dataset_hash": dataset_hash(examples),
        "seed": resolved_seed,
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    summary = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "dataset_hash": checkpoint["dataset_hash"],
        "model_hash": file_hash(checkpoint_path),
        "parameter_count": model.parameter_count(),
        "seed": resolved_seed,
        "sequence_mode": resolved_sequence_mode,
        "max_sequence_len": resolved_max_sequence_len,
        "epochs": resolved_epochs,
        "batch_size": resolved_batch_size,
        "learning_rate": resolved_learning_rate,
        "loss_weights": asdict(loss_weights),
        "sensory_training": {
            "numeric_action_aux_weight": numeric_action_aux_weight,
            "numeric_action_aux_zero_text": numeric_action_aux_zero_text,
            "numeric_action_aux_zero_hash": numeric_action_aux_zero_hash,
        },
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def numeric_only_observation_vectors(
    observation_vectors: torch.Tensor,
    *,
    zero_hash: bool = True,
) -> torch.Tensor:
    """Keep structured numeric state features while optionally removing hash bins."""

    numeric_vectors = observation_vectors.clone()
    if zero_hash:
        input_dim = numeric_vectors.shape[-1]
        numeric_dim = min(StateVectorizer(hash_bins=0).numeric_dim, input_dim)
        if input_dim > numeric_dim:
            numeric_vectors[..., numeric_dim:] = 0.0
    return numeric_vectors


def load_reflexcore_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
