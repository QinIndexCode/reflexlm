from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from reflexlm.core.schema import (
    ComputerObservation,
    MotorAction,
    ReflexCoreTrainingExample,
    action_to_index,
    dataset_hash,
)
from reflexlm.models.features import (
    MAX_CANDIDATE_SLOTS,
    StateVectorizer,
    candidate_commands,
    candidate_files,
    serialize_state_as_text,
)
from reflexlm.schema import ActionDecision, ActionType, SystemStateFrame, TrajectoryRecord

DEFAULT_MAX_TEXT_TOKENS = 64
DEFAULT_VOCAB_SIZE = 4096
FORBIDDEN_OBSERVATION_KEYS = {
    "oracle_action",
    "gold_action",
    "hidden_action",
    "target_action",
    "oracle_label",
    "gold_label",
}


def hash_text_tokens(
    text: str,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_tokens: int = DEFAULT_MAX_TEXT_TOKENS,
) -> list[int]:
    if vocab_size < 2:
        raise ValueError("vocab_size must reserve at least pad and one content token")
    tokens: list[int] = []
    for piece in text.replace("\n", " ").split():
        digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
        tokens.append(1 + (int.from_bytes(digest, "little") % (vocab_size - 1)))
        if len(tokens) >= max_tokens:
            break
    return tokens


def observation_vector_from_state(
    state: SystemStateFrame,
    *,
    vectorizer: StateVectorizer | None = None,
) -> list[float]:
    vec = (vectorizer or StateVectorizer()).vectorize_state(state)
    return [float(item) for item in vec.tolist()]


def observation_from_state(
    state: SystemStateFrame,
    *,
    vectorizer: StateVectorizer | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_text_tokens: int = DEFAULT_MAX_TEXT_TOKENS,
) -> ComputerObservation:
    text = serialize_state_as_text(state, include_internal_hints=False)
    return ComputerObservation.from_state_frame(
        state,
        text_tokens=hash_text_tokens(
            text,
            vocab_size=vocab_size,
            max_tokens=max_text_tokens,
        ),
        vector=observation_vector_from_state(state, vectorizer=vectorizer),
    )


def _safe_action(record: TrajectoryRecord) -> ActionDecision:
    if record.action is not None:
        return record.action
    return ActionDecision(type=ActionType.WAIT, reason="missing_action", confidence=0.0)


def example_from_trajectory(
    record: TrajectoryRecord,
    *,
    vectorizer: StateVectorizer | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_text_tokens: int = DEFAULT_MAX_TEXT_TOKENS,
) -> ReflexCoreTrainingExample:
    vectorizer = vectorizer or StateVectorizer()
    return ReflexCoreTrainingExample(
        episode_id=record.episode_id,
        t=record.t,
        observation=observation_from_state(
            record.state,
            vectorizer=vectorizer,
            vocab_size=vocab_size,
            max_text_tokens=max_text_tokens,
        ),
        action=MotorAction.from_decision(_safe_action(record)),
        next_observation=observation_from_state(
            record.next_state,
            vectorizer=vectorizer,
            vocab_size=vocab_size,
            max_text_tokens=max_text_tokens,
        ),
        reward=record.reward,
        done=record.done,
        source=record.source.value,
    )


def build_reflexcore_examples(
    records: Iterable[TrajectoryRecord],
    *,
    vectorizer: StateVectorizer | None = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    max_text_tokens: int = DEFAULT_MAX_TEXT_TOKENS,
) -> list[ReflexCoreTrainingExample]:
    vectorizer = vectorizer or StateVectorizer()
    examples = [
        example_from_trajectory(
            record,
            vectorizer=vectorizer,
            vocab_size=vocab_size,
            max_text_tokens=max_text_tokens,
        )
        for record in records
    ]
    for example in examples:
        assert_no_hidden_oracle_fields(example)
    return examples


def write_reflexcore_jsonl(path: Path, examples: list[ReflexCoreTrainingExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.model_dump_json())
            handle.write("\n")


def read_reflexcore_jsonl(path: Path) -> list[ReflexCoreTrainingExample]:
    examples: list[ReflexCoreTrainingExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(ReflexCoreTrainingExample.model_validate(json.loads(line)))
    return examples


def split_examples_by_episode(
    examples: list[ReflexCoreTrainingExample],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 13,
) -> dict[str, list[ReflexCoreTrainingExample]]:
    grouped: dict[str, list[ReflexCoreTrainingExample]] = defaultdict(list)
    for example in examples:
        grouped[example.episode_id].append(example)
    episode_ids = sorted(grouped)
    rng = Random(seed)
    rng.shuffle(episode_ids)
    train_cut = int(len(episode_ids) * train_ratio)
    val_cut = int(len(episode_ids) * (train_ratio + val_ratio))
    split_ids = {
        "train": set(episode_ids[:train_cut]),
        "val": set(episode_ids[train_cut:val_cut]),
        "test": set(episode_ids[val_cut:]),
    }
    return {
        split: sorted(
            [example for episode_id in ids for example in grouped[episode_id]],
            key=lambda item: (item.episode_id, item.t),
        )
        for split, ids in split_ids.items()
    }


def split_hashes(
    splits: dict[str, list[ReflexCoreTrainingExample]],
) -> dict[str, str]:
    return {name: dataset_hash(examples) for name, examples in splits.items()}


def assert_no_hidden_oracle_fields(example: ReflexCoreTrainingExample) -> None:
    payload = example.observation.model_dump(mode="json")
    keys: set[str] = set()

    def visit(value: object, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                keys.add(name.lower())
                visit(item, name)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{prefix}[{index}]")

    visit(payload)
    forbidden = [key for key in keys if key.split(".")[-1] in FORBIDDEN_OBSERVATION_KEYS]
    if forbidden:
        raise ValueError(f"forbidden hidden/oracle observation field: {forbidden[0]}")


class ReflexCoreTorchDataset(Dataset):
    def __init__(
        self,
        examples: list[ReflexCoreTrainingExample],
        *,
        max_text_tokens: int = DEFAULT_MAX_TEXT_TOKENS,
    ) -> None:
        if not examples:
            raise ValueError("ReflexCoreTorchDataset requires at least one example")
        self.examples = examples
        self.max_text_tokens = max_text_tokens
        self.input_dim = len(examples[0].observation.vector)
        if self.input_dim <= 0:
            raise ValueError("examples must contain precomputed observation vectors")
        for example in examples:
            if len(example.observation.vector) != self.input_dim:
                raise ValueError("all examples must share observation vector dimension")
            if len(example.next_observation.vector) != self.input_dim:
                raise ValueError("next observation vector dimension mismatch")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tensors = tensors_for_example(
            self.examples[index],
            max_text_tokens=self.max_text_tokens,
            input_dim=self.input_dim,
        )
        return {
            key: value.unsqueeze(0)
            for key, value in tensors.items()
        }


class ReflexCoreEpisodeDataset(Dataset):
    """Episode-grouped dataset for recurrent/transformer sequence training."""

    def __init__(
        self,
        examples: list[ReflexCoreTrainingExample],
        *,
        max_text_tokens: int = DEFAULT_MAX_TEXT_TOKENS,
        max_sequence_len: int | None = None,
    ) -> None:
        if not examples:
            raise ValueError("ReflexCoreEpisodeDataset requires at least one example")
        self.max_text_tokens = max_text_tokens
        self.input_dim = len(examples[0].observation.vector)
        if self.input_dim <= 0:
            raise ValueError("examples must contain precomputed observation vectors")
        grouped: dict[str, list[ReflexCoreTrainingExample]] = defaultdict(list)
        for example in examples:
            if len(example.observation.vector) != self.input_dim:
                raise ValueError("all examples must share observation vector dimension")
            grouped[example.episode_id].append(example)
        sequences: list[list[ReflexCoreTrainingExample]] = []
        for _episode_id, episode_examples in sorted(grouped.items()):
            ordered = sorted(episode_examples, key=lambda item: item.t)
            if max_sequence_len is None or max_sequence_len <= 0:
                sequences.append(ordered)
            else:
                for start in range(0, len(ordered), max_sequence_len):
                    sequences.append(ordered[start : start + max_sequence_len])
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        items = [
            tensors_for_example(
                example,
                max_text_tokens=self.max_text_tokens,
                input_dim=self.input_dim,
            )
            for example in self.sequences[index]
        ]
        return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


def tensors_for_example(
    example: ReflexCoreTrainingExample,
    *,
    max_text_tokens: int,
    input_dim: int,
) -> dict[str, torch.Tensor]:
    obs_vec = torch.tensor(example.observation.vector, dtype=torch.float32)
    next_vec = torch.tensor(example.next_observation.vector, dtype=torch.float32)
    if len(obs_vec) != input_dim or len(next_vec) != input_dim:
        raise ValueError("example vector dimension mismatch")
    text_tokens = _pad_tokens(example.observation.text_tokens, max_text_tokens)
    next_tokens = _pad_tokens(example.next_observation.text_tokens, max_text_tokens)
    action_index = action_to_index(example.action.type)
    command_slot = _slot_target(example.action.command, example.observation.candidate_commands)
    file_slot = _slot_target(example.action.file_target, example.observation.candidate_files)
    risk = _risk_target(example)
    salience = _salience_target(example)
    prediction_error = _prediction_error_target(example, obs_vec, next_vec, input_dim)
    first_next_token = int(next_tokens[0].item())
    return {
        "observation_vectors": obs_vec,
        "next_state": next_vec,
        "text_tokens": text_tokens,
        "text_targets": torch.tensor(first_next_token, dtype=torch.long),
        "action_indices": torch.tensor(action_index, dtype=torch.long),
        "command_slot_indices": torch.tensor(command_slot, dtype=torch.long),
        "file_slot_indices": torch.tensor(file_slot, dtype=torch.long),
        "risk_targets": torch.tensor([risk], dtype=torch.float32),
        "salience_targets": torch.tensor([salience], dtype=torch.float32),
        "prediction_error_targets": torch.tensor([prediction_error], dtype=torch.float32),
        "next_state_loss_mask": _next_state_loss_mask(input_dim),
        "loss_mask": torch.tensor([1.0], dtype=torch.float32),
    }


def _pad_tokens(tokens: list[int], max_tokens: int) -> torch.Tensor:
    clipped = tokens[:max_tokens]
    return torch.tensor(clipped + [0] * (max_tokens - len(clipped)), dtype=torch.long)


def _slot_target(value: str | None, candidates: list[str]) -> int:
    if value is None:
        return -100
    try:
        return candidates.index(value)
    except ValueError:
        return -100


def _risk_target(example: ReflexCoreTrainingExample) -> float:
    if example.observation.safety.dangerous_command_detected:
        return 1.0
    if example.action.type in {ActionType.BLOCK, ActionType.STOP_PROCESS}:
        return 0.8
    if example.action.type == ActionType.RUN_COMMAND:
        return 0.4
    return 0.1


def _salience_target(example: ReflexCoreTrainingExample) -> float:
    obs = example.observation
    return float(
        example.action.type not in {ActionType.WAIT, ActionType.DONE}
        or obs.filesystem.external_change_detected
        or obs.safety.dangerous_command_detected
        or bool(obs.terminal.stderr_delta.strip())
    )


def _prediction_error_target(
    example: ReflexCoreTrainingExample,
    obs_vec: torch.Tensor,
    next_vec: torch.Tensor,
    input_dim: int,
) -> float:
    observed_error = example.next_observation.runtime_evidence.observed_prediction_error
    if observed_error is not None and np.isfinite(float(observed_error)):
        return max(float(observed_error), 0.0)
    feature_mask = _next_state_loss_mask(input_dim).numpy()
    diff = (next_vec.numpy() - obs_vec.numpy()) * feature_mask
    active_features = max(float(feature_mask.sum()), 1.0)
    return float(
        np.linalg.norm(diff) / max(np.sqrt(active_features), 1.0)
    )


def _next_state_loss_mask(input_dim: int) -> torch.Tensor:
    numeric_dim = StateVectorizer(hash_bins=0).numeric_dim
    if input_dim < numeric_dim:
        return torch.ones(input_dim, dtype=torch.float32)
    hash_bins = input_dim - numeric_dim
    mask = StateVectorizer(hash_bins=hash_bins).world_model_target_mask()
    return torch.tensor(mask, dtype=torch.float32)


def collate_reflexcore_batch(
    items: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


def collate_reflexcore_sequence_batch(
    items: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("cannot collate empty ReflexCore sequence batch")
    max_len = max(item["observation_vectors"].shape[0] for item in items)
    input_dim = items[0]["observation_vectors"].shape[-1]
    text_len = items[0]["text_tokens"].shape[-1]
    batch: dict[str, torch.Tensor] = {
        "observation_vectors": torch.zeros(len(items), max_len, input_dim, dtype=torch.float32),
        "next_state": torch.zeros(len(items), max_len, input_dim, dtype=torch.float32),
        "text_tokens": torch.zeros(len(items), max_len, text_len, dtype=torch.long),
        "text_targets": torch.full((len(items), max_len), -100, dtype=torch.long),
        "action_indices": torch.full((len(items), max_len), -100, dtype=torch.long),
        "command_slot_indices": torch.full((len(items), max_len), -100, dtype=torch.long),
        "file_slot_indices": torch.full((len(items), max_len), -100, dtype=torch.long),
        "risk_targets": torch.zeros(len(items), max_len, 1, dtype=torch.float32),
        "salience_targets": torch.zeros(len(items), max_len, 1, dtype=torch.float32),
        "prediction_error_targets": torch.zeros(len(items), max_len, 1, dtype=torch.float32),
        "next_state_loss_mask": torch.zeros(len(items), max_len, input_dim, dtype=torch.float32),
        "loss_mask": torch.zeros(len(items), max_len, 1, dtype=torch.float32),
    }
    for batch_index, item in enumerate(items):
        seq_len = item["observation_vectors"].shape[0]
        for key, value in item.items():
            if key in batch:
                batch[key][batch_index, :seq_len] = value
    return batch


def slot_bounds_ok(examples: list[ReflexCoreTrainingExample]) -> bool:
    for example in examples:
        if len(example.observation.candidate_commands) > MAX_CANDIDATE_SLOTS:
            return False
        if len(example.observation.candidate_files) > MAX_CANDIDATE_SLOTS:
            return False
    return True
