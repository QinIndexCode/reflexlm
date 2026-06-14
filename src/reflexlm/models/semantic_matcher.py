from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from reflexlm.models.features import candidate_commands
from reflexlm.schema import SystemStateFrame


_WORD_RE = re.compile(r"[a-z0-9_]+")
_SEMANTIC_STOP_WORDS = {
    "a",
    "allowlisted",
    "an",
    "and",
    "another",
    "by",
    "bounded",
    "candidate",
    "command",
    "exe",
    "failure",
    "from",
    "is",
    "of",
    "print",
    "process",
    "python",
    "recover",
    "recovery",
    "terminal",
    "the",
    "to",
    "using",
    "visible",
    "with",
}


def _semantic_stem(word: str) -> str:
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _semantic_words(text: str) -> list[str]:
    lowered = text.lower()
    return [
        _semantic_stem(word)
        for word in _WORD_RE.findall(lowered)
        if word not in _SEMANTIC_STOP_WORDS
        and not word.startswith("phase2")
        and not word.isdigit()
    ]


def _features(text: str, *, bins: int) -> torch.Tensor:
    words = _semantic_words(text)
    tokens = [f"w:{word}" for word in words]
    tokens.extend(
        f"b:{left}_{right}" for left, right in zip(words, words[1:])
    )
    compact = " ".join(words)
    for size in (3, 4, 5):
        tokens.extend(
            f"c{size}:{compact[index:index + size]}"
            for index in range(max(0, len(compact) - size + 1))
        )
    vector = torch.zeros(bins, dtype=torch.float32)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        vector[int.from_bytes(digest, "little") % bins] += 1.0
    return F.normalize(vector, dim=0) if torch.any(vector) else vector


def _state_text(state: SystemStateFrame) -> str:
    return " ".join(
        str(part).strip()
        for part in [
            state.goal.description,
            state.terminal.stdout_delta,
            state.terminal.stderr_delta,
            state.terminal.last_command,
            " ".join(state.runtime_evidence.terminal_observations),
            " ".join(state.filesystem.dirty_files),
            " ".join(state.filesystem.changed_paths),
        ]
        if part is not None and str(part).strip()
    )


def _receptor_text(state: SystemStateFrame) -> str:
    return " ".join(
        str(part).strip()
        for part in [
            state.terminal.stdout_delta,
            state.terminal.stderr_delta,
            " ".join(state.runtime_evidence.terminal_observations),
            " ".join(state.filesystem.dirty_files),
            " ".join(state.filesystem.changed_paths),
        ]
        if part is not None and str(part).strip()
    )


def _receptor_frames(state: SystemStateFrame) -> list[str]:
    frames = [
        *state.runtime_evidence.terminal_observations,
        state.terminal.stdout_delta,
        state.terminal.stderr_delta,
        " ".join(state.filesystem.dirty_files),
        " ".join(state.filesystem.changed_paths),
    ]
    return list(
        dict.fromkeys(
            str(frame).strip()
            for frame in frames
            if frame is not None and str(frame).strip()
        )
    )


def _command_semantic_text(command: str) -> str:
    explicit_intent = re.search(
        r"--intent(?:=|\s+)(?:\"([^\"]+)\"|'([^']+)'|([^\s]+))",
        command,
        flags=re.IGNORECASE,
    )
    if explicit_intent is not None:
        return next(
            value.strip()
            for value in explicit_intent.groups()
            if value is not None and value.strip()
        )
    quoted_parts = [
        double or single
        for double, single in re.findall(r'"([^"]+)"|\'([^\']+)\'', command)
        if double or single
    ]
    semantic_source = max(quoted_parts, key=len) if quoted_parts else command
    return " ".join(
        word
        for word in re.findall(r"[A-Za-z0-9_]+", semantic_source)
        if word.lower() not in _SEMANTIC_STOP_WORDS
        and not word.lower().startswith("phase2")
        and not word.isdigit()
    )


class RecencyWeightedSemanticMatcher:
    """Aggregate per-frame semantic affordances without flattening temporal order."""

    def __init__(self, matcher: Any, *, recency_decay: float = 0.25) -> None:
        if not 0.0 <= recency_decay <= 1.0:
            raise ValueError("recency_decay must be between zero and one")
        self.matcher = matcher
        self.recency_decay = float(recency_decay)

    def score_texts(self, observation: str, commands: list[str]) -> list[float]:
        return [float(value) for value in self.matcher.score_texts(observation, commands)]

    def score_state(self, state: SystemStateFrame) -> list[float]:
        commands = candidate_commands(state)
        frames = _receptor_frames(state)
        if not frames:
            return self.score_texts("", commands)
        frame_scores = [self.score_texts(frame, commands) for frame in frames]
        weights = [
            self.recency_decay ** (len(frames) - index - 1)
            for index in range(len(frames))
        ]
        normalizer = max(sum(weights), 1.0e-12)
        return [
            sum(weight * scores[command_index] for weight, scores in zip(weights, frame_scores))
            / normalizer
            for command_index in range(len(commands))
        ]

    def metadata(self) -> dict[str, Any]:
        base_metadata = (
            self.matcher.metadata()
            if callable(getattr(self.matcher, "metadata", None))
            else {"matcher_family": type(self.matcher).__name__}
        )
        return {
            "matcher_family": "recency_weighted_temporal_receptor",
            "recency_decay": self.recency_decay,
            "frame_source": "ordered_runtime_receptor_frames",
            "persistent_receptor_history": True,
            "base_matcher": base_metadata,
        }


class _DualProjection(nn.Module):
    def __init__(self, *, bins: int, embedding_dim: int) -> None:
        super().__init__()
        self.observation = nn.Linear(bins, embedding_dim, bias=False)
        self.command = nn.Linear(bins, embedding_dim, bias=False)

    def encode_observation(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.observation(values), dim=-1)

    def encode_command(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.command(values), dim=-1)


@dataclass(slots=True)
class SemanticMatcherTrainingSummary:
    concepts: int
    observation_variants: int
    command_variants: int
    epochs: int
    final_loss: float
    training_top1_accuracy: float


class HashedDualEncoderSemanticMatcher:
    """Learn bounded observation-command compatibility without runtime ontology lookup."""

    def __init__(
        self,
        *,
        bins: int = 1024,
        embedding_dim: int = 64,
        seed: int = 13,
        lexical_residual_weight: float = 0.0,
    ) -> None:
        torch.manual_seed(seed)
        self.bins = bins
        self.embedding_dim = embedding_dim
        self.seed = seed
        self.lexical_residual_weight = max(0.0, float(lexical_residual_weight))
        self.model = _DualProjection(bins=bins, embedding_dim=embedding_dim)
        self.training_summary: SemanticMatcherTrainingSummary | None = None

    def fit(
        self,
        concept_groups: dict[str, dict[str, list[str]]],
        *,
        epochs: int = 800,
        learning_rate: float = 0.03,
        temperature: float = 0.08,
    ) -> SemanticMatcherTrainingSummary:
        if len(concept_groups) < 2:
            raise ValueError("semantic matcher requires at least two concept groups")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(self.seed)
        names = sorted(concept_groups)
        final_loss = 0.0
        for _ in range(epochs):
            observations = []
            commands = []
            for name in names:
                group = concept_groups[name]
                observation_variants = group.get("observations", [])
                command_variants = group.get("commands", [])
                if not observation_variants or not command_variants:
                    raise ValueError(f"semantic group {name!r} requires observations and commands")
                observation_index = int(
                    torch.randint(len(observation_variants), (1,), generator=generator).item()
                )
                command_index = int(
                    torch.randint(len(command_variants), (1,), generator=generator).item()
                )
                observations.append(
                    _features(observation_variants[observation_index], bins=self.bins)
                )
                commands.append(_features(command_variants[command_index], bins=self.bins))
            observation_vectors = torch.stack(observations)
            command_vectors = torch.stack(commands)
            observation_embeddings = self.model.encode_observation(observation_vectors)
            command_embeddings = self.model.encode_command(command_vectors)
            logits = observation_embeddings @ command_embeddings.T / temperature
            labels = torch.arange(len(names))
            loss = (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)
            ) / 2.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
        accuracy = self._training_accuracy(concept_groups)
        self.training_summary = SemanticMatcherTrainingSummary(
            concepts=len(names),
            observation_variants=sum(
                len(group["observations"]) for group in concept_groups.values()
            ),
            command_variants=sum(
                len(group["commands"]) for group in concept_groups.values()
            ),
            epochs=epochs,
            final_loss=final_loss,
            training_top1_accuracy=accuracy,
        )
        return self.training_summary

    def _training_accuracy(self, concept_groups: dict[str, dict[str, list[str]]]) -> float:
        names = sorted(concept_groups)
        command_prototypes = torch.stack(
            [
                self.model.encode_command(
                    torch.stack(
                        [_features(text, bins=self.bins) for text in concept_groups[name]["commands"]]
                    )
                ).mean(dim=0)
                for name in names
            ]
        )
        correct = 0
        total = 0
        with torch.inference_mode():
            for index, name in enumerate(names):
                for text in concept_groups[name]["observations"]:
                    observation = self.model.encode_observation(
                        _features(text, bins=self.bins).view(1, -1)
                    )[0]
                    correct += int(
                        int((observation @ command_prototypes.T).argmax().item()) == index
                    )
                    total += 1
        return correct / max(total, 1)

    def score_texts(self, observation: str, commands: list[str]) -> list[float]:
        if not commands:
            return []
        with torch.inference_mode():
            observation_embedding = self.model.encode_observation(
                _features(observation, bins=self.bins).view(1, -1)
            )[0]
            command_embeddings = self.model.encode_command(
                torch.stack([_features(command, bins=self.bins) for command in commands])
            )
            learned_scores = [
                float(value) for value in (command_embeddings @ observation_embedding).tolist()
            ]
        if self.lexical_residual_weight <= 0.0:
            return learned_scores
        observation_words = set(_semantic_words(observation))
        scores = []
        for learned_score, command in zip(learned_scores, commands):
            command_words = set(_semantic_words(command))
            union = observation_words | command_words
            lexical_score = len(observation_words & command_words) / max(len(union), 1)
            scores.append(learned_score + self.lexical_residual_weight * lexical_score)
        return scores

    def score_state(self, state: SystemStateFrame) -> list[float]:
        return self.score_texts(_state_text(state), candidate_commands(state))

    def metadata(self) -> dict[str, Any]:
        return {
            "matcher_family": "hashed_dual_encoder",
            "bins": self.bins,
            "embedding_dim": self.embedding_dim,
            "seed": self.seed,
            "lexical_residual_weight": self.lexical_residual_weight,
            "runtime_ontology_lookup": False,
            "training_summary": (
                asdict(self.training_summary) if self.training_summary is not None else None
            ),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bins": self.bins,
                "embedding_dim": self.embedding_dim,
                "seed": self.seed,
                "lexical_residual_weight": self.lexical_residual_weight,
                "model_state_dict": self.model.state_dict(),
                "training_summary": (
                    asdict(self.training_summary) if self.training_summary is not None else None
                ),
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "HashedDualEncoderSemanticMatcher":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        matcher = cls(
            bins=int(payload["bins"]),
            embedding_dim=int(payload["embedding_dim"]),
            seed=int(payload["seed"]),
            lexical_residual_weight=float(payload.get("lexical_residual_weight", 0.0)),
        )
        matcher.model.load_state_dict(payload["model_state_dict"])
        summary = payload.get("training_summary")
        if summary is not None:
            matcher.training_summary = SemanticMatcherTrainingSummary(**summary)
        return matcher


class CausalLMConditionalSemanticMatcher:
    """Score bounded actions by their conditional association with receptor text."""

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        model_name: str,
        device: str | torch.device | None = None,
        max_length: int = 192,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.max_length = max(32, int(max_length))

    @classmethod
    def from_pretrained(
        cls,
        model_name: str | Path,
        *,
        device: str = "cpu",
        dtype: str = "auto",
        local_files_only: bool = True,
        max_length: int = 192,
    ) -> "CausalLMConditionalSemanticMatcher":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }
        if dtype != "auto":
            load_kwargs["dtype"] = getattr(torch, dtype)
        if device != "cpu":
            load_kwargs["device_map"] = device
        model = AutoModelForCausalLM.from_pretrained(str(model_name), **load_kwargs)
        if device == "cpu":
            model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(model_name), local_files_only=local_files_only)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=str(model_name),
            device=device,
            max_length=max_length,
        )

    def _token_ids(self, text: str, *, add_special_tokens: bool) -> list[int]:
        payload = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            truncation=True,
            max_length=self.max_length,
        )
        values = payload["input_ids"]
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(value) for value in values]

    def _conditional_log_likelihood(self, observation: str, command: str) -> float:
        prefix = (
            "Observed system failure: "
            f"{observation}\nBest bounded recovery action:"
        )
        prefix_ids = self._token_ids(prefix, add_special_tokens=True)
        completion_ids = self._token_ids(f" {command}", add_special_tokens=False)
        if not prefix_ids or not completion_ids:
            return float("-inf")
        available = max(1, self.max_length - len(completion_ids))
        prefix_ids = prefix_ids[-available:]
        input_ids = torch.tensor(
            [prefix_ids + completion_ids],
            dtype=torch.long,
            device=self.device,
        )
        with torch.inference_mode():
            outputs: SimpleNamespace = self.model(input_ids=input_ids)
            logits = outputs.logits[:, :-1].float()
        labels = input_ids[:, 1:]
        start = max(0, len(prefix_ids) - 1)
        token_log_probs = F.log_softmax(logits, dim=-1).gather(
            -1, labels.unsqueeze(-1)
        ).squeeze(-1)
        return float(token_log_probs[:, start:].mean().item())

    def score_texts(self, observation: str, commands: list[str]) -> list[float]:
        neutral = "an unspecified process exited unsuccessfully"
        scores = []
        for command in commands:
            semantic_command = _command_semantic_text(command)
            conditioned = self._conditional_log_likelihood(observation, semantic_command)
            baseline = self._conditional_log_likelihood(neutral, semantic_command)
            scores.append(conditioned - baseline)
        return scores

    def score_state(self, state: SystemStateFrame) -> list[float]:
        return self.score_texts(_receptor_text(state), candidate_commands(state))

    def metadata(self) -> dict[str, Any]:
        return {
            "matcher_family": "causal_lm_conditional_pmi",
            "model_name": self.model_name,
            "max_length": self.max_length,
            "state_text_source": "receptor_only",
            "runtime_ontology_lookup": False,
            "free_form_action_generation": False,
        }


@dataclass(slots=True)
class FrozenEncoderDualTrainingSummary:
    training_pairs: int
    epochs: int
    final_loss: float
    training_top1_accuracy: float


class FrozenEncoderDualSemanticMatcher:
    """Learn a small action head on top of frozen cortex text embeddings."""

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        model_name: str,
        device: str | torch.device | None = None,
        max_length: int = 128,
        projection_dim: int = 96,
        seed: int = 29,
    ) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.max_length = max(32, int(max_length))
        self.projection_dim = int(projection_dim)
        self.seed = int(seed)
        self.embedding_dim = int(getattr(getattr(model, "config", None), "hidden_size"))
        torch.manual_seed(self.seed)
        self.observation_projection = nn.Linear(self.embedding_dim, self.projection_dim, bias=False)
        self.command_projection = nn.Linear(self.embedding_dim, self.projection_dim, bias=False)
        self.training_summary: FrozenEncoderDualTrainingSummary | None = None
        self._embedding_cache: dict[str, torch.Tensor] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name: str | Path,
        *,
        device: str = "cpu",
        dtype: str = "auto",
        local_files_only: bool = True,
        max_length: int = 128,
        projection_dim: int = 96,
        seed: int = 29,
    ) -> "FrozenEncoderDualSemanticMatcher":
        from transformers import AutoModel, AutoTokenizer

        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
        }
        if dtype != "auto":
            load_kwargs["dtype"] = getattr(torch, dtype)
        if device != "cpu":
            load_kwargs["device_map"] = device
        model = AutoModel.from_pretrained(str(model_name), **load_kwargs)
        if device == "cpu":
            model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(model_name), local_files_only=local_files_only)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=str(model_name),
            device=device,
            max_length=max_length,
            projection_dim=projection_dim,
            seed=seed,
        )

    def _embed_texts(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, self.embedding_dim)
        missing = [text for text in texts if text not in self._embedding_cache]
        for offset in range(0, len(missing), 32):
            batch = missing[offset : offset + 32]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
                if hasattr(value, "to")
            }
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state.float()
            mask = encoded.get("attention_mask")
            if mask is None:
                pooled = hidden[:, -1, :]
            else:
                weights = mask.unsqueeze(-1).float()
                pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
            for text, vector in zip(batch, pooled.detach().cpu()):
                self._embedding_cache[text] = F.normalize(vector, dim=0)
        return torch.stack([self._embedding_cache[text] for text in texts])

    def fit(
        self,
        pairs: list[tuple[str, str]],
        *,
        epochs: int = 350,
        learning_rate: float = 0.03,
        temperature: float = 0.08,
    ) -> FrozenEncoderDualTrainingSummary:
        if len(pairs) < 2:
            raise ValueError("frozen encoder matcher requires at least two training pairs")
        observations = [left for left, _ in pairs]
        commands = list(dict.fromkeys(right for _, right in pairs))
        command_indices = {command: index for index, command in enumerate(commands)}
        observation_embeddings = self._embed_texts(observations)
        command_embeddings = self._embed_texts(commands)
        optimizer = torch.optim.AdamW(
            list(self.observation_projection.parameters())
            + list(self.command_projection.parameters()),
            lr=learning_rate,
            weight_decay=0.01,
        )
        labels = torch.tensor([command_indices[right] for _, right in pairs])
        final_loss = 0.0
        for _ in range(epochs):
            observation_projected = F.normalize(
                self.observation_projection(observation_embeddings),
                dim=-1,
            )
            command_projected = F.normalize(
                self.command_projection(command_embeddings),
                dim=-1,
            )
            logits = observation_projected @ command_projected.T / temperature
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
        with torch.inference_mode():
            observation_projected = F.normalize(
                self.observation_projection(observation_embeddings),
                dim=-1,
            )
            command_projected = F.normalize(
                self.command_projection(command_embeddings),
                dim=-1,
            )
            accuracy = float(
                (
                    (observation_projected @ command_projected.T).argmax(dim=1)
                    == labels
                )
                .float()
                .mean()
                .item()
            )
        self.training_summary = FrozenEncoderDualTrainingSummary(
            training_pairs=len(pairs),
            epochs=epochs,
            final_loss=final_loss,
            training_top1_accuracy=accuracy,
        )
        return self.training_summary

    def score_texts(self, observation: str, commands: list[str]) -> list[float]:
        if not commands:
            return []
        command_texts = [_command_semantic_text(command) for command in commands]
        observation_embedding = self._embed_texts([observation])
        command_embeddings = self._embed_texts(command_texts)
        with torch.inference_mode():
            observation_projected = F.normalize(
                self.observation_projection(observation_embedding),
                dim=-1,
            )
            command_projected = F.normalize(
                self.command_projection(command_embeddings),
                dim=-1,
            )
            return [
                float(value)
                for value in (command_projected @ observation_projected[0]).tolist()
            ]

    def score_state(self, state: SystemStateFrame) -> list[float]:
        return self.score_texts(_receptor_text(state), candidate_commands(state))

    def metadata(self) -> dict[str, Any]:
        return {
            "matcher_family": "frozen_encoder_dual_semantic_head",
            "model_name": self.model_name,
            "max_length": self.max_length,
            "projection_dim": self.projection_dim,
            "state_text_source": "receptor_only",
            "runtime_ontology_lookup": False,
            "free_form_action_generation": False,
            "training_summary": (
                asdict(self.training_summary) if self.training_summary else None
            ),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "reflexlm.frozen_encoder_dual_semantic_matcher.v1",
                "model_name": self.model_name,
                "max_length": self.max_length,
                "projection_dim": self.projection_dim,
                "seed": self.seed,
                "observation_projection_state_dict": self.observation_projection.state_dict(),
                "command_projection_state_dict": self.command_projection.state_dict(),
                "training_summary": (
                    asdict(self.training_summary) if self.training_summary else None
                ),
            },
            output,
        )
        return output

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        dtype: str = "auto",
        local_files_only: bool = True,
        model_name: str | Path | None = None,
    ) -> "FrozenEncoderDualSemanticMatcher":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if payload.get("schema_version") != "reflexlm.frozen_encoder_dual_semantic_matcher.v1":
            raise ValueError("unsupported frozen encoder semantic matcher schema")
        matcher = cls.from_pretrained(
            model_name or payload["model_name"],
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            max_length=int(payload["max_length"]),
            projection_dim=int(payload["projection_dim"]),
            seed=int(payload["seed"]),
        )
        matcher.observation_projection.load_state_dict(
            payload["observation_projection_state_dict"]
        )
        matcher.command_projection.load_state_dict(
            payload["command_projection_state_dict"]
        )
        summary = payload.get("training_summary")
        if isinstance(summary, dict):
            matcher.training_summary = FrozenEncoderDualTrainingSummary(**summary)
        return matcher
