from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn

from reflexlm.models.features import MAX_CANDIDATE_SLOTS, ROUTE_ORDER
from reflexlm.schema import ActionType, InternalTarget


@dataclass(slots=True)
class ReflexCoreV0Config:
    input_dim: int
    vocab_size: int = 4096
    text_embedding_dim: int = 128
    hidden_dim: int = 256
    transformer_layers: int = 2
    transformer_heads: int = 4
    gru_layers: int = 1
    max_command_slots: int = MAX_CANDIDATE_SLOTS
    max_file_slots: int = MAX_CANDIDATE_SLOTS
    dropout: float = 0.1
    prediction_error_calibration_scale: float = 0.02
    prediction_error_mode: str = "delta_plus_calibration"
    prediction_error_conditioning: str = "state"
    action_vector_residual: bool = False

    @classmethod
    def smoke(cls, *, input_dim: int, vocab_size: int = 512) -> "ReflexCoreV0Config":
        return cls(
            input_dim=input_dim,
            vocab_size=vocab_size,
            text_embedding_dim=32,
            hidden_dim=64,
            transformer_layers=1,
            transformer_heads=2,
            gru_layers=1,
            dropout=0.0,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ReflexCoreV0(nn.Module):
    """Small sensory-motor language core for bounded computer interaction.

    Inputs are structured observation vectors plus optional visible-text token
    ids. Outputs include language, motor, slot, risk/salience, prediction-error,
    and next-state heads.
    """

    def __init__(self, config: ReflexCoreV0Config) -> None:
        super().__init__()
        self.config = config
        self.text_embedding = nn.Embedding(
            config.vocab_size,
            config.text_embedding_dim,
            padding_idx=0,
        )
        self.vector_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.text_projector = nn.Sequential(
            nn.Linear(config.text_embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        if config.transformer_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.transformer_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
        else:
            self.transformer = nn.Identity()
        if config.gru_layers > 0:
            self.recurrent = nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.gru_layers,
                dropout=config.dropout if config.gru_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            self.recurrent = None
        self.action_embedding = nn.Embedding(len(ActionType), config.hidden_dim)
        self.text_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.action_head = nn.Linear(config.hidden_dim, len(ActionType))
        self.action_vector_head = (
            nn.Linear(config.hidden_dim, len(ActionType))
            if config.action_vector_residual
            else None
        )
        self.target_head = nn.Linear(config.hidden_dim, len(InternalTarget))
        self.route_head = nn.Linear(config.hidden_dim, len(ROUTE_ORDER))
        self.command_slot_head = nn.Linear(config.hidden_dim, config.max_command_slots)
        self.file_slot_head = nn.Linear(config.hidden_dim, config.max_file_slots)
        self.risk_head = nn.Linear(config.hidden_dim, 1)
        self.salience_head = nn.Linear(config.hidden_dim, 1)
        if config.prediction_error_conditioning == "state":
            prediction_error_input_dim = config.hidden_dim
        elif config.prediction_error_conditioning == "state_action":
            prediction_error_input_dim = config.hidden_dim * 2
        else:
            raise ValueError(
                "prediction_error_conditioning must be 'state' or 'state_action'"
            )
        self.prediction_error_head = nn.Linear(prediction_error_input_dim, 1)
        nn.init.zeros_(self.prediction_error_head.weight)
        nn.init.zeros_(self.prediction_error_head.bias)
        self.next_state_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.input_dim),
        )
        final_next_state = self.next_state_head[-1]
        if isinstance(final_next_state, nn.Linear):
            nn.init.zeros_(final_next_state.weight)
            nn.init.zeros_(final_next_state.bias)

    def forward(
        self,
        observation_vectors: Tensor,
        text_tokens: Tensor | None = None,
        *,
        action_indices: Tensor | None = None,
        hidden: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        squeeze_time = observation_vectors.ndim == 2
        if squeeze_time:
            observation_vectors = observation_vectors.unsqueeze(1)
        if observation_vectors.ndim != 3:
            raise ValueError("observation_vectors must have shape [batch, seq, input_dim]")
        batch_size, seq_len, _input_dim = observation_vectors.shape
        vector_state = self.vector_encoder(observation_vectors.float())
        text_state = self._encode_text_tokens(
            text_tokens,
            batch_size=batch_size,
            seq_len=seq_len,
            device=observation_vectors.device,
        )
        state = self.fusion(torch.cat([vector_state, text_state], dim=-1))
        state = self.transformer(state)
        if self.recurrent is not None:
            state, next_hidden = self.recurrent(state, hidden)
        else:
            next_hidden = hidden
        if action_indices is None:
            action_logits = self.action_head(state)
            if self.action_vector_head is not None:
                action_logits = action_logits + self.action_vector_head(vector_state)
            action_indices = action_logits.argmax(dim=-1)
        else:
            action_logits = self.action_head(state)
            if self.action_vector_head is not None:
                action_logits = action_logits + self.action_vector_head(vector_state)
            if action_indices.ndim == 1:
                action_indices = action_indices.unsqueeze(1)
        action_indices = action_indices.clamp(min=0, max=len(ActionType) - 1)
        action_state = self.action_embedding(action_indices)
        next_state_input = torch.cat([state, action_state], dim=-1)
        next_state_delta = self.next_state_head(next_state_input)
        predicted_delta_norm = torch.linalg.vector_norm(
            next_state_delta,
            dim=-1,
            keepdim=True,
        ) / max(math.sqrt(float(self.config.input_dim)), 1.0)
        if self.config.prediction_error_conditioning == "state":
            prediction_error_input = state
        elif self.config.prediction_error_conditioning == "state_action":
            prediction_error_input = next_state_input
        else:
            raise ValueError(
                "prediction_error_conditioning must be 'state' or 'state_action'"
            )
        prediction_error_head = (
            torch.sigmoid(self.prediction_error_head(prediction_error_input))
            * self.config.prediction_error_calibration_scale
        )
        if self.config.prediction_error_mode == "direct":
            prediction_error = prediction_error_head
        elif self.config.prediction_error_mode == "delta_plus_calibration":
            prediction_error = predicted_delta_norm + prediction_error_head
        else:
            raise ValueError(
                "prediction_error_mode must be 'direct' or 'delta_plus_calibration'"
            )
        outputs: dict[str, Tensor | None] = {
            "memory": state,
            "hidden": next_hidden,
            "text_logits": self.text_head(state),
            "action_logits": action_logits,
            "target_logits": self.target_head(state),
            "route_logits": self.route_head(state),
            "command_slot_logits": self.command_slot_head(state),
            "file_slot_logits": self.file_slot_head(state),
            "risk": torch.sigmoid(self.risk_head(state)),
            "salience": torch.sigmoid(self.salience_head(state)),
            "prediction_error": prediction_error,
            "next_state": observation_vectors.float() + next_state_delta,
        }
        if squeeze_time:
            for key, value in list(outputs.items()):
                if key != "hidden" and isinstance(value, Tensor) and value.ndim >= 3:
                    outputs[key] = value.squeeze(1)
        return outputs

    def _encode_text_tokens(
        self,
        text_tokens: Tensor | None,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        if text_tokens is None:
            return torch.zeros(
                batch_size,
                seq_len,
                self.config.hidden_dim,
                dtype=torch.float32,
                device=device,
            )
        if text_tokens.ndim == 2:
            text_tokens = text_tokens.unsqueeze(1)
        if text_tokens.ndim != 3:
            raise ValueError("text_tokens must have shape [batch, seq, tokens]")
        text_tokens = text_tokens.to(device=device, dtype=torch.long)
        text_tokens = text_tokens.clamp(min=0, max=self.config.vocab_size - 1)
        embedded = self.text_embedding(text_tokens)
        mask = (text_tokens > 0).unsqueeze(-1)
        denom = mask.sum(dim=2).clamp(min=1)
        pooled = (embedded * mask).sum(dim=2) / denom
        if pooled.shape[1] != seq_len:
            if pooled.shape[1] == 1:
                pooled = pooled.expand(-1, seq_len, -1)
            else:
                raise ValueError("text token sequence length does not match observations")
        return self.text_projector(pooled)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
