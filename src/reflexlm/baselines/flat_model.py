from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from reflexlm.models.features import ACTION_ORDER, MAX_CANDIDATE_SLOTS


@dataclass(slots=True)
class FlatTextModelConfig:
    input_dim: int
    hidden_dim: int = 1280
    gru_layers: int = 2
    dropout: float = 0.1

    @classmethod
    def smoke(cls, input_dim: int) -> "FlatTextModelConfig":
        return cls(input_dim=input_dim, hidden_dim=128, gru_layers=1, dropout=0.0)


class FlatTextBaselineModel(nn.Module):
    def __init__(self, config: FlatTextModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
        )
        self.action_head = nn.Linear(config.hidden_dim, len(ACTION_ORDER))
        self.command_slot_head = nn.Linear(config.hidden_dim, MAX_CANDIDATE_SLOTS)
        self.file_slot_head = nn.Linear(config.hidden_dim, MAX_CANDIDATE_SLOTS)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(inputs)
        temporal_out, hidden_out = self.temporal(encoded, hidden)
        return {
            "hidden": hidden_out,
            "action_logits": self.action_head(temporal_out),
            "command_slot_logits": self.command_slot_head(temporal_out),
            "file_slot_logits": self.file_slot_head(temporal_out),
        }

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

