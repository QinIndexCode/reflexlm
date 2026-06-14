from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from reflexlm.models.features import ACTION_ORDER, MAX_CANDIDATE_SLOTS, ROUTE_ORDER
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER


@dataclass(slots=True)
class NSIModelConfig:
    input_dim: int
    hidden_dim: int = 1280
    encoder_depth: int = 4
    gru_layers: int = 2
    dropout: float = 0.1
    route_conditioned_action: bool = False
    auxiliary_heads: bool = True
    action_conditioned_world_model: bool = True
    residual_world_model: bool = False

    @classmethod
    def smoke(cls, input_dim: int) -> "NSIModelConfig":
        return cls(
            input_dim=input_dim,
            hidden_dim=128,
            encoder_depth=2,
            gru_layers=1,
            dropout=0.0,
        )


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class NSIReflexModel(nn.Module):
    def __init__(self, config: NSIModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.encoder_blocks = nn.ModuleList(
            [FeedForwardBlock(config.hidden_dim, config.dropout) for _ in range(config.encoder_depth)]
        )
        self.temporal = (
            nn.GRU(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.gru_layers,
                batch_first=True,
                dropout=config.dropout if config.gru_layers > 1 else 0.0,
            )
            if config.gru_layers > 0
            else None
        )
        self.memory_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(config.hidden_dim, len(ACTION_ORDER))
        self.target_head = nn.Linear(config.hidden_dim, len(INTERNAL_TARGET_ORDER))
        self.route_action_heads = (
            nn.ModuleList([nn.Linear(config.hidden_dim, len(ACTION_ORDER)) for _ in ROUTE_ORDER])
            if config.route_conditioned_action
            else None
        )
        self.command_slot_head = nn.Linear(config.hidden_dim, MAX_CANDIDATE_SLOTS)
        self.file_slot_head = nn.Linear(config.hidden_dim, MAX_CANDIDATE_SLOTS)
        self.route_head = (
            nn.Linear(config.hidden_dim, len(ROUTE_ORDER))
            if config.auxiliary_heads or config.route_conditioned_action
            else None
        )
        self.salience_head = nn.Linear(config.hidden_dim, 1) if config.auxiliary_heads else None
        self.urgency_head = nn.Linear(config.hidden_dim, 1) if config.auxiliary_heads else None
        self.risk_head = nn.Linear(config.hidden_dim, 1) if config.auxiliary_heads else None
        self.novelty_head = nn.Linear(config.hidden_dim, 1) if config.auxiliary_heads else None
        self.prediction_error_head = (
            nn.Linear(config.hidden_dim, 1) if config.auxiliary_heads else None
        )
        self.next_state_head = (
            (
                nn.Sequential(
                    nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, config.input_dim),
                )
                if config.action_conditioned_world_model
                else nn.Linear(config.hidden_dim, config.input_dim)
            )
            if config.auxiliary_heads
            else None
        )
        self.world_model_action_embedding = (
            nn.Embedding(len(ACTION_ORDER), config.hidden_dim)
            if config.auxiliary_heads and config.action_conditioned_world_model
            else None
        )

    def predict_next_state(
        self,
        memory: torch.Tensor,
        inputs: torch.Tensor,
        *,
        action_indices: torch.Tensor | None = None,
        action_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.next_state_head is None:
            raise RuntimeError("next-state prediction requires auxiliary_heads=True")
        if self.config.action_conditioned_world_model:
            if self.world_model_action_embedding is None:
                raise RuntimeError(
                    "action_conditioned_world_model=True requires action embedding"
                )
            if action_indices is not None:
                action_context = self.world_model_action_embedding(action_indices)
            elif action_logits is not None:
                action_context = torch.matmul(
                    torch.softmax(action_logits, dim=-1),
                    self.world_model_action_embedding.weight,
                )
            else:
                raise ValueError(
                    "action-conditioned next-state prediction requires action indices or logits"
                )
            world_model_input = torch.cat([memory, action_context], dim=-1)
        else:
            world_model_input = memory
        next_state_prediction = self.next_state_head(world_model_input)
        if self.config.residual_world_model:
            next_state_prediction = inputs + next_state_prediction
        return next_state_prediction

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor | None = None,
        action_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.input_projection(inputs)
        for block in self.encoder_blocks:
            x = block(x)
        if self.temporal is None:
            gru_out = x
            hidden_out = None
        else:
            gru_out, hidden_out = self.temporal(x, hidden)
        memory = self.memory_projection(gru_out)
        action_logits = self.action_head(memory)
        if self.config.route_conditioned_action and self.route_action_heads is not None:
            if self.route_head is None:
                raise RuntimeError("route_conditioned_action requires route_head")
            route_logits = self.route_head(memory)
            route_probs = torch.softmax(route_logits, dim=-1)
            expert_logits = torch.stack(
                [head(memory) for head in self.route_action_heads],
                dim=-2,
            )
            action_logits = (expert_logits * route_probs.unsqueeze(-1)).sum(dim=-2)
        outputs = {
            "memory": memory,
            "hidden": hidden_out,
            "action_logits": action_logits,
            "target_logits": self.target_head(memory),
            "command_slot_logits": self.command_slot_head(memory),
            "file_slot_logits": self.file_slot_head(memory),
        }
        if self.config.auxiliary_heads:
            if (
                self.route_head is None
                or self.salience_head is None
                or self.urgency_head is None
                or self.risk_head is None
                or self.novelty_head is None
                or self.prediction_error_head is None
                or self.next_state_head is None
            ):
                raise RuntimeError("auxiliary_heads=True requires all auxiliary heads")
            outputs.update(
                {
                    "route_logits": self.route_head(memory),
                    "salience": self.salience_head(memory).squeeze(-1),
                    "urgency": self.urgency_head(memory).squeeze(-1),
                    "risk": self.risk_head(memory).squeeze(-1),
                    "novelty": self.novelty_head(memory).squeeze(-1),
                    "prediction_error": self.prediction_error_head(memory).squeeze(-1),
                }
            )
            outputs["next_state"] = self.predict_next_state(
                memory,
                inputs,
                action_indices=action_indices,
                action_logits=action_logits,
            )
        return outputs

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())
