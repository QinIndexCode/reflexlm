from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class FusionAdapterConfig:
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    fallback_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    latent_dim: int = 1280
    hidden_size: int = 1536
    prefix_tokens: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class LatentPrefixAdapter(nn.Module):
    """Maps NSI latents to prefix embeddings for Stage 2 fusion."""

    def __init__(self, config: FusionAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.prefix_tokens * config.hidden_size),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        prefix = self.project(latent)
        return prefix.view(latent.shape[0], self.config.prefix_tokens, self.config.hidden_size)


def build_stage2_fusion_bundle(config: FusionAdapterConfig) -> dict[str, object]:
    """Return a configuration bundle for later PEFT integration.

    The repository does not claim Stage 2 has already been run. This helper exists
    so the fusion route is codified and can be wired to `peft` once the model
    weights are downloaded locally.
    """

    return {
        "base_model_name": config.base_model_name,
        "fallback_model_name": config.fallback_model_name,
        "prefix_tokens": config.prefix_tokens,
        "lora": {
            "r": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
    }

