from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F
from torch import nn

from reflexlm.llm.candidate_features import (
    CANDIDATE_FEATURE_DIM,
    COMMAND_IDENTITY_FEATURE_END,
    COMMAND_IDENTITY_FEATURE_START,
    COMMAND_INTENT_COUNT,
    PAIRWISE_COMMAND_POLICIES,
)
from reflexlm.models.features import ACTION_ORDER, MAX_CANDIDATE_SLOTS, ROUTE_ORDER
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER

COMMAND_CANDIDATE_ENCODERS = ("backbone", "features_only")
OPEN_REPAIR_CAPABILITY_NAMES = (
    "patch_proposal_head",
    "test_selection_head",
    "rollback_safety_head",
    "stop_condition_head",
    "bounded_edit_scope_policy",
    "progress_monitor_receptors",
    "verification_state_receptors",
)
PATCH_OPERATION_ORDER = (
    "replace_symbol",
    "replace_attribute",
    "insert_import",
    "replace_literal",
    "insert_guard",
)
PATCH_TEMPLATE_ORDER = (
    "call_attribute_restoration",
    "import_restoration",
    "symbol_reference_restoration",
    "literal_restoration",
    "guard_restoration",
)


@dataclass(slots=True)
class NativeCortexHeadConfig:
    backbone_hidden_dim: int
    nsi_latent_dim: int
    head_hidden_dim: int = 512
    dropout: float = 0.05
    inject_nsi_latent: bool = True
    latent_fusion: str = "concat"
    command_candidate_feature_dim: int = CANDIDATE_FEATURE_DIM
    command_candidate_encoder: str = "backbone"
    use_pairwise_command_reranker: bool = False
    pairwise_command_fusion: str = "replace"
    pairwise_command_policy: str = "all"
    pairwise_command_max_length: int | None = None
    pairwise_command_top_k: int | None = None
    command_identity_logit_bias: float = 0.0
    open_repair_heads_enabled: bool = False


class NativeCortexActionHeads(nn.Module):
    """Explicit heads for a shared-backbone semantic cortex.

    The module consumes a backbone state embedding plus an optional NSI latent.
    It does not generate JSON. Runtime serializes the selected head outputs into
    the fixed motor schema after inhibition, route, and slot decisions.
    """

    def __init__(self, config: NativeCortexHeadConfig) -> None:
        super().__init__()
        self.config = config
        if config.latent_fusion not in {"concat", "additive"}:
            raise ValueError(f"unsupported latent_fusion={config.latent_fusion!r}")
        if config.pairwise_command_fusion not in {"replace", "residual"}:
            raise ValueError(f"unsupported pairwise_command_fusion={config.pairwise_command_fusion!r}")
        if config.pairwise_command_policy not in PAIRWISE_COMMAND_POLICIES:
            raise ValueError(f"unsupported pairwise_command_policy={config.pairwise_command_policy!r}")
        if config.pairwise_command_top_k is not None and config.pairwise_command_top_k <= 0:
            raise ValueError("pairwise_command_top_k must be positive when provided")
        if config.command_identity_logit_bias < 0:
            raise ValueError("command_identity_logit_bias must be non-negative")
        if config.command_candidate_encoder not in COMMAND_CANDIDATE_ENCODERS:
            raise ValueError(f"unsupported command_candidate_encoder={config.command_candidate_encoder!r}")
        input_dim = config.backbone_hidden_dim
        if config.inject_nsi_latent and config.latent_fusion == "concat":
            input_dim += config.nsi_latent_dim
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.latent_projection = (
            nn.Sequential(
                nn.Linear(config.nsi_latent_dim, config.head_hidden_dim),
                nn.LayerNorm(config.head_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            if config.inject_nsi_latent and config.latent_fusion == "additive"
            else None
        )
        self.action_head = nn.Linear(config.head_hidden_dim, len(ACTION_ORDER))
        self.target_head = nn.Linear(config.head_hidden_dim, len(INTERNAL_TARGET_ORDER))
        self.route_head = nn.Linear(config.head_hidden_dim, len(ROUTE_ORDER))
        self.command_intent_head = nn.Linear(config.head_hidden_dim, COMMAND_INTENT_COUNT)
        self.command_slot_head = nn.Linear(config.head_hidden_dim, MAX_CANDIDATE_SLOTS)
        self.file_slot_head = nn.Linear(config.head_hidden_dim, MAX_CANDIDATE_SLOTS)
        self.command_candidate_projection = nn.Sequential(
            nn.Linear(config.backbone_hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.command_feature_projection = nn.Sequential(
            nn.Linear(config.command_candidate_feature_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.command_candidate_scorer = nn.Sequential(
            nn.Linear(config.head_hidden_dim * 4, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
        self.command_pair_projection = nn.Sequential(
            nn.Linear(config.backbone_hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.command_pair_scorer = nn.Sequential(
            nn.Linear(config.head_hidden_dim * 4, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
        self.file_candidate_projection = nn.Sequential(
            nn.Linear(config.backbone_hidden_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.file_candidate_scorer = nn.Sequential(
            nn.Linear(config.head_hidden_dim * 4, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
        self.confidence_head = nn.Linear(config.head_hidden_dim, 1)
        self.inhibition_head = nn.Linear(config.head_hidden_dim, 1)
        self.salience_head = nn.Linear(config.head_hidden_dim, 1)
        self.risk_head = nn.Linear(config.head_hidden_dim, 1)
        self.prediction_error_head = nn.Linear(config.head_hidden_dim, 1)
        if config.open_repair_heads_enabled:
            self.patch_proposal_head = nn.Linear(config.head_hidden_dim, 2)
            self.test_selection_head = nn.Linear(config.head_hidden_dim, MAX_CANDIDATE_SLOTS)
            self.rollback_safety_head = nn.Linear(config.head_hidden_dim, 2)
            self.stop_condition_head = nn.Linear(config.head_hidden_dim, 2)
            self.bounded_edit_scope_head = nn.Linear(config.head_hidden_dim, 2)
            self.progress_monitor_head = nn.Linear(config.head_hidden_dim, 3)
            self.verification_state_head = nn.Linear(config.head_hidden_dim, 3)
            self.patch_operation_head = nn.Linear(config.head_hidden_dim, len(PATCH_OPERATION_ORDER))
            self.patch_target_file_slot_head = nn.Linear(config.head_hidden_dim, MAX_CANDIDATE_SLOTS)
            self.patch_template_slot_head = nn.Linear(config.head_hidden_dim, MAX_CANDIDATE_SLOTS)

    def _candidate_logits(
        self,
        hidden: torch.Tensor,
        candidate_embeddings: torch.Tensor | None,
        candidate_mask: torch.Tensor | None,
        *,
        projection: nn.Module,
        scorer: nn.Module,
        candidate_features: torch.Tensor | None = None,
        feature_projection: nn.Module | None = None,
        command_intent_logits: torch.Tensor | None = None,
        command_candidate_intents: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if candidate_embeddings is None:
            if candidate_features is None or feature_projection is None:
                return None
            projection_dtype = next(feature_projection.parameters()).dtype
            hidden = hidden.to(dtype=projection_dtype)
            projected = feature_projection(candidate_features.to(dtype=projection_dtype))
        else:
            projection_dtype = next(projection.parameters()).dtype
            candidate_embeddings = candidate_embeddings.to(dtype=projection_dtype)
            hidden = hidden.to(dtype=projection_dtype)
            projected = projection(candidate_embeddings)
            if candidate_features is not None and feature_projection is not None:
                projected = projected + feature_projection(candidate_features.to(dtype=projection_dtype))
        expanded = hidden.unsqueeze(1).expand_as(projected)
        pair_features = torch.cat(
            [
                expanded,
                projected,
                expanded * projected,
                torch.abs(expanded - projected),
            ],
            dim=-1,
        )
        logits = scorer(pair_features).squeeze(-1)
        if command_intent_logits is not None and command_candidate_intents is not None:
            intent_count = int(command_intent_logits.shape[-1])
            one_hot = F.one_hot(
                command_candidate_intents.clamp(min=0, max=intent_count - 1),
                num_classes=intent_count,
            ).to(dtype=projection_dtype)
            logits = logits + (one_hot * command_intent_logits.unsqueeze(1).to(dtype=projection_dtype)).sum(dim=-1)
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask.bool(), -1.0e4)
        return logits

    def _apply_command_identity_prior(
        self,
        logits: torch.Tensor,
        candidate_features: torch.Tensor | None,
        candidate_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Bias candidate logits with runtime-visible command-identity evidence.

        The feature extractor computes candidate-identity evidence from structured
        runtime receptors. This deterministic prior prevents the learned scorer
        from erasing a clear bounded identity signal while keeping the mechanism
        disabled by default for older adapters.
        """

        bias = float(self.config.command_identity_logit_bias or 0.0)
        if bias <= 0.0 or candidate_features is None:
            return logits
        if candidate_features.shape[-1] < COMMAND_IDENTITY_FEATURE_END:
            return logits
        identity = candidate_features[
            ..., COMMAND_IDENTITY_FEATURE_START:COMMAND_IDENTITY_FEATURE_END
        ].to(dtype=logits.dtype)
        prior = identity.sum(dim=-1) * bias
        if candidate_mask is not None:
            prior = prior.masked_fill(~candidate_mask.bool(), 0.0)
        return logits + prior

    def forward(
        self,
        backbone_state: torch.Tensor,
        *,
        nsi_latent: torch.Tensor | None = None,
        command_candidate_embeddings: torch.Tensor | None = None,
        command_candidate_mask: torch.Tensor | None = None,
        command_candidate_features: torch.Tensor | None = None,
        command_candidate_intents: torch.Tensor | None = None,
        command_pair_embeddings: torch.Tensor | None = None,
        command_pair_mask: torch.Tensor | None = None,
        file_candidate_embeddings: torch.Tensor | None = None,
        file_candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent_hidden: torch.Tensor | None = None
        if self.config.inject_nsi_latent:
            if nsi_latent is None:
                raise ValueError("nsi_latent is required when inject_nsi_latent=True")
            latent = nsi_latent.to(dtype=backbone_state.dtype)
            if self.config.latent_fusion == "concat":
                x = torch.cat([backbone_state, latent], dim=-1)
            else:
                x = backbone_state
                assert self.latent_projection is not None
                latent_hidden = self.latent_projection(
                    latent.to(dtype=next(self.latent_projection.parameters()).dtype)
                )
        else:
            x = backbone_state
        x = x.to(dtype=next(self.trunk.parameters()).dtype)
        hidden = self.trunk(x)
        if latent_hidden is not None:
            hidden = hidden + latent_hidden.to(dtype=hidden.dtype)
        outputs = {
            "action_logits": self.action_head(hidden),
            "target_logits": self.target_head(hidden),
            "route_logits": self.route_head(hidden),
            "command_intent_logits": self.command_intent_head(hidden),
            "command_slot_logits": self.command_slot_head(hidden),
            "file_slot_logits": self.file_slot_head(hidden),
            "confidence": torch.sigmoid(self.confidence_head(hidden)).squeeze(-1),
            "inhibition": torch.sigmoid(self.inhibition_head(hidden)).squeeze(-1),
            "salience": torch.sigmoid(self.salience_head(hidden)).squeeze(-1),
            "risk": torch.sigmoid(self.risk_head(hidden)).squeeze(-1),
            "prediction_error": torch.sigmoid(self.prediction_error_head(hidden)).squeeze(-1),
        }
        if self.config.open_repair_heads_enabled:
            outputs.update(
                {
                    "patch_proposal_logits": self.patch_proposal_head(hidden),
                    "test_selection_logits": self.test_selection_head(hidden),
                    "rollback_safety_logits": self.rollback_safety_head(hidden),
                    "stop_condition_logits": self.stop_condition_head(hidden),
                    "bounded_edit_scope_logits": self.bounded_edit_scope_head(hidden),
                    "progress_monitor_logits": self.progress_monitor_head(hidden),
                    "verification_state_logits": self.verification_state_head(hidden),
                    "patch_operation_logits": self.patch_operation_head(hidden),
                    "patch_target_file_slot_logits": self.patch_target_file_slot_head(hidden),
                    "patch_template_slot_logits": self.patch_template_slot_head(hidden),
                }
            )
        command_candidate_logits = self._candidate_logits(
            hidden,
            command_candidate_embeddings,
            command_candidate_mask,
            projection=self.command_candidate_projection,
            scorer=self.command_candidate_scorer,
            candidate_features=command_candidate_features,
            feature_projection=self.command_feature_projection,
            command_intent_logits=outputs["command_intent_logits"],
            command_candidate_intents=command_candidate_intents,
        )
        if command_candidate_logits is not None:
            command_candidate_logits = self._apply_command_identity_prior(
                command_candidate_logits,
                command_candidate_features,
                command_candidate_mask,
            )
            outputs["command_lightweight_candidate_logits"] = command_candidate_logits
            outputs["command_candidate_logits"] = command_candidate_logits
        if self.config.use_pairwise_command_reranker:
            command_pair_logits = self._candidate_logits(
                hidden,
                command_pair_embeddings,
                command_pair_mask,
                projection=self.command_pair_projection,
                scorer=self.command_pair_scorer,
            )
            if command_pair_logits is not None:
                outputs["command_pair_logits"] = command_pair_logits
                # Pairwise scoring is optional because full candidate cross
                # encoding is too expensive for the default 8GB path. The
                # residual mode keeps lightweight visible-state features active
                # instead of letting the cross-encoder erase them.
                if self.config.pairwise_command_fusion == "residual" and command_candidate_logits is not None:
                    if command_pair_mask is None:
                        pair_contribution = command_pair_logits
                    else:
                        pair_contribution = torch.where(
                            command_pair_mask.bool(),
                            command_pair_logits,
                            torch.zeros_like(command_pair_logits),
                        )
                    outputs["command_candidate_logits"] = command_candidate_logits + pair_contribution
                elif command_candidate_logits is not None and command_pair_mask is not None:
                    outputs["command_candidate_logits"] = torch.where(
                        command_pair_mask.bool(),
                        command_pair_logits,
                        command_candidate_logits,
                    )
                else:
                    outputs["command_candidate_logits"] = command_pair_logits
        file_candidate_logits = self._candidate_logits(
            hidden,
            file_candidate_embeddings,
            file_candidate_mask,
            projection=self.file_candidate_projection,
            scorer=self.file_candidate_scorer,
        )
        if file_candidate_logits is not None:
            outputs["file_candidate_logits"] = file_candidate_logits
        return outputs


class QwenBackboneHeadAdapter(nn.Module):
    """Qwen backbone wrapper with native action heads.

    This is the Phase 2C direction: a shared language backbone supplies semantic
    state, NSI supplies native system-state latent input, and explicit heads
    choose actions. It intentionally avoids text JSON generation.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        head_config: NativeCortexHeadConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = NativeCortexActionHeads(head_config)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        nsi_latent: torch.Tensor,
        command_input_ids: torch.Tensor | None = None,
        command_attention_mask: torch.Tensor | None = None,
        command_candidate_mask: torch.Tensor | None = None,
        command_candidate_features: torch.Tensor | None = None,
        command_candidate_intents: torch.Tensor | None = None,
        command_pair_input_ids: torch.Tensor | None = None,
        command_pair_attention_mask: torch.Tensor | None = None,
        command_pair_mask: torch.Tensor | None = None,
        file_input_ids: torch.Tensor | None = None,
        file_attention_mask: torch.Tensor | None = None,
        file_candidate_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        def pool_hidden(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            input_ids = input_ids.long()
            attention_mask = attention_mask.long() if attention_mask is not None else None
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )
            hidden_states = outputs.hidden_states[-1]
            if attention_mask is None:
                return hidden_states[:, -1, :]
            last_indices = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            row_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            return hidden_states[row_indices, last_indices]

        def pool_candidates(
            candidate_input_ids: torch.Tensor | None,
            candidate_attention_mask: torch.Tensor | None,
            candidate_mask: torch.Tensor | None,
        ) -> torch.Tensor | None:
            if candidate_input_ids is None:
                return None
            batch_size, candidate_count, sequence_length = candidate_input_ids.shape
            flattened_ids = candidate_input_ids.reshape(batch_size * candidate_count, sequence_length)
            flattened_mask = (
                candidate_attention_mask.reshape(batch_size * candidate_count, sequence_length)
                if candidate_attention_mask is not None
                else None
            )
            if candidate_mask is None:
                flat_valid = torch.ones(
                    batch_size * candidate_count,
                    dtype=torch.bool,
                    device=flattened_ids.device,
                )
            else:
                flat_valid = candidate_mask.reshape(batch_size * candidate_count).bool().to(flattened_ids.device)
            output_dtype = next(self.backbone.parameters()).dtype
            pooled = torch.zeros(
                batch_size * candidate_count,
                self.heads.config.backbone_hidden_dim,
                dtype=output_dtype,
                device=flattened_ids.device,
            )
            if bool(flat_valid.any().item()):
                pooled_valid = pool_hidden(
                    flattened_ids[flat_valid],
                    flattened_mask[flat_valid] if flattened_mask is not None else None,
                )
                pooled = pooled.to(device=pooled_valid.device, dtype=pooled_valid.dtype)
                pooled[flat_valid.to(pooled.device)] = pooled_valid
            return pooled.reshape(batch_size, candidate_count, -1)

        pooled = pool_hidden(input_ids, attention_mask)
        command_embeddings = pool_candidates(command_input_ids, command_attention_mask, command_candidate_mask)
        command_pair_embeddings = (
            pool_candidates(command_pair_input_ids, command_pair_attention_mask, command_pair_mask)
            if self.heads.config.use_pairwise_command_reranker
            else None
        )
        file_embeddings = pool_candidates(file_input_ids, file_attention_mask, file_candidate_mask)
        return self.heads(
            pooled,
            nsi_latent=nsi_latent,
            command_candidate_embeddings=command_embeddings,
            command_candidate_mask=command_candidate_mask,
            command_candidate_features=command_candidate_features,
            command_candidate_intents=command_candidate_intents,
            command_pair_embeddings=command_pair_embeddings,
            command_pair_mask=command_pair_mask,
            file_candidate_embeddings=file_embeddings,
            file_candidate_mask=file_candidate_mask,
        )
