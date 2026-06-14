from __future__ import annotations

import json
import hashlib
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from reflexlm.experiment import create_experiment_run, stable_config_hash
from reflexlm.llm.candidate_features import (
    CANDIDATE_FEATURE_DIM,
    COMMAND_INTENT_COUNT,
    PAIRWISE_COMMAND_POLICIES,
    build_candidate_pair_prompt,
    command_candidate_feature_rows,
    command_candidate_intent_indices,
    command_intent_for_text,
    command_intent_target,
    guard_command_identity_reference,
    pairwise_command_candidate_mask,
    source_overlap_command_slot_prediction,
)
from reflexlm.llm.native_cortex import (
    COMMAND_CANDIDATE_ENCODERS,
    OPEN_REPAIR_CAPABILITY_NAMES,
    PATCH_OPERATION_ORDER,
    PATCH_TEMPLATE_ORDER,
    NativeCortexHeadConfig,
    QwenBackboneHeadAdapter,
)
from reflexlm.llm.receptor_latent import (
    COMMAND_IDENTITY_LATENT_FIELDS,
    DESCRIPTOR_FAILURE_FAMILY_LATENT_FIELDS,
    DESCRIPTOR_FAILURE_FAMILY_ORDER,
    DEBUG_ACTION_STAGE_LATENT_FIELDS,
    DEBUG_ACTION_STAGE_ORDER,
    RECEPTOR_FAILURE_SIGNAL_ORDER,
)
from reflexlm.models.features import ACTION_ORDER, MAX_CANDIDATE_SLOTS, ROUTE_ORDER


BASE_NSI_LATENT_FIELDS = ("salience", "risk", "prediction_error", "confidence")
RECEPTOR_FAILURE_SIGNAL_GAIN = 4.0
DEBUG_ACTION_STAGE_GAIN = 4.0
DESCRIPTOR_FAILURE_FAMILY_GAIN = 4.0
NSI_LATENT_FIELDS = (
    *BASE_NSI_LATENT_FIELDS,
    *(f"reflex_action:{action.value}" for action in ACTION_ORDER),
    *(f"route:{route.value}" for route in ROUTE_ORDER),
    *(f"receptor_failure:{signal}" for signal in RECEPTOR_FAILURE_SIGNAL_ORDER),
    *DEBUG_ACTION_STAGE_LATENT_FIELDS,
    *DESCRIPTOR_FAILURE_FAMILY_LATENT_FIELDS,
    *COMMAND_IDENTITY_LATENT_FIELDS,
)


def _assert_requested_device_available(device: str) -> None:
    """Fail fast when an evidence run requests CUDA from a CPU-only torch env."""

    normalized = str(device or "").lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Use a CUDA-enabled Python environment, for example "
            ".venv312-gpu\\Scripts\\python.exe or .venv312-qwen7b-stable\\Scripts\\python.exe."
        )


def nsi_latent_values(reference: dict[str, Any]) -> list[float]:
    values = [
        float(reference.get(field_name, 0.0) or 0.0)
        for field_name in BASE_NSI_LATENT_FIELDS
    ]
    reflex_action = str(reference.get("reflex_action") or "")
    values.extend(
        1.0 if reflex_action == action.value else 0.0
        for action in ACTION_ORDER
    )
    route_name = str(reference.get("route_name") or "")
    values.extend(
        1.0 if route_name == route.value else 0.0
        for route in ROUTE_ORDER
    )
    receptor_signal = str(reference.get("receptor_failure_signal") or "other")
    values.extend(
        RECEPTOR_FAILURE_SIGNAL_GAIN if receptor_signal == signal else 0.0
        for signal in RECEPTOR_FAILURE_SIGNAL_ORDER
    )
    debug_action_stage = str(reference.get("debug_action_stage") or "other")
    values.extend(
        DEBUG_ACTION_STAGE_GAIN if debug_action_stage == stage else 0.0
        for stage in DEBUG_ACTION_STAGE_ORDER
    )
    descriptor_failure_family = str(reference.get("descriptor_failure_family") or "other")
    values.extend(
        DESCRIPTOR_FAILURE_FAMILY_GAIN if descriptor_failure_family == family else 0.0
        for family in DESCRIPTOR_FAILURE_FAMILY_ORDER
    )
    values.extend(float(reference.get(field, 0.0) or 0.0) for field in COMMAND_IDENTITY_LATENT_FIELDS)
    return values


@dataclass(slots=True)
class NativeHeadTrainConfig:
    base_model_name: str
    adapter_name: str
    quantization: str = "4bit"
    learning_rate: float = 1e-4
    epochs: int = 1
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 512
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    head_hidden_dim: int = 512
    head_dropout: float = 0.05
    seed: int = 13
    device: str = "cuda"
    progress_log_interval_steps: int = 50
    checkpoint_interval_steps: int = 0
    checkpoint_dir: str | None = None
    resume_from_checkpoint: str | None = None
    max_train_records: int | None = None
    max_val_records: int | None = None
    debug_command_oversample: int = 1
    balance_command_slots: bool = False
    balance_debug_command_intents: bool = False
    balance_patch_descriptor_labels: bool = False
    use_pairwise_command_reranker: bool = False
    pairwise_command_fusion: str = "residual"
    pairwise_command_policy: str = "all"
    pairwise_command_max_length: int | None = None
    pairwise_command_top_k: int | None = None
    command_identity_logit_bias: float = 0.0
    command_candidate_encoder: str = "backbone"
    latent_fusion: str = "additive"
    open_repair_heads_enabled: bool = False
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "action": 1.0,
            "internal_target": 1.0,
            "route": 0.5,
            "command_intent": 0.5,
            "command_slot": 0.3,
            "file_slot": 0.3,
            "confidence": 0.1,
            "inhibition": 0.3,
            "salience": 0.1,
            "risk": 0.1,
            "prediction_error": 0.1,
            "patch_proposal": 0.3,
            "test_selection": 0.3,
            "rollback_safety": 0.3,
            "stop_condition": 0.3,
            "bounded_edit_scope": 0.3,
            "progress_monitor": 0.3,
            "verification_state": 0.3,
        }
    )


CHECKPOINT_STATE_FILENAME = "trainer_state.pt"
CHECKPOINT_METADATA_FILENAME = "trainer_state.json"
CHECKPOINT_VERSION = 1


class Phase2CHeadJsonlDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        path: str | Path,
        *,
        limit: int | None = None,
        debug_command_oversample: int = 1,
        balance_command_slots: bool = False,
        balance_debug_command_intents: bool = False,
        balance_patch_descriptor_labels: bool = False,
    ) -> None:
        self.rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if balance_debug_command_intents:
            self.rows = _balance_debug_command_intent_rows(self.rows)
        if balance_command_slots:
            self.rows = _balance_command_slot_rows(self.rows)
        if balance_patch_descriptor_labels:
            self.rows = _balance_patch_descriptor_rows(self.rows)
        if debug_command_oversample > 1:
            self.rows = _oversample_debug_command_rows(self.rows, debug_command_oversample)
        if limit is not None:
            self.rows = _balanced_limited_rows(self.rows, max(limit, 0))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _slot_label(row: dict[str, Any], key: str) -> int:
    try:
        return int(row.get(key, -100))
    except (TypeError, ValueError):
        return -100


def _balanced_limit_primary_key(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("task_type", "")),
        str(row.get("head_scope", "")),
        str(row.get("action_type", "")),
    )


def _balanced_limit_secondary_key(row: dict[str, Any]) -> tuple[str, ...]:
    action_type = str(row.get("action_type", ""))
    key = [action_type]
    if action_type == "RUN_COMMAND":
        command_slot = _slot_label(row, "command_slot")
        if command_slot != -100:
            key.append(f"command_slot:{command_slot}")
            command_intent = str(row.get("command_intent") or command_intent_for_text(row.get("command")))
            key.append(f"command_intent:{command_intent}")
    elif action_type == "READ_FILE":
        file_slot = _slot_label(row, "file_slot")
        if file_slot != -100:
            key.append(f"file_slot:{file_slot}")
    patch_operation = _slot_label(row, "patch_operation_label")
    patch_template = _slot_label(row, "patch_template_slot")
    if patch_operation != -100:
        key.append(f"patch_operation:{patch_operation}")
    if patch_template != -100:
        key.append(f"patch_template:{patch_template}")
    return tuple(key)


def _round_robin_by_key(
    rows: list[dict[str, Any]],
    key_fn: Any,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    selected: list[dict[str, Any]] = []
    group_keys = sorted(groups)
    index = 0
    while True:
        added = False
        for key in group_keys:
            group = groups[key]
            if index < len(group):
                selected.append(group[index])
                added = True
        if not added:
            break
        index += 1
    return selected


def _balanced_limited_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    primary_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        primary_groups[_balanced_limit_primary_key(row)].append(row)
    ordered_groups = {
        key: _round_robin_by_key(group, _balanced_limit_secondary_key)
        for key, group in primary_groups.items()
    }
    selected: list[dict[str, Any]] = []
    group_keys = sorted(ordered_groups)
    index = 0
    while len(selected) < limit:
        added = False
        for key in group_keys:
            group = ordered_groups[key]
            if index < len(group):
                selected.append(group[index])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        index += 1
    return selected


def _canonical_rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _command_slot_baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    slot_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    total = 0
    correct = 0
    for row in rows:
        command_slot = _slot_label(row, "command_slot")
        if command_slot == -100:
            continue
        candidates = list(row.get("candidate_commands") or [])[:MAX_CANDIDATE_SLOTS]
        prediction = source_overlap_command_slot_prediction(str(row.get("state_prompt") or ""), candidates)
        is_correct = int(prediction == command_slot)
        total += 1
        correct += is_correct
        intent = str(row.get("command_intent") or command_intent_for_text(row.get("command")))
        totals[intent]["total"] += 1
        totals[intent]["correct"] += is_correct
        slot_key = str(command_slot)
        slot_totals[slot_key]["total"] += 1
        slot_totals[slot_key]["correct"] += is_correct

    def finalize(bucket: dict[str, int]) -> dict[str, Any]:
        bucket_total = int(bucket["total"])
        bucket_correct = int(bucket["correct"])
        return {
            "total": bucket_total,
            "correct": bucket_correct,
            "accuracy": bucket_correct / bucket_total if bucket_total else 0.0,
        }

    return {
        "method": "source_overlap_command_slot_prediction",
        "uses_labels_for_prediction": False,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_intent": {key: finalize(value) for key, value in sorted(totals.items())},
        "by_gold_slot": {key: finalize(value) for key, value in sorted(slot_totals.items())},
    }


def _slot_intent_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    command_intents: dict[str, int] = defaultdict(int)
    command_slots: dict[str, int] = defaultdict(int)
    actions: dict[str, int] = defaultdict(int)
    head_scopes: dict[str, int] = defaultdict(int)
    command_rows = 0
    for row in rows:
        action = str(row.get("action_type") or "")
        actions[action] += 1
        head_scopes[str(row.get("head_scope") or "")] += 1
        command_slot = _slot_label(row, "command_slot")
        if command_slot == -100:
            continue
        command_rows += 1
        command_slots[str(command_slot)] += 1
        intent = str(row.get("command_intent") or command_intent_for_text(row.get("command")))
        command_intents[intent] += 1
    return {
        "rows": len(rows),
        "command_slot_rows": command_rows,
        "actions": dict(sorted(actions.items())),
        "head_scopes": dict(sorted(head_scopes.items())),
        "command_intents": dict(sorted(command_intents.items())),
        "command_slots": dict(sorted(command_slots.items(), key=lambda item: item[0])),
    }


def _patch_descriptor_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    operation_labels: dict[str, int] = defaultdict(int)
    operation_names: dict[str, int] = defaultdict(int)
    template_slots: dict[str, int] = defaultdict(int)
    template_names: dict[str, int] = defaultdict(int)
    operation_template_pairs: dict[str, int] = defaultdict(int)
    descriptor_rows = 0
    for row in rows:
        operation = _slot_label(row, "patch_operation_label")
        template = _slot_label(row, "patch_template_slot")
        if operation == -100 and template == -100:
            continue
        descriptor_rows += 1
        operation_key = str(operation)
        template_key = str(template)
        operation_labels[operation_key] += 1
        template_slots[template_key] += 1
        if 0 <= operation < len(PATCH_OPERATION_ORDER):
            operation_key = PATCH_OPERATION_ORDER[operation]
            operation_names[operation_key] += 1
        if 0 <= template < len(PATCH_TEMPLATE_ORDER):
            template_key = PATCH_TEMPLATE_ORDER[template]
            template_names[template_key] += 1
        operation_template_pairs[f"{operation_key}|{template_key}"] += 1
    return {
        "rows": len(rows),
        "descriptor_rows": descriptor_rows,
        "patch_operation_labels": dict(sorted(operation_labels.items())),
        "patch_operation_names": dict(sorted(operation_names.items())),
        "patch_template_slots": dict(sorted(template_slots.items())),
        "patch_template_names": dict(sorted(template_names.items())),
        "patch_operation_template_pairs": dict(sorted(operation_template_pairs.items())),
    }


def _pairwise_candidate_encoding_stats(
    rows: list[dict[str, Any]],
    *,
    enabled: bool = True,
    policy: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    if policy not in PAIRWISE_COMMAND_POLICIES:
        raise ValueError(f"unsupported pairwise_command_policy={policy!r}")
    command_rows = 0
    rows_with_any_candidate = 0
    rows_with_pairwise = 0
    valid_candidates = 0
    scored_candidates = 0
    max_valid_candidates = 0
    max_scored_candidates = 0
    for row in rows:
        if int(row.get("command_slot", -100)) == -100:
            continue
        candidates = list(row.get("candidate_commands") or [])[:MAX_CANDIDATE_SLOTS]
        command_rows += 1
        if candidates:
            rows_with_any_candidate += 1
        mask = pairwise_command_candidate_mask(
            candidates,
            policy,
            visible_state_text=str(row.get("state_prompt", "")),
            top_k=top_k,
        )
        valid_count = len(candidates)
        scored_count = sum(1 for value in mask if value)
        valid_candidates += valid_count
        scored_candidates += scored_count
        max_valid_candidates = max(max_valid_candidates, valid_count)
        max_scored_candidates = max(max_scored_candidates, scored_count)
        if scored_count > 0:
            rows_with_pairwise += 1
    if not enabled:
        rows_with_pairwise = 0
        scored_candidates = 0
        max_scored_candidates = 0
    return {
        "enabled": enabled,
        "policy": policy,
        "top_k": top_k,
        "rows": len(rows),
        "command_slot_rows": command_rows,
        "rows_with_any_candidate": rows_with_any_candidate,
        "rows_with_pairwise_scoring": rows_with_pairwise,
        "valid_command_candidates": valid_candidates,
        "pairwise_scored_candidates": scored_candidates,
        "skipped_command_candidates": max(valid_candidates - scored_candidates, 0),
        "max_valid_candidates_per_row": max_valid_candidates,
        "max_pairwise_scored_candidates_per_row": max_scored_candidates,
        "pairwise_to_valid_candidate_ratio": (
            scored_candidates / valid_candidates if valid_candidates else 0.0
        ),
    }


def _oversample_debug_command_rows(
    rows: list[dict[str, Any]],
    factor: int,
) -> list[dict[str, Any]]:
    factor = max(int(factor), 1)
    if factor <= 1:
        return rows
    expanded: list[dict[str, Any]] = []
    for row in rows:
        expanded.append(row)
        is_debug_command = (
            row.get("head_scope") == "debug_cortex"
            and row.get("action_type") == "RUN_COMMAND"
            and int(row.get("command_slot", -100)) != -100
        )
        if is_debug_command:
            expanded.extend([row for _ in range(factor - 1)])
    return expanded


def _balance_debug_command_intent_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        is_debug_command = (
            row.get("head_scope") == "debug_cortex"
            and row.get("action_type") == "RUN_COMMAND"
            and int(row.get("command_slot", -100)) != -100
        )
        if not is_debug_command:
            passthrough.append(row)
            continue
        groups[command_intent_for_text(str(row.get("command") or ""))].append(row)
    if not groups:
        return rows
    target_count = max(len(group) for group in groups.values())
    balanced = list(passthrough)
    for intent_name in sorted(groups):
        group = groups[intent_name]
        repeats = (target_count + len(group) - 1) // len(group)
        balanced.extend((group * repeats)[:target_count])
    return balanced


def _balance_command_slot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        command_slot = _slot_label(row, "command_slot")
        if command_slot == -100:
            passthrough.append(row)
            continue
        groups[command_slot].append(row)
    if not groups:
        return rows
    target_count = max(len(group) for group in groups.values())
    balanced = list(passthrough)
    for slot in sorted(groups):
        group = groups[slot]
        repeats = (target_count + len(group) - 1) // len(group)
        balanced.extend((group * repeats)[:target_count])
    return balanced


def _balance_patch_descriptor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        patch_operation = _slot_label(row, "patch_operation_label")
        patch_template = _slot_label(row, "patch_template_slot")
        if patch_operation == -100 and patch_template == -100:
            passthrough.append(row)
            continue
        groups[(patch_operation, patch_template)].append(row)
    if not groups:
        return rows
    target_count = max(len(group) for group in groups.values())
    balanced = list(passthrough)
    for key in sorted(groups):
        group = groups[key]
        repeats = (target_count + len(group) - 1) // len(group)
        balanced.extend((group * repeats)[:target_count])
    return balanced


def _nsi_latent_from_row(row: dict[str, Any]) -> torch.Tensor:
    reference = guard_command_identity_reference(
        str(row.get("state_prompt") or ""),
        list(row.get("candidate_commands") or [])[:MAX_CANDIDATE_SLOTS],
        row.get("nsi_reference") or {},
    )
    return torch.tensor(nsi_latent_values(reference), dtype=torch.float32)


def _tokenize_head_example(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_length: int,
) -> dict[str, torch.Tensor]:
    tokenized = tokenizer(
        row["state_prompt"],
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(tokenized.get("attention_mask", [1] * len(input_ids)), dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "nsi_latent": _nsi_latent_from_row(row),
        "action_labels": torch.tensor(int(row["action_index"]), dtype=torch.long),
        "target_labels": torch.tensor(int(row["internal_target_index"]), dtype=torch.long),
        "route_labels": torch.tensor(int(row["route_index"]), dtype=torch.long),
        "command_intent_labels": torch.tensor(
            command_intent_target(row.get("command")),
            dtype=torch.long,
        ),
        "command_slot_labels": torch.tensor(int(row["command_slot"]), dtype=torch.long),
        "file_slot_labels": torch.tensor(int(row["file_slot"]), dtype=torch.long),
        "confidence_targets": torch.tensor(float(row["confidence_target"]), dtype=torch.float32),
        "inhibition_targets": torch.tensor(float(row["inhibition_target"]), dtype=torch.float32),
        "salience_targets": torch.tensor(float(row["salience_target"]), dtype=torch.float32),
        "risk_targets": torch.tensor(float(row["risk_target"]), dtype=torch.float32),
        "prediction_error_targets": torch.tensor(
            float(row["prediction_error_target"]),
            dtype=torch.float32,
        ),
        "patch_proposal_labels": torch.tensor(_slot_label(row, "patch_proposal_label"), dtype=torch.long),
        "test_selection_labels": torch.tensor(_slot_label(row, "test_selection_slot"), dtype=torch.long),
        "rollback_safety_labels": torch.tensor(_slot_label(row, "rollback_safety_label"), dtype=torch.long),
        "stop_condition_labels": torch.tensor(_slot_label(row, "stop_condition_label"), dtype=torch.long),
        "bounded_edit_scope_labels": torch.tensor(_slot_label(row, "bounded_edit_scope_label"), dtype=torch.long),
        "progress_monitor_labels": torch.tensor(_slot_label(row, "progress_monitor_label"), dtype=torch.long),
        "verification_state_labels": torch.tensor(_slot_label(row, "verification_state_label"), dtype=torch.long),
        "patch_operation_labels": torch.tensor(_slot_label(row, "patch_operation_label"), dtype=torch.long),
        "patch_target_file_slot_labels": torch.tensor(_slot_label(row, "patch_target_file_slot"), dtype=torch.long),
        "patch_template_slot_labels": torch.tensor(_slot_label(row, "patch_template_slot"), dtype=torch.long),
    }


def _tokenize_candidate_text(
    tokenizer: Any,
    text: str,
    *,
    kind: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    tokenized = tokenizer(
        f"{kind} candidate:\n{text}",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(tokenized.get("attention_mask", [1] * len(input_ids)), dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _tokenize_candidate_pair_text(
    tokenizer: Any,
    state_prompt: str,
    candidate_text: str,
    *,
    kind: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    prompt = build_candidate_pair_prompt(
        state_prompt,
        candidate_text,
        kind=kind,
    )
    tokenized = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(tokenized.get("attention_mask", [1] * len(input_ids)), dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _collate_candidate_texts(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    key: str,
    kind: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    candidate_max_length = max(8, min(max_length, 64))
    tokenized_rows: list[list[dict[str, torch.Tensor]]] = []
    max_len = 1
    for row in rows:
        candidates = list(row.get(key) or [])[:MAX_CANDIDATE_SLOTS]
        tokenized_candidates: list[dict[str, torch.Tensor]] = []
        for index in range(MAX_CANDIDATE_SLOTS):
            text = candidates[index] if index < len(candidates) else ""
            tokenized = _tokenize_candidate_text(
                tokenizer,
                text,
                kind=kind,
                max_length=candidate_max_length,
            )
            max_len = max(max_len, int(tokenized["input_ids"].shape[0]))
            tokenized_candidates.append(tokenized)
        tokenized_rows.append(tokenized_candidates)

    pad_id = tokenizer.pad_token_id
    input_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    candidate_masks: list[torch.Tensor] = []
    for row, tokenized_candidates in zip(rows, tokenized_rows):
        candidates = list(row.get(key) or [])[:MAX_CANDIDATE_SLOTS]
        input_candidates: list[torch.Tensor] = []
        attention_candidates: list[torch.Tensor] = []
        candidate_mask = torch.zeros(MAX_CANDIDATE_SLOTS, dtype=torch.bool)
        for index, tokenized in enumerate(tokenized_candidates):
            pad = max_len - tokenized["input_ids"].shape[0]
            input_candidates.append(nn.functional.pad(tokenized["input_ids"], (0, pad), value=pad_id))
            attention_candidates.append(nn.functional.pad(tokenized["attention_mask"], (0, pad), value=0))
            if index < len(candidates):
                candidate_mask[index] = True
        input_rows.append(torch.stack(input_candidates))
        mask_rows.append(torch.stack(attention_candidates))
        candidate_masks.append(candidate_mask)
    return {
        "input_ids": torch.stack(input_rows),
        "attention_mask": torch.stack(mask_rows),
        "candidate_mask": torch.stack(candidate_masks),
    }


def _collate_candidate_masks(
    rows: list[dict[str, Any]],
    *,
    key: str,
) -> torch.Tensor:
    masks: list[torch.Tensor] = []
    for row in rows:
        candidates = list(row.get(key) or [])[:MAX_CANDIDATE_SLOTS]
        mask = torch.zeros(MAX_CANDIDATE_SLOTS, dtype=torch.bool)
        for index in range(len(candidates)):
            mask[index] = True
        masks.append(mask)
    return torch.stack(masks)


def _collate_candidate_pair_texts(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    key: str,
    kind: str,
    max_length: int,
    pairwise_command_policy: str = "all",
    pairwise_command_max_length: int | None = None,
    pairwise_command_top_k: int | None = None,
) -> dict[str, torch.Tensor]:
    pair_max_length = max(32, int(pairwise_command_max_length or max_length))
    tokenized_rows: list[list[dict[str, torch.Tensor]]] = []
    pairwise_masks: list[list[bool]] = []
    max_len = 1
    for row in rows:
        candidates = list(row.get(key) or [])[:MAX_CANDIDATE_SLOTS]
        pairwise_mask = (
            pairwise_command_candidate_mask(
                candidates,
                pairwise_command_policy,
                visible_state_text=str(row["state_prompt"]),
                top_k=pairwise_command_top_k,
            )
            if key == "candidate_commands"
            else [index < len(candidates) for index in range(MAX_CANDIDATE_SLOTS)]
        )
        pairwise_masks.append(pairwise_mask)
        tokenized_candidates: list[dict[str, torch.Tensor]] = []
        for index in range(MAX_CANDIDATE_SLOTS):
            if index < len(candidates) and pairwise_mask[index]:
                tokenized = _tokenize_candidate_pair_text(
                    tokenizer,
                    str(row["state_prompt"]),
                    candidates[index],
                    kind=kind,
                    max_length=pair_max_length,
                )
            else:
                tokenized = tokenizer(
                    "",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=1,
                )
                tokenized = {
                    "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(
                        tokenized.get("attention_mask", [1] * len(tokenized["input_ids"])),
                        dtype=torch.long,
                    ),
                }
            max_len = max(max_len, int(tokenized["input_ids"].shape[0]))
            tokenized_candidates.append(tokenized)
        tokenized_rows.append(tokenized_candidates)

    pad_id = tokenizer.pad_token_id
    input_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    candidate_masks: list[torch.Tensor] = []
    for tokenized_candidates, pairwise_mask in zip(tokenized_rows, pairwise_masks):
        input_candidates: list[torch.Tensor] = []
        attention_candidates: list[torch.Tensor] = []
        candidate_mask = torch.zeros(MAX_CANDIDATE_SLOTS, dtype=torch.bool)
        for index, tokenized in enumerate(tokenized_candidates):
            pad = max_len - tokenized["input_ids"].shape[0]
            input_candidates.append(nn.functional.pad(tokenized["input_ids"], (0, pad), value=pad_id))
            attention_candidates.append(nn.functional.pad(tokenized["attention_mask"], (0, pad), value=0))
            if pairwise_mask[index]:
                candidate_mask[index] = True
        input_rows.append(torch.stack(input_candidates))
        mask_rows.append(torch.stack(attention_candidates))
        candidate_masks.append(candidate_mask)
    return {
        "input_ids": torch.stack(input_rows),
        "attention_mask": torch.stack(mask_rows),
        "candidate_mask": torch.stack(candidate_masks),
    }


def _resize_candidate_feature_rows(
    rows: list[list[float]],
    feature_dim: int,
) -> list[list[float]]:
    resized: list[list[float]] = []
    for row in rows:
        values = list(row)
        if len(values) > feature_dim:
            values = values[:feature_dim]
        elif len(values) < feature_dim:
            values.extend([0.0] * (feature_dim - len(values)))
        resized.append(values)
    return resized


def _collate_head_rows(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
    use_pairwise_command_reranker: bool = False,
    pairwise_command_policy: str = "all",
    pairwise_command_max_length: int | None = None,
    pairwise_command_top_k: int | None = None,
    command_candidate_encoder: str = "backbone",
    command_candidate_feature_dim: int = CANDIDATE_FEATURE_DIM,
) -> dict[str, torch.Tensor]:
    if command_candidate_encoder not in COMMAND_CANDIDATE_ENCODERS:
        raise ValueError(f"unsupported command_candidate_encoder={command_candidate_encoder!r}")
    batch = [_tokenize_head_example(tokenizer, row, max_length=max_length) for row in rows]
    pad_id = tokenizer.pad_token_id
    max_len = max(item["input_ids"].shape[0] for item in batch)
    collated: dict[str, list[torch.Tensor]] = {
        "input_ids": [],
        "attention_mask": [],
        "nsi_latent": [],
        "action_labels": [],
        "target_labels": [],
        "route_labels": [],
        "command_intent_labels": [],
        "command_slot_labels": [],
        "file_slot_labels": [],
        "confidence_targets": [],
        "inhibition_targets": [],
        "salience_targets": [],
        "risk_targets": [],
        "prediction_error_targets": [],
        "patch_proposal_labels": [],
        "test_selection_labels": [],
        "rollback_safety_labels": [],
        "stop_condition_labels": [],
        "bounded_edit_scope_labels": [],
        "progress_monitor_labels": [],
        "verification_state_labels": [],
        "patch_operation_labels": [],
        "patch_target_file_slot_labels": [],
        "patch_template_slot_labels": [],
    }
    for item in batch:
        pad = max_len - item["input_ids"].shape[0]
        collated["input_ids"].append(nn.functional.pad(item["input_ids"], (0, pad), value=pad_id))
        collated["attention_mask"].append(nn.functional.pad(item["attention_mask"], (0, pad), value=0))
        for key in collated:
            if key not in {"input_ids", "attention_mask"}:
                collated[key].append(item[key])
    output = {key: torch.stack(values) for key, values in collated.items()}
    if any(int(row.get("command_slot", -100)) != -100 for row in rows):
        if command_candidate_encoder == "backbone":
            command_candidates = _collate_candidate_texts(
                tokenizer,
                rows,
                key="candidate_commands",
                kind="Command",
                max_length=max_length,
            )
            output["command_input_ids"] = command_candidates["input_ids"]
            output["command_attention_mask"] = command_candidates["attention_mask"]
            output["command_candidate_mask"] = command_candidates["candidate_mask"]
        else:
            output["command_candidate_mask"] = _collate_candidate_masks(rows, key="candidate_commands")
        output["command_candidate_features"] = torch.tensor(
            [
                _resize_candidate_feature_rows(
                    command_candidate_feature_rows(
                        str(row["state_prompt"]),
                        list(row.get("candidate_commands") or [])[:MAX_CANDIDATE_SLOTS],
                        nsi_reference=row.get("nsi_reference") or {},
                    ),
                    command_candidate_feature_dim,
                )
                for row in rows
            ],
            dtype=torch.float32,
        )
        output["command_candidate_intents"] = torch.tensor(
            [
                command_candidate_intent_indices(
                    list(row.get("candidate_commands") or [])[:MAX_CANDIDATE_SLOTS],
                )
                for row in rows
            ],
            dtype=torch.long,
        )
        if use_pairwise_command_reranker:
            command_pairs = _collate_candidate_pair_texts(
                tokenizer,
                rows,
                key="candidate_commands",
                kind="Command",
                max_length=max_length,
                pairwise_command_policy=pairwise_command_policy,
                pairwise_command_max_length=pairwise_command_max_length,
                pairwise_command_top_k=pairwise_command_top_k,
            )
            output["command_pair_input_ids"] = command_pairs["input_ids"]
            output["command_pair_attention_mask"] = command_pairs["attention_mask"]
            output["command_pair_mask"] = command_pairs["candidate_mask"]
    if any(int(row.get("file_slot", -100)) != -100 for row in rows):
        file_candidates = _collate_candidate_texts(
            tokenizer,
            rows,
            key="candidate_files",
            kind="File",
            max_length=max_length,
        )
        output["file_input_ids"] = file_candidates["input_ids"]
        output["file_attention_mask"] = file_candidates["attention_mask"]
        output["file_candidate_mask"] = file_candidates["candidate_mask"]
    return output


def compute_native_head_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    loss_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = NativeHeadTrainConfig(base_model_name="", adapter_name="").loss_weights
    if loss_weights:
        weights.update(loss_weights)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    zero = outputs["action_logits"].sum() * 0.0

    def slot_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels != -100
        if int(valid.sum().item()) == 0:
            return zero
        return nn.functional.cross_entropy(logits[valid], labels[valid])

    losses = {
        "action": ce_loss(outputs["action_logits"], batch["action_labels"]),
        "internal_target": ce_loss(outputs["target_logits"], batch["target_labels"]),
        "route": ce_loss(outputs["route_logits"], batch["route_labels"]),
        "command_intent": slot_loss(
            outputs["command_intent_logits"],
            batch["command_intent_labels"],
        ),
        "command_slot": slot_loss(
            outputs.get("command_candidate_logits", outputs["command_slot_logits"]),
            batch["command_slot_labels"],
        ),
        "file_slot": slot_loss(
            outputs.get("file_candidate_logits", outputs["file_slot_logits"]),
            batch["file_slot_labels"],
        ),
        "confidence": mse_loss(outputs["confidence"], batch["confidence_targets"]),
        "inhibition": mse_loss(outputs["inhibition"], batch["inhibition_targets"]),
        "salience": mse_loss(outputs["salience"], batch["salience_targets"]),
        "risk": mse_loss(outputs["risk"], batch["risk_targets"]),
        "prediction_error": mse_loss(outputs["prediction_error"], batch["prediction_error_targets"]),
    }
    optional_ce_losses = {
        "patch_proposal": ("patch_proposal_logits", "patch_proposal_labels"),
        "test_selection": ("test_selection_logits", "test_selection_labels"),
        "rollback_safety": ("rollback_safety_logits", "rollback_safety_labels"),
        "stop_condition": ("stop_condition_logits", "stop_condition_labels"),
        "bounded_edit_scope": ("bounded_edit_scope_logits", "bounded_edit_scope_labels"),
        "progress_monitor": ("progress_monitor_logits", "progress_monitor_labels"),
        "verification_state": ("verification_state_logits", "verification_state_labels"),
        "patch_operation": ("patch_operation_logits", "patch_operation_labels"),
        "patch_target_file_slot": ("patch_target_file_slot_logits", "patch_target_file_slot_labels"),
        "patch_template_slot": ("patch_template_slot_logits", "patch_template_slot_labels"),
    }
    for name, (output_key, label_key) in optional_ce_losses.items():
        if output_key in outputs and label_key in batch:
            losses[name] = slot_loss(outputs[output_key], batch[label_key])
    total = sum(losses[name] * weights.get(name, 1.0) for name in losses)
    return total, {name: float(value.detach().cpu().item()) for name, value in losses.items()}


def _build_quantization_config(config: NativeHeadTrainConfig) -> Any:
    from transformers import BitsAndBytesConfig

    if config.quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    if config.quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if config.quantization == "none":
        return None
    raise ValueError(f"Unsupported quantization mode: {config.quantization}")


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _checkpoint_neutral_config(config: NativeHeadTrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("checkpoint_dir", None)
    payload.pop("checkpoint_interval_steps", None)
    payload.pop("resume_from_checkpoint", None)
    return payload


def _open_repair_training_contract(config: NativeHeadTrainConfig) -> dict[str, Any]:
    learned_targets_enabled = bool(config.open_repair_heads_enabled)
    return {
        "sealed_feedback_used": False,
        "learned_patch_candidate_targets": learned_targets_enabled,
        "recorded_patch_artifact_as_generation_target": False,
        "symbolic_generator_as_generation_target": False,
        "freeform_patch_text_target": False,
        "json_text_target": False,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "patch_descriptor_heads_enabled": learned_targets_enabled,
        "patch_proposal_strategy": (
            "learned_bounded_candidate" if learned_targets_enabled else "none"
        ),
    }


def _native_head_training_identity_hash(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    output_dir: str | Path,
    config: NativeHeadTrainConfig,
    train_rows_hash: str | None = None,
    val_rows_hash: str | None = None,
) -> str:
    payload = {
        "train_jsonl": str(Path(train_jsonl).resolve()),
        "val_jsonl": str(Path(val_jsonl).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "config": _checkpoint_neutral_config(config),
    }
    if train_rows_hash is not None:
        payload["train_rows_hash"] = train_rows_hash
    if val_rows_hash is not None:
        payload["val_rows_hash"] = val_rows_hash
    return stable_config_hash(payload)


def _checkpoint_root_for_run(run: Any, config: NativeHeadTrainConfig) -> Path:
    return Path(config.checkpoint_dir) if config.checkpoint_dir else run.path / "checkpoints"


def _checkpoint_path(root: str | Path, *, epoch: int, step: int) -> Path:
    return Path(root) / f"epoch{epoch:04d}-step{step:06d}"


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _save_native_head_training_checkpoint(
    *,
    checkpoint_root: str | Path,
    model: QwenBackboneHeadAdapter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    total_steps: int,
    identity_hash: str,
    stats: dict[str, Any],
) -> Path:
    from peft import get_peft_model_state_dict

    checkpoint_dir = _checkpoint_path(checkpoint_root, epoch=epoch, step=step)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "identity_hash": identity_hash,
        "epoch": epoch,
        "step": step,
        "total_steps": total_steps,
        "state_file": CHECKPOINT_STATE_FILENAME,
        "stats": stats,
    }
    state = {
        "backbone_adapter": get_peft_model_state_dict(model.backbone),
        "heads": model.heads.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, checkpoint_dir / CHECKPOINT_STATE_FILENAME)
    (checkpoint_dir / CHECKPOINT_METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return checkpoint_dir


def _load_native_head_training_checkpoint(
    *,
    checkpoint_dir: str | Path,
    model: QwenBackboneHeadAdapter,
    optimizer: torch.optim.Optimizer,
    expected_identity_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    from peft import set_peft_model_state_dict

    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / CHECKPOINT_METADATA_FILENAME
    state_path = checkpoint_path / CHECKPOINT_STATE_FILENAME
    if not metadata_path.exists() or not state_path.exists():
        raise FileNotFoundError(
            f"checkpoint must contain {CHECKPOINT_METADATA_FILENAME} and {CHECKPOINT_STATE_FILENAME}: "
            f"{checkpoint_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint_version={metadata.get('checkpoint_version')!r}")
    if metadata.get("identity_hash") != expected_identity_hash:
        raise ValueError(
            "checkpoint identity mismatch; refusing to resume with different data, output, "
            "or non-checkpoint training config"
        )
    state = torch.load(state_path, map_location=device, weights_only=False)
    set_peft_model_state_dict(model.backbone, state["backbone_adapter"])
    model.heads.load_state_dict(state["heads"])
    optimizer.load_state_dict(state["optimizer"])
    _move_optimizer_state_to_device(optimizer, device)
    return metadata


def _write_progress_snapshot(
    run: Any,
    *,
    stage: str,
    epoch: int,
    step: int,
    total_steps: int,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "total_steps": total_steps,
        "metrics": metrics,
    }
    run.write_json("progress/latest.json", payload)
    progress_history_path = run.path / "progress" / "history.jsonl"
    existing_rows = []
    if progress_history_path.exists():
        existing_rows = [
            json.loads(line)
            for line in progress_history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    existing_rows.append(payload)
    run.write_jsonl("progress/history.jsonl", existing_rows)


def _classification_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int | None = None,
) -> tuple[int, int]:
    if ignore_index is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        mask = labels != ignore_index
    if int(mask.sum().item()) == 0:
        return 0, 0
    predictions = logits.argmax(dim=-1)
    correct = int((predictions[mask] == labels[mask]).sum().item())
    total = int(mask.sum().item())
    return correct, total


def _action_name(index: int) -> str:
    if 0 <= index < len(ACTION_ORDER):
        return ACTION_ORDER[index].value
    return f"unknown:{index}"


def _update_action_accuracy_by_target(
    buckets: dict[str, dict[str, Any]],
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    for predicted_index, target_index in zip(
        predictions.detach().cpu().tolist(),
        labels.detach().cpu().tolist(),
    ):
        target_name = _action_name(int(target_index))
        predicted_name = _action_name(int(predicted_index))
        bucket = buckets.setdefault(
            target_name,
            {"correct": 0, "total": 0, "predictions": defaultdict(int)},
        )
        bucket["total"] += 1
        bucket["correct"] += int(predicted_name == target_name)
        bucket["predictions"][predicted_name] += 1


def _finalize_action_accuracy_by_target(
    buckets: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    finalized: dict[str, dict[str, Any]] = {}
    for target_name, bucket in sorted(buckets.items()):
        total = int(bucket["total"])
        correct = int(bucket["correct"])
        finalized[target_name] = {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "predictions": dict(sorted(bucket["predictions"].items())),
        }
    return finalized


def _evaluate_native_head_model(
    model: QwenBackboneHeadAdapter,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_weights: dict[str, float],
) -> dict[str, Any]:
    model.eval()
    started = time.perf_counter()
    loss_total = 0.0
    batches = 0
    pairwise_encoded_candidates = 0
    correct_totals = {
        "action": [0, 0],
        "internal_target": [0, 0],
        "route": [0, 0],
        "command_intent": [0, 0],
        "command_slot": [0, 0],
        "file_slot": [0, 0],
        "patch_operation": [0, 0],
        "patch_target_file_slot": [0, 0],
        "patch_template_slot": [0, 0],
    }
    action_accuracy_by_target: dict[str, dict[str, Any]] = {}
    slot_confusion: dict[str, dict[str, dict[str, int]]] = {
        "command_slot": defaultdict(lambda: defaultdict(int)),
        "file_slot": defaultdict(lambda: defaultdict(int)),
        "patch_operation": defaultdict(lambda: defaultdict(int)),
        "patch_target_file_slot": defaultdict(lambda: defaultdict(int)),
        "patch_template_slot": defaultdict(lambda: defaultdict(int)),
    }
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            if "command_pair_mask" in batch:
                pairwise_encoded_candidates += int(batch["command_pair_mask"].sum().item())
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                nsi_latent=batch["nsi_latent"],
                command_input_ids=batch.get("command_input_ids"),
                command_attention_mask=batch.get("command_attention_mask"),
                command_candidate_mask=batch.get("command_candidate_mask"),
                command_candidate_features=batch.get("command_candidate_features"),
                command_candidate_intents=batch.get("command_candidate_intents"),
                command_pair_input_ids=batch.get("command_pair_input_ids"),
                command_pair_attention_mask=batch.get("command_pair_attention_mask"),
                command_pair_mask=batch.get("command_pair_mask"),
                file_input_ids=batch.get("file_input_ids"),
                file_attention_mask=batch.get("file_attention_mask"),
                file_candidate_mask=batch.get("file_candidate_mask"),
            )
            loss, _ = compute_native_head_loss(outputs, batch, loss_weights=loss_weights)
            loss_total += float(loss.item())
            batches += 1
            _update_action_accuracy_by_target(
                action_accuracy_by_target,
                predictions=outputs["action_logits"].argmax(dim=-1),
                labels=batch["action_labels"],
            )
            for name, logits_key, label_key, ignore_index in [
                ("action", "action_logits", "action_labels", None),
                ("internal_target", "target_logits", "target_labels", None),
                ("route", "route_logits", "route_labels", None),
                ("command_intent", "command_intent_logits", "command_intent_labels", -100),
                (
                    "command_slot",
                    "command_candidate_logits"
                    if "command_candidate_logits" in outputs
                    else "command_slot_logits",
                    "command_slot_labels",
                    -100,
                ),
                (
                    "file_slot",
                    "file_candidate_logits"
                    if "file_candidate_logits" in outputs
                    else "file_slot_logits",
                    "file_slot_labels",
                    -100,
                ),
                (
                    "patch_operation",
                    "patch_operation_logits",
                    "patch_operation_labels",
                    -100,
                ),
                (
                    "patch_target_file_slot",
                    "patch_target_file_slot_logits",
                    "patch_target_file_slot_labels",
                    -100,
                ),
                (
                    "patch_template_slot",
                    "patch_template_slot_logits",
                    "patch_template_slot_labels",
                    -100,
                ),
            ]:
                if logits_key not in outputs or label_key not in batch:
                    continue
                correct, total = _classification_accuracy(
                    outputs[logits_key],
                    batch[label_key],
                    ignore_index=ignore_index,
                )
                correct_totals[name][0] += correct
                correct_totals[name][1] += total
                if name in slot_confusion:
                    labels = batch[label_key].detach().cpu()
                    predictions = outputs[logits_key].argmax(dim=-1).detach().cpu()
                    valid = labels != -100
                    for target, prediction in zip(labels[valid].tolist(), predictions[valid].tolist()):
                        slot_confusion[name][str(int(target))][str(int(prediction))] += 1
    elapsed_seconds = time.perf_counter() - started
    metrics = {
        "loss": loss_total / max(batches, 1),
        "elapsed_seconds": elapsed_seconds,
        "batches_per_second": batches / elapsed_seconds if elapsed_seconds > 0.0 else 0.0,
        "pairwise_encoded_candidates": float(pairwise_encoded_candidates),
    }
    for name, (correct, total) in correct_totals.items():
        metrics[f"{name}_accuracy"] = correct / total if total else 0.0
        metrics[f"{name}_count"] = float(total)
    metrics["action_accuracy_by_target"] = _finalize_action_accuracy_by_target(
        action_accuracy_by_target
    )
    metrics["slot_confusion"] = {
        name: {
            target: dict(sorted(predictions.items(), key=lambda item: int(item[0])))
            for target, predictions in sorted(confusion.items(), key=lambda item: int(item[0]))
        }
        for name, confusion in slot_confusion.items()
        if confusion
    }
    return metrics


def _collect_native_head_prediction_records(
    model: QwenBackboneHeadAdapter,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    head_config: NativeCortexHeadConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            batch = _collate_head_rows(
                tokenizer,
                [row],
                max_length=max_length,
                use_pairwise_command_reranker=head_config.use_pairwise_command_reranker,
                pairwise_command_policy=head_config.pairwise_command_policy,
                pairwise_command_max_length=head_config.pairwise_command_max_length,
                pairwise_command_top_k=head_config.pairwise_command_top_k,
                command_candidate_encoder=head_config.command_candidate_encoder,
                command_candidate_feature_dim=head_config.command_candidate_feature_dim,
            )
            batch = _move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                nsi_latent=batch["nsi_latent"],
                command_input_ids=batch.get("command_input_ids"),
                command_attention_mask=batch.get("command_attention_mask"),
                command_candidate_mask=batch.get("command_candidate_mask"),
                command_candidate_features=batch.get("command_candidate_features"),
                command_candidate_intents=batch.get("command_candidate_intents"),
                command_pair_input_ids=batch.get("command_pair_input_ids"),
                command_pair_attention_mask=batch.get("command_pair_attention_mask"),
                command_pair_mask=batch.get("command_pair_mask"),
                file_input_ids=batch.get("file_input_ids"),
                file_attention_mask=batch.get("file_attention_mask"),
                file_candidate_mask=batch.get("file_candidate_mask"),
            )
            command_logits = (
                outputs["command_candidate_logits"]
                if "command_candidate_logits" in outputs
                else outputs.get("command_slot_logits")
            )
            command_slot_label = _slot_label(row, "command_slot")
            command_slot_prediction = (
                int(command_logits.argmax(dim=-1).detach().cpu()[0].item())
                if command_logits is not None
                else None
            )
            candidates = list(row.get("candidate_commands") or [])[:MAX_CANDIDATE_SLOTS]
            source_overlap_prediction = (
                source_overlap_command_slot_prediction(str(row.get("state_prompt") or ""), candidates)
                if candidates
                else None
            )
            nsi_reference = row.get("nsi_reference") if isinstance(row.get("nsi_reference"), dict) else {}
            records.append(
                {
                    "row_index": index,
                    "example_id": row.get("example_id"),
                    "episode_id": row.get("episode_id"),
                    "source_trace": row.get("source_trace"),
                    "command_slot_label": command_slot_label,
                    "command_slot_prediction": command_slot_prediction,
                    "command_slot_correct": command_slot_prediction == command_slot_label,
                    "source_overlap_prediction": source_overlap_prediction,
                    "source_overlap_correct": source_overlap_prediction == command_slot_label,
                    "candidate_commands": candidates,
                    "command_identity_scores": {
                        f"slot:{slot}": float(
                            nsi_reference.get(f"command_identity_slot:{slot}", 0.0) or 0.0
                        )
                        for slot in range(MAX_CANDIDATE_SLOTS)
                    },
                    "command_identity_margin": float(
                        nsi_reference.get("command_identity_margin", 0.0) or 0.0
                    ),
                    "command_identity_confidence": float(
                        nsi_reference.get("command_identity_confidence", 0.0) or 0.0
                    ),
                }
            )
    return records


def train_native_head_adapter(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    output_dir: str | Path,
    config: NativeHeadTrainConfig,
    run_root: str | Path | None = None,
) -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModel, AutoTokenizer

    if config.pairwise_command_policy not in PAIRWISE_COMMAND_POLICIES:
        raise ValueError(f"unsupported pairwise_command_policy={config.pairwise_command_policy!r}")
    if config.pairwise_command_max_length is not None and config.pairwise_command_max_length <= 0:
        raise ValueError("pairwise_command_max_length must be positive when provided")
    if config.pairwise_command_top_k is not None and config.pairwise_command_top_k <= 0:
        raise ValueError("pairwise_command_top_k must be positive when provided")
    if config.command_identity_logit_bias < 0:
        raise ValueError("command_identity_logit_bias must be non-negative")
    if config.command_candidate_encoder not in COMMAND_CANDIDATE_ENCODERS:
        raise ValueError(f"unsupported command_candidate_encoder={config.command_candidate_encoder!r}")
    if config.checkpoint_interval_steps < 0:
        raise ValueError("checkpoint_interval_steps must be non-negative")

    _assert_requested_device_available(config.device)
    torch.manual_seed(config.seed)
    output_dir = Path(output_dir)
    run = create_experiment_run(
        kind="phase2c_native_head_training",
        name=config.adapter_name,
        config={
            "train_jsonl": str(Path(train_jsonl).resolve()),
            "val_jsonl": str(Path(val_jsonl).resolve()),
            "output_dir": str(output_dir.resolve()),
            "config": asdict(config),
        },
        run_root=run_root,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        config.base_model_name,
        device_map="auto" if config.device == "cuda" else None,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        quantization_config=_build_quantization_config(config),
    )
    backbone.config.use_cache = False
    try:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        backbone.gradient_checkpointing_enable()
    backbone = prepare_model_for_kbit_training(backbone)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
        target_modules=list(config.target_modules),
    )
    backbone = get_peft_model(backbone, lora_config)
    head_config = NativeCortexHeadConfig(
        backbone_hidden_dim=int(backbone.config.hidden_size),
        nsi_latent_dim=len(NSI_LATENT_FIELDS),
        head_hidden_dim=config.head_hidden_dim,
        dropout=config.head_dropout,
        command_candidate_feature_dim=CANDIDATE_FEATURE_DIM,
        command_candidate_encoder=config.command_candidate_encoder,
        latent_fusion=config.latent_fusion,
        use_pairwise_command_reranker=config.use_pairwise_command_reranker,
        pairwise_command_fusion=config.pairwise_command_fusion,
        pairwise_command_policy=config.pairwise_command_policy,
        pairwise_command_max_length=config.pairwise_command_max_length or config.max_length,
        pairwise_command_top_k=config.pairwise_command_top_k,
        command_identity_logit_bias=config.command_identity_logit_bias,
        open_repair_heads_enabled=config.open_repair_heads_enabled,
    )
    model = QwenBackboneHeadAdapter(backbone, head_config=head_config)
    device = next(model.backbone.parameters()).device
    model.heads.to(device)
    model.train()

    train_dataset = Phase2CHeadJsonlDataset(
        train_jsonl,
        limit=config.max_train_records,
        debug_command_oversample=config.debug_command_oversample,
        balance_command_slots=config.balance_command_slots,
        balance_debug_command_intents=config.balance_debug_command_intents,
        balance_patch_descriptor_labels=config.balance_patch_descriptor_labels,
    )
    val_dataset = Phase2CHeadJsonlDataset(val_jsonl, limit=config.max_val_records)
    train_rows_hash = _canonical_rows_sha256(train_dataset.rows)
    val_rows_hash = _canonical_rows_sha256(val_dataset.rows)
    training_identity_hash = _native_head_training_identity_hash(
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        output_dir=output_dir,
        config=config,
        train_rows_hash=train_rows_hash,
        val_rows_hash=val_rows_hash,
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=lambda rows: _collate_head_rows(
            tokenizer,
            rows,
            max_length=config.max_length,
            use_pairwise_command_reranker=config.use_pairwise_command_reranker,
            pairwise_command_policy=config.pairwise_command_policy,
            pairwise_command_max_length=config.pairwise_command_max_length,
            pairwise_command_top_k=config.pairwise_command_top_k,
            command_candidate_encoder=config.command_candidate_encoder,
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        collate_fn=lambda rows: _collate_head_rows(
            tokenizer,
            rows,
            max_length=config.max_length,
            use_pairwise_command_reranker=config.use_pairwise_command_reranker,
            pairwise_command_policy=config.pairwise_command_policy,
            pairwise_command_max_length=config.pairwise_command_max_length,
            pairwise_command_top_k=config.pairwise_command_top_k,
            command_candidate_encoder=config.command_candidate_encoder,
        ),
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    checkpoint_root = _checkpoint_root_for_run(run, config)
    resumed_checkpoint: dict[str, Any] | None = None
    if config.resume_from_checkpoint:
        resumed_checkpoint = _load_native_head_training_checkpoint(
            checkpoint_dir=config.resume_from_checkpoint,
            model=model,
            optimizer=optimizer,
            expected_identity_hash=training_identity_hash,
            device=device,
        )
    history: list[dict[str, Any]] = []
    resumed_epoch = int(resumed_checkpoint.get("epoch", 0)) if resumed_checkpoint else 0
    for epoch in range(config.epochs):
        if resumed_checkpoint and epoch + 1 < resumed_epoch:
            continue
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        resume_step = 0
        resume_stats: dict[str, Any] = {}
        if resumed_checkpoint and int(resumed_checkpoint.get("epoch", 0)) == epoch + 1:
            resume_step = int(resumed_checkpoint.get("step", 0))
            resume_stats = dict(resumed_checkpoint.get("stats", {}) or {})
        train_loss_total = float(resume_stats.get("train_loss_total", 0.0) or 0.0)
        train_batches = int(resume_stats.get("train_batches", 0) or 0)
        train_pairwise_encoded_candidates = int(
            resume_stats.get("train_pairwise_encoded_candidates", 0) or 0
        )
        component_totals: dict[str, float] = {
            str(name): float(value)
            for name, value in dict(resume_stats.get("component_totals", {}) or {}).items()
        }
        first_train_loss_raw = resume_stats.get("first_train_loss")
        first_train_loss: float | None = (
            float(first_train_loss_raw) if first_train_loss_raw is not None else None
        )
        last_checkpoint_step = resume_step
        for step, batch in enumerate(train_loader, start=1):
            if step <= resume_step:
                continue
            batch = _move_batch_to_device(batch, device)
            if "command_pair_mask" in batch:
                train_pairwise_encoded_candidates += int(batch["command_pair_mask"].sum().item())
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                nsi_latent=batch["nsi_latent"],
                command_input_ids=batch.get("command_input_ids"),
                command_attention_mask=batch.get("command_attention_mask"),
                command_candidate_mask=batch.get("command_candidate_mask"),
                command_candidate_features=batch.get("command_candidate_features"),
                command_candidate_intents=batch.get("command_candidate_intents"),
                command_pair_input_ids=batch.get("command_pair_input_ids"),
                command_pair_attention_mask=batch.get("command_pair_attention_mask"),
                command_pair_mask=batch.get("command_pair_mask"),
                file_input_ids=batch.get("file_input_ids"),
                file_attention_mask=batch.get("file_attention_mask"),
                file_candidate_mask=batch.get("file_candidate_mask"),
            )
            loss, components = compute_native_head_loss(
                outputs,
                batch,
                loss_weights=config.loss_weights,
            )
            scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()
            did_optimizer_step = False
            if step % config.gradient_accumulation_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                did_optimizer_step = True
            if first_train_loss is None:
                first_train_loss = float(loss.item())
            train_loss_total += float(loss.item())
            train_batches += 1
            for name, value in components.items():
                component_totals[name] = component_totals.get(name, 0.0) + value
            if (
                step == 1
                or step == len(train_loader)
                or step % max(config.progress_log_interval_steps, 1) == 0
            ):
                elapsed_seconds = time.perf_counter() - epoch_started
                progress_metrics = {
                    "running_train_loss": train_loss_total / max(train_batches, 1),
                    "last_batch_loss": float(loss.item()),
                    "micro_batch_size": config.micro_batch_size,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "elapsed_seconds": elapsed_seconds,
                    "steps_per_second": step / elapsed_seconds if elapsed_seconds > 0.0 else 0.0,
                    "pairwise_encoded_candidates_seen": train_pairwise_encoded_candidates,
                }
                _write_progress_snapshot(
                    run,
                    stage="train",
                    epoch=epoch + 1,
                    step=step,
                    total_steps=len(train_loader),
                    metrics=progress_metrics,
                )
                print(
                    json.dumps(
                        {
                            "event": "native_head_train_progress",
                            "epoch": epoch + 1,
                            "step": step,
                            "total_steps": len(train_loader),
                            **progress_metrics,
                        }
                    ),
                    flush=True,
                )
            if (
                config.checkpoint_interval_steps > 0
                and did_optimizer_step
                and (
                    step == len(train_loader)
                    or step - last_checkpoint_step >= config.checkpoint_interval_steps
                )
            ):
                checkpoint_stats = {
                    "train_loss_total": train_loss_total,
                    "train_batches": train_batches,
                    "train_pairwise_encoded_candidates": train_pairwise_encoded_candidates,
                    "component_totals": component_totals,
                    "first_train_loss": first_train_loss,
                }
                checkpoint_path = _save_native_head_training_checkpoint(
                    checkpoint_root=checkpoint_root,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    step=step,
                    total_steps=len(train_loader),
                    identity_hash=training_identity_hash,
                    stats=checkpoint_stats,
                )
                last_checkpoint_step = step
                print(
                    json.dumps(
                        {
                            "event": "native_head_train_checkpoint",
                            "epoch": epoch + 1,
                            "step": step,
                            "total_steps": len(train_loader),
                            "checkpoint_path": str(checkpoint_path),
                        }
                    ),
                    flush=True,
                )
        train_components = {
            name: value / max(train_batches, 1)
            for name, value in sorted(component_totals.items())
        }
        train_elapsed_seconds = time.perf_counter() - epoch_started
        val_metrics = _evaluate_native_head_model(
            model,
            val_loader,
            device=device,
            loss_weights=config.loss_weights,
        )
        epoch_summary = {
            "epoch": epoch + 1,
            "first_train_loss": float(first_train_loss or 0.0),
            "train_loss": train_loss_total / max(train_batches, 1),
            "train_components": train_components,
            "train_elapsed_seconds": train_elapsed_seconds,
            "train_steps_per_second": train_batches / train_elapsed_seconds if train_elapsed_seconds > 0.0 else 0.0,
            "train_pairwise_encoded_candidates": train_pairwise_encoded_candidates,
            "val_metrics": val_metrics,
        }
        history.append(epoch_summary)
        _write_progress_snapshot(
            run,
            stage="validation",
            epoch=epoch + 1,
            step=len(val_loader),
            total_steps=len(val_loader),
            metrics=val_metrics,
        )
        print(json.dumps({"event": "native_head_epoch_complete", **epoch_summary}), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(output_dir / "backbone_adapter")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    torch.save(model.heads.state_dict(), output_dir / "native_heads.pt")
    (output_dir / "head_config.json").write_text(
        json.dumps(asdict(head_config), indent=2),
        encoding="utf-8",
    )
    summary = {
        "base_model_name": config.base_model_name,
        "adapter_name": config.adapter_name,
        "adapter_output_dir": str(output_dir.resolve()),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "config_hash": run.config_hash,
        "use_pairwise_command_reranker": config.use_pairwise_command_reranker,
        "pairwise_command_fusion": config.pairwise_command_fusion,
        "pairwise_command_policy": config.pairwise_command_policy,
        "pairwise_command_max_length": config.pairwise_command_max_length or config.max_length,
        "pairwise_command_top_k": config.pairwise_command_top_k,
        "command_identity_logit_bias": config.command_identity_logit_bias,
        "command_candidate_encoder": config.command_candidate_encoder,
        "command_candidate_feature_dim": head_config.command_candidate_feature_dim,
        "open_repair_heads_enabled": config.open_repair_heads_enabled,
        "open_repair_capabilities": {
            name: bool(config.open_repair_heads_enabled)
            for name in OPEN_REPAIR_CAPABILITY_NAMES
        },
        "learned_patch_descriptor_heads": {
            "enabled": bool(config.open_repair_heads_enabled),
            "patch_operation_order": list(PATCH_OPERATION_ORDER),
            "patch_template_order": list(PATCH_TEMPLATE_ORDER),
            "target_file_slot_classes": MAX_CANDIDATE_SLOTS,
            "template_slot_classes": MAX_CANDIDATE_SLOTS,
        },
        "open_repair_training_contract": _open_repair_training_contract(config),
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "nsi_latent_field_gains": {
            "receptor_failure_signal": RECEPTOR_FAILURE_SIGNAL_GAIN,
            "debug_action_stage": DEBUG_ACTION_STAGE_GAIN,
            "descriptor_failure_family": DESCRIPTOR_FAILURE_FAMILY_GAIN,
        },
        "effective_split_hashes": {
            "phase2c_head_train": train_rows_hash,
            "phase2c_head_val": val_rows_hash,
        },
        "slot_intent_distribution": {
            "train": _slot_intent_distribution(train_dataset.rows),
            "val": _slot_intent_distribution(val_dataset.rows),
        },
        "patch_descriptor_distribution": {
            "train": _patch_descriptor_distribution(train_dataset.rows),
            "val": _patch_descriptor_distribution(val_dataset.rows),
        },
        "loss_weights": dict(sorted(config.loss_weights.items())),
        "source_overlap_command_slot_baseline": {
            "train": _command_slot_baseline_metrics(train_dataset.rows),
            "val": _command_slot_baseline_metrics(val_dataset.rows),
        },
        "pairwise_candidate_encoding": {
            "train": _pairwise_candidate_encoding_stats(
                train_dataset.rows,
                enabled=config.use_pairwise_command_reranker,
                policy=config.pairwise_command_policy,
                top_k=config.pairwise_command_top_k,
            ),
            "val": _pairwise_candidate_encoding_stats(
                val_dataset.rows,
                enabled=config.use_pairwise_command_reranker,
                policy=config.pairwise_command_policy,
                top_k=config.pairwise_command_top_k,
            ),
        },
        "history": history,
        "config": asdict(config),
        "head_config": asdict(head_config),
        "json_text_target": False,
        "checkpointing": {
            "training_identity_hash": training_identity_hash,
            "checkpoint_interval_steps": config.checkpoint_interval_steps,
            "checkpoint_dir": str(checkpoint_root.resolve())
            if config.checkpoint_interval_steps > 0 or config.resume_from_checkpoint
            else None,
            "resume_from_checkpoint": config.resume_from_checkpoint,
            "resumed": resumed_checkpoint is not None,
            "resumed_checkpoint": resumed_checkpoint,
        },
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }
    summary["run_manifest"] = run.finalize(summary)
    run.write_json("training_summary.json", summary)
    return summary


def evaluate_native_head_adapter(
    *,
    eval_jsonl: str | Path,
    adapter_dir: str | Path,
    base_model_name: str,
    quantization: str = "4bit",
    device: str = "cuda",
    max_length: int | None = None,
    batch_size: int = 1,
    eval_split: str = "eval",
    max_eval_records: int | None = None,
    loss_weights: dict[str, float] | None = None,
    include_prediction_records: bool = False,
) -> dict[str, Any]:
    """Evaluate a saved native-head adapter on a head JSONL split.

    This is intentionally inference-only. It reloads the same head config saved
    with the adapter so evaluation cannot silently drift from training/runtime
    semantics.
    """
    _assert_requested_device_available(device)

    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    adapter_path = Path(adapter_dir)
    head_config_path = adapter_path / "head_config.json"
    native_heads_path = adapter_path / "native_heads.pt"
    backbone_adapter_path = adapter_path / "backbone_adapter"
    if not head_config_path.exists() or not native_heads_path.exists() or not backbone_adapter_path.exists():
        raise FileNotFoundError(
            "adapter_dir must contain backbone_adapter, native_heads.pt, and head_config.json: "
            f"{adapter_path}"
        )

    loader_config = NativeHeadTrainConfig(
        base_model_name=base_model_name,
        adapter_name=adapter_path.name,
        quantization=quantization,
        device=device,
    )
    tokenizer_path = adapter_path / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path) if tokenizer_path.exists() else base_model_name
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        base_model_name,
        device_map="auto" if device == "cuda" else None,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        quantization_config=_build_quantization_config(loader_config),
    )
    backbone = PeftModel.from_pretrained(backbone, backbone_adapter_path)
    head_config_payload = json.loads(head_config_path.read_text(encoding="utf-8"))
    head_config = NativeCortexHeadConfig(**head_config_payload)
    model = QwenBackboneHeadAdapter(backbone, head_config=head_config)
    try:
        head_state = torch.load(native_heads_path, map_location="cpu", weights_only=True)
    except TypeError:
        head_state = torch.load(native_heads_path, map_location="cpu")
    model.heads.load_state_dict(head_state, strict=False)
    eval_device = next(model.backbone.parameters()).device
    model.heads.to(eval_device)
    model.eval()

    effective_max_length = int(max_length or head_config.pairwise_command_max_length or 512)
    dataset = Phase2CHeadJsonlDataset(eval_jsonl, limit=max_eval_records)
    rows_hash = _canonical_rows_sha256(dataset.rows)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: _collate_head_rows(
            tokenizer,
            rows,
            max_length=effective_max_length,
            use_pairwise_command_reranker=head_config.use_pairwise_command_reranker,
            pairwise_command_policy=head_config.pairwise_command_policy,
            pairwise_command_max_length=head_config.pairwise_command_max_length,
            pairwise_command_top_k=head_config.pairwise_command_top_k,
            command_candidate_encoder=head_config.command_candidate_encoder,
            command_candidate_feature_dim=head_config.command_candidate_feature_dim,
        ),
    )
    metrics = _evaluate_native_head_model(
        model,
        loader,
        device=eval_device,
        loss_weights=loss_weights or NativeHeadTrainConfig(
            base_model_name=base_model_name,
            adapter_name=adapter_path.name,
        ).loss_weights,
    )
    pairwise_stats = _pairwise_candidate_encoding_stats(
        dataset.rows,
        enabled=head_config.use_pairwise_command_reranker,
        policy=head_config.pairwise_command_policy,
        top_k=head_config.pairwise_command_top_k,
    )
    report = {
        "artifact_family": "phase2c_native_head_adapter_eval",
        "base_model_name": base_model_name,
        "adapter_output_dir": str(adapter_path.resolve()),
        "eval_jsonl": str(Path(eval_jsonl).resolve()),
        "eval_split": eval_split,
        "eval_examples": len(dataset),
        "eval_rows_hash": rows_hash,
        "effective_split_hashes": {
            f"phase2c_head_{eval_split}": rows_hash,
        },
        "use_pairwise_command_reranker": head_config.use_pairwise_command_reranker,
        "pairwise_command_fusion": head_config.pairwise_command_fusion,
        "pairwise_command_policy": head_config.pairwise_command_policy,
        "pairwise_command_max_length": head_config.pairwise_command_max_length,
        "pairwise_command_top_k": head_config.pairwise_command_top_k,
        "command_candidate_encoder": head_config.command_candidate_encoder,
        "command_candidate_feature_dim": head_config.command_candidate_feature_dim,
        "command_identity_logit_bias": head_config.command_identity_logit_bias,
        "open_repair_heads_enabled": head_config.open_repair_heads_enabled,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "device": str(eval_device),
        "config": {
            "device": device,
            "quantization": quantization,
            "max_length": effective_max_length,
            "batch_size": batch_size,
            "max_eval_records": max_eval_records,
        },
        "slot_intent_distribution": {
            eval_split: _slot_intent_distribution(dataset.rows),
        },
        "source_overlap_command_slot_baseline": {
            eval_split: _command_slot_baseline_metrics(dataset.rows),
        },
        "pairwise_candidate_encoding": {
            eval_split: pairwise_stats,
        },
        "eval_metrics": metrics,
        "history": [
            {
                "eval_split": eval_split,
                "val_metrics": metrics,
            }
        ],
        "head_config": asdict(head_config),
        "claim_boundary": (
            "Adapter evaluation measures one supplied non-sealed or holdout split only; "
            "it does not establish sealed transfer, production autonomy, or open-ended "
            "debugging generalization."
        ),
    }
    if include_prediction_records:
        report["prediction_records"] = _collect_native_head_prediction_records(
            model,
            tokenizer,
            dataset.rows,
            device=eval_device,
            max_length=effective_max_length,
            head_config=head_config,
        )
    return report
