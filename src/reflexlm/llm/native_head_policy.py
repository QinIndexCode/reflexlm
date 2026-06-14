from __future__ import annotations

import json
import inspect
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from reflexlm.eval import PolicyStats, SequenceModelPolicy
from reflexlm.llm.candidate_features import (
    build_candidate_pair_prompt,
    command_candidate_feature_rows,
    command_candidate_intent_indices,
    command_intent_for_text,
    guard_command_identity_reference,
    normalize_command_candidate_feature_ablation_groups,
    pairwise_command_candidate_mask,
    redact_structured_command_identity_text,
    zero_command_candidate_feature_groups,
)
from reflexlm.llm.head_dataset import build_phase2c_head_state_prompt_from_state
from reflexlm.llm.native_cortex import (
    OPEN_REPAIR_CAPABILITY_NAMES,
    PATCH_OPERATION_ORDER,
    PATCH_TEMPLATE_ORDER,
    NativeCortexHeadConfig,
    QwenBackboneHeadAdapter,
)
from reflexlm.llm.native_head_training import (
    BASE_NSI_LATENT_FIELDS,
    _build_quantization_config,
    NativeHeadTrainConfig,
    nsi_latent_values,
)
from reflexlm.llm.receptor_latent import (
    debug_action_stage_signal,
    descriptor_failure_family_signal,
    receptor_failure_signal,
    runtime_command_identity_signal,
)
from reflexlm.models.features import (
    ACTION_ORDER,
    MAX_CANDIDATE_SLOTS,
    ROUTE_ORDER,
    candidate_commands,
    candidate_files,
    valid_action_mask,
)
from reflexlm.runtime.nervous_system import (
    INTERNAL_TARGET_ORDER,
    SynapticMotorPlan,
    internal_target_for_state,
    serialize_motor_action,
)
from reflexlm.runtime.plasticity import SynapticPlasticityMemory
from reflexlm.schema import ActionDecision, ActionType, InternalTarget, RouteName, SystemStateFrame
from reflexlm.train import load_model_checkpoint


MODEL_LOAD_STRATEGIES = ("auto", "single_device")


def _resolve_prediction_error_escalation_threshold(
    configured_threshold: float | None,
    nsi_payload: dict[str, Any],
) -> tuple[float, str]:
    training_summary = nsi_payload.get("training_summary")
    calibration = (
        training_summary.get("prediction_error_calibration", {})
        if isinstance(training_summary, dict)
        else {}
    )
    calibrated_threshold = calibration.get("threshold") if isinstance(calibration, dict) else None
    resolved = configured_threshold if configured_threshold is not None else calibrated_threshold
    if resolved is None:
        resolved = 0.45
        source = "legacy_default"
    elif configured_threshold is not None:
        source = "configured_override"
    else:
        source = "checkpoint_calibration"
    if not 0.0 <= float(resolved) <= 1.0:
        raise ValueError("prediction_error_escalation_threshold must be between 0 and 1")
    return float(resolved), source


def _model_load_kwargs(
    *,
    device: str,
    model_load_strategy: str,
    offload_state_dict: bool,
    offload_folder: str | Path | None = None,
) -> dict[str, Any]:
    if model_load_strategy not in MODEL_LOAD_STRATEGIES:
        raise ValueError(
            "model_load_strategy must be one of: "
            + ", ".join(MODEL_LOAD_STRATEGIES)
        )
    if model_load_strategy == "single_device" and device != "cuda":
        raise ValueError("single_device model loading requires device='cuda'")
    device_map: str | dict[str, int] | None
    if device != "cuda":
        device_map = None
    elif model_load_strategy == "single_device":
        device_map = {"": 0}
    else:
        device_map = "auto"
    payload: dict[str, Any] = {
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "offload_state_dict": bool(offload_state_dict),
    }
    if offload_state_dict and offload_folder is not None:
        payload["offload_folder"] = str(offload_folder)
    return payload


def _hardlink_or_copy_file(source: Path, target: Path) -> None:
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _runtime_compatible_peft_adapter_path(adapter_path: Path, config_cls: type) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Return an adapter path whose config is accepted by the installed PEFT.

    PEFT adapter configs can include forward-version fields. For reproducible
    runtime evaluation, keep original artifacts immutable and materialize a
    temporary view with unsupported config keys removed only when needed.
    """
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return adapter_path, None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    signature = inspect.signature(config_cls.__init__)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}
    }
    unknown = sorted(key for key in payload if key not in accepted)
    if not unknown:
        return adapter_path, None

    temp_dir = tempfile.TemporaryDirectory(prefix="reflexlm-peft-compat-")
    compat_path = Path(temp_dir.name)
    sanitized = {key: value for key, value in payload.items() if key in accepted}
    (compat_path / "adapter_config.json").write_text(
        json.dumps(sanitized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for child in adapter_path.iterdir():
        if child.name == "adapter_config.json":
            continue
        target = compat_path / child.name
        if child.is_file():
            _hardlink_or_copy_file(child, target)
        elif child.is_dir():
            shutil.copytree(child, target)
    return compat_path, temp_dir


class NativeHeadPolicy:
    """Phase 2C runtime policy.

    Low-level reflex/inhibition/routing remains on the NSI reference path. The
    Qwen native-head path is invoked only for internal cortex escalation and
    never emits JSON text as the motor channel.
    """

    def __init__(
        self,
        *,
        base_model_name: str,
        native_head_path: str | Path,
        nsi_checkpoint_path: str | Path,
        quantization: str = "4bit",
        nsi_device: str = "cpu",
        device: str = "cuda",
        cpu_offload: bool = False,
        model_load_strategy: str = "auto",
        offload_state_dict: bool = False,
        max_length: int = 512,
        policy_label: str = "phase2c_native_heads",
        zero_nsi_latent: bool = False,
        enable_debug_continuation: bool = True,
        continuation_control: str = "normal",
        enable_native_head_calls: bool = True,
        prediction_error_escalation_threshold: float | None = None,
        disabled_command_candidate_feature_groups: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        from peft import LoraConfig, PeftModel
        from transformers import AutoModel, AutoTokenizer

        self.policy_label = policy_label
        self.native_head_path = Path(native_head_path)
        self._peft_compat_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._model_offload_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.stats = PolicyStats()
        self.last_call: dict[str, Any] = {}
        self._debug_continuation: dict[str, Any] | None = None
        self._last_cache_reset_reason: str | None = None
        self.plasticity_memory = SynapticPlasticityMemory()
        self._last_plasticity_state: SystemStateFrame | None = None
        self._last_plasticity_command: str | None = None
        self._last_plasticity_prediction: dict[str, Any] = {}
        self.max_length = max_length
        self.zero_nsi_latent = zero_nsi_latent
        if continuation_control not in {"normal", "cache_erased", "wrong_cache"}:
            raise ValueError(
                "continuation_control must be one of: normal, cache_erased, wrong_cache"
            )
        self.continuation_control = continuation_control
        self.enable_debug_continuation = (
            enable_debug_continuation and continuation_control != "cache_erased"
        )
        self.enable_native_head_calls = enable_native_head_calls
        self.disabled_command_candidate_feature_groups = (
            normalize_command_candidate_feature_ablation_groups(
                disabled_command_candidate_feature_groups
            )
        )
        nsi_model, nsi_vectorizer, nsi_payload = load_model_checkpoint(
            nsi_checkpoint_path,
            device=nsi_device,
        )
        (
            self.prediction_error_escalation_threshold,
            self.prediction_error_escalation_threshold_source,
        ) = _resolve_prediction_error_escalation_threshold(
            prediction_error_escalation_threshold,
            nsi_payload,
        )
        self.nsi_policy = SequenceModelPolicy(
            nsi_model,
            nsi_vectorizer,
            policy_label="phase2c_low_level_nsi_reference",
            use_legal_action_mask=True,
            training_summary=nsi_payload.get("training_summary", {}),
        )
        tokenizer_path = self.native_head_path / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path) if tokenizer_path.exists() else base_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        loader_config = NativeHeadTrainConfig(
            base_model_name=base_model_name,
            adapter_name=policy_label,
            quantization=quantization,
            device=device,
        )
        if offload_state_dict:
            self._model_offload_tmpdir = tempfile.TemporaryDirectory(
                prefix="reflexlm-model-load-offload-"
            )
        model_load_kwargs = _model_load_kwargs(
            device=device,
            model_load_strategy=model_load_strategy,
            offload_state_dict=offload_state_dict,
            offload_folder=(
                self._model_offload_tmpdir.name
                if self._model_offload_tmpdir is not None
                else None
            ),
        )
        backbone = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype="auto",
            quantization_config=_build_quantization_config(loader_config),
            **model_load_kwargs,
        )
        adapter_path, peft_tmpdir = _runtime_compatible_peft_adapter_path(
            self.native_head_path / "backbone_adapter",
            LoraConfig,
        )
        self._peft_compat_tmpdir = peft_tmpdir
        backbone = PeftModel.from_pretrained(backbone, adapter_path)
        head_config_payload = json.loads((self.native_head_path / "head_config.json").read_text(encoding="utf-8"))
        head_config = NativeCortexHeadConfig(**head_config_payload)
        self.nsi_latent_dim = int(head_config.nsi_latent_dim)
        self.use_pairwise_command_reranker = bool(head_config.use_pairwise_command_reranker)
        self.pairwise_command_fusion = head_config.pairwise_command_fusion
        self.pairwise_command_policy = head_config.pairwise_command_policy
        self.pairwise_command_max_length = int(head_config.pairwise_command_max_length or self.max_length)
        self.pairwise_command_top_k = head_config.pairwise_command_top_k
        self.command_candidate_encoder = head_config.command_candidate_encoder
        self.command_candidate_feature_dim = int(head_config.command_candidate_feature_dim)
        self.open_repair_heads_enabled = bool(head_config.open_repair_heads_enabled)
        self.open_repair_capabilities = {
            name: self.open_repair_heads_enabled for name in OPEN_REPAIR_CAPABILITY_NAMES
        }
        self.model = QwenBackboneHeadAdapter(backbone, head_config=head_config)
        try:
            head_state = torch.load(
                self.native_head_path / "native_heads.pt",
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            head_state = torch.load(self.native_head_path / "native_heads.pt", map_location="cpu")
        self.model.heads.load_state_dict(head_state, strict=False)
        self.device = next(self.model.backbone.parameters()).device
        self.model.heads.to(self.device)
        self.model.eval()
        self.cpu_offload = cpu_offload
        self.model_load_strategy = model_load_strategy
        self.offload_state_dict = bool(offload_state_dict)

    def reset(self) -> None:
        self.stats = PolicyStats()
        self.last_call = {}
        self._debug_continuation = None
        self._last_cache_reset_reason = None
        self._last_plasticity_state = None
        self._last_plasticity_command = None
        self._last_plasticity_prediction = {}
        self.nsi_policy.reset()

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "phase2c_native_heads",
            "policy_label": self.policy_label,
            "native_head_path": str(self.native_head_path),
            "json_text_target": False,
            "zero_nsi_latent": self.zero_nsi_latent,
            "debug_continuation_cache": self.enable_debug_continuation,
            "continuation_cache_enabled": self.enable_debug_continuation,
            "continuation_control": self.continuation_control,
            "native_head_calls_enabled": self.enable_native_head_calls,
            "prediction_error_escalation_threshold": self.prediction_error_escalation_threshold,
            "prediction_error_escalation_threshold_source": (
                self.prediction_error_escalation_threshold_source
            ),
            "model_load_strategy": self.model_load_strategy,
            "offload_state_dict": self.offload_state_dict,
            "use_pairwise_command_reranker": self.use_pairwise_command_reranker,
            "pairwise_command_fusion": self.pairwise_command_fusion,
            "pairwise_command_policy": self.pairwise_command_policy,
            "pairwise_command_max_length": self.pairwise_command_max_length,
            "pairwise_command_top_k": self.pairwise_command_top_k,
            "command_candidate_encoder": self.command_candidate_encoder,
            "command_candidate_feature_dim": self.command_candidate_feature_dim,
            "disabled_command_candidate_feature_groups": list(
                self.disabled_command_candidate_feature_groups
            ),
            "open_repair_heads_enabled": self.open_repair_heads_enabled,
            "open_repair_capabilities": dict(self.open_repair_capabilities),
            "plasticity_memory_schema": "reflexlm.synaptic_plasticity.v1",
            "plasticity_feedback_events": self.plasticity_memory.feedback_events,
        }

    def load_plasticity_memory(self, path: str | Path) -> None:
        self.plasticity_memory = SynapticPlasticityMemory.load(path)

    def save_plasticity_memory(self, path: str | Path) -> None:
        self.plasticity_memory.save(path)

    def clear_plasticity_memory(self) -> None:
        self._plasticity_memory().clear()

    def set_plasticity_control(self, control: str) -> None:
        self._plasticity_memory().control = control

    def _plasticity_memory(self) -> SynapticPlasticityMemory:
        memory = getattr(self, "plasticity_memory", None)
        if not isinstance(memory, SynapticPlasticityMemory):
            memory = SynapticPlasticityMemory()
            self.plasticity_memory = memory
        return memory

    def observe_feedback(
        self,
        *,
        verified_success: bool,
        verifier: str = "post_execution_verifier",
    ) -> dict[str, Any]:
        if self._last_plasticity_state is None or not self._last_plasticity_command:
            return {
                "accepted": False,
                "reason": "no_prior_bounded_command_decision",
                "verifier": verifier,
            }
        feedback = self._plasticity_memory().observe_feedback(
            self._last_plasticity_state,
            command=self._last_plasticity_command,
            verified_success=verified_success,
            verifier=verifier,
        )
        return feedback

    def _command_identity_reference(self, state: SystemStateFrame) -> dict[str, Any]:
        receptor_reference = runtime_command_identity_signal(state)
        receptor_confidence = float(
            receptor_reference.get("command_identity_confidence", 0.0) or 0.0
        )
        plasticity_prediction = self._plasticity_memory().predict(state)
        self._last_plasticity_prediction = plasticity_prediction
        if receptor_confidence > 0.0 or not plasticity_prediction.get("memory_hit"):
            return receptor_reference
        return self._plasticity_memory().reference(state)

    def _open_repair_runtime_outputs(self, outputs: dict[str, torch.Tensor]) -> dict[str, Any]:
        if not self.open_repair_heads_enabled:
            return {
                "open_repair_heads_enabled": False,
                "open_repair_capabilities": dict(self.open_repair_capabilities),
            }
        def argmax_or_none(key: str) -> int | None:
            logits = outputs.get(key)
            if logits is None:
                return None
            return int(torch.argmax(logits, dim=-1).detach().cpu().item())

        patch_operation_index = argmax_or_none("patch_operation_logits")
        patch_template_slot = argmax_or_none("patch_template_slot_logits")
        return {
            "open_repair_heads_enabled": True,
            "open_repair_capabilities": dict(self.open_repair_capabilities),
            "open_repair_head_outputs": {
                "patch_proposal": argmax_or_none("patch_proposal_logits"),
                "test_selection_slot": argmax_or_none("test_selection_logits"),
                "rollback_safety": argmax_or_none("rollback_safety_logits"),
                "stop_condition": argmax_or_none("stop_condition_logits"),
                "bounded_edit_scope": argmax_or_none("bounded_edit_scope_logits"),
                "progress_monitor": argmax_or_none("progress_monitor_logits"),
                "verification_state": argmax_or_none("verification_state_logits"),
            },
            "learned_patch_descriptor_outputs": {
                "patch_operation_index": patch_operation_index,
                "patch_operation": (
                    PATCH_OPERATION_ORDER[patch_operation_index]
                    if patch_operation_index is not None
                    and 0 <= patch_operation_index < len(PATCH_OPERATION_ORDER)
                    else None
                ),
                "patch_target_file_slot": argmax_or_none("patch_target_file_slot_logits"),
                "patch_template_slot": patch_template_slot,
                "patch_template": (
                    PATCH_TEMPLATE_ORDER[patch_template_slot]
                    if patch_template_slot is not None
                    and 0 <= patch_template_slot < len(PATCH_TEMPLATE_ORDER)
                    else None
                ),
            },
            "open_repair_execution_policy": "heads_expose_control_state_only_no_patch_execution",
        }

    def _nsi_latent(
        self,
        state: SystemStateFrame,
        nsi_reference_override: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if self.zero_nsi_latent:
            return torch.zeros((1, self.nsi_latent_dim), dtype=torch.float32, device=self.device)
        debug = self.nsi_policy.last_call
        reference = {
            field_name: float(debug.get(field_name, 0.0))
            for field_name in BASE_NSI_LATENT_FIELDS
        }
        try:
            action_index = int(debug.get("action_index", -1))
        except (TypeError, ValueError):
            action_index = -1
        if 0 <= action_index < len(ACTION_ORDER):
            reference["reflex_action"] = ACTION_ORDER[action_index].value
        reference["route_name"] = str(debug.get("route_name") or "")
        reference["receptor_failure_signal"] = receptor_failure_signal(state)
        reference["debug_action_stage"] = debug_action_stage_signal(state)
        reference["descriptor_failure_family"] = descriptor_failure_family_signal(state)
        reference.update(self._command_identity_reference(state))
        if nsi_reference_override:
            reference.update(dict(nsi_reference_override))
        reference = guard_command_identity_reference(
            build_phase2c_head_state_prompt_from_state(state),
            candidate_commands(state),
            reference,
        )
        values = nsi_latent_values(reference)
        if len(values) < self.nsi_latent_dim:
            values.extend([0.0] * (self.nsi_latent_dim - len(values)))
        elif len(values) > self.nsi_latent_dim:
            values = values[: self.nsi_latent_dim]
        return torch.tensor([values], dtype=torch.float32, device=self.device)

    def _internal_target_with_prediction_error(
        self,
        state: SystemStateFrame,
    ) -> tuple[InternalTarget, str]:
        threshold = float(getattr(self, "prediction_error_escalation_threshold", 0.45))
        target = internal_target_for_state(state)
        if target != InternalTarget.REFLEX_MOTOR:
            return target, "state_receptor"
        debug = self.nsi_policy.last_call
        homeostatic = debug.get("homeostatic_decision")
        if isinstance(homeostatic, dict):
            homeostatic_target = str(homeostatic.get("target") or "")
            if homeostatic_target in {
                InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX.value,
                InternalTarget.INHIBIT.value,
            }:
                return (
                    InternalTarget(homeostatic_target),
                    str(homeostatic.get("reason") or "homeostatic_control"),
                )
        if debug.get("temporal_observation_available") is not True:
            return target, "state_receptor"
        prediction_error = float(debug.get("prediction_error", 0.0))
        if prediction_error > threshold:
            return (
                InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX,
                "observed_temporal_prediction_error",
            )
        return target, "state_receptor"

    def _low_level_action(
        self,
        state: SystemStateFrame,
        *,
        nsi_reference_override: dict[str, Any] | None,
    ) -> ActionDecision:
        if not nsi_reference_override:
            return self.nsi_policy.act(state)
        reference = dict(nsi_reference_override)
        reflex_action = str(reference.get("reflex_action") or "")
        try:
            action_index = ACTION_ORDER.index(ActionType(reflex_action))
        except ValueError:
            action_index = ACTION_ORDER.index(ActionType.WAIT)
        self.nsi_policy.last_call = {
            **reference,
            "action_index": action_index,
            "route_name": str(reference.get("route_name") or ""),
            "nsi_reference_override": True,
            "temporal_observation_available": False,
        }
        return ActionDecision(
            type=ActionType.WAIT,
            confidence=float(reference.get("confidence", 0.0)),
            reason="nsi_reference_override_fallback",
        )

    def _encode_state(self, state: SystemStateFrame) -> dict[str, torch.Tensor]:
        prompt = build_phase2c_head_state_prompt_from_state(state)
        tokenized = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in tokenized.items()}

    def _candidate_identity_text_ablation_enabled(self) -> bool:
        return "candidate_identity" in self.disabled_command_candidate_feature_groups

    def _state_for_native_head_text(self, state: SystemStateFrame) -> SystemStateFrame:
        """Apply text-channel ablations before Qwen native-head encoding.

        Candidate-identity controls must not only zero numeric feature columns:
        the same sidecar can otherwise leak through the backbone prompt via
        ``command_identity_tokens=...`` in command candidates or receptor text.
        The original state is still used for non-ablated NSI latent paths.
        """

        if not self._candidate_identity_text_ablation_enabled():
            return state
        redacted_commands = [
            redact_structured_command_identity_text(command)
            for command in candidate_commands(state)
        ]
        return state.model_copy(
            deep=True,
            update={
                "goal": state.goal.model_copy(
                    update={
                        "description": redact_structured_command_identity_text(
                            state.goal.description
                        ),
                        "command_allowlist": redacted_commands,
                    }
                ),
                "terminal": state.terminal.model_copy(
                    update={
                        "stdout_delta": redact_structured_command_identity_text(
                            state.terminal.stdout_delta
                        ),
                        "stderr_delta": redact_structured_command_identity_text(
                            state.terminal.stderr_delta
                        ),
                    }
                ),
            },
        )

    def _encode_candidate_texts(self, candidates: list[str], *, kind: str) -> dict[str, torch.Tensor]:
        max_length = max(8, min(self.max_length, 64))
        input_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        candidate_mask = torch.zeros(1, MAX_CANDIDATE_SLOTS, dtype=torch.bool, device=self.device)
        pad_id = int(self.tokenizer.pad_token_id or 0)
        max_len = 1
        tokenized_candidates = []
        for index in range(MAX_CANDIDATE_SLOTS):
            text = candidates[index] if index < len(candidates) else ""
            tokenized = self.tokenizer(
                f"{kind} candidate:\n{text}",
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            ids = tokenized["input_ids"][0]
            mask = tokenized.get("attention_mask", torch.ones_like(ids))[0]
            max_len = max(max_len, int(ids.shape[0]))
            tokenized_candidates.append((ids, mask))
            if index < len(candidates):
                candidate_mask[0, index] = True
        for ids, mask in tokenized_candidates:
            pad = max_len - ids.shape[0]
            input_rows.append(torch.nn.functional.pad(ids, (0, pad), value=pad_id))
            mask_rows.append(torch.nn.functional.pad(mask, (0, pad), value=0))
        return {
            "input_ids": torch.stack(input_rows).unsqueeze(0).to(self.device),
            "attention_mask": torch.stack(mask_rows).unsqueeze(0).to(self.device),
            "candidate_mask": candidate_mask,
        }

    def _candidate_mask_tensor(self, candidates: list[str]) -> torch.Tensor:
        mask = torch.zeros(1, MAX_CANDIDATE_SLOTS, dtype=torch.bool, device=self.device)
        for index in range(min(len(candidates), MAX_CANDIDATE_SLOTS)):
            mask[0, index] = True
        return mask

    def _command_candidate_features(
        self,
        state: SystemStateFrame,
        candidates: list[str],
        nsi_reference_override: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        nsi_reference = None if self.zero_nsi_latent else self._command_identity_reference(state)
        if nsi_reference is not None and nsi_reference_override:
            nsi_reference = {**nsi_reference, **dict(nsi_reference_override)}
        features = command_candidate_feature_rows(
            build_phase2c_head_state_prompt_from_state(state),
            candidates,
            nsi_reference=nsi_reference,
        )
        for row in features:
            if len(row) > self.command_candidate_feature_dim:
                del row[self.command_candidate_feature_dim :]
            elif len(row) < self.command_candidate_feature_dim:
                row.extend([0.0] * (self.command_candidate_feature_dim - len(row)))
        zero_command_candidate_feature_groups(
            features,
            self.disabled_command_candidate_feature_groups,
            feature_dim=self.command_candidate_feature_dim,
        )
        return torch.tensor([features], dtype=torch.float32, device=self.device)

    def _command_candidate_intents(self, candidates: list[str]) -> torch.Tensor:
        return torch.tensor(
            [command_candidate_intent_indices(candidates)],
            dtype=torch.long,
            device=self.device,
        )

    def _encode_candidate_pair_texts(
        self,
        state: SystemStateFrame,
        candidates: list[str],
        *,
        kind: str,
    ) -> dict[str, torch.Tensor]:
        state_prompt = build_phase2c_head_state_prompt_from_state(state)
        max_length = max(32, self.pairwise_command_max_length)
        pairwise_mask = pairwise_command_candidate_mask(
            candidates,
            self.pairwise_command_policy,
            visible_state_text=state_prompt,
            top_k=self.pairwise_command_top_k,
        )
        input_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        candidate_mask = torch.zeros(1, MAX_CANDIDATE_SLOTS, dtype=torch.bool, device=self.device)
        pad_id = int(self.tokenizer.pad_token_id or 0)
        max_len = 1
        tokenized_candidates = []
        for index in range(MAX_CANDIDATE_SLOTS):
            if index < len(candidates) and pairwise_mask[index]:
                prompt = build_candidate_pair_prompt(
                    state_prompt,
                    candidates[index],
                    kind=kind,
                )
                tokenized = self.tokenizer(
                    prompt,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
            else:
                tokenized = self.tokenizer(
                    "",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=1,
                    return_tensors="pt",
                )
            ids = tokenized["input_ids"][0]
            mask = tokenized.get("attention_mask", torch.ones_like(ids))[0]
            max_len = max(max_len, int(ids.shape[0]))
            tokenized_candidates.append((ids, mask))
            if pairwise_mask[index]:
                candidate_mask[0, index] = True
        for ids, mask in tokenized_candidates:
            pad = max_len - ids.shape[0]
            input_rows.append(torch.nn.functional.pad(ids, (0, pad), value=pad_id))
            mask_rows.append(torch.nn.functional.pad(mask, (0, pad), value=0))
        return {
            "input_ids": torch.stack(input_rows).unsqueeze(0).to(self.device),
            "attention_mask": torch.stack(mask_rows).unsqueeze(0).to(self.device),
            "candidate_mask": candidate_mask,
        }

    def _cortex_action_from_heads(
        self,
        outputs: dict[str, torch.Tensor],
        state: SystemStateFrame,
    ) -> tuple[ActionDecision, SynapticMotorPlan]:
        action_logits = outputs["action_logits"][0]
        mask = torch.tensor(valid_action_mask(state), dtype=torch.bool, device=action_logits.device)
        action_logits = action_logits.masked_fill(~mask, -1.0e4)
        action_index = int(action_logits.argmax().item())
        route_index = int(outputs["route_logits"][0].argmax().item())
        command_logits = outputs.get("command_candidate_logits", outputs["command_slot_logits"])
        file_logits = outputs.get("file_candidate_logits", outputs["file_slot_logits"])
        command_slot = int(command_logits[0].argmax().item())
        file_slot = int(file_logits[0].argmax().item())
        confidence = float(torch.softmax(action_logits, dim=-1).max().item())
        action_type = ACTION_ORDER[action_index]
        if action_type == ActionType.RUN_COMMAND and not candidate_commands(state):
            action_type = ActionType.WAIT
        if action_type == ActionType.READ_FILE and not candidate_files(state):
            action_type = ActionType.WAIT
        cortex_plan = SynapticMotorPlan(
            internal_target=InternalTarget.REFLEX_MOTOR,
            action_type=action_type,
            route_name=ROUTE_ORDER[route_index],
            command_slot=command_slot,
            file_slot=file_slot,
            confidence=confidence,
            reason="debug_cortex_head",
        )
        return serialize_motor_action(cortex_plan, state), cortex_plan

    def _debug_cache_key(self, state: SystemStateFrame, *, include_last_command: bool = True) -> str:
        """State-derived cache key for one visible debug transition chain.

        The key deliberately excludes episode ids, scenario templates, oracle
        labels, and hidden recovery hints. It captures only stable receptor
        affordances that remain valid across READ_STDERR -> READ_FILE ->
        RUN_COMMAND transitions inside one debug episode.
        """

        return json.dumps(
            {
                "task_type": state.goal.task_type.value,
                "description": state.goal.description,
                **({"last_command": state.terminal.last_command or ""} if include_last_command else {}),
                "commands": candidate_commands(state),
                "watched_paths": list(state.goal.watched_paths),
            },
            sort_keys=True,
        )

    def _reset_debug_continuation(self, reason: str) -> None:
        self._debug_continuation = None
        self._last_cache_reset_reason = reason

    def _visible_failure_signal(self, state: SystemStateFrame) -> str:
        if "low-level receptor latent" in state.goal.description.lower():
            return "latent_required"
        visible_failure = f"{state.terminal.stderr_delta} {state.terminal.stdout_delta}".lower()
        if "snapshot" in visible_failure and (
            "mismatch" in visible_failure or "update" in visible_failure
        ):
            return "snapshot_update"
        if (
            "modulenotfounderror" in visible_failure
            or "no module named" in visible_failure
            or "missing dependency" in visible_failure
            or "dependency missing" in visible_failure
        ):
            return "dependency_install"
        if "assertionerror" in visible_failure or "assertion failure" in visible_failure:
            return "assertion_inspection"
        return "other"

    def _select_command_by_intent(self, state: SystemStateFrame, intent: str) -> tuple[str | None, int | None]:
        for index, command in enumerate(candidate_commands(state)):
            if command_intent_for_text(command) == intent:
                return command, index
        return None, None

    def _select_last_command(
        self,
        state: SystemStateFrame,
        *,
        prior_command: str | None = None,
    ) -> tuple[str | None, int | None]:
        last_command = state.terminal.last_command or ""
        commands = candidate_commands(state)
        if last_command in commands:
            return last_command, commands.index(last_command)
        if prior_command and prior_command in commands:
            return prior_command, commands.index(prior_command)
        for index, command in enumerate(commands):
            if command_intent_for_text(command) == "test_rerun" and "--snapshot-update" not in command:
                return command, index
        return None, None

    def _counterfactual_prior_command(self, state: SystemStateFrame) -> str:
        """Return a non-oracle counterfactual cache command for intervention evals.

        The wrong-cache control must not read gold slots or scenario metadata.
        It deterministically swaps the current prior command to another visible
        allowlisted command, which is sufficient for Phase2L paired profiles
        because the paired counterfactual command is in the same candidate set.
        """

        prior_command = state.terminal.last_command or ""
        for command in candidate_commands(state):
            if command and command != prior_command:
                return command
        return prior_command

    def _continuation_prior_command(self, state: SystemStateFrame) -> tuple[str, bool]:
        if self.continuation_control == "wrong_cache":
            return self._counterfactual_prior_command(state), True
        return state.terminal.last_command or "", False

    def _select_source_file(self, state: SystemStateFrame) -> tuple[str | None, int | None]:
        files = candidate_files(state)
        if not files:
            return None, None
        dirty = set(state.filesystem.dirty_files)
        for index, file_target in enumerate(files):
            if file_target in dirty:
                return file_target, index
        return files[0], 0

    def _build_debug_continuation(
        self,
        state: SystemStateFrame,
        cortex_action: ActionDecision,
    ) -> dict[str, Any] | None:
        if not self.enable_debug_continuation:
            return None
        key = self._debug_cache_key(state)
        stable_key = self._debug_cache_key(state, include_last_command=False)
        prior_command, wrong_cache_injected = self._continuation_prior_command(state)
        if cortex_action.type == ActionType.READ_STDERR:
            signal = self._visible_failure_signal(state)
            if signal in {"snapshot_update", "dependency_install"}:
                return {
                    "version": "phase2f_debug_continuation_v1",
                    "key": key,
                    "stable_key": stable_key,
                    "prior_command": prior_command,
                    "wrong_cache_injected": wrong_cache_injected,
                    "continuation_control": self.continuation_control,
                    "next": "run_command_intent",
                    "intent": signal,
                    "created_from": cortex_action.type.value,
                }
            if signal == "assertion_inspection":
                return {
                    "version": "phase2f_debug_continuation_v1",
                    "key": key,
                    "stable_key": stable_key,
                    "prior_command": prior_command,
                    "wrong_cache_injected": wrong_cache_injected,
                    "continuation_control": self.continuation_control,
                    "next": "read_source_file",
                    "created_from": cortex_action.type.value,
                }
        if cortex_action.type == ActionType.READ_FILE:
            return {
                "version": "phase2f_debug_continuation_v1",
                "key": key,
                "stable_key": stable_key,
                "prior_command": prior_command,
                "wrong_cache_injected": wrong_cache_injected,
                "continuation_control": self.continuation_control,
                "next": "run_last_command",
                "created_from": cortex_action.type.value,
            }
        return None

    def _debug_receptor_reflex(
        self,
        state: SystemStateFrame,
        target: InternalTarget,
    ) -> ActionDecision | None:
        """Let the low-level receptor read fresh stderr before cortex actions.

        The Debug Cortex should reason over an observed failure signal, not
        skip directly from a terminal-error receptor frame to a motor command.
        This keeps sensing in the NSI path and avoids spending a 7B call on the
        first raw-error frame.
        """

        if target != InternalTarget.ESCALATE_TO_DEBUG_CORTEX:
            return None
        if not state.terminal.stderr_delta.strip():
            return None
        mask = valid_action_mask(state)
        read_stderr_index = ACTION_ORDER.index(ActionType.READ_STDERR)
        if float(mask[read_stderr_index]) <= 0.0:
            return None
        return ActionDecision(
            type=ActionType.READ_STDERR,
            confidence=0.96,
            reason="debug_receptor_read_stderr",
        )

    def _debug_continuation_action(
        self,
        state: SystemStateFrame,
        target: InternalTarget,
    ) -> tuple[ActionDecision | None, SynapticMotorPlan | None, str | None]:
        if not self.enable_debug_continuation:
            return None, None, None
        cache = self._debug_continuation
        if cache is None:
            return None, None, None
        if target not in {
            InternalTarget.ESCALATE_TO_DEBUG_CORTEX,
            InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX,
        }:
            self._reset_debug_continuation("non_debug_route")
            return None, None, "non_debug_route"
        if (
            state.safety.dangerous_command_detected
            or state.filesystem.external_change_detected
            or state.filesystem.stale_cache_detected
            or state.filesystem.conflict_detected
        ):
            self._reset_debug_continuation("visible_safety_or_stale_state")
            return None, None, "visible_safety_or_stale_state"
        current_key = self._debug_cache_key(state)
        if cache.get("key") != current_key:
            stable_key = cache.get("stable_key")
            current_stable_key = self._debug_cache_key(state, include_last_command=False)
            last_command_missing = not (state.terminal.last_command or "").strip()
            if not (stable_key == current_stable_key and last_command_missing):
                self._reset_debug_continuation("debug_context_changed")
                return None, None, "debug_context_changed"

        next_step = str(cache.get("next") or "")
        action: ActionDecision | None = None
        plan: SynapticMotorPlan | None = None
        if next_step == "run_command_intent":
            command, slot = self._select_command_by_intent(state, str(cache.get("intent") or ""))
            if command is not None and slot is not None:
                action = ActionDecision(
                    type=ActionType.RUN_COMMAND,
                    command=command,
                    confidence=0.92,
                    reason="debug_continuation_command_intent",
                )
                plan = SynapticMotorPlan(
                    internal_target=target,
                    action_type=ActionType.RUN_COMMAND,
                    route_name=RouteName.DEBUG,
                    command_slot=slot,
                    confidence=action.confidence,
                    reason=action.reason,
                )
                self._debug_continuation = None
        elif next_step == "read_source_file":
            file_target, slot = self._select_source_file(state)
            if file_target is not None and slot is not None:
                action = ActionDecision(
                    type=ActionType.READ_FILE,
                    file_target=file_target,
                    confidence=0.90,
                    reason="debug_continuation_source_inspection",
                )
                plan = SynapticMotorPlan(
                    internal_target=target,
                    action_type=ActionType.READ_FILE,
                    route_name=RouteName.DEBUG,
                    file_slot=slot,
                    confidence=action.confidence,
                    reason=action.reason,
                )
                self._debug_continuation = {
                    **cache,
                    "next": "run_last_command",
                    "created_from": action.type.value,
                }
        elif next_step == "run_last_command":
            visible = f"{state.terminal.stdout_delta} {state.terminal.stderr_delta}".lower()
            if "semantic disambiguation required" in visible:
                self._reset_debug_continuation("semantic_command_ambiguity")
                return None, None, self._last_cache_reset_reason
            if "source inspected" in visible or "rerun" in visible:
                command, slot = self._select_last_command(
                    state,
                    prior_command=str(cache.get("prior_command") or ""),
                )
                if command is not None and slot is not None:
                    action = ActionDecision(
                        type=ActionType.RUN_COMMAND,
                        command=command,
                        confidence=0.92,
                        reason="debug_continuation_last_command",
                    )
                    plan = SynapticMotorPlan(
                        internal_target=target,
                        action_type=ActionType.RUN_COMMAND,
                        route_name=RouteName.DEBUG,
                        command_slot=slot,
                        confidence=action.confidence,
                        reason=action.reason,
                    )
                    self._debug_continuation = None
        if action is None:
            self._reset_debug_continuation(f"unsupported_continuation:{next_step}")
            return None, None, self._last_cache_reset_reason
        return action, plan, None

    def _continuation_only_action(
        self,
        state: SystemStateFrame,
        target: InternalTarget,
    ) -> tuple[ActionDecision | None, SynapticMotorPlan | None, str | None]:
        if target not in {
            InternalTarget.ESCALATE_TO_DEBUG_CORTEX,
            InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX,
        }:
            return None, None, None
        if (
            state.safety.dangerous_command_detected
            or state.filesystem.external_change_detected
            or state.filesystem.stale_cache_detected
            or state.filesystem.conflict_detected
        ):
            return None, None, "visible_safety_or_stale_state"

        signal = self._visible_failure_signal(state)
        action: ActionDecision | None = None
        plan: SynapticMotorPlan | None = None
        if signal in {"snapshot_update", "dependency_install"}:
            command, slot = self._select_command_by_intent(state, signal)
            if command is not None and slot is not None:
                action = ActionDecision(
                    type=ActionType.RUN_COMMAND,
                    command=command,
                    confidence=0.90,
                    reason="continuation_only_command_intent",
                )
                plan = SynapticMotorPlan(
                    internal_target=target,
                    action_type=ActionType.RUN_COMMAND,
                    route_name=RouteName.DEBUG,
                    command_slot=slot,
                    confidence=action.confidence,
                    reason=action.reason,
                )
        elif signal == "assertion_inspection":
            file_target, slot = self._select_source_file(state)
            if file_target is not None and slot is not None:
                action = ActionDecision(
                    type=ActionType.READ_FILE,
                    file_target=file_target,
                    confidence=0.88,
                    reason="continuation_only_source_inspection",
                )
                plan = SynapticMotorPlan(
                    internal_target=target,
                    action_type=ActionType.READ_FILE,
                    route_name=RouteName.DEBUG,
                    file_slot=slot,
                    confidence=action.confidence,
                    reason=action.reason,
                )
        else:
            visible = f"{state.terminal.stdout_delta} {state.terminal.stderr_delta}".lower()
            if "source inspected" in visible or "rerun" in visible:
                command, slot = self._select_last_command(state)
                if command is not None and slot is not None:
                    action = ActionDecision(
                        type=ActionType.RUN_COMMAND,
                        command=command,
                        confidence=0.90,
                        reason="continuation_only_last_command",
                    )
                    plan = SynapticMotorPlan(
                        internal_target=target,
                        action_type=ActionType.RUN_COMMAND,
                        route_name=RouteName.DEBUG,
                        command_slot=slot,
                        confidence=action.confidence,
                        reason=action.reason,
                    )
        if action is None or plan is None:
            return None, None, "continuation_only_no_visible_plan"
        return action, plan, None

    def act(
        self,
        state: SystemStateFrame,
        *,
        nsi_reference_override: dict[str, Any] | None = None,
    ) -> ActionDecision:
        started = time.perf_counter()
        self._last_plasticity_state = state
        self._last_plasticity_command = None
        low_level_action = self._low_level_action(
            state,
            nsi_reference_override=nsi_reference_override,
        )
        target, target_source = self._internal_target_with_prediction_error(state)
        prediction_error_escalation_threshold = float(
            getattr(self, "prediction_error_escalation_threshold", 0.45)
        )
        self.nsi_policy.last_call["internal_target_source"] = target_source
        self.nsi_policy.last_call["prediction_error_escalation_threshold"] = (
            prediction_error_escalation_threshold
        )
        self.nsi_policy.last_call["prediction_error_escalated"] = (
            target_source == "observed_temporal_prediction_error"
        )
        if target not in {
            InternalTarget.ESCALATE_TO_DEBUG_CORTEX,
            InternalTarget.ESCALATE_TO_SEMANTIC_CORTEX,
        }:
            self.last_call = {
                "action_source": "low_level_nsi",
                "internal_target": target.value,
                "route": low_level_action.reason,
                "nsi_debug": self.nsi_policy.last_call,
                "qwen_called": False,
                "cache_hit": False,
                "cache_reset_reason": self._last_cache_reset_reason,
                "slot_source": None,
                "decision_latency_ms": round((time.perf_counter() - started) * 1000, 6),
            }
            if low_level_action.type == ActionType.RUN_COMMAND:
                self._last_plasticity_command = low_level_action.command
            return low_level_action

        receptor_action = self._debug_receptor_reflex(state, target)
        if receptor_action is not None:
            self._debug_continuation = self._build_debug_continuation(state, receptor_action)
            self.last_call = {
                "action_source": "low_level_debug_receptor",
                "internal_target": target.value,
                "route": RouteName.DEBUG.value,
                "nsi_debug": self.nsi_policy.last_call,
                "json_text_target": False,
                "qwen_called": False,
                "cache_hit": False,
                "cache_reset_reason": self._last_cache_reset_reason,
                "debug_continuation_cached": self._debug_continuation is not None,
                "continuation_control": self.continuation_control,
                "wrong_cache_injected": bool(
                    self._debug_continuation
                    and self._debug_continuation.get("wrong_cache_injected")
                ),
                "slot_source": receptor_action.reason,
                "decision_latency_ms": round((time.perf_counter() - started) * 1000, 6),
            }
            if receptor_action.type == ActionType.RUN_COMMAND:
                self._last_plasticity_command = receptor_action.command
            return receptor_action

        if self.enable_native_head_calls:
            continuation_action, continuation_plan, cache_reset_reason = self._debug_continuation_action(
                state,
                target,
            )
        else:
            continuation_action, continuation_plan, cache_reset_reason = self._continuation_only_action(
                state,
                target,
            )
        if continuation_action is not None and continuation_plan is not None:
            self.last_call = {
                "action_source": (
                    "native_head_continuation_cache"
                    if self.enable_native_head_calls
                    else "continuation_only"
                ),
                "internal_target": target.value,
                "route": RouteName.DEBUG.value,
                "cortex_plan": continuation_plan.to_dict(),
                "nsi_debug": self.nsi_policy.last_call,
                "json_text_target": False,
                "qwen_called": False,
                "cache_hit": bool(self.enable_native_head_calls),
                "cache_reset_reason": cache_reset_reason,
                "continuation_control": self.continuation_control,
                "wrong_cache_injected": bool(
                    self.enable_native_head_calls
                    and self.continuation_control == "wrong_cache"
                ),
                "slot_source": continuation_action.reason,
                "decision_latency_ms": round((time.perf_counter() - started) * 1000, 6),
            }
            if continuation_action.type == ActionType.RUN_COMMAND:
                self._last_plasticity_command = continuation_action.command
            return continuation_action

        if not self.enable_native_head_calls:
            fallback = low_level_action
            self.last_call = {
                "action_source": "continuation_only_fallback",
                "internal_target": target.value,
                "route": RouteName.DEBUG.value,
                "nsi_debug": self.nsi_policy.last_call,
                "json_text_target": False,
                "qwen_called": False,
                "cache_hit": False,
                "cache_reset_reason": cache_reset_reason,
                "slot_source": "continuation_only_no_visible_plan",
                "decision_latency_ms": round((time.perf_counter() - started) * 1000, 6),
            }
            if fallback.type == ActionType.RUN_COMMAND:
                self._last_plasticity_command = fallback.command
            return fallback

        native_text_state = self._state_for_native_head_text(state)
        encoded = self._encode_state(native_text_state)
        commands = candidate_commands(native_text_state)
        files = candidate_files(native_text_state)
        if commands and self.command_candidate_encoder == "backbone":
            command_encoded = self._encode_candidate_texts(commands, kind="Command")
        elif commands:
            command_encoded = {"candidate_mask": self._candidate_mask_tensor(commands)}
        else:
            command_encoded = {}
        command_pair_encoded = (
            self._encode_candidate_pair_texts(native_text_state, commands, kind="Command")
            if commands and self.use_pairwise_command_reranker
            else {}
        )
        pairwise_encoded_candidates = int(
            command_pair_encoded.get("candidate_mask", torch.zeros(1, 0)).sum().item()
        )
        file_encoded = self._encode_candidate_texts(files, kind="File") if files else {}
        with torch.inference_mode():
            outputs = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                nsi_latent=self._nsi_latent(
                    state,
                    nsi_reference_override=nsi_reference_override,
                ),
                command_input_ids=command_encoded.get("input_ids"),
                command_attention_mask=command_encoded.get("attention_mask"),
                command_candidate_mask=command_encoded.get("candidate_mask"),
                command_candidate_features=(
                    self._command_candidate_features(
                        native_text_state,
                        commands,
                        nsi_reference_override=nsi_reference_override,
                    )
                    if commands
                    else None
                ),
                command_candidate_intents=(
                    self._command_candidate_intents(commands) if commands else None
                ),
                command_pair_input_ids=command_pair_encoded.get("input_ids"),
                command_pair_attention_mask=command_pair_encoded.get("attention_mask"),
                command_pair_mask=command_pair_encoded.get("candidate_mask"),
                file_input_ids=file_encoded.get("input_ids"),
                file_attention_mask=file_encoded.get("attention_mask"),
                file_candidate_mask=file_encoded.get("candidate_mask"),
            )
        cortex_action, cortex_plan = self._cortex_action_from_heads(outputs, state)
        escalation_plan = SynapticMotorPlan(
            internal_target=target,
            action_type=None,
            route_name=RouteName.DEBUG
            if target == InternalTarget.ESCALATE_TO_DEBUG_CORTEX
            else RouteName.PLANNER,
            confidence=cortex_action.confidence,
            reason=target.value.lower(),
        )
        action = serialize_motor_action(escalation_plan, state, cortex_action=cortex_action)
        self._debug_continuation = self._build_debug_continuation(state, cortex_action)
        self.stats.model_calls += 1
        self.stats.token_cost += int(encoded["attention_mask"].sum().item())
        target_index = int(outputs["target_logits"][0].argmax().item())
        self.last_call = {
            "action_source": "native_head_cortex",
            "internal_target": target.value,
            "raw_internal_target": INTERNAL_TARGET_ORDER[target_index].value,
            "route": cortex_plan.route_name.value,
            "cortex_plan": cortex_plan.to_dict(),
            "nsi_debug": self.nsi_policy.last_call,
            "json_text_target": False,
            "qwen_called": True,
            "cache_hit": False,
            "cache_reset_reason": self._last_cache_reset_reason,
            "debug_continuation_cached": self._debug_continuation is not None,
            "continuation_control": self.continuation_control,
            "wrong_cache_injected": bool(
                self._debug_continuation
                and self._debug_continuation.get("wrong_cache_injected")
            ),
            "pairwise_command_policy": self.pairwise_command_policy,
            "pairwise_command_max_length": self.pairwise_command_max_length,
            "pairwise_command_top_k": self.pairwise_command_top_k,
            "command_candidate_encoder": self.command_candidate_encoder,
            "disabled_command_candidate_feature_groups": list(
                self.disabled_command_candidate_feature_groups
            ),
            "pairwise_encoded_candidates": pairwise_encoded_candidates,
            **self._open_repair_runtime_outputs(outputs),
            "slot_source": cortex_action.reason or "native_head_logits",
            "plasticity_prediction": dict(self._last_plasticity_prediction),
            "decision_latency_ms": round((time.perf_counter() - started) * 1000, 6),
        }
        if action.type == ActionType.RUN_COMMAND:
            self._last_plasticity_command = action.command
        return action
