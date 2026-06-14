from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from reflexlm.llm.candidate_features import normalize_command_candidate_feature_ablation_groups
from reflexlm.llm.native_cortex import OPEN_REPAIR_CAPABILITY_NAMES
from reflexlm.llm.native_head_policy import MODEL_LOAD_STRATEGIES, NativeHeadPolicy
from reflexlm.models.features import candidate_commands
from reflexlm.models.semantic_matcher import (
    FrozenEncoderDualSemanticMatcher,
    RecencyWeightedSemanticMatcher,
)


PACKAGE_MANIFEST_NAME = "native_nervous_package.json"
PATCH_PROPOSAL_STRATEGIES = (
    "none",
    "recorded_candidate_selection",
    "symbolic_runtime_generator",
    "learned_bounded_candidate",
)


def _file_sha256(path: str | Path) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_native_nervous_package(
    output_dir: str | Path,
    *,
    base_model_name: str,
    native_head_path: str | Path,
    low_level_checkpoint_path: str | Path,
    quantization: str = "4bit",
    max_length: int = 256,
    nsi_device: str = "cpu",
    device: str = "cuda",
    model_load_strategy: str = "auto",
    offload_state_dict: bool = False,
    policy_label: str = "phase2d_native_nervous_package",
    zero_nsi_latent: bool = False,
    continuation_cache_enabled: bool = True,
    continuation_control: str = "normal",
    native_head_calls_enabled: bool = True,
    disabled_command_candidate_feature_groups: list[str] | tuple[str, ...] | None = None,
    open_repair_capabilities: dict[str, bool] | None = None,
    patch_proposal_strategy: str = "none",
    learned_patch_generation_enabled: bool = False,
    patch_candidate_schema_version: str | None = None,
    verification_cortex_path: str | Path | None = None,
    verification_cortex_model_name: str | Path | None = None,
    verification_cortex_recency_decay: float = 0.25,
    structured_runtime_cortex_checkpoint_path: str | Path | None = None,
    structured_runtime_cortex_python_identity: str | None = None,
) -> dict[str, Any]:
    """Write a single runtime package manifest for Phase2D policy loading.

    The package intentionally stores references to the heavy Qwen model, LoRA
    adapter, and low-level NSI checkpoint rather than copying multi-GB files.
    The runtime loads them as one policy package and reports package metadata
    in evaluation artifacts.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    requested_feature_groups = list(disabled_command_candidate_feature_groups or [])
    if zero_nsi_latent and "candidate_identity" not in requested_feature_groups:
        requested_feature_groups.append("candidate_identity")
    feature_ablation_groups = normalize_command_candidate_feature_ablation_groups(
        requested_feature_groups
    )
    if continuation_control not in {"normal", "cache_erased", "wrong_cache"}:
        raise ValueError(
            "continuation_control must be one of: normal, cache_erased, wrong_cache"
        )
    if patch_proposal_strategy not in PATCH_PROPOSAL_STRATEGIES:
        raise ValueError(
            "patch_proposal_strategy must be one of: "
            + ", ".join(PATCH_PROPOSAL_STRATEGIES)
        )
    if model_load_strategy not in MODEL_LOAD_STRATEGIES:
        raise ValueError(
            "model_load_strategy must be one of: "
            + ", ".join(MODEL_LOAD_STRATEGIES)
        )
    if model_load_strategy == "single_device" and device != "cuda":
        raise ValueError("single_device model loading requires device='cuda'")
    if learned_patch_generation_enabled and patch_proposal_strategy != "learned_bounded_candidate":
        raise ValueError(
            "learned_patch_generation_enabled requires "
            "patch_proposal_strategy='learned_bounded_candidate'"
        )
    effective_continuation_enabled = (
        continuation_cache_enabled and continuation_control != "cache_erased"
    )
    normalized_open_repair_caps = {
        name: bool((open_repair_capabilities or {}).get(name, False))
        for name in OPEN_REPAIR_CAPABILITY_NAMES
    }
    manifest = {
        "package_family": "phase2d_native_nervous_package",
        "policy_label": policy_label,
        "base_model_name": str(base_model_name),
        "native_head_path": str(native_head_path),
        "low_level_checkpoint_path": str(low_level_checkpoint_path),
        "quantization": quantization,
        "max_length": int(max_length),
        "nsi_device": nsi_device,
        "device": device,
        "model_load_strategy": model_load_strategy,
        "offload_state_dict": bool(offload_state_dict),
        "json_text_target": False,
        "zero_nsi_latent": zero_nsi_latent,
        "low_level_reference_packaged": True,
        "cortex_invocation_policy": "ESCALATE_TO_DEBUG_CORTEX only",
        "debug_continuation_cache": effective_continuation_enabled,
        "continuation_cache_enabled": effective_continuation_enabled,
        "continuation_control": continuation_control,
        "native_head_calls_enabled": native_head_calls_enabled,
        "disabled_command_candidate_feature_groups": list(feature_ablation_groups),
        "open_repair_capabilities": normalized_open_repair_caps,
        "patch_proposal_strategy": patch_proposal_strategy,
        "learned_patch_generation_enabled": bool(learned_patch_generation_enabled),
        "patch_candidate_schema_version": patch_candidate_schema_version,
        "verification_cortex_path": (
            str(verification_cortex_path) if verification_cortex_path is not None else None
        ),
        "verification_cortex_model_name": (
            str(verification_cortex_model_name)
            if verification_cortex_model_name is not None
            else None
        ),
        "verification_cortex_recency_decay": float(verification_cortex_recency_decay),
        "structured_runtime_cortex_checkpoint_path": (
            str(structured_runtime_cortex_checkpoint_path)
            if structured_runtime_cortex_checkpoint_path is not None
            else None
        ),
        "structured_runtime_cortex_python_identity": structured_runtime_cortex_python_identity,
        "motor_output": "explicit_heads_runtime_serialization",
    }
    (output_path / PACKAGE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


class NativeNervousPolicyPackage:
    """Phase2D policy package.

    This keeps the deployment/evaluation interface single-package even though
    the current low-level reflex module is still loaded from the validated
    small NSI checkpoint. The gate remains responsible for detecting whether
    low-level states call Qwen.
    """

    def __init__(
        self,
        package_path: str | Path,
        *,
        continuation_control: str | None = None,
        disabled_command_candidate_feature_groups: list[str] | tuple[str, ...] | None = None,
        device: str | None = None,
        quantization: str | None = None,
        model_load_strategy: str | None = None,
        offload_state_dict: bool | None = None,
        verification_cortex_path: str | Path | None = None,
        verification_cortex_model_name: str | Path | None = None,
        verification_cortex_device: str = "cpu",
        verification_cortex_dtype: str = "auto",
        structured_runtime_cortex_checkpoint_path: str | Path | None = None,
        structured_runtime_cortex_python_identity: str | None = None,
        load_native_head_policy: bool = True,
        load_verification_cortex: bool = True,
    ) -> None:
        self.package_path = Path(package_path)
        manifest_path = (
            self.package_path
            if self.package_path.is_file()
            else self.package_path / PACKAGE_MANIFEST_NAME
        )
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runtime_overrides = {
            key: value
            for key, value in {
                "device": device,
                "quantization": quantization,
                "model_load_strategy": model_load_strategy,
                "offload_state_dict": offload_state_dict,
            }.items()
            if value is not None
        }
        if (
            runtime_overrides.get("model_load_strategy") == "single_device"
            and runtime_overrides.get("device", self.manifest.get("device")) != "cuda"
        ):
            raise ValueError("single_device model loading requires device='cuda'")
        self.manifest.update(runtime_overrides)
        self.runtime_overrides = runtime_overrides
        self.verification_cortex_path = (
            Path(verification_cortex_path)
            if verification_cortex_path is not None
            else (
                Path(str(self.manifest["verification_cortex_path"]))
                if self.manifest.get("verification_cortex_path")
                else None
            )
        )
        self.verification_cortex_model_name = (
            str(verification_cortex_model_name)
            if verification_cortex_model_name is not None
            else self.manifest.get("verification_cortex_model_name")
        )
        self.structured_runtime_cortex_checkpoint_path = (
            Path(structured_runtime_cortex_checkpoint_path)
            if structured_runtime_cortex_checkpoint_path is not None
            else (
                Path(str(self.manifest["structured_runtime_cortex_checkpoint_path"]))
                if self.manifest.get("structured_runtime_cortex_checkpoint_path")
                else None
            )
        )
        self.structured_runtime_cortex_python_identity = (
            structured_runtime_cortex_python_identity
            if structured_runtime_cortex_python_identity is not None
            else self.manifest.get("structured_runtime_cortex_python_identity")
        )
        if disabled_command_candidate_feature_groups is None:
            feature_ablation_groups = self.manifest.get(
                "disabled_command_candidate_feature_groups",
                [],
            )
        else:
            feature_ablation_groups = disabled_command_candidate_feature_groups
        if self.manifest.get("zero_nsi_latent") is True:
            feature_ablation_groups = [*feature_ablation_groups, "candidate_identity"]
        if continuation_control is not None:
            if continuation_control not in {"normal", "cache_erased", "wrong_cache"}:
                raise ValueError(
                    "continuation_control must be one of: normal, cache_erased, wrong_cache"
                )
            self.manifest["continuation_control"] = continuation_control
            self.manifest["continuation_cache_enabled"] = continuation_control != "cache_erased"
            self.manifest["debug_continuation_cache"] = continuation_control != "cache_erased"
        self.disabled_command_candidate_feature_groups = (
            normalize_command_candidate_feature_ablation_groups(feature_ablation_groups)
        )
        self.policy = (
            NativeHeadPolicy(
                base_model_name=self.manifest["base_model_name"],
                native_head_path=self.manifest["native_head_path"],
                nsi_checkpoint_path=self.manifest["low_level_checkpoint_path"],
                quantization=self.manifest.get("quantization", "4bit"),
                nsi_device=self.manifest.get("nsi_device", "cpu"),
                device=self.manifest.get("device", "cuda"),
                model_load_strategy=str(
                    self.manifest.get("model_load_strategy", "auto")
                ),
                offload_state_dict=bool(
                    self.manifest.get("offload_state_dict", False)
                ),
                max_length=int(self.manifest.get("max_length", 256)),
                policy_label=self.manifest.get(
                    "policy_label", "phase2d_native_nervous_package"
                ),
                zero_nsi_latent=bool(self.manifest.get("zero_nsi_latent", False)),
                enable_debug_continuation=bool(
                    self.manifest.get(
                        "continuation_cache_enabled",
                        self.manifest.get("debug_continuation_cache", True),
                    )
                ),
                continuation_control=str(
                    self.manifest.get("continuation_control", "normal")
                ),
                enable_native_head_calls=bool(
                    self.manifest.get("native_head_calls_enabled", True)
                ),
                prediction_error_escalation_threshold=(
                    float(self.manifest["prediction_error_escalation_threshold"])
                    if self.manifest.get("prediction_error_escalation_threshold")
                    is not None
                    else None
                ),
                disabled_command_candidate_feature_groups=list(
                    self.disabled_command_candidate_feature_groups
                ),
            )
            if load_native_head_policy
            else None
        )
        self.verification_cortex = None
        if load_verification_cortex and self.verification_cortex_path is not None:
            base_matcher = FrozenEncoderDualSemanticMatcher.load(
                self.verification_cortex_path,
                device=verification_cortex_device,
                dtype=verification_cortex_dtype,
                model_name=self.verification_cortex_model_name,
            )
            self.verification_cortex = RecencyWeightedSemanticMatcher(
                base_matcher,
                recency_decay=float(
                    self.manifest.get("verification_cortex_recency_decay", 0.25)
                ),
            )

    def _native_head_policy(self) -> NativeHeadPolicy:
        if self.policy is None:
            raise RuntimeError("native head policy was not loaded for this package view")
        return self.policy

    @property
    def stats(self) -> Any:
        return self._native_head_policy().stats

    @property
    def last_call(self) -> dict[str, Any]:
        return self._native_head_policy().last_call

    def reset(self) -> None:
        self._native_head_policy().reset()

    def load_plasticity_memory(self, path: str | Path) -> None:
        self._native_head_policy().load_plasticity_memory(path)

    def save_plasticity_memory(self, path: str | Path) -> None:
        self._native_head_policy().save_plasticity_memory(path)

    def clear_plasticity_memory(self) -> None:
        self._native_head_policy().clear_plasticity_memory()

    def set_plasticity_control(self, control: str) -> None:
        self._native_head_policy().set_plasticity_control(control)

    def observe_feedback(
        self,
        *,
        verified_success: bool,
        verifier: str = "post_execution_verifier",
    ) -> dict[str, Any]:
        return self._native_head_policy().observe_feedback(
            verified_success=verified_success,
            verifier=verifier,
        )

    def decide_verification(self, state) -> dict[str, Any]:
        if self.verification_cortex is None:
            raise RuntimeError("native nervous package has no verification cortex")
        commands = candidate_commands(state)
        scores = [float(value) for value in self.verification_cortex.score_state(state)]
        selected_slot = max(range(len(scores)), key=scores.__getitem__) if scores else None
        return {
            "selected_slot": selected_slot,
            "selected_command": (
                commands[selected_slot] if selected_slot is not None else None
            ),
            "scores": scores,
            "verification_cortex_metadata": self.verification_cortex.metadata(),
            "package_internal_expert": True,
        }

    def create_structured_runtime_policy(
        self,
        *,
        enable_online_homeostatic_adaptation: bool | None = None,
        enable_cross_episode_homeostatic_memory: bool | None = None,
    ):
        if self.structured_runtime_cortex_checkpoint_path is None:
            raise RuntimeError("native nervous package has no structured runtime cortex")
        from reflexlm.eval import SequenceModelPolicy
        from reflexlm.train import load_model_checkpoint

        model, vectorizer, checkpoint_payload = load_model_checkpoint(
            self.structured_runtime_cortex_checkpoint_path,
            device="cpu",
        )
        policy = SequenceModelPolicy(
            model,
            vectorizer,
            policy_label="package_internal_structured_runtime_cortex",
            training_summary=checkpoint_payload.get("training_summary", {}),
            authorize_bounded_debug_cortex_recovery=True,
            use_synaptic_motor_plan=True,
            enable_online_homeostatic_adaptation=enable_online_homeostatic_adaptation,
            enable_cross_episode_homeostatic_memory=(
                enable_cross_episode_homeostatic_memory
            ),
            homeostatic_persistence_scope=json.dumps(
                {
                    "package_manifest_sha256": _file_sha256(self.manifest_path),
                    "structured_runtime_cortex_checkpoint_sha256": _file_sha256(
                        self.structured_runtime_cortex_checkpoint_path
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        return _PackageInternalPolicyAdapter(
            policy,
            package_manifest_path=self.manifest_path,
            expert_name="structured_runtime_cortex",
            checkpoint_path=self.structured_runtime_cortex_checkpoint_path,
            python_identity=self.structured_runtime_cortex_python_identity,
        )

    def metadata(self) -> dict[str, Any]:
        payload = (
            self._native_head_policy().metadata()
            if self.policy is not None
            else {
                "policy_family": "phase2d_native_nervous_package",
                "policy_label": self.manifest.get(
                    "policy_label",
                    "phase2d_native_nervous_package",
                ),
            }
        )
        payload.update(
            {
                "policy_family": "phase2d_native_nervous_package",
                "policy_label": self.manifest.get(
                    "policy_label",
                    "phase2d_native_nervous_package",
                ),
                "package_path": str(self.package_path),
                "package_manifest_path": str(self.manifest_path),
                "package_family": self.manifest.get("package_family"),
                "runtime_overrides": dict(self.runtime_overrides),
                "low_level_reference_packaged": bool(
                    self.manifest.get("low_level_reference_packaged")
                ),
                "zero_nsi_latent": bool(self.manifest.get("zero_nsi_latent", False)),
                "json_text_target": False,
                "debug_continuation_cache": bool(
                    self.manifest.get("debug_continuation_cache", False)
                ),
                "continuation_cache_enabled": bool(
                    self.manifest.get(
                        "continuation_cache_enabled",
                        self.manifest.get("debug_continuation_cache", False),
                    )
                ),
                "continuation_control": str(
                    self.manifest.get("continuation_control", "normal")
                ),
                "native_head_calls_enabled": bool(
                    self.manifest.get("native_head_calls_enabled", True)
                ),
                "model_load_strategy": str(
                    self.manifest.get("model_load_strategy", "auto")
                ),
                "offload_state_dict": bool(
                    self.manifest.get("offload_state_dict", False)
                ),
                "disabled_command_candidate_feature_groups": list(
                    self.disabled_command_candidate_feature_groups
                ),
                "open_repair_capabilities": dict(
                    self.manifest.get("open_repair_capabilities") or {}
                ),
                "patch_proposal_strategy": str(
                    self.manifest.get("patch_proposal_strategy", "none")
                ),
                "learned_patch_generation_enabled": bool(
                    self.manifest.get("learned_patch_generation_enabled", False)
                ),
                "patch_candidate_schema_version": self.manifest.get(
                    "patch_candidate_schema_version"
                ),
                "native_head_policy_loaded": self.policy is not None,
                "verification_cortex_packaged": self.verification_cortex_path is not None,
                "verification_cortex_loaded": self.verification_cortex is not None,
                "verification_cortex_path": (
                    str(self.verification_cortex_path)
                    if self.verification_cortex_path is not None
                    else None
                ),
                "verification_cortex_model_name": self.verification_cortex_model_name,
                "verification_cortex_metadata": (
                    self.verification_cortex.metadata()
                    if self.verification_cortex is not None
                    else None
                ),
                "structured_runtime_cortex_packaged": (
                    self.structured_runtime_cortex_checkpoint_path is not None
                ),
                "structured_runtime_cortex_checkpoint_path": (
                    str(self.structured_runtime_cortex_checkpoint_path)
                    if self.structured_runtime_cortex_checkpoint_path is not None
                    else None
                ),
                "structured_runtime_cortex_python_identity": (
                    self.structured_runtime_cortex_python_identity
                ),
            }
        )
        return payload

    def act(self, state, *, nsi_reference_override: dict[str, Any] | None = None):
        return self._native_head_policy().act(
            state,
            nsi_reference_override=nsi_reference_override,
        )


class _PackageInternalPolicyAdapter:
    PYTHON_IDENTITY_MAPPING_SCOPE = "command_executable_prefix"

    def __init__(
        self,
        policy,
        *,
        package_manifest_path: Path,
        expert_name: str,
        checkpoint_path: Path,
        python_identity: str | None = None,
    ) -> None:
        self.policy = policy
        self.package_manifest_path = package_manifest_path
        self.expert_name = expert_name
        self.checkpoint_path = checkpoint_path
        self.python_identity = python_identity

    @property
    def stats(self):
        return self.policy.stats

    @property
    def last_call(self):
        return self.policy.last_call

    def reset(self) -> None:
        self.policy.reset()

    def save_homeostatic_state(
        self,
        path: str | Path,
        *,
        authenticity_key: str | bytes | None = None,
    ) -> dict[str, object]:
        return self.policy.save_homeostatic_state(
            path,
            authenticity_key=authenticity_key,
        )

    def load_homeostatic_state(
        self,
        path: str | Path,
        *,
        authenticity_key: str | bytes | None = None,
    ) -> dict[str, object]:
        return self.policy.load_homeostatic_state(
            path,
            authenticity_key=authenticity_key,
        )

    def act(self, state):
        policy_state = self._canonicalize_state(state)
        selected = self.policy.act(policy_state)
        selected = self._runtime_action(selected)
        self.policy.last_call = {
            **dict(self.policy.last_call),
            "package_python_identity_canonicalized": self._canonicalization_active(),
            "package_python_identity": self.python_identity,
            "package_runtime_python": sys.executable,
            "package_python_identity_mapping_scope": self.PYTHON_IDENTITY_MAPPING_SCOPE,
        }
        return selected

    def metadata(self) -> dict[str, Any]:
        return {
            "policy_family": "package_internal_cortical_expert",
            "package_internal_expert": True,
            "expert_name": self.expert_name,
            "package_manifest_path": str(self.package_manifest_path),
            "checkpoint_path": str(self.checkpoint_path),
            "python_identity": self.python_identity,
            "runtime_python": sys.executable,
            "python_identity_canonicalization": self._canonicalization_active(),
            "python_identity_mapping_scope": self.PYTHON_IDENTITY_MAPPING_SCOPE,
            "expert_policy": self.policy.metadata(),
        }

    def _canonicalization_active(self) -> bool:
        return bool(
            self.python_identity
            and sys.executable
            and self.python_identity.lower() != sys.executable.lower()
        )

    def _canonicalize_text(self, value: str | None) -> str | None:
        if value is None or not self._canonicalization_active():
            return value
        return self._replace_command_executable(
            value,
            source=sys.executable,
            target=str(self.python_identity),
        )

    def _runtime_text(self, value: str | None) -> str | None:
        if value is None or not self._canonicalization_active():
            return value
        return self._replace_command_executable(
            value,
            source=str(self.python_identity),
            target=sys.executable,
        )

    @staticmethod
    def _replace_command_executable(value: str, *, source: str, target: str) -> str:
        leading_length = len(value) - len(value.lstrip())
        leading = value[:leading_length]
        command = value[leading_length:]
        for quote in ("", '"'):
            source_prefix = f"{quote}{source}{quote}"
            if not command.lower().startswith(source_prefix.lower()):
                continue
            suffix = command[len(source_prefix) :]
            if suffix and not suffix[0].isspace():
                continue
            return f"{leading}{quote}{target}{quote}{suffix}"
        return value

    def _canonicalize_state(self, state):
        if not self._canonicalization_active():
            return state
        goal = state.goal.model_copy(
            update={
                "command_allowlist": [
                    self._canonicalize_text(command)
                    for command in state.goal.command_allowlist
                ]
            }
        )
        terminal = state.terminal.model_copy(
            update={
                "last_command": self._canonicalize_text(state.terminal.last_command),
            }
        )
        safety = state.safety.model_copy(
            update={
                "command_candidate": self._canonicalize_text(
                    state.safety.command_candidate
                ),
            }
        )
        return state.model_copy(
            update={"goal": goal, "terminal": terminal, "safety": safety}
        )

    def _runtime_action(self, action):
        if not self._canonicalization_active() or action.command is None:
            return action
        return action.model_copy(update={"command": self._runtime_text(action.command)})
