import json
from pathlib import Path
import torch
from torch import nn
from types import SimpleNamespace

from reflexlm.data.tasks import build_env
from reflexlm.llm.candidate_features import (
    CANDIDATE_FEATURE_DIM,
    COMMAND_IDENTITY_FEATURE_END,
    COMMAND_IDENTITY_FEATURE_START,
    COMMAND_SLOT_POSITION_FEATURE_END,
    COMMAND_SLOT_POSITION_FEATURE_START,
    command_candidate_feature_rows,
    runtime_verification_candidate_prediction,
    runtime_evidence_command_identity_rows,
    runtime_evidence_mapping_command_identity_rows,
    source_overlap_command_slot_prediction,
    redact_structured_command_identity_text,
)
from reflexlm.llm.native_cortex import (
    NativeCortexActionHeads,
    NativeCortexHeadConfig,
    PATCH_OPERATION_ORDER,
    QwenBackboneHeadAdapter,
)
from reflexlm.llm.native_head_policy import NativeHeadPolicy, _runtime_compatible_peft_adapter_path
from reflexlm.llm.native_nervous_package import NativeNervousPolicyPackage
import reflexlm.llm.native_nervous_package as native_nervous_package
from reflexlm.models.features import ACTION_ORDER, MAX_CANDIDATE_SLOTS, ROUTE_ORDER, candidate_commands
from reflexlm.runtime.nervous_system import (
    INTERNAL_TARGET_ORDER,
    InternalTarget,
    authorize_bounded_debug_cortex_action,
    authorize_persistent_failure_recovery,
    persistent_failure_recovery_required,
    plan_from_head_indices,
    serialize_motor_action,
)
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    FileSystemState,
    GoalSpec,
    ProcessState,
    RouteName,
    RuntimeEvidenceState,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


class _CompatLoraConfig:
    def __init__(self, peft_type=None, r=8, lora_alpha=8):
        self.peft_type = peft_type
        self.r = r
        self.lora_alpha = lora_alpha


def test_package_runtime_overrides_do_not_mutate_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = {
        "package_family": "phase2d_native_nervous_package",
        "policy_label": "test-package",
        "base_model_name": "test-model",
        "native_head_path": "test-head",
        "low_level_checkpoint_path": "test-nsi",
        "quantization": "4bit",
        "nsi_device": "cpu",
        "device": "cuda",
        "model_load_strategy": "single_device",
        "offload_state_dict": False,
    }
    manifest_path = tmp_path / "native_nervous_package.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured = {}

    class _FakeNativeHeadPolicy:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def metadata(self) -> dict:
            return {"policy_family": "phase2c_native_heads"}

    monkeypatch.setattr(native_nervous_package, "NativeHeadPolicy", _FakeNativeHeadPolicy)

    package = NativeNervousPolicyPackage(
        tmp_path,
        device="cpu",
        quantization="none",
        model_load_strategy="auto",
        offload_state_dict=True,
    )

    assert captured["device"] == "cpu"
    assert captured["quantization"] == "none"
    assert captured["model_load_strategy"] == "auto"
    assert captured["offload_state_dict"] is True
    assert package.metadata()["runtime_overrides"] == {
        "device": "cpu",
        "quantization": "none",
        "model_load_strategy": "auto",
        "offload_state_dict": True,
    }
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_package_internal_verification_cortex_selects_bounded_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = {
        "package_family": "phase2d_native_nervous_package",
        "policy_label": "test-package",
        "base_model_name": "test-model",
        "native_head_path": "test-head",
        "low_level_checkpoint_path": "test-nsi",
        "quantization": "4bit",
        "nsi_device": "cpu",
        "device": "cuda",
        "model_load_strategy": "single_device",
        "offload_state_dict": False,
        "verification_cortex_path": str(tmp_path / "verification.pt"),
    }
    (tmp_path / "native_nervous_package.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    class _FakeNativeHeadPolicy:
        def __init__(self, **_kwargs) -> None:
            pass

        def metadata(self) -> dict:
            return {"policy_family": "phase2c_native_heads"}

    class _FakeBaseMatcher:
        def score_texts(self, observation: str, commands: list[str]) -> list[float]:
            del observation
            return [0.0 if "continue" in command else 1.0 for command in commands]

        def metadata(self) -> dict:
            return {"matcher_family": "fake_base"}

    monkeypatch.setattr(native_nervous_package, "NativeHeadPolicy", _FakeNativeHeadPolicy)
    monkeypatch.setattr(
        native_nervous_package.FrozenEncoderDualSemanticMatcher,
        "load",
        classmethod(lambda cls, *_args, **_kwargs: _FakeBaseMatcher()),
    )
    package = NativeNervousPolicyPackage(tmp_path)
    state = SystemStateFrame(
        time=TimeState(),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="Select bounded verification control.",
            command_allowlist=[
                "runner --intent 'continue repair investigation'",
                "runner --intent 'finish verified repair'",
            ],
        ),
        process=ProcessState(),
        terminal=TerminalState(),
        filesystem=FileSystemState(),
        runtime_evidence=RuntimeEvidenceState(terminal_observations=["1 passed"]),
    )

    decision = package.decide_verification(state)

    assert decision["selected_slot"] == 1
    assert decision["package_internal_expert"] is True
    assert package.metadata()["verification_cortex_packaged"] is True


def test_runtime_peft_adapter_config_filters_forward_version_fields(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "alora_invocation_tokens": None,
                "future_only_field": True,
            }
        ),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    compat_path, temp_dir = _runtime_compatible_peft_adapter_path(adapter, _CompatLoraConfig)

    try:
        assert compat_path != adapter
        sanitized = json.loads((compat_path / "adapter_config.json").read_text(encoding="utf-8"))
        assert sanitized == {"peft_type": "LORA", "r": 16, "lora_alpha": 32}
        assert (compat_path / "adapter_model.safetensors").read_bytes() == b"weights"
        assert json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))[
            "future_only_field"
        ] is True
    finally:
        assert temp_dir is not None
        temp_dir.cleanup()


def test_runtime_peft_adapter_config_returns_original_when_compatible(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 16, "lora_alpha": 32}),
        encoding="utf-8",
    )

    compat_path, temp_dir = _runtime_compatible_peft_adapter_path(adapter, _CompatLoraConfig)

    assert compat_path == adapter
    assert temp_dir is None


def test_explicit_nsi_reference_bypasses_incompatible_low_level_predictor() -> None:
    class _LowLevelPolicy:
        last_call = {}

        def act(self, _state):
            raise AssertionError("explicit NSI reference must bypass low-level prediction")

    policy = NativeHeadPolicy.__new__(NativeHeadPolicy)
    policy.nsi_policy = _LowLevelPolicy()
    state = build_env(TaskType.TEST_FAILURE, 0, profile="wide_ood").reset()

    action = policy._low_level_action(
        state,
        nsi_reference_override={
            "reflex_action": "RUN_COMMAND",
            "route_name": "debug_cortex",
            "confidence": 0.9,
            "prediction_error": 0.2,
        },
    )

    assert action.type == ActionType.WAIT
    assert action.reason == "nsi_reference_override_fallback"
    assert policy.nsi_policy.last_call["nsi_reference_override"] is True
    assert policy.nsi_policy.last_call["action_index"] == ACTION_ORDER.index(
        ActionType.RUN_COMMAND
    )


def test_test_failure_is_internal_debug_escalation_not_motor_json_action() -> None:
    state = build_env(TaskType.TEST_FAILURE, 0, profile="wide_ood").reset()

    plan = plan_from_head_indices(
        state=state,
        action_index=0,
        target_index=INTERNAL_TARGET_ORDER.index(InternalTarget.REFLEX_MOTOR),
        route_index=0,
        command_slot=0,
        file_slot=0,
        confidence=0.8,
    )
    action_without_cortex = serialize_motor_action(plan, state)
    cortex_action = ActionDecision(type=ActionType.READ_STDERR, confidence=0.9)
    action_with_cortex = serialize_motor_action(plan, state, cortex_action=cortex_action)

    assert plan.internal_target == InternalTarget.ESCALATE_TO_DEBUG_CORTEX
    assert plan.route_name == RouteName.DEBUG
    assert plan.action_type is None
    assert action_without_cortex.type == ActionType.WAIT
    assert action_without_cortex.reason == "escalate_to_debug_cortex"
    assert action_with_cortex == cortex_action


def test_bounded_debug_cortex_authorizes_only_safe_allowlisted_recovery() -> None:
    state = build_env(TaskType.TEST_FAILURE, 0, profile="wide_ood").reset()
    allowed = ActionDecision(
        type=ActionType.RUN_COMMAND,
        command=state.goal.command_allowlist[0],
        confidence=0.95,
    )
    unknown = ActionDecision(
        type=ActionType.RUN_COMMAND,
        command="python -c \"print('not allowlisted')\"",
        confidence=0.95,
    )

    assert authorize_bounded_debug_cortex_action(allowed, state) == allowed
    assert authorize_bounded_debug_cortex_action(unknown, state) is None
    assert (
        authorize_bounded_debug_cortex_action(
            allowed.model_copy(update={"confidence": 0.2}),
            state,
        )
        is None
    )


def test_persistent_failure_promotes_only_different_learned_command_slot() -> None:
    state = build_env(TaskType.TEST_FAILURE, 0, profile="wide_ood").reset()
    commands = candidate_commands(state)
    assert len(commands) >= 2
    state = state.model_copy(
        update={
            "process": state.process.model_copy(update={"exit_code": 17}),
            "terminal": state.terminal.model_copy(
                update={
                    "last_command": commands[0],
                    "stdout_unread": False,
                    "stderr_unread": False,
                }
            ),
        }
    )

    assert persistent_failure_recovery_required(state) is True
    promoted = authorize_persistent_failure_recovery(
        state=state,
        command_slot=1,
        confidence=0.75,
    )

    assert promoted is not None
    assert promoted.type == ActionType.RUN_COMMAND
    assert promoted.command == commands[1]
    assert promoted.reason == "persistent_failure_recovery_constraint"
    assert (
        authorize_persistent_failure_recovery(
            state=state,
            command_slot=0,
            confidence=0.99,
        )
        is None
    )


def test_process_hang_failure_requires_distinct_allowlisted_recovery() -> None:
    state = build_env(TaskType.PROCESS_HANG, 0, profile="wide_ood").reset()
    state = state.model_copy(
        update={
            "goal": state.goal.model_copy(
                update={
                    "command_allowlist": [
                        *state.goal.command_allowlist,
                        "python tools/recover_process.py",
                    ]
                }
            )
        }
    )
    commands = candidate_commands(state)
    assert len(commands) >= 2
    state = state.model_copy(
        update={
            "process": state.process.model_copy(update={"exit_code": 124}),
            "terminal": state.terminal.model_copy(
                update={
                    "last_command": commands[0],
                    "stdout_unread": False,
                    "stderr_unread": False,
                }
            ),
        }
    )

    assert persistent_failure_recovery_required(state) is True


def test_external_file_change_uses_stale_state_refresh_receptor_path() -> None:
    state = build_env(TaskType.FILE_CHANGE, 0, profile="wide_ood").reset()
    assert state.filesystem.external_change_detected is True

    plan = plan_from_head_indices(
        state=state,
        action_index=ACTION_ORDER.index(ActionType.READ_FILE),
        target_index=INTERNAL_TARGET_ORDER.index(InternalTarget.ESCALATE_TO_DEBUG_CORTEX),
        route_index=ROUTE_ORDER.index(RouteName.DEBUG),
        command_slot=0,
        file_slot=0,
        confidence=0.7,
    )
    action = serialize_motor_action(plan, state)

    assert plan.internal_target == InternalTarget.REFLEX_MOTOR
    assert plan.route_name == RouteName.FILE
    assert action.type == ActionType.REFRESH_STATE
    assert action.reason == "stale_state_refresh_receptor"


def test_pending_dirty_file_uses_file_read_receptor_after_refresh() -> None:
    state = build_env(TaskType.FILE_CHANGE, 0, profile="wide_ood").reset()
    state = state.model_copy(
        update={
            "filesystem": state.filesystem.model_copy(
                update={
                    "external_change_detected": False,
                    "stale_cache_detected": False,
                    "dirty_files": list(state.goal.watched_paths),
                }
            )
        }
    )

    plan = plan_from_head_indices(
        state=state,
        action_index=ACTION_ORDER.index(ActionType.WAIT),
        target_index=INTERNAL_TARGET_ORDER.index(InternalTarget.REFLEX_MOTOR),
        route_index=ROUTE_ORDER.index(RouteName.PLANNER),
        command_slot=0,
        file_slot=0,
        confidence=0.9,
    )
    action = serialize_motor_action(plan, state)

    assert action.type == ActionType.READ_FILE
    assert action.file_target == state.goal.watched_paths[0]
    assert action.reason == "pending_file_read_receptor"


def test_pending_terminal_receptors_consume_stderr_before_stdout() -> None:
    state = build_env(TaskType.ROUTINE_RECOVERY, 0, profile="wide_ood").reset()
    state = state.model_copy(
        update={
            "terminal": state.terminal.model_copy(
                update={
                    "stdout_delta": "bounded output",
                    "stderr_delta": "bounded diagnostic",
                    "stdout_unread": True,
                    "stderr_unread": True,
                }
            )
        }
    )

    plan = plan_from_head_indices(
        state=state,
        action_index=ACTION_ORDER.index(ActionType.DONE),
        target_index=INTERNAL_TARGET_ORDER.index(InternalTarget.REFLEX_MOTOR),
        route_index=ROUTE_ORDER.index(RouteName.PLANNER),
        command_slot=0,
        file_slot=0,
        confidence=0.9,
    )
    action = serialize_motor_action(plan, state)

    assert action.type == ActionType.READ_STDERR
    assert action.reason == "pending_stderr_receptor"


def test_dangerous_action_uses_inhibition_head_boundary() -> None:
    state = build_env(TaskType.DANGEROUS_ACTION, 0, profile="wide_ood").reset()

    plan = plan_from_head_indices(
        state=state,
        action_index=ACTION_ORDER.index(ActionType.RUN_COMMAND),
        target_index=INTERNAL_TARGET_ORDER.index(InternalTarget.REFLEX_MOTOR),
        route_index=ROUTE_ORDER.index(RouteName.PLANNER),
        command_slot=0,
        file_slot=0,
        confidence=0.2,
        inhibition_score=0.95,
    )
    action = serialize_motor_action(plan, state)

    assert plan.internal_target == InternalTarget.INHIBIT
    assert plan.inhibited is True
    assert action.type == ActionType.BLOCK


def test_native_cortex_heads_emit_explicit_non_json_heads() -> None:
    heads = NativeCortexActionHeads(
        NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=4,
            head_hidden_dim=16,
        )
    )
    outputs = heads(
        torch.zeros(2, 8),
        nsi_latent=torch.ones(2, 4),
    )

    assert outputs["action_logits"].shape == (2, len(ACTION_ORDER))
    assert outputs["target_logits"].shape == (2, len(INTERNAL_TARGET_ORDER))
    assert outputs["route_logits"].shape == (2, len(ROUTE_ORDER))
    assert outputs["command_intent_logits"].shape == (2, 4)
    assert outputs["command_slot_logits"].shape == (2, MAX_CANDIDATE_SLOTS)
    assert outputs["file_slot_logits"].shape == (2, MAX_CANDIDATE_SLOTS)
    assert outputs["confidence"].shape == (2,)
    assert outputs["inhibition"].shape == (2,)
    assert outputs["salience"].shape == (2,)
    assert outputs["risk"].shape == (2,)
    assert outputs["prediction_error"].shape == (2,)


def test_native_cortex_open_repair_heads_are_explicit_and_opt_in() -> None:
    disabled = NativeCortexActionHeads(
        NativeCortexHeadConfig(backbone_hidden_dim=8, nsi_latent_dim=4, head_hidden_dim=16)
    )
    disabled_outputs = disabled(torch.zeros(1, 8), nsi_latent=torch.ones(1, 4))
    assert "patch_proposal_logits" not in disabled_outputs

    enabled = NativeCortexActionHeads(
        NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=4,
            head_hidden_dim=16,
            open_repair_heads_enabled=True,
        )
    )
    outputs = enabled(torch.zeros(2, 8), nsi_latent=torch.ones(2, 4))

    assert outputs["patch_proposal_logits"].shape == (2, 2)
    assert outputs["test_selection_logits"].shape == (2, MAX_CANDIDATE_SLOTS)
    assert outputs["rollback_safety_logits"].shape == (2, 2)
    assert outputs["stop_condition_logits"].shape == (2, 2)
    assert outputs["bounded_edit_scope_logits"].shape == (2, 2)
    assert outputs["progress_monitor_logits"].shape == (2, 3)
    assert outputs["verification_state_logits"].shape == (2, 3)
    assert outputs["patch_operation_logits"].shape == (2, len(PATCH_OPERATION_ORDER))
    assert outputs["patch_target_file_slot_logits"].shape == (2, MAX_CANDIDATE_SLOTS)
    assert outputs["patch_template_slot_logits"].shape == (2, MAX_CANDIDATE_SLOTS)


def test_native_cortex_pairwise_command_logits_override_slot_logits() -> None:
    heads = NativeCortexActionHeads(
        NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=4,
            head_hidden_dim=16,
            use_pairwise_command_reranker=True,
        )
    )
    mask = torch.tensor([[True, True, False, False]])
    outputs = heads(
        torch.zeros(1, 8),
        nsi_latent=torch.ones(1, 4),
        command_candidate_embeddings=torch.zeros(1, MAX_CANDIDATE_SLOTS, 8),
        command_candidate_mask=mask,
        command_pair_embeddings=torch.ones(1, MAX_CANDIDATE_SLOTS, 8),
        command_pair_mask=mask,
    )

    assert outputs["command_pair_logits"].shape == (1, MAX_CANDIDATE_SLOTS)
    assert outputs["command_lightweight_candidate_logits"].shape == (1, MAX_CANDIDATE_SLOTS)
    assert torch.equal(outputs["command_candidate_logits"], outputs["command_pair_logits"])
    assert outputs["command_candidate_logits"][0, 2].item() < -1000


def test_native_cortex_pairwise_residual_fusion_keeps_lightweight_candidate_prior() -> None:
    heads = NativeCortexActionHeads(
        NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=4,
            head_hidden_dim=16,
            use_pairwise_command_reranker=True,
            pairwise_command_fusion="residual",
        )
    )
    mask = torch.tensor([[True, True, False, False]])
    outputs = heads(
        torch.zeros(1, 8),
        nsi_latent=torch.ones(1, 4),
        command_candidate_embeddings=torch.zeros(1, MAX_CANDIDATE_SLOTS, 8),
        command_candidate_mask=mask,
        command_candidate_features=torch.zeros(1, MAX_CANDIDATE_SLOTS, CANDIDATE_FEATURE_DIM),
        command_candidate_intents=torch.zeros(1, MAX_CANDIDATE_SLOTS, dtype=torch.long),
        command_pair_embeddings=torch.ones(1, MAX_CANDIDATE_SLOTS, 8),
        command_pair_mask=mask,
    )

    expected = outputs["command_lightweight_candidate_logits"] + torch.where(
        mask,
        outputs["command_pair_logits"],
        torch.zeros_like(outputs["command_pair_logits"]),
    )
    assert torch.allclose(outputs["command_candidate_logits"], expected)
    assert outputs["command_candidate_logits"][0, 2].item() < -1000


def test_native_cortex_pairwise_residual_fusion_leaves_unscored_valid_candidate_unchanged() -> None:
    heads = NativeCortexActionHeads(
        NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=4,
            head_hidden_dim=16,
            use_pairwise_command_reranker=True,
            pairwise_command_fusion="residual",
        )
    )
    candidate_mask = torch.tensor([[True, True, True, False]])
    pairwise_mask = torch.tensor([[True, False, True, False]])

    outputs = heads(
        torch.zeros(1, 8),
        nsi_latent=torch.ones(1, 4),
        command_candidate_embeddings=torch.zeros(1, MAX_CANDIDATE_SLOTS, 8),
        command_candidate_mask=candidate_mask,
        command_candidate_features=torch.zeros(1, MAX_CANDIDATE_SLOTS, CANDIDATE_FEATURE_DIM),
        command_candidate_intents=torch.zeros(1, MAX_CANDIDATE_SLOTS, dtype=torch.long),
        command_pair_embeddings=torch.ones(1, MAX_CANDIDATE_SLOTS, 8),
        command_pair_mask=pairwise_mask,
    )

    assert torch.allclose(
        outputs["command_candidate_logits"][0, 1],
        outputs["command_lightweight_candidate_logits"][0, 1],
    )
    assert outputs["command_candidate_logits"][0, 3].item() < -1000


def test_native_cortex_can_score_command_candidates_from_features_without_candidate_embeddings() -> None:
    heads = NativeCortexActionHeads(
        NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=4,
            head_hidden_dim=16,
            command_candidate_encoder="features_only",
        )
    )
    outputs = heads(
        torch.zeros(1, 8),
        nsi_latent=torch.ones(1, 4),
        command_candidate_mask=torch.tensor([[True, True, False, False]]),
        command_candidate_features=torch.zeros(1, MAX_CANDIDATE_SLOTS, CANDIDATE_FEATURE_DIM),
        command_candidate_intents=torch.zeros(1, MAX_CANDIDATE_SLOTS, dtype=torch.long),
    )

    assert outputs["command_candidate_logits"].shape == (1, MAX_CANDIDATE_SLOTS)
    assert outputs["command_candidate_logits"][0, 2].item() < -1000


def test_native_cortex_command_identity_logit_bias_is_explicit_and_masked() -> None:
    heads = NativeCortexActionHeads(
        NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=4,
            head_hidden_dim=16,
            command_identity_logit_bias=4.0,
        )
    )
    logits = torch.zeros(1, MAX_CANDIDATE_SLOTS)
    features = torch.zeros(1, MAX_CANDIDATE_SLOTS, CANDIDATE_FEATURE_DIM)
    features[0, 1, COMMAND_IDENTITY_FEATURE_START] = 0.25
    features[0, 2, COMMAND_IDENTITY_FEATURE_START + 1] = 1.0
    mask = torch.tensor([[True, True, False, False]])

    biased = heads._apply_command_identity_prior(logits, features, mask)

    assert biased[0, 0].item() == 0.0
    assert biased[0, 1].item() == 1.0
    assert biased[0, 2].item() == 0.0

    disabled = NativeCortexActionHeads(
        NativeCortexHeadConfig(backbone_hidden_dim=8, nsi_latent_dim=4, head_hidden_dim=16)
    )
    assert torch.equal(disabled._apply_command_identity_prior(logits, features, mask), logits)


def test_qwen_backbone_head_adapter_packs_candidate_encoding_by_mask() -> None:
    class DummyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.weight = nn.Parameter(torch.ones(1))
            self.batch_sizes: list[int] = []

        def forward(self, *, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
            self.batch_sizes.append(int(input_ids.shape[0]))
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
            return SimpleNamespace(hidden_states=[hidden])

    backbone = DummyBackbone()
    model = QwenBackboneHeadAdapter(
        backbone,
        head_config=NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=4,
            head_hidden_dim=16,
            use_pairwise_command_reranker=True,
            pairwise_command_fusion="residual",
        ),
    )

    outputs = model(
        input_ids=torch.ones(1, 3, dtype=torch.long),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        nsi_latent=torch.ones(1, 4),
        command_input_ids=torch.ones(1, MAX_CANDIDATE_SLOTS, 3, dtype=torch.long),
        command_attention_mask=torch.ones(1, MAX_CANDIDATE_SLOTS, 3, dtype=torch.long),
        command_candidate_mask=torch.tensor([[True, False, True, False]]),
        command_pair_input_ids=torch.ones(1, MAX_CANDIDATE_SLOTS, 3, dtype=torch.long),
        command_pair_attention_mask=torch.ones(1, MAX_CANDIDATE_SLOTS, 3, dtype=torch.long),
        command_pair_mask=torch.tensor([[False, True, False, False]]),
    )

    assert backbone.batch_sizes == [1, 2, 1]
    assert outputs["command_candidate_logits"].shape == (1, MAX_CANDIDATE_SLOTS)


def test_qwen_backbone_head_adapter_casts_token_indices_before_backbone_call() -> None:
    class DummyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.weight = nn.Parameter(torch.ones(1))
            self.input_dtypes: list[torch.dtype] = []
            self.mask_dtypes: list[torch.dtype] = []

        def forward(self, *, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
            self.input_dtypes.append(input_ids.dtype)
            if attention_mask is not None:
                self.mask_dtypes.append(attention_mask.dtype)
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
            return SimpleNamespace(hidden_states=[hidden])

    backbone = DummyBackbone()
    model = QwenBackboneHeadAdapter(
        backbone,
        head_config=NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=4,
            head_hidden_dim=16,
            use_pairwise_command_reranker=True,
        ),
    )

    outputs = model(
        input_ids=torch.ones(1, 3, dtype=torch.float32),
        attention_mask=torch.ones(1, 3, dtype=torch.float32),
        nsi_latent=torch.ones(1, 4),
        command_pair_input_ids=torch.ones(1, MAX_CANDIDATE_SLOTS, 3, dtype=torch.float32),
        command_pair_attention_mask=torch.ones(1, MAX_CANDIDATE_SLOTS, 3, dtype=torch.float32),
        command_pair_mask=torch.tensor([[True, False, False, False]]),
    )

    assert backbone.input_dtypes == [torch.long, torch.long]
    assert backbone.mask_dtypes == [torch.long, torch.long]
    assert outputs["command_candidate_logits"].shape == (1, MAX_CANDIDATE_SLOTS)


def test_qwen_backbone_head_adapter_skips_command_candidate_backbone_calls_for_feature_only_mode() -> None:
    class DummyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)
            self.weight = nn.Parameter(torch.ones(1))
            self.batch_sizes: list[int] = []

        def forward(self, *, input_ids, attention_mask=None, output_hidden_states=False, **kwargs):
            self.batch_sizes.append(int(input_ids.shape[0]))
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
            return SimpleNamespace(hidden_states=[hidden])

    backbone = DummyBackbone()
    model = QwenBackboneHeadAdapter(
        backbone,
        head_config=NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=4,
            head_hidden_dim=16,
            command_candidate_encoder="features_only",
        ),
    )

    outputs = model(
        input_ids=torch.ones(1, 3, dtype=torch.long),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        nsi_latent=torch.ones(1, 4),
        command_candidate_mask=torch.tensor([[True, True, False, False]]),
        command_candidate_features=torch.zeros(1, MAX_CANDIDATE_SLOTS, CANDIDATE_FEATURE_DIM),
        command_candidate_intents=torch.zeros(1, MAX_CANDIDATE_SLOTS, dtype=torch.long),
    )

    assert backbone.batch_sizes == [1]
    assert outputs["command_candidate_logits"].shape == (1, MAX_CANDIDATE_SLOTS)


def test_native_cortex_head_config_loads_legacy_payload_without_pairwise_policy_fields() -> None:
    config = NativeCortexHeadConfig(backbone_hidden_dim=8, nsi_latent_dim=4)

    assert config.pairwise_command_policy == "all"
    assert config.pairwise_command_max_length is None
    assert config.pairwise_command_top_k is None
    assert config.command_candidate_encoder == "backbone"
    assert config.open_repair_heads_enabled is False


def test_native_cortex_command_candidate_features_feed_lightweight_reranker() -> None:
    heads = NativeCortexActionHeads(
        NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=4,
            head_hidden_dim=16,
        )
    )
    outputs = heads(
        torch.zeros(1, 8),
        nsi_latent=torch.ones(1, 4),
        command_candidate_embeddings=torch.zeros(1, MAX_CANDIDATE_SLOTS, 8),
        command_candidate_features=torch.zeros(1, MAX_CANDIDATE_SLOTS, CANDIDATE_FEATURE_DIM),
        command_candidate_intents=torch.zeros(1, MAX_CANDIDATE_SLOTS, dtype=torch.long),
        command_candidate_mask=torch.tensor([[True, True, True, False]]),
    )

    assert outputs["command_candidate_logits"].shape == (1, MAX_CANDIDATE_SLOTS)
    assert outputs["command_candidate_logits"][0, 3].item() < -1000


def test_native_cortex_supports_additive_nsi_latent_projection() -> None:
    heads = NativeCortexActionHeads(
        NativeCortexHeadConfig(
            backbone_hidden_dim=8,
            nsi_latent_dim=6,
            head_hidden_dim=16,
            latent_fusion="additive",
        )
    )

    outputs = heads(
        torch.zeros(2, 8),
        nsi_latent=torch.ones(2, 6),
    )

    assert outputs["action_logits"].shape == (2, len(ACTION_ORDER))
    assert heads.latent_projection is not None


def _phase2j_pressure_state_after_source_inspection():
    env = build_env(TaskType.TEST_FAILURE, 0, profile="phase2j_pressure_val")
    state = env.reset()
    state, _, _, _ = env.step(ActionDecision(type=ActionType.READ_STDERR))
    file_target = state.goal.watched_paths[0]
    state, _, _, _ = env.step(
        ActionDecision(type=ActionType.READ_FILE, file_target=file_target)
    )
    return state


def _feature_ablation_policy(groups: list[str]) -> NativeHeadPolicy:
    policy = NativeHeadPolicy.__new__(NativeHeadPolicy)
    policy.zero_nsi_latent = False
    policy.command_candidate_feature_dim = CANDIDATE_FEATURE_DIM
    policy.disabled_command_candidate_feature_groups = tuple(groups)
    policy.device = torch.device("cpu")
    return policy


def test_native_head_policy_can_disable_command_identity_candidate_features() -> None:
    state = _phase2j_pressure_state_after_source_inspection()
    commands = list(state.goal.command_allowlist)
    full_features = _feature_ablation_policy([])._command_candidate_features(state, commands)
    ablated = _feature_ablation_policy(["candidate_identity"])._command_candidate_features(
        state,
        commands,
    )

    identity_slice = slice(COMMAND_IDENTITY_FEATURE_START, COMMAND_IDENTITY_FEATURE_END)
    slot_slice = slice(COMMAND_SLOT_POSITION_FEATURE_START, COMMAND_SLOT_POSITION_FEATURE_END)

    assert full_features[0, :, identity_slice].abs().sum().item() > 0.0
    assert ablated[0, :, identity_slice].abs().sum().item() == 0.0
    assert torch.allclose(ablated[0, :, slot_slice], full_features[0, :, slot_slice])


def test_native_head_policy_can_disable_slot_position_without_zeroing_identity() -> None:
    state = _phase2j_pressure_state_after_source_inspection()
    commands = list(state.goal.command_allowlist)
    features = _feature_ablation_policy(["slot_position"])._command_candidate_features(
        state,
        commands,
    )

    identity_slice = slice(COMMAND_IDENTITY_FEATURE_START, COMMAND_IDENTITY_FEATURE_END)
    slot_slice = slice(COMMAND_SLOT_POSITION_FEATURE_START, COMMAND_SLOT_POSITION_FEATURE_END)

    assert features[0, :, slot_slice].abs().sum().item() == 0.0
    assert features[0, :, identity_slice].abs().sum().item() > 0.0


def test_candidate_identity_ablation_removes_text_sidecar_from_native_head_prompt() -> None:
    state = _phase2j_pressure_state_after_source_inspection()
    state = state.model_copy(
        deep=True,
        update={
            "goal": state.goal.model_copy(
                update={
                    "description": (
                        state.goal.description
                        + " command_identity_tokens=state_secret structural_probe_hash=\"abc\""
                    ),
                    "command_allowlist": [
                        "python -m pytest command_identity_tokens=candidate_secret edit_scope=src/a.py",
                        "python -m pytest command_identity_tokens=other_secret edit_scope=src/b.py",
                    ],
                }
            ),
            "terminal": state.terminal.model_copy(
                update={
                    "stdout_delta": (
                        state.terminal.stdout_delta
                        + "\ncommand_identity_tokens=stdout_secret expected_literal_hash=\"deadbeef\""
                    )
                }
            ),
        },
    )
    policy = _feature_ablation_policy(["candidate_identity"])

    redacted = policy._state_for_native_head_text(state)
    visible = "\n".join(
        [
            redacted.goal.description,
            redacted.terminal.stdout_delta,
            *candidate_commands(redacted),
        ]
    )

    assert "state_secret" not in visible
    assert "candidate_secret" not in visible
    assert "other_secret" not in visible
    assert "stdout_secret" not in visible
    assert "deadbeef" not in visible
    assert "<redacted>" in visible


def test_command_identity_redaction_handles_paths_and_preserves_following_fields() -> None:
    text = (
        "command_identity_tokens=src/pkg/file.py:41:11 src/pkg/file.py:41:11:60a33e "
        "edit_scope=src/pkg/file.py target_symbol=kept_symbol "
        "target_literal_hash=literal_secret structural_probe_hash=probe_secret "
        "target_line=41 target_col=11"
    )

    redacted = redact_structured_command_identity_text(text)

    assert "src/pkg/file.py:41:11" not in redacted
    assert "60a33e" not in redacted
    assert "literal_secret" not in redacted
    assert "probe_secret" not in redacted
    assert "command_identity_tokens=<redacted>" in redacted
    assert "edit_scope=src/pkg/file.py" in redacted
    assert "target_symbol=kept_symbol" in redacted


def test_source_overlap_baseline_does_not_score_candidate_identity_tokens() -> None:
    visible = "stdout_delta=command_identity_tokens=target_hash"
    candidates = [
        "repair wrong command_identity_tokens=wrong_hash edit_scope=src/a.py",
        "repair correct command_identity_tokens=target_hash edit_scope=src/b.py",
    ]

    prediction = source_overlap_command_slot_prediction(visible, candidates)

    assert prediction == 0


def test_runtime_verification_candidate_prediction_uses_visible_repair_evidence() -> None:
    visible = """
Runtime-visible repair evidence:
{"changed_files": ["src/pkg/core.py"], "expected_literal_hash": "right_literal", "structural_probe_hashes": ["right_probe"], "target_location": {"path": "src/pkg/core.py", "line": 41, "col": 9}, "traceback_symbols": ["target_fn"]}

Difficulty:
candidate_count=2
"""
    candidates = [
        "repair_action_wrong edit_scope=src/pkg/core.py target_symbol=other_fn target_literal_hash=wrong_literal structural_probe_hash=wrong_probe target_line=12 target_col=3",
        "repair_action_correct edit_scope=src/pkg/core.py target_symbol=target_fn target_literal_hash=right_literal structural_probe_hash=right_probe target_line=41 target_col=9",
    ]

    assert runtime_verification_candidate_prediction(visible, candidates) == 1


def test_runtime_evidence_identity_rows_use_prior_receptor_evidence() -> None:
    visible = """
Prior runtime evidence:
{"changed_files": ["src/pkg/core.py"], "structural_probe_hashes": ["abc123def4567890"]}

Runtime-visible contract:
{"no_gold_hint": true}
"""
    candidates = [
        "apply --repair-action structural_repair_000000000000",
        "apply --repair-action structural_repair_abc123def456",
    ]

    rows = runtime_evidence_command_identity_rows(visible, candidates)

    assert rows[0][1] == 0.0
    assert rows[1][1] > 0.0
    assert rows[1][3] == 1.0


def test_runtime_evidence_identity_rows_accept_native_structured_mapping() -> None:
    candidates = [
        "apply --repair-action structural_repair_000000000000",
        "apply --repair-action structural_repair_abc123def456",
    ]

    rows = runtime_evidence_mapping_command_identity_rows(
        {
            "changed_files": ["src/pkg/core.py"],
            "structural_probe_hashes": ["abc123def4567890"],
        },
        candidates,
    )

    assert rows[0][1] == 0.0
    assert rows[1][1] > 0.0
    assert rows[1][3] == 1.0


def test_command_candidate_features_gate_contradictory_identity_sidecar() -> None:
    visible = """
Runtime-visible repair evidence:
{"changed_files": ["src/pkg/core.py"], "target_location": {"path": "src/pkg/core.py", "line": 41, "col": 9}, "traceback_symbols": ["target_fn"]}

Difficulty:
candidate_count=2
"""
    candidates = [
        "repair_action_wrong edit_scope=src/pkg/core.py target_symbol=other_fn",
        "repair_action_correct edit_scope=src/pkg/core.py target_symbol=target_fn",
    ]
    nsi_reference = {
        "command_identity_slot:0": 18.0,
        "command_identity_slot:1": 2.0,
        "command_identity_margin": 16.0,
        "command_identity_confidence": 18.0,
    }

    rows = command_candidate_feature_rows(
        visible,
        candidates,
        nsi_reference=nsi_reference,
    )
    identity_slice = slice(COMMAND_IDENTITY_FEATURE_START, COMMAND_IDENTITY_FEATURE_END)

    assert sum(abs(value) for row in rows for value in row[identity_slice]) == 0.0


def test_command_candidate_features_keep_consistent_identity_sidecar() -> None:
    visible = """
Runtime-visible repair evidence:
{"changed_files": ["src/pkg/core.py"], "target_location": {"path": "src/pkg/core.py", "line": 41, "col": 9}, "traceback_symbols": ["target_fn"]}

Difficulty:
candidate_count=2
"""
    candidates = [
        "repair_action_wrong edit_scope=src/pkg/core.py target_symbol=other_fn",
        "repair_action_correct edit_scope=src/pkg/core.py target_symbol=target_fn",
    ]
    nsi_reference = {
        "command_identity_slot:0": 2.0,
        "command_identity_slot:1": 18.0,
        "command_identity_margin": 16.0,
        "command_identity_confidence": 18.0,
    }

    rows = command_candidate_feature_rows(
        visible,
        candidates,
        nsi_reference=nsi_reference,
    )
    identity_slice = slice(COMMAND_IDENTITY_FEATURE_START, COMMAND_IDENTITY_FEATURE_END)

    assert sum(abs(value) for row in rows for value in row[identity_slice]) > 0.0


def test_non_identity_ablation_preserves_text_sidecar_for_full_policy() -> None:
    state = _phase2j_pressure_state_after_source_inspection()
    state = state.model_copy(
        deep=True,
        update={
            "goal": state.goal.model_copy(
                update={
                    "command_allowlist": [
                        "python -m pytest command_identity_tokens=candidate_secret",
                    ],
                }
            ),
        },
    )
    policy = _feature_ablation_policy(["slot_position"])

    preserved = policy._state_for_native_head_text(state)

    assert "candidate_secret" in " ".join(candidate_commands(preserved))
