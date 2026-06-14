import json
from pathlib import Path

import pytest
import torch

from reflexlm.llm.native_head_training import (
    DEBUG_ACTION_STAGE_GAIN,
    DESCRIPTOR_FAILURE_FAMILY_GAIN,
    NativeHeadTrainConfig,
    Phase2CHeadJsonlDataset,
    NSI_LATENT_FIELDS,
    _balanced_limited_rows,
    _balance_command_slot_rows,
    _balance_debug_command_intent_rows,
    _balance_patch_descriptor_rows,
    _assert_requested_device_available,
    _canonical_rows_sha256,
    _checkpoint_path,
    _collate_head_rows,
    _command_slot_baseline_metrics,
    _evaluate_native_head_model,
    _finalize_action_accuracy_by_target,
    _native_head_training_identity_hash,
    _open_repair_training_contract,
    _pairwise_candidate_encoding_stats,
    _patch_descriptor_distribution,
    _oversample_debug_command_rows,
    _update_action_accuracy_by_target,
    compute_native_head_loss,
    nsi_latent_values,
)
from reflexlm.llm.head_dataset import build_phase2c_head_state_prompt_from_state
from reflexlm.llm.receptor_latent import (
    COMMAND_IDENTITY_LATENT_FIELDS,
    DESCRIPTOR_FAILURE_FAMILY_LATENT_FIELDS,
    DEBUG_ACTION_STAGE_LATENT_FIELDS,
    debug_action_stage_signal,
    runtime_command_identity_signal,
)
from reflexlm.llm.candidate_features import (
    CANDIDATE_FEATURE_DIM,
    COMMAND_IDENTITY_CANDIDATE_FEATURE_DIM,
    COMMAND_SLOT_POSITION_FEATURE_DIM,
    COMMAND_INTENT_COUNT,
    build_candidate_pair_prompt,
    command_candidate_feature_rows,
    command_candidate_source_overlap_rows,
    pairwise_command_candidate_mask,
    compact_visible_state_for_candidate_pair,
    source_overlap_command_slot_prediction,
)
from reflexlm.data.tasks import build_env
from reflexlm.models.features import ACTION_ORDER, MAX_CANDIDATE_SLOTS, ROUTE_ORDER
from reflexlm.runtime.nervous_system import INTERNAL_TARGET_ORDER
from reflexlm.schema import ActionDecision, ActionType, TaskType


class FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=True,
        max_length=32,
    ):
        tokens = list(range(1, min(len(text.split()) + 1, max_length + 1)))
        if add_special_tokens:
            tokens = [99] + tokens
        tokens = tokens[:max_length]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


def _row(**overrides):
    payload = {
        "state_prompt": "Phase 2C receptor state only",
        "candidate_commands": [],
        "candidate_files": [],
        "nsi_reference": {
            "salience": 0.7,
            "risk": 0.2,
            "prediction_error": 0.1,
            "confidence": 0.9,
        },
        "action_index": 0,
        "internal_target_index": 0,
        "route_index": 0,
        "command_intent": None,
        "command_slot": -100,
        "file_slot": -100,
        "confidence_target": 1.0,
        "inhibition_target": 0.0,
        "salience_target": 0.5,
        "risk_target": 0.2,
        "prediction_error_target": 0.1,
    }
    payload.update(overrides)
    return payload


def test_requested_cuda_device_fails_fast_when_torch_is_cpu_only(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        _assert_requested_device_available("cuda")


def test_requested_cpu_device_allows_cpu_only_torch(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    _assert_requested_device_available("cpu")


def test_action_accuracy_by_target_records_confusion() -> None:
    buckets = {}
    _update_action_accuracy_by_target(
        buckets,
        predictions=torch.tensor([0, 2, 2]),
        labels=torch.tensor([0, 1, 2]),
    )

    report = _finalize_action_accuracy_by_target(buckets)

    assert report[ACTION_ORDER[0].value]["accuracy"] == 1.0
    assert report[ACTION_ORDER[1].value]["accuracy"] == 0.0
    assert report[ACTION_ORDER[1].value]["predictions"] == {ACTION_ORDER[2].value: 1}
    assert report[ACTION_ORDER[2].value]["accuracy"] == 1.0


def test_phase2c_native_head_collate_uses_sidecar_latent_not_target_text() -> None:
    batch = _collate_head_rows(FakeTokenizer(), [_row(), _row(action_index=1)], max_length=16)

    assert batch["input_ids"].shape[0] == 2
    assert batch["nsi_latent"].shape == (2, len(NSI_LATENT_FIELDS))
    assert torch.allclose(batch["nsi_latent"][0, :4], torch.tensor([0.7, 0.2, 0.1, 0.9]))
    assert "target_text" not in batch


def test_phase2c_native_head_loss_handles_empty_slot_labels() -> None:
    batch = _collate_head_rows(FakeTokenizer(), [_row(), _row(action_index=1)], max_length=16)
    outputs = {
        "action_logits": torch.zeros(2, len(ACTION_ORDER), requires_grad=True),
        "target_logits": torch.zeros(2, len(INTERNAL_TARGET_ORDER), requires_grad=True),
        "route_logits": torch.zeros(2, len(ROUTE_ORDER), requires_grad=True),
        "command_intent_logits": torch.zeros(2, COMMAND_INTENT_COUNT, requires_grad=True),
        "command_slot_logits": torch.zeros(2, MAX_CANDIDATE_SLOTS, requires_grad=True),
        "file_slot_logits": torch.zeros(2, MAX_CANDIDATE_SLOTS, requires_grad=True),
        "confidence": torch.full((2,), 0.5, requires_grad=True),
        "inhibition": torch.full((2,), 0.5, requires_grad=True),
        "salience": torch.full((2,), 0.5, requires_grad=True),
        "risk": torch.full((2,), 0.5, requires_grad=True),
        "prediction_error": torch.full((2,), 0.5, requires_grad=True),
    }

    loss, components = compute_native_head_loss(outputs, batch)

    assert torch.isfinite(loss)
    assert components["command_slot"] == 0.0
    assert components["file_slot"] == 0.0


def test_phase2c_native_head_loss_handles_optional_open_repair_labels() -> None:
    batch = _collate_head_rows(
        FakeTokenizer(),
        [
            _row(
                patch_proposal_label=1,
                test_selection_slot=0,
                rollback_safety_label=1,
                stop_condition_label=0,
                bounded_edit_scope_label=1,
                progress_monitor_label=2,
                verification_state_label=1,
            )
        ],
        max_length=16,
    )
    outputs = {
        "action_logits": torch.zeros(1, len(ACTION_ORDER), requires_grad=True),
        "target_logits": torch.zeros(1, len(INTERNAL_TARGET_ORDER), requires_grad=True),
        "route_logits": torch.zeros(1, len(ROUTE_ORDER), requires_grad=True),
        "command_intent_logits": torch.zeros(1, COMMAND_INTENT_COUNT, requires_grad=True),
        "command_slot_logits": torch.zeros(1, MAX_CANDIDATE_SLOTS, requires_grad=True),
        "file_slot_logits": torch.zeros(1, MAX_CANDIDATE_SLOTS, requires_grad=True),
        "confidence": torch.full((1,), 0.5, requires_grad=True),
        "inhibition": torch.full((1,), 0.5, requires_grad=True),
        "salience": torch.full((1,), 0.5, requires_grad=True),
        "risk": torch.full((1,), 0.5, requires_grad=True),
        "prediction_error": torch.full((1,), 0.5, requires_grad=True),
        "patch_proposal_logits": torch.zeros(1, 2, requires_grad=True),
        "test_selection_logits": torch.zeros(1, MAX_CANDIDATE_SLOTS, requires_grad=True),
        "rollback_safety_logits": torch.zeros(1, 2, requires_grad=True),
        "stop_condition_logits": torch.zeros(1, 2, requires_grad=True),
        "bounded_edit_scope_logits": torch.zeros(1, 2, requires_grad=True),
        "progress_monitor_logits": torch.zeros(1, 3, requires_grad=True),
        "verification_state_logits": torch.zeros(1, 3, requires_grad=True),
    }

    loss, components = compute_native_head_loss(outputs, batch)

    assert torch.isfinite(loss)
    assert components["patch_proposal"] > 0.0
    assert components["test_selection"] > 0.0
    assert components["rollback_safety"] > 0.0
    assert components["stop_condition"] > 0.0
    assert components["bounded_edit_scope"] > 0.0
    assert components["progress_monitor"] > 0.0
    assert components["verification_state"] > 0.0


def test_phase2c_training_contract_records_learned_bounded_targets() -> None:
    config = NativeHeadTrainConfig(
        base_model_name="qwen",
        adapter_name="adapter",
        open_repair_heads_enabled=True,
    )

    contract = _open_repair_training_contract(config)

    assert contract["sealed_feedback_used"] is False
    assert contract["learned_patch_candidate_targets"] is True
    assert contract["recorded_patch_artifact_as_generation_target"] is False
    assert contract["symbolic_generator_as_generation_target"] is False
    assert contract["freeform_patch_text_target"] is False
    assert contract["json_text_target"] is False
    assert contract["low_level_qwen_calls_target"] == 0
    assert contract["patch_proposal_strategy"] == "learned_bounded_candidate"


def test_phase2c_native_head_collate_adds_candidate_command_tensors() -> None:
    batch = _collate_head_rows(
        FakeTokenizer(),
        [
            _row(
                command_slot=2,
                candidate_commands=[
                    "python -m pytest -q tests/test_auth.py",
                    "python -m pip install -r requirements.txt",
                    "python -m pytest -q tests/test_snapshots.py --snapshot-update",
                ],
                nsi_reference={
                    "command_identity_slot:0": 0.0,
                    "command_identity_slot:1": 0.0,
                    "command_identity_slot:2": 0.9,
                    "command_identity_slot:3": 0.0,
                    "command_identity_margin": 0.9,
                    "command_identity_confidence": 0.9,
                },
            )
        ],
        max_length=32,
    )

    identity_start = CANDIDATE_FEATURE_DIM - COMMAND_IDENTITY_CANDIDATE_FEATURE_DIM
    assert batch["command_input_ids"].shape[:2] == (1, MAX_CANDIDATE_SLOTS)
    assert batch["command_candidate_mask"].tolist() == [[True, True, True, False]]
    assert batch["command_candidate_features"].shape == (1, MAX_CANDIDATE_SLOTS, CANDIDATE_FEATURE_DIM)
    assert torch.isclose(
        batch["command_candidate_features"][0, 2, identity_start],
        torch.tensor(0.9),
    )
    assert batch["command_candidate_intents"].shape == (1, MAX_CANDIDATE_SLOTS)


def test_command_candidate_features_include_slot_position_and_nsi_identity_without_prompt_leak() -> None:
    candidates = [
        "python -m pytest -q tests/phase2j_hard_val/redaction/test_command_identity_redaction.py::test_owner_cartographer",
        "python -m pytest -q tests/phase2j_hard_val/redaction/test_command_identity_redaction.py::test_owner_harbor",
        "python -m pytest -q tests/phase2j_hard_val/redaction/test_command_identity_redaction.py::test_owner_quartz",
        "python -m pytest -q tests/phase2j_hard_val/redaction/test_command_identity_redaction.py::test_owner_ember",
    ]
    prompt = (
        "stdout_delta=Source inspected: semantic disambiguation required. "
        "command_identity_tokens=<redacted>."
    )
    nsi_reference = {
        "command_identity_slot:0": 0.0,
        "command_identity_slot:1": 0.75,
        "command_identity_slot:2": 0.0,
        "command_identity_slot:3": 0.0,
        "command_identity_margin": 0.75,
        "command_identity_confidence": 0.75,
    }

    rows = command_candidate_feature_rows(prompt, candidates, nsi_reference=nsi_reference)

    position_start = CANDIDATE_FEATURE_DIM - COMMAND_IDENTITY_CANDIDATE_FEATURE_DIM - COMMAND_SLOT_POSITION_FEATURE_DIM
    identity_start = CANDIDATE_FEATURE_DIM - COMMAND_IDENTITY_CANDIDATE_FEATURE_DIM
    assert rows[1][position_start : position_start + COMMAND_SLOT_POSITION_FEATURE_DIM] == [
        0.0,
        1.0,
        0.0,
        0.0,
    ]
    assert rows[1][identity_start:] == [0.75, 1.0, 0.75, 0.75]
    assert rows[0][identity_start:] == [0.0, 0.0, 0.0, 0.0]


def test_phase2c_native_head_collate_can_skip_command_backbone_candidate_encoding() -> None:
    batch = _collate_head_rows(
        FakeTokenizer(),
        [
            _row(
                command_slot=1,
                candidate_commands=[
                    "python -m pytest -q tests/test_auth.py",
                    "python -m pytest -q tests/test_auth.py::test_login_redirect",
                ],
            )
        ],
        max_length=32,
        command_candidate_encoder="features_only",
    )

    assert "command_input_ids" not in batch
    assert "command_attention_mask" not in batch
    assert batch["command_candidate_mask"].tolist() == [[True, True, False, False]]
    assert batch["command_candidate_features"].shape == (1, MAX_CANDIDATE_SLOTS, CANDIDATE_FEATURE_DIM)
    assert batch["command_candidate_intents"].shape == (1, MAX_CANDIDATE_SLOTS)


def test_command_candidate_features_ignore_candidate_section_self_overlap() -> None:
    prompt = "\n".join(
        [
            "Visible transition summary:",
            "source inspected archive hash coverage should rerun archive manifest test",
            "",
            "Candidate commands:",
            "- python -m pytest -q tests/test_dataset_generation.py::test_external_trace_generation_seals_and_refuses_overwrite",
            "- python -m pytest -q tests/test_phase2f_archive_and_tables.py::test_archive_manifest_records_all_required_hashes",
            "",
            "Candidate files:",
            "- src/reflexlm/cli/archive_phase2f_evidence.py",
        ]
    )
    rows = command_candidate_feature_rows(
        prompt,
        [
            "python -m pytest -q tests/test_dataset_generation.py::test_external_trace_generation_seals_and_refuses_overwrite",
            "python -m pytest -q tests/test_phase2f_archive_and_tables.py::test_archive_manifest_records_all_required_hashes",
        ],
    )

    # Index 8 is candidate-token overlap. The dataset command appears only in
    # the candidate section, so it should not win via self-overlap.
    assert rows[1][8] > rows[0][8]


def test_command_candidate_features_add_source_evidence_overlap_without_last_command_bias() -> None:
    prompt = "\n".join(
        [
            "failure_signal=other",
            "source_inspected=True",
            (
                "stdout_delta=Source inspected: semantic disambiguation required. "
                "Source-visible selected test terms: test billing invoice rules py "
                "test_invoice_total_rounds_tax."
            ),
            "last_command=python -m pytest -q tests/billing/test_invoice_rules.py::test_invoice_total_uses_discount",
            "",
            "Candidate commands:",
            "- python -m pytest -q tests/billing/test_invoice_rules.py::test_invoice_total_uses_discount",
            "- python -m pytest -q tests/billing/test_invoice_rules.py::test_invoice_total_rounds_tax",
            "",
            "Head constraints:",
            "- RUN_COMMAND must select a command slot.",
        ]
    )
    candidates = [
        "python -m pytest -q tests/billing/test_invoice_rules.py::test_invoice_total_uses_discount",
        "python -m pytest -q tests/billing/test_invoice_rules.py::test_invoice_total_rounds_tax",
    ]

    rows = command_candidate_feature_rows(prompt, candidates)
    source_rows = command_candidate_source_overlap_rows(prompt, candidates)

    assert len(rows[0]) == CANDIDATE_FEATURE_DIM
    assert source_rows[1][1] > source_rows[0][1]
    assert source_overlap_command_slot_prediction(prompt, candidates) == 1


def test_runtime_command_identity_signal_uses_visible_evidence_not_candidate_self_overlap() -> None:
    state = build_env(TaskType.TEST_FAILURE, 0, profile="phase2i_semantic_val").reset()
    state = state.model_copy(
        update={
            "goal": state.goal.model_copy(
                update={
                    "command_allowlist": [
                        "python -m pytest -q tests/billing/test_invoice_rules.py::test_invoice_total_uses_discount",
                        "python -m pytest -q tests/billing/test_invoice_rules.py::test_invoice_total_rounds_tax",
                    ]
                }
            ),
            "terminal": state.terminal.model_copy(
                update={
                    "stdout_delta": (
                        "Source inspected: semantic disambiguation required. "
                        "The failure is in invoice total rounds tax."
                    ),
                    "last_command": (
                        "python -m pytest -q "
                        "tests/billing/test_invoice_rules.py::test_invoice_total_uses_discount"
                    ),
                }
            ),
        }
    )

    signal = runtime_command_identity_signal(state)

    assert signal["command_identity_slot:1"] > signal["command_identity_slot:0"]
    assert signal["command_identity_confidence"] > 0.0
    assert signal["command_identity_margin"] > 0.0


def test_runtime_command_identity_signal_uses_structured_sidecar_not_source_overlap() -> None:
    candidates = [
        "python -m pytest -q tests/phase2j_hard_val/redaction/test_command_identity_redaction.py::test_owner_cartographer",
        "python -m pytest -q tests/phase2j_hard_val/redaction/test_command_identity_redaction.py::test_owner_harbor",
        "python -m pytest -q tests/phase2j_hard_val/redaction/test_command_identity_redaction.py::test_owner_quartz",
        "python -m pytest -q tests/phase2j_hard_val/redaction/test_command_identity_redaction.py::test_owner_ember",
    ]
    state = build_env(TaskType.TEST_FAILURE, 0, profile="phase2j_semantic_val").reset()
    state = state.model_copy(
        update={
            "goal": state.goal.model_copy(update={"command_allowlist": candidates}),
            "terminal": state.terminal.model_copy(
                update={
                    "stdout_delta": (
                        "Source inspected: semantic disambiguation required. "
                        "Runtime inspection produced a structured command-identity sidecar. "
                        "phase2j_command_identity_tokens=harbor."
                    ),
                    "last_command": candidates[0],
                }
            ),
        }
    )
    prompt = build_phase2c_head_state_prompt_from_state(state)

    signal = runtime_command_identity_signal(state)
    source_rows = command_candidate_source_overlap_rows(prompt, candidates)
    feature_rows = command_candidate_feature_rows(prompt, candidates)
    identity_feature_rows = command_candidate_feature_rows(
        prompt,
        candidates,
        nsi_reference=signal,
    )
    identity_start = CANDIDATE_FEATURE_DIM - COMMAND_IDENTITY_CANDIDATE_FEATURE_DIM

    assert "phase2j_command_identity_tokens=harbor" not in prompt
    assert signal["command_identity_slot:1"] > signal["command_identity_slot:0"]
    assert signal["command_identity_confidence"] > 0.0
    assert source_overlap_command_slot_prediction(prompt, candidates) != 1
    assert source_rows[1][1] == 0.0
    assert feature_rows[1][20] == 0.0
    assert identity_feature_rows[1][identity_start] > identity_feature_rows[0][identity_start]


def test_source_overlap_baseline_ignores_candidate_repair_action_metadata() -> None:
    prompt = "\n".join(
        [
            "Phase2AA bounded patch candidate state.",
            "Runtime-visible repair evidence:",
            '{"changed_files": ["src/example.py"], "structural_probe_hashes": []}',
            "",
            "Candidate repair actions:",
            "- repair_action=structural_repair_alpha; target_symbol=alpha_symbol",
            "- repair_action=structural_repair_bravo; target_symbol=bravo_symbol",
            "",
            "Candidate commands:",
            "- structural_repair_alpha command_identity_tokens=alpha_hash",
            "- structural_repair_bravo command_identity_tokens=bravo_hash",
            "",
            "Head constraints:",
            "- RUN_COMMAND must select one repair action command slot.",
        ]
    )
    candidates = [
        "structural_repair_alpha command_identity_tokens=alpha_hash",
        "structural_repair_bravo command_identity_tokens=bravo_hash",
    ]

    rows = command_candidate_source_overlap_rows(prompt, candidates)

    assert rows[0][1] == 0.0
    assert rows[1][1] == 0.0
    assert source_overlap_command_slot_prediction(prompt, candidates) == 0


def test_source_overlap_baseline_ignores_candidate_verification_command_metadata() -> None:
    prompt = "\n".join(
        [
            "Phase2AU policy-required runtime-delta native-head input.",
            "runtime_visible_error=ambiguous failure without candidate identity",
            "",
            "Candidate verification commands:",
            "- phase2au_apply_candidate --repair-action repair_alpha structural_probe_hash=alpha_hash target_symbol=alpha_symbol --verify pytest",
            "- phase2au_apply_candidate --repair-action repair_bravo structural_probe_hash=bravo_hash target_symbol=bravo_symbol --verify pytest",
        ]
    )
    candidates = [
        "phase2au_apply_candidate --repair-action repair_alpha structural_probe_hash=alpha_hash target_symbol=alpha_symbol --verify pytest",
        "phase2au_apply_candidate --repair-action repair_bravo structural_probe_hash=bravo_hash target_symbol=bravo_symbol --verify pytest",
    ]

    rows = command_candidate_source_overlap_rows(prompt, candidates)

    assert rows[0][1] == 0.0
    assert rows[1][1] == 0.0
    assert source_overlap_command_slot_prediction(prompt, candidates) == 0


def test_source_overlap_baseline_redacts_structural_probe_identity_hashes() -> None:
    prompt = "\n".join(
        [
            "Phase2AA bounded patch candidate state.",
            "Runtime-visible repair evidence:",
            '{"structural_probe_hashes": ["alpha_hash"], "changed_files": ["src/example.py"]}',
            "",
            "Candidate commands:",
            "- structural_repair_alpha command_identity_tokens=alpha_hash",
            "- structural_repair_bravo command_identity_tokens=bravo_hash",
            "",
            "Head constraints:",
            "- RUN_COMMAND must select one repair action command slot.",
        ]
    )
    candidates = [
        "structural_repair_alpha command_identity_tokens=alpha_hash",
        "structural_repair_bravo command_identity_tokens=bravo_hash",
    ]

    rows = command_candidate_source_overlap_rows(prompt, candidates)

    assert rows[0][1] == 0.0
    assert rows[1][1] == 0.0
    assert source_overlap_command_slot_prediction(prompt, candidates) == 0


def test_debug_action_stage_signal_tracks_visible_debug_transition() -> None:
    env = build_env(TaskType.TEST_FAILURE, 0, profile="phase2j_source_overlap_hard_actiongate_val")
    raw_state = env.reset()
    parsed_state, _, _, _ = env.step(ActionDecision(type=ActionType.READ_STDERR))
    inspected_state, _, _, _ = env.step(
        ActionDecision(
            type=ActionType.READ_FILE,
            file_target=parsed_state.goal.watched_paths[0],
        )
    )

    assert debug_action_stage_signal(raw_state) == "raw_failure_output"
    assert debug_action_stage_signal(parsed_state) == "parsed_failure_summary"
    assert debug_action_stage_signal(inspected_state) == "source_inspected"
    assert "debug_action_stage=raw_failure_output" in build_phase2c_head_state_prompt_from_state(
        raw_state
    )
    assert "debug_action_stage=parsed_failure_summary" in build_phase2c_head_state_prompt_from_state(
        parsed_state
    )
    assert "debug_action_stage=source_inspected" in build_phase2c_head_state_prompt_from_state(
        inspected_state
    )


def test_nsi_latent_values_appends_command_identity_fields() -> None:
    reference = {
        "debug_action_stage": "parsed_failure_summary",
        "descriptor_failure_family": "attribute_missing_runtime",
        "command_identity_slot:0": 0.125,
        "command_identity_slot:2": 0.75,
        "command_identity_margin": 0.5,
        "command_identity_confidence": 0.75,
    }

    values = nsi_latent_values(reference)
    indexed = dict(zip(NSI_LATENT_FIELDS, values))

    assert len(values) == len(NSI_LATENT_FIELDS)
    assert set(COMMAND_IDENTITY_LATENT_FIELDS).issubset(NSI_LATENT_FIELDS)
    assert set(DEBUG_ACTION_STAGE_LATENT_FIELDS).issubset(NSI_LATENT_FIELDS)
    assert set(DESCRIPTOR_FAILURE_FAMILY_LATENT_FIELDS).issubset(NSI_LATENT_FIELDS)
    assert indexed["debug_action_stage:raw_failure_output"] == 0.0
    assert indexed["debug_action_stage:parsed_failure_summary"] == DEBUG_ACTION_STAGE_GAIN
    assert indexed["debug_action_stage:source_inspected"] == 0.0
    assert indexed["descriptor_failure_family:other"] == 0.0
    assert (
        indexed["descriptor_failure_family:attribute_missing_runtime"]
        == DESCRIPTOR_FAILURE_FAMILY_GAIN
    )
    assert indexed["command_identity_slot:0"] == 0.125
    assert indexed["command_identity_slot:1"] == 0.0
    assert indexed["command_identity_slot:2"] == 0.75
    assert indexed["command_identity_margin"] == 0.5
    assert indexed["command_identity_confidence"] == 0.75


def test_command_slot_baseline_metrics_records_source_overlap_and_split_hash() -> None:
    rows = [
        _row(
            command="python -m pytest -q tests/billing/test_invoice_rules.py::test_invoice_total_rounds_tax",
            command_intent="test_rerun",
            command_slot=1,
            candidate_commands=[
                "python -m pytest -q tests/billing/test_invoice_rules.py::test_invoice_total_uses_discount",
                "python -m pytest -q tests/billing/test_invoice_rules.py::test_invoice_total_rounds_tax",
            ],
            state_prompt=(
                "stdout_delta=Source inspected: invoice total rounds tax\n"
                "last_command=python -m pytest -q tests/billing/test_invoice_rules.py::test_invoice_total_uses_discount"
            ),
        )
    ]

    metrics = _command_slot_baseline_metrics(rows)

    assert _canonical_rows_sha256(rows) == _canonical_rows_sha256(list(rows))
    assert metrics["uses_labels_for_prediction"] is False
    assert metrics["accuracy"] == 1.0
    assert metrics["by_intent"]["test_rerun"]["accuracy"] == 1.0


def test_native_head_checkpoint_identity_ignores_runtime_checkpoint_controls(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text(json.dumps(_row(action_index=0)) + "\n", encoding="utf-8")
    val.write_text(json.dumps(_row(action_index=1)) + "\n", encoding="utf-8")
    base = NativeHeadTrainConfig(
        base_model_name="model",
        adapter_name="adapter",
    )
    resumed = NativeHeadTrainConfig(
        base_model_name="model",
        adapter_name="adapter",
        checkpoint_interval_steps=50,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        resume_from_checkpoint=str(tmp_path / "checkpoints" / "epoch0001-step000050"),
    )

    base_hash = _native_head_training_identity_hash(
        train_jsonl=train,
        val_jsonl=val,
        output_dir=tmp_path / "adapter",
        config=base,
        train_rows_hash="train-hash",
        val_rows_hash="val-hash",
    )
    resumed_hash = _native_head_training_identity_hash(
        train_jsonl=train,
        val_jsonl=val,
        output_dir=tmp_path / "adapter",
        config=resumed,
        train_rows_hash="train-hash",
        val_rows_hash="val-hash",
    )

    assert base_hash == resumed_hash


def test_native_head_checkpoint_identity_changes_for_training_semantics(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text(json.dumps(_row(action_index=0)) + "\n", encoding="utf-8")
    val.write_text(json.dumps(_row(action_index=1)) + "\n", encoding="utf-8")
    base = NativeHeadTrainConfig(base_model_name="model", adapter_name="adapter")
    changed = NativeHeadTrainConfig(
        base_model_name="model",
        adapter_name="adapter",
        learning_rate=2e-4,
    )

    assert _native_head_training_identity_hash(
        train_jsonl=train,
        val_jsonl=val,
        output_dir=tmp_path / "adapter",
        config=base,
        train_rows_hash="train-hash",
        val_rows_hash="val-hash",
    ) != _native_head_training_identity_hash(
        train_jsonl=train,
        val_jsonl=val,
        output_dir=tmp_path / "adapter",
        config=changed,
        train_rows_hash="train-hash",
        val_rows_hash="val-hash",
    )


def test_native_head_checkpoint_path_is_step_addressable(tmp_path: Path) -> None:
    assert _checkpoint_path(tmp_path, epoch=1, step=52) == tmp_path / "epoch0001-step000052"


def test_pairwise_command_prompt_keeps_candidate_and_compact_receptor_evidence() -> None:
    long_boilerplate = "\n".join(f"irrelevant_line_{index}=x" for index in range(200))
    state_prompt = "\n".join(
        [
            "Phase 2C native nervous interface state input.",
            long_boilerplate,
            "failure_signal=assertion_inspection",
            "source_inspected=True",
            "stdout_delta=Source inspected: choose the archive manifest determinism test.",
            "stderr_delta=AssertionError: archive manifest hashes changed",
            "last_command=python -m pytest -q tests/test_wrong.py",
            "watched_paths=src/reflexlm/cli/archive_phase2f_evidence.py,tests/test_phase2f_archive_and_tables.py",
            "",
            "Candidate commands:",
            "- python -m pytest -q tests/test_wrong.py",
            "",
            "Head constraints:",
            "- RUN_COMMAND must select a command slot.",
        ]
    )
    candidate = (
        "python -m pytest -q "
        "tests/test_phase2f_archive_and_tables.py::test_phase2f_archive_manifest_hashes_are_deterministic"
    )

    compact = compact_visible_state_for_candidate_pair(state_prompt)
    prompt = build_candidate_pair_prompt(state_prompt, candidate, kind="Command")

    assert candidate in prompt
    assert prompt.index(candidate) < prompt.index("Compact visible state evidence:")
    assert "stdout_delta=Source inspected" in compact
    assert "stderr_delta=AssertionError" in compact
    assert "Candidate commands:" not in compact
    assert "Head constraints:" not in compact


def test_phase2c_native_head_collate_can_add_pairwise_command_tensors() -> None:
    batch = _collate_head_rows(
        FakeTokenizer(),
        [
            _row(
                command_slot=2,
                candidate_commands=[
                    "python -m pytest -q tests/test_auth.py",
                    "python -m pip install -r requirements.txt",
                    "python -m pytest -q tests/test_snapshots.py --snapshot-update",
                ],
            )
        ],
        max_length=32,
        use_pairwise_command_reranker=True,
    )

    assert batch["command_pair_input_ids"].shape[:2] == (1, MAX_CANDIDATE_SLOTS)
    assert batch["command_pair_mask"].tolist() == [[True, True, True, False]]


def test_pairwise_command_policy_masks_only_same_intent_competition() -> None:
    candidates = [
        "python -m pytest -q tests/test_auth.py::test_login",
        "python -m pip install -r requirements.txt",
        "python -m pytest -q tests/test_billing.py::test_total",
        "python -m pytest -q tests/test_snapshots.py --snapshot-update",
    ]

    assert pairwise_command_candidate_mask(candidates, "all") == [True, True, True, True]
    assert pairwise_command_candidate_mask(candidates, "ambiguous_intent") == [
        True,
        False,
        True,
        False,
    ]


def test_pairwise_command_policy_top_k_uses_visible_source_overlap_within_intent() -> None:
    candidates = [
        "python -m pytest -q tests/test_auth.py::test_login_redirect",
        "python -m pytest -q tests/test_billing.py::test_total",
        "python -m pip install -r requirements.txt",
        "python -m pytest -q tests/test_auth.py::test_logout",
    ]
    visible_state = "\n".join(
        [
            "failure_signal=assertion",
            "source_inspected=tests/test_auth.py",
            "goal_description=fix auth login redirect",
        ]
    )

    assert pairwise_command_candidate_mask(
        candidates,
        "ambiguous_intent",
        visible_state_text=visible_state,
        top_k=2,
    ) == [True, False, False, True]


def test_phase2c_native_head_collate_ambiguous_intent_pairwise_skips_uncontested_candidates() -> None:
    batch = _collate_head_rows(
        FakeTokenizer(),
        [
            _row(
                command_slot=0,
                candidate_commands=[
                    "python -m pytest -q tests/test_auth.py::test_login",
                    "python -m pip install -r requirements.txt",
                    "python -m pytest -q tests/test_auth.py::test_logout",
                ],
            )
        ],
        max_length=256,
        use_pairwise_command_reranker=True,
        pairwise_command_policy="ambiguous_intent",
        pairwise_command_max_length=96,
    )

    assert batch["command_pair_input_ids"].shape[:2] == (1, MAX_CANDIDATE_SLOTS)
    assert batch["command_pair_mask"].tolist() == [[True, False, True, False]]
    assert int(batch["command_pair_attention_mask"][0, 1].sum().item()) <= 1


def test_phase2c_native_head_collate_top_k_pairwise_keeps_best_visible_same_intent_candidates() -> None:
    batch = _collate_head_rows(
        FakeTokenizer(),
        [
            _row(
                state_prompt="\n".join(
                    [
                        "failure_signal=assertion",
                        "source_inspected=tests/test_auth.py",
                        "goal_description=fix auth login redirect",
                    ]
                ),
                command_slot=0,
                candidate_commands=[
                    "python -m pytest -q tests/test_auth.py::test_login_redirect",
                    "python -m pytest -q tests/test_billing.py::test_total",
                    "python -m pip install -r requirements.txt",
                    "python -m pytest -q tests/test_auth.py::test_logout",
                ],
            )
        ],
        max_length=256,
        use_pairwise_command_reranker=True,
        pairwise_command_policy="ambiguous_intent",
        pairwise_command_max_length=96,
        pairwise_command_top_k=2,
    )

    assert batch["command_pair_mask"].tolist() == [[True, False, False, True]]
    assert int(batch["command_pair_attention_mask"][0, 1].sum().item()) <= 1


def test_pairwise_candidate_encoding_stats_respects_top_k_and_command_rows_only() -> None:
    rows = [
        _row(
            state_prompt="source_inspected=tests/test_auth.py\ngoal_description=auth login failure",
            command_slot=0,
            candidate_commands=[
                "python -m pytest -q tests/test_auth.py::test_login",
                "python -m pytest -q tests/test_billing.py::test_total",
                "python -m pytest -q tests/test_auth.py::test_logout",
            ],
        ),
        _row(
            command_slot=-100,
            candidate_commands=[
                "python -m pytest -q tests/test_auth.py::test_login",
                "python -m pytest -q tests/test_billing.py::test_total",
            ],
        ),
    ]

    stats = _pairwise_candidate_encoding_stats(rows, policy="ambiguous_intent", top_k=1)

    assert stats["command_slot_rows"] == 1
    assert stats["valid_command_candidates"] == 3
    assert stats["pairwise_scored_candidates"] == 1


def test_phase2c_native_head_collate_can_resize_candidate_features_for_legacy_heads() -> None:
    legacy_feature_dim = CANDIDATE_FEATURE_DIM - 5
    batch = _collate_head_rows(
        FakeTokenizer(),
        [
            _row(
                command_slot=1,
                candidate_commands=[
                    "python -m pytest -q tests/test_auth.py",
                    "python -m pytest -q tests/test_auth.py::test_login_redirect",
                ],
            )
        ],
        max_length=32,
        command_candidate_feature_dim=legacy_feature_dim,
    )

    assert batch["command_candidate_features"].shape == (1, MAX_CANDIDATE_SLOTS, legacy_feature_dim)


def test_phase2c_native_head_loss_prefers_candidate_logits_for_command_slots() -> None:
    batch = _collate_head_rows(
        FakeTokenizer(),
        [
            _row(
                command_slot=2,
                candidate_commands=[
                    "python -m pytest -q tests/test_auth.py",
                    "python -m pip install -r requirements.txt",
                    "python -m pytest -q tests/test_snapshots.py --snapshot-update",
                ],
            )
        ],
        max_length=32,
    )
    outputs = {
        "action_logits": torch.zeros(1, len(ACTION_ORDER), requires_grad=True),
        "target_logits": torch.zeros(1, len(INTERNAL_TARGET_ORDER), requires_grad=True),
        "route_logits": torch.zeros(1, len(ROUTE_ORDER), requires_grad=True),
        "command_intent_logits": torch.zeros(1, COMMAND_INTENT_COUNT, requires_grad=True),
        "command_slot_logits": torch.zeros(1, MAX_CANDIDATE_SLOTS, requires_grad=True),
        "command_candidate_logits": torch.tensor([[0.0, 0.0, 4.0, -10000.0]], requires_grad=True),
        "file_slot_logits": torch.zeros(1, MAX_CANDIDATE_SLOTS, requires_grad=True),
        "confidence": torch.full((1,), 0.5, requires_grad=True),
        "inhibition": torch.full((1,), 0.5, requires_grad=True),
        "salience": torch.full((1,), 0.5, requires_grad=True),
        "risk": torch.full((1,), 0.5, requires_grad=True),
        "prediction_error": torch.full((1,), 0.5, requires_grad=True),
    }

    loss, components = compute_native_head_loss(outputs, batch)

    assert torch.isfinite(loss)
    assert components["command_slot"] < 0.1


def test_phase2c_eval_reports_learned_patch_descriptor_metrics() -> None:
    batch = _collate_head_rows(
        FakeTokenizer(),
        [
            _row(
                patch_operation_label=2,
                patch_target_file_slot=1,
                patch_template_slot=3,
            )
        ],
        max_length=32,
    )

    class FakeModel:
        def eval(self) -> None:
            return None

        def __call__(self, **_kwargs):
            def logits(width: int, target: int) -> torch.Tensor:
                values = torch.zeros(1, width)
                values[0, target] = 8.0
                return values

            return {
                "action_logits": logits(len(ACTION_ORDER), 0),
                "target_logits": logits(len(INTERNAL_TARGET_ORDER), 0),
                "route_logits": logits(len(ROUTE_ORDER), 0),
                "command_intent_logits": torch.zeros(1, COMMAND_INTENT_COUNT),
                "command_slot_logits": torch.zeros(1, MAX_CANDIDATE_SLOTS),
                "file_slot_logits": torch.zeros(1, MAX_CANDIDATE_SLOTS),
                "patch_operation_logits": logits(5, 2),
                "patch_target_file_slot_logits": logits(MAX_CANDIDATE_SLOTS, 1),
                "patch_template_slot_logits": logits(MAX_CANDIDATE_SLOTS, 3),
                "confidence": torch.full((1,), 1.0),
                "inhibition": torch.full((1,), 0.0),
                "salience": torch.full((1,), 0.5),
                "risk": torch.full((1,), 0.2),
                "prediction_error": torch.full((1,), 0.1),
            }

    metrics = _evaluate_native_head_model(
        FakeModel(),
        [batch],
        device=torch.device("cpu"),
        loss_weights={},
    )

    assert metrics["patch_operation_accuracy"] == 1.0
    assert metrics["patch_operation_count"] == 1.0
    assert metrics["patch_target_file_slot_accuracy"] == 1.0
    assert metrics["patch_template_slot_accuracy"] == 1.0
    assert metrics["slot_confusion"]["patch_template_slot"] == {"3": {"3": 1}}


def test_phase2c_head_dataset_limit_supports_canary_runs(tmp_path: Path) -> None:
    path = tmp_path / "heads.jsonl"
    rows = [
        {**_row(action_index=0), "task_type": "a", "head_scope": "reflex", "action_type": "WAIT"},
        {**_row(action_index=0), "task_type": "a", "head_scope": "reflex", "action_type": "WAIT"},
        {**_row(action_index=1), "task_type": "b", "head_scope": "debug", "action_type": "READ_STDERR"},
        {**_row(action_index=1), "task_type": "b", "head_scope": "debug", "action_type": "READ_STDERR"},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    dataset = Phase2CHeadJsonlDataset(path, limit=2)

    assert len(dataset) == 2
    assert dataset[1]["action_index"] == 1


def test_balanced_limited_rows_round_robins_task_scope_action() -> None:
    rows = [
        {"task_type": "a", "head_scope": "reflex", "action_type": "WAIT", "id": "a1"},
        {"task_type": "a", "head_scope": "reflex", "action_type": "WAIT", "id": "a2"},
        {"task_type": "b", "head_scope": "debug", "action_type": "READ_STDERR", "id": "b1"},
        {"task_type": "b", "head_scope": "debug", "action_type": "READ_STDERR", "id": "b2"},
        {"task_type": "c", "head_scope": "inhibition", "action_type": "BLOCK", "id": "c1"},
    ]

    selected = _balanced_limited_rows(rows, 3)

    assert [row["id"] for row in selected] == ["a1", "b1", "c1"]


def test_balanced_limited_rows_round_robins_debug_command_slots() -> None:
    rows = []
    for index, slot in enumerate([0, 0, 0, 0, 1, 2, 3]):
        rows.append(
            {
                "task_type": "test_failure_reflex",
                "head_scope": "debug_cortex",
                "action_type": "RUN_COMMAND",
                "command_slot": slot,
                "id": f"slot{slot}-{index}",
            }
        )

    selected = _balanced_limited_rows(rows, 4)

    assert sorted(row["command_slot"] for row in selected) == [0, 1, 2, 3]


def test_balanced_limited_rows_round_robins_command_intents_within_slots() -> None:
    rows = []
    for index in range(4):
        rows.append(
            {
                "task_type": "test_failure_reflex",
                "head_scope": "debug_cortex",
                "action_type": "RUN_COMMAND",
                "command_slot": 0,
                "command": "python -m pip install -r requirements.txt",
                "id": f"dep-{index}",
            }
        )
    rows.append(
        {
            "task_type": "test_failure_reflex",
            "head_scope": "debug_cortex",
            "action_type": "RUN_COMMAND",
            "command_slot": 0,
            "command": "python -m pytest -q tests/test_api.py::test_contract",
            "id": "rerun-0",
        }
    )

    selected = _balanced_limited_rows(rows, 2)

    assert {row["id"].split("-")[0] for row in selected} == {"dep", "rerun"}


def test_balanced_limited_rows_balances_actions_before_command_slots() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "task_type": "test_failure_reflex",
                "head_scope": "debug_cortex",
                "action_type": "READ_STDERR",
                "id": f"stderr-{index}",
            }
        )
        rows.append(
            {
                "task_type": "test_failure_reflex",
                "head_scope": "debug_cortex",
                "action_type": "READ_FILE",
                "file_slot": 0,
                "id": f"file-{index}",
            }
        )
    for slot in range(4):
        for index in range(12):
            rows.append(
                {
                    "task_type": "test_failure_reflex",
                    "head_scope": "debug_cortex",
                    "action_type": "RUN_COMMAND",
                    "command_slot": slot,
                    "command": f"python -m pytest -q tests/test_{slot}.py::test_case_{index}",
                    "id": f"run-{slot}-{index}",
                }
            )

    selected = _balanced_limited_rows(rows, 12)
    action_counts = {}
    run_slots = []
    for row in selected:
        action_counts[row["action_type"]] = action_counts.get(row["action_type"], 0) + 1
        if row["action_type"] == "RUN_COMMAND":
            run_slots.append(row["command_slot"])

    assert action_counts == {"READ_FILE": 4, "READ_STDERR": 4, "RUN_COMMAND": 4}
    assert sorted(run_slots) == [0, 1, 2, 3]


def test_debug_command_oversampling_only_repeats_valid_debug_commands() -> None:
    rows = [
        {"head_scope": "debug_cortex", "action_type": "READ_STDERR", "command_slot": -100, "id": "read"},
        {"head_scope": "debug_cortex", "action_type": "RUN_COMMAND", "command_slot": 2, "id": "run"},
        {"head_scope": "reflex_layer", "action_type": "RUN_COMMAND", "command_slot": 1, "id": "reflex"},
    ]

    expanded = _oversample_debug_command_rows(rows, 3)

    assert [row["id"] for row in expanded].count("run") == 3
    assert [row["id"] for row in expanded].count("read") == 1
    assert [row["id"] for row in expanded].count("reflex") == 1


def test_balance_debug_command_intents_equalizes_debug_run_command_categories() -> None:
    rows = [
        {
            "head_scope": "debug_cortex",
            "action_type": "RUN_COMMAND",
            "command_slot": 0,
            "command": "python -m pip install -r requirements.txt",
            "id": "dep1",
        },
        {
            "head_scope": "debug_cortex",
            "action_type": "RUN_COMMAND",
            "command_slot": 0,
            "command": "python -m pip install -r requirements.txt",
            "id": "dep2",
        },
        {
            "head_scope": "debug_cortex",
            "action_type": "RUN_COMMAND",
            "command_slot": 1,
            "command": "python -m pytest -q tests/test_snapshots.py --snapshot-update",
            "id": "snap1",
        },
        {
            "head_scope": "debug_cortex",
            "action_type": "READ_STDERR",
            "command_slot": -100,
            "id": "read",
        },
    ]

    balanced = _balance_debug_command_intent_rows(rows)
    ids = [row["id"] for row in balanced]

    assert ids.count("dep1") + ids.count("dep2") == 2
    assert ids.count("snap1") == 2
    assert ids.count("read") == 1


def test_balance_command_slot_rows_equalizes_all_labeled_command_slots() -> None:
    rows = [
        {"command_slot": 2, "id": "slot2-a"},
        {"command_slot": 2, "id": "slot2-b"},
        {"command_slot": 2, "id": "slot2-c"},
        {"command_slot": 3, "id": "slot3-a"},
        {"command_slot": -100, "id": "passthrough"},
    ]

    balanced = _balance_command_slot_rows(rows)
    ids = [row["id"] for row in balanced]

    assert ids.count("passthrough") == 1
    assert sum(1 for row in balanced if row.get("command_slot") == 2) == 3
    assert sum(1 for row in balanced if row.get("command_slot") == 3) == 3


def test_balance_patch_descriptor_rows_equalizes_operation_template_pairs() -> None:
    rows = [
        {"patch_operation_label": 2, "patch_template_slot": 1, "id": "import-a"},
        {"patch_operation_label": 2, "patch_template_slot": 1, "id": "import-b"},
        {"patch_operation_label": 2, "patch_template_slot": 1, "id": "import-c"},
        {"patch_operation_label": 1, "patch_template_slot": 0, "id": "attr-a"},
        {"patch_operation_label": -100, "patch_template_slot": -100, "id": "passthrough"},
    ]

    balanced = _balance_patch_descriptor_rows(rows)
    ids = [row["id"] for row in balanced]

    assert ids.count("passthrough") == 1
    assert sum(1 for row in balanced if row.get("patch_operation_label") == 2) == 3
    assert sum(1 for row in balanced if row.get("patch_operation_label") == 1) == 3


def test_phase2c_dataset_can_balance_command_slots_before_training(tmp_path: Path) -> None:
    path = tmp_path / "heads.jsonl"
    rows = [
        _row(command_slot=2, action_type="RUN_COMMAND", id="slot2-a"),
        _row(command_slot=2, action_type="RUN_COMMAND", id="slot2-b"),
        _row(command_slot=3, action_type="RUN_COMMAND", id="slot3-a"),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    dataset = Phase2CHeadJsonlDataset(path, balance_command_slots=True)

    assert len(dataset) == 4
    assert sum(1 for row in dataset.rows if row["command_slot"] == 2) == 2
    assert sum(1 for row in dataset.rows if row["command_slot"] == 3) == 2


def test_phase2c_dataset_can_balance_patch_descriptor_labels_before_training(tmp_path: Path) -> None:
    path = tmp_path / "heads.jsonl"
    rows = [
        _row(patch_operation_label=2, patch_template_slot=1, id="import-a"),
        _row(patch_operation_label=2, patch_template_slot=1, id="import-b"),
        _row(patch_operation_label=1, patch_template_slot=0, id="attr-a"),
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    dataset = Phase2CHeadJsonlDataset(path, balance_patch_descriptor_labels=True)

    assert len(dataset) == 4
    assert sum(1 for row in dataset.rows if row["patch_operation_label"] == 2) == 2
    assert sum(1 for row in dataset.rows if row["patch_operation_label"] == 1) == 2


def test_patch_descriptor_distribution_records_label_and_name_counts() -> None:
    rows = [
        _row(patch_operation_label=2, patch_template_slot=1),
        _row(patch_operation_label=2, patch_template_slot=1),
        _row(patch_operation_label=1, patch_template_slot=0),
        _row(patch_operation_label=-100, patch_template_slot=-100),
    ]

    distribution = _patch_descriptor_distribution(rows)

    assert distribution["rows"] == 4
    assert distribution["descriptor_rows"] == 3
    assert distribution["patch_operation_labels"] == {"1": 1, "2": 2}
    assert distribution["patch_operation_names"] == {
        "insert_import": 2,
        "replace_attribute": 1,
    }
    assert distribution["patch_template_names"] == {
        "call_attribute_restoration": 1,
        "import_restoration": 2,
    }
    assert distribution["patch_operation_template_pairs"] == {
        "insert_import|import_restoration": 2,
        "replace_attribute|call_attribute_restoration": 1,
    }


def test_balanced_limit_preserves_patch_descriptor_label_diversity() -> None:
    rows = [
        _row(
            action_type="RUN_COMMAND",
            command_slot=0,
            patch_operation_label=2,
            patch_template_slot=1,
            id=f"import-{index}",
        )
        for index in range(6)
    ] + [
        _row(
            action_type="RUN_COMMAND",
            command_slot=0,
            patch_operation_label=1,
            patch_template_slot=0,
            id=f"attr-{index}",
        )
        for index in range(2)
    ]

    limited = _balanced_limited_rows(rows, 4)

    assert sum(1 for row in limited if row["patch_operation_label"] == 2) == 2
    assert sum(1 for row in limited if row["patch_operation_label"] == 1) == 2
