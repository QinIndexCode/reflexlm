import json
from pathlib import Path

from reflexlm.cli.check_phase2i_prepackage_gates import build_phase2i_prepackage_gate_report


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(**overrides: object) -> dict:
    payload = {
        "adapter_name": "phase2i_intentbalanced",
        "train_examples": 1024,
        "val_examples": 512,
        "effective_split_hashes": {
            "phase2c_head_train": "train-hash",
            "phase2c_head_val": "val-hash",
        },
        "slot_intent_distribution": {
            "train": {"command_intents": {"dependency_install": 1, "test_rerun": 1}},
            "val": {"command_intents": {"dependency_install": 1, "test_rerun": 1}},
        },
        "source_overlap_command_slot_baseline": {"train": {}, "val": {}},
        "history": [
            {
                "val_metrics": {
                    "command_slot_accuracy": 0.9,
                    "command_slot_count": 109.0,
                    "command_intent_accuracy": 1.0,
                }
            }
        ],
        "config": {
            "max_train_records": 1024,
            "max_val_records": 512,
            "use_pairwise_command_reranker": True,
            "pairwise_command_fusion": "residual",
            "pairwise_command_policy": "ambiguous_intent",
            "pairwise_command_max_length": 96,
            "pairwise_command_top_k": 2,
            "command_candidate_encoder": "features_only",
            "latent_fusion": "additive",
        },
        "head_config": {
            "command_candidate_feature_dim": 24,
            "use_pairwise_command_reranker": True,
            "pairwise_command_fusion": "residual",
            "pairwise_command_policy": "ambiguous_intent",
            "pairwise_command_max_length": 96,
            "pairwise_command_top_k": 2,
            "command_candidate_encoder": "features_only",
        },
        "json_text_target": False,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
    }
    payload.update(overrides)
    return payload


def _audit(**overrides: object) -> dict:
    payload = {
        "passed": True,
        "checks": {
            "phase2i_effective_split_hashes_present": True,
            "phase2i_train_val_command_intent_coverage": True,
            "phase2i_head_val_target_command_coverage": True,
            "external_v3_has_no_phase2i_command_overlap": True,
        },
        "effective_split_hashes": {
            "phase2i_head_train": "train-hash",
            "phase2i_head_val": "val-hash",
        },
    }
    payload.update(overrides)
    return payload


def _latent_audit(**overrides: object) -> dict:
    payload = {
        "audit_family": "phase2i_latent_necessity",
        "passed": True,
        "checks": {
            "nsi_latent_command_identity_available": True,
        },
    }
    payload.update(overrides)
    return payload


def test_phase2i_prepackage_gate_accepts_full_pairwise_summary(tmp_path: Path) -> None:
    summary_json = _write(tmp_path / "summary.json", _summary())
    audit_json = _write(tmp_path / "audit.json", _audit())

    report = build_phase2i_prepackage_gate_report(
        training_summary_json=summary_json,
        data_audit_json=audit_json,
        expected_adapter_name="phase2i_intentbalanced",
        expected_pairwise_command_policy="ambiguous_intent",
        expected_pairwise_command_max_length=96,
        expected_pairwise_command_top_k=2,
        expected_command_candidate_encoder="features_only",
    )

    assert report["passed"] is True
    assert report["checks"]["pairwise_enabled"] is True
    assert report["checks"]["pairwise_command_policy_expected"] is True
    assert report["checks"]["pairwise_command_max_length_expected"] is True
    assert report["checks"]["pairwise_command_top_k_expected"] is True
    assert report["checks"]["command_candidate_encoder_expected"] is True
    assert report["checks"]["summary_hashes_match_data_audit"] is True


def test_phase2i_prepackage_gate_accepts_latent_necessity_audit(tmp_path: Path) -> None:
    summary_json = _write(tmp_path / "summary.json", _summary())
    audit_json = _write(tmp_path / "audit.json", _audit())
    latent_json = _write(tmp_path / "latent.json", _latent_audit())

    report = build_phase2i_prepackage_gate_report(
        training_summary_json=summary_json,
        data_audit_json=audit_json,
        latent_necessity_audit_json=latent_json,
    )

    assert report["passed"] is True
    assert report["checks"]["latent_necessity_audit_passed"] is True
    assert report["checks"]["latent_necessity_architecture_identifiable"] is True


def test_phase2i_prepackage_gate_rejects_failed_latent_necessity_audit(
    tmp_path: Path,
) -> None:
    summary_json = _write(tmp_path / "summary.json", _summary())
    audit_json = _write(tmp_path / "audit.json", _audit())
    latent_json = _write(
        tmp_path / "latent.json",
        _latent_audit(
            passed=False,
            checks={"nsi_latent_command_identity_available": False},
        ),
    )

    report = build_phase2i_prepackage_gate_report(
        training_summary_json=summary_json,
        data_audit_json=audit_json,
        latent_necessity_audit_json=latent_json,
    )

    assert report["passed"] is False
    assert report["checks"]["latent_necessity_audit_passed"] is False
    assert report["checks"]["latent_necessity_architecture_identifiable"] is False


def test_phase2i_prepackage_gate_rejects_lightweight_diagnostic(tmp_path: Path) -> None:
    summary_json = _write(
        tmp_path / "summary.json",
        _summary(
            train_examples=128,
            config={
                "max_train_records": 128,
                "max_val_records": 512,
                "use_pairwise_command_reranker": False,
                "pairwise_command_fusion": "residual",
                "pairwise_command_policy": "ambiguous_intent",
                "pairwise_command_max_length": 96,
                "pairwise_command_top_k": 2,
                "command_candidate_encoder": "features_only",
                "latent_fusion": "additive",
            },
            head_config={
                "command_candidate_feature_dim": 24,
                "use_pairwise_command_reranker": False,
                "pairwise_command_fusion": "residual",
                "pairwise_command_policy": "ambiguous_intent",
                "pairwise_command_max_length": 96,
                "pairwise_command_top_k": 2,
                "command_candidate_encoder": "features_only",
            },
            effective_split_hashes={
                "phase2c_head_train": "short-train-hash",
                "phase2c_head_val": "val-hash",
            },
        ),
    )
    audit_json = _write(tmp_path / "audit.json", _audit())

    report = build_phase2i_prepackage_gate_report(
        training_summary_json=summary_json,
        data_audit_json=audit_json,
    )

    assert report["passed"] is False
    assert report["checks"]["pairwise_enabled"] is False
    assert report["checks"]["train_examples_match_expected"] is False
    assert report["checks"]["summary_hashes_match_data_audit"] is False


def test_phase2i_prepackage_gate_rejects_pairwise_policy_mismatch(tmp_path: Path) -> None:
    summary_json = _write(tmp_path / "summary.json", _summary())
    audit_json = _write(tmp_path / "audit.json", _audit())

    report = build_phase2i_prepackage_gate_report(
        training_summary_json=summary_json,
        data_audit_json=audit_json,
        expected_pairwise_command_policy="all",
        expected_pairwise_command_max_length=256,
        expected_pairwise_command_top_k=1,
        expected_command_candidate_encoder="backbone",
    )

    assert report["passed"] is False
    assert report["checks"]["pairwise_command_policy_expected"] is False
    assert report["checks"]["pairwise_command_max_length_expected"] is False
    assert report["checks"]["pairwise_command_top_k_expected"] is False
    assert report["checks"]["command_candidate_encoder_expected"] is False


def test_phase2i_prepackage_gate_rejects_old_failed_summary(tmp_path: Path) -> None:
    summary_json = _write(
        tmp_path / "summary.json",
        _summary(
            history=[{"val_metrics": {"command_slot_accuracy": 0.486, "command_slot_count": 109.0}}],
            config={
                "max_train_records": 1024,
                "max_val_records": 512,
                "use_pairwise_command_reranker": True,
                "pairwise_command_fusion": "residual",
                "latent_fusion": "additive",
            },
            head_config={
                "command_candidate_feature_dim": 19,
                "use_pairwise_command_reranker": True,
            },
            no_json_motor_target=None,
            low_level_qwen_calls_target=None,
            slot_intent_distribution=None,
        ),
    )
    audit_json = _write(tmp_path / "audit.json", _audit())

    report = build_phase2i_prepackage_gate_report(
        training_summary_json=summary_json,
        data_audit_json=audit_json,
    )

    assert report["passed"] is False
    assert report["checks"]["val_command_slot_accuracy_min"] is False
    assert report["checks"]["command_candidate_feature_dim_expected"] is False
    assert report["checks"]["pairwise_command_policy_recorded"] is False
    assert report["checks"]["command_candidate_encoder_recorded"] is False
    assert report["checks"]["low_level_qwen_calls_target_zero"] is False
    assert report["checks"]["slot_intent_distribution_present"] is False
