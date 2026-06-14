import json
from pathlib import Path

from reflexlm.cli.audit_phase2ar_cross_model_reproduction import (
    audit_phase2ar_cross_model_reproduction,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _execution(rows: int = 32, success_rate: float = 1.0) -> dict:
    return {
        "rows": rows,
        "success_rate": success_rate,
        "patch_mode": "runtime_symbolic_structural",
        "claim_boundary": "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
    }


def _training(base_model: str) -> dict:
    return {
        "effective_split_hashes": {"phase2c_head_train": "a", "phase2c_head_val": "b"},
        "config": {
            "base_model_name": base_model,
            "quantization": "4bit",
            "learning_rate": 0.0001,
            "epochs": 5,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_length": 256,
            "lora_rank": 16,
            "lora_alpha": 32,
            "command_candidate_encoder": "features_only",
            "latent_fusion": "additive",
            "open_repair_heads_enabled": True,
            "seed": 13,
        },
    }


def test_phase2ar_cross_model_audit_accepts_same_contract_different_model(
    tmp_path: Path,
) -> None:
    report = audit_phase2ar_cross_model_reproduction(
        primary_execution_summary_json=_write(tmp_path / "primary_exec.json", _execution()),
        cross_model_execution_summary_json=_write(tmp_path / "cross_exec.json", _execution()),
        primary_training_summary_json=_write(tmp_path / "primary_train.json", _training("qwen7b")),
        cross_model_training_summary_json=_write(tmp_path / "cross_train.json", _training("qwen3b")),
    )

    assert report["passed"] is True
    assert report["checks"]["base_models_differ"] is True
    assert "phase2ar_qwen7b_to_qwen3b_same_family_reproduction_supported" in report["supported_claims"]
    assert "sealed_cross_model_transfer" in report["unsupported_claims"]


def test_phase2ar_cross_model_audit_rejects_same_model(tmp_path: Path) -> None:
    report = audit_phase2ar_cross_model_reproduction(
        primary_execution_summary_json=_write(tmp_path / "primary_exec.json", _execution()),
        cross_model_execution_summary_json=_write(tmp_path / "cross_exec.json", _execution()),
        primary_training_summary_json=_write(tmp_path / "primary_train.json", _training("qwen7b")),
        cross_model_training_summary_json=_write(tmp_path / "cross_train.json", _training("qwen7b")),
    )

    assert report["passed"] is False
    assert report["checks"]["base_models_differ"] is False


def test_phase2ar_cross_model_audit_rejects_config_drift(tmp_path: Path) -> None:
    cross_training = _training("qwen3b")
    cross_training["config"]["epochs"] = 1
    report = audit_phase2ar_cross_model_reproduction(
        primary_execution_summary_json=_write(tmp_path / "primary_exec.json", _execution()),
        cross_model_execution_summary_json=_write(tmp_path / "cross_exec.json", _execution()),
        primary_training_summary_json=_write(tmp_path / "primary_train.json", _training("qwen7b")),
        cross_model_training_summary_json=_write(tmp_path / "cross_train.json", cross_training),
    )

    assert report["passed"] is False
    assert report["checks"]["training_contract_matches_except_model"] is False
