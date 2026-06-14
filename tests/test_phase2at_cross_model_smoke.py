import json
from pathlib import Path

from reflexlm.cli.audit_phase2at_cross_model_smoke import (
    build_phase2at_cross_model_smoke_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(base_model: str, *, split_hash: str = "val-hash") -> dict:
    return {
        "base_model_name": base_model,
        "adapter_name": f"adapter-{base_model}",
        "train_examples": 32,
        "val_examples": 32,
        "effective_split_hashes": {
            "phase2c_head_train": "train-hash",
            "phase2c_head_val": split_hash,
        },
        "learned_patch_descriptor_heads": {
            "enabled": True,
            "patch_operation_order": ["replace_attribute", "insert_import"],
            "patch_template_order": ["call_attribute_restoration", "import_restoration"],
        },
        "config": {
            "base_model_name": base_model,
            "adapter_name": f"adapter-{base_model}",
            "quantization": "4bit",
            "learning_rate": 0.0001,
            "epochs": 5,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_length": 256,
            "lora_rank": 16,
            "lora_alpha": 32,
            "seed": 13,
            "device": "cuda",
            "progress_log_interval_steps": 12,
            "max_train_records": 32,
            "max_val_records": 32,
            "use_pairwise_command_reranker": False,
            "command_candidate_encoder": "features_only",
            "latent_fusion": "additive",
            "open_repair_heads_enabled": True,
        },
    }


def _postflight(summary_path: Path, *, template: float = 0.91, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "metrics": {
            "command_slot_accuracy": 1.0,
            "source_overlap_accuracy": 0.5,
            "model_minus_source_overlap_accuracy": 0.5,
            "patch_operation_accuracy": 0.91,
            "patch_target_file_slot_accuracy": 1.0,
            "patch_template_slot_accuracy": template,
        },
        "inputs": {"training_summary_json": str(summary_path)},
    }


def _model_artifacts(tmp_path: Path, model: str, **kwargs) -> Path:
    safe_name = model.replace("\\", "_").replace("/", "_")
    summary = _write(tmp_path / f"{safe_name}.summary.json", _summary(model, **kwargs))
    return _write(tmp_path / f"{safe_name}.postflight.json", _postflight(summary))


def test_phase2at_cross_model_smoke_ignores_logging_interval_mismatch(
    tmp_path: Path,
) -> None:
    first = _model_artifacts(tmp_path, "qwen1.5b")
    second_summary_payload = _summary("qwen3b")
    second_summary_payload["config"]["progress_log_interval_steps"] = 24
    second_summary = _write(tmp_path / "qwen3b.summary.json", second_summary_payload)
    second = _write(tmp_path / "qwen3b.postflight.json", _postflight(second_summary))

    report = build_phase2at_cross_model_smoke_report(postflight_jsons=[first, second])

    assert report["passed"] is True
    assert report["checks"]["training_contract_consistent_except_model_and_names"] is True


def test_phase2at_cross_model_smoke_accepts_two_models_same_contract(
    tmp_path: Path,
) -> None:
    report = build_phase2at_cross_model_smoke_report(
        postflight_jsons=[
            _model_artifacts(tmp_path, "qwen1.5b"),
            _model_artifacts(tmp_path, "qwen3b"),
        ]
    )

    assert report["passed"] is True
    assert report["checks"]["model_count_minimum_met"] is True
    assert report["supported_claims"] == [
        "phase2at_initial_same_split_cross_model_descriptor_smoke_supported"
    ]
    assert "sealed_cross_model_transfer" in report["unsupported_claims"]


def test_phase2at_cross_model_smoke_rejects_same_model(tmp_path: Path) -> None:
    report = build_phase2at_cross_model_smoke_report(
        postflight_jsons=[
            _model_artifacts(tmp_path, "qwen1.5b"),
            _model_artifacts(tmp_path, "qwen1.5b"),
        ]
    )

    assert report["passed"] is False
    assert report["checks"]["model_count_minimum_met"] is False


def test_phase2at_cross_model_smoke_rejects_split_mismatch(tmp_path: Path) -> None:
    report = build_phase2at_cross_model_smoke_report(
        postflight_jsons=[
            _model_artifacts(tmp_path, "qwen1.5b"),
            _model_artifacts(tmp_path, "qwen3b", split_hash="different"),
        ]
    )

    assert report["passed"] is False
    assert report["checks"]["split_hashes_consistent"] is False


def test_phase2at_cross_model_smoke_rejects_weak_model(tmp_path: Path) -> None:
    primary = _model_artifacts(tmp_path, "qwen1.5b")
    summary = _write(tmp_path / "qwen3b.summary.json", _summary("qwen3b"))
    weak = _write(tmp_path / "qwen3b.postflight.json", _postflight(summary, template=0.8))

    report = build_phase2at_cross_model_smoke_report(postflight_jsons=[primary, weak])

    assert report["passed"] is False
    assert report["checks"]["patch_template_slot_min_met"] is False
