import json
from pathlib import Path

from reflexlm.cli.audit_phase2at_multiseed_smoke import (
    build_phase2at_multiseed_smoke_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(seed: int, *, split_hash: str = "val-hash", epochs: int = 5) -> dict:
    return {
        "base_model_name": "artifacts\\models\\Qwen2.5-1.5B-Instruct",
        "adapter_name": f"adapter-seed-{seed}",
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
            "base_model_name": "artifacts\\models\\Qwen2.5-1.5B-Instruct",
            "adapter_name": f"adapter-seed-{seed}",
            "quantization": "4bit",
            "learning_rate": 0.0001,
            "epochs": epochs,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_length": 256,
            "lora_rank": 16,
            "lora_alpha": 32,
            "seed": seed,
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


def _postflight(summary_path: Path, *, passed: bool = True, template: float = 0.91) -> dict:
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


def _seed_artifacts(tmp_path: Path, seed: int, **kwargs) -> Path:
    summary = _write(tmp_path / f"seed{seed}.summary.json", _summary(seed, **kwargs))
    return _write(tmp_path / f"seed{seed}.postflight.json", _postflight(summary))


def test_phase2at_multiseed_smoke_ignores_logging_interval_mismatch(
    tmp_path: Path,
) -> None:
    seed13 = _seed_artifacts(tmp_path, 13)
    summary17_payload = _summary(17)
    summary17_payload["config"]["progress_log_interval_steps"] = 24
    summary17 = _write(tmp_path / "seed17.summary.json", summary17_payload)
    seed17 = _write(tmp_path / "seed17.postflight.json", _postflight(summary17))
    seed23 = _seed_artifacts(tmp_path, 23)

    report = build_phase2at_multiseed_smoke_report(
        postflight_jsons=[seed13, seed17, seed23]
    )

    assert report["passed"] is True
    assert report["checks"]["training_contract_consistent_except_seed_and_names"] is True


def test_phase2at_multiseed_smoke_accepts_three_consistent_passing_seeds(
    tmp_path: Path,
) -> None:
    report = build_phase2at_multiseed_smoke_report(
        postflight_jsons=[
            _seed_artifacts(tmp_path, 13),
            _seed_artifacts(tmp_path, 17),
            _seed_artifacts(tmp_path, 23),
        ]
    )

    assert report["passed"] is True
    assert report["metrics"]["unique_seeds"] == [13, 17, 23]
    assert report["metrics"]["metric_summary"]["patch_template_slot_accuracy"]["min"] == 0.91
    assert report["supported_claims"] == [
        "phase2at_same_model_three_seed_nonsealed_descriptor_smoke_supported"
    ]
    assert "epoch_making_architecture" in report["unsupported_claims"]


def test_phase2at_multiseed_smoke_rejects_failed_seed(tmp_path: Path) -> None:
    seed13 = _seed_artifacts(tmp_path, 13)
    seed17 = _seed_artifacts(tmp_path, 17)
    summary23 = _write(tmp_path / "seed23.summary.json", _summary(23))
    seed23 = _write(tmp_path / "seed23.postflight.json", _postflight(summary23, passed=False))

    report = build_phase2at_multiseed_smoke_report(
        postflight_jsons=[seed13, seed17, seed23]
    )

    assert report["passed"] is False
    assert report["checks"]["all_postflights_passed"] is False


def test_phase2at_multiseed_smoke_rejects_contract_mismatch(tmp_path: Path) -> None:
    report = build_phase2at_multiseed_smoke_report(
        postflight_jsons=[
            _seed_artifacts(tmp_path, 13),
            _seed_artifacts(tmp_path, 17, epochs=4),
            _seed_artifacts(tmp_path, 23),
        ]
    )

    assert report["passed"] is False
    assert report["checks"]["training_contract_consistent_except_seed_and_names"] is False


def test_phase2at_multiseed_smoke_rejects_split_hash_mismatch(tmp_path: Path) -> None:
    report = build_phase2at_multiseed_smoke_report(
        postflight_jsons=[
            _seed_artifacts(tmp_path, 13),
            _seed_artifacts(tmp_path, 17, split_hash="different-val-hash"),
            _seed_artifacts(tmp_path, 23),
        ]
    )

    assert report["passed"] is False
    assert report["checks"]["split_hashes_consistent"] is False


def test_phase2at_multiseed_smoke_rejects_fewer_than_three_seeds(
    tmp_path: Path,
) -> None:
    report = build_phase2at_multiseed_smoke_report(
        postflight_jsons=[
            _seed_artifacts(tmp_path, 13),
            _seed_artifacts(tmp_path, 17),
        ]
    )

    assert report["passed"] is False
    assert report["checks"]["unique_seed_minimum_met"] is False


def test_phase2at_multiseed_smoke_rejects_weak_descriptor_min(
    tmp_path: Path,
) -> None:
    seed13 = _seed_artifacts(tmp_path, 13)
    seed17 = _seed_artifacts(tmp_path, 17)
    summary23 = _write(tmp_path / "seed23.summary.json", _summary(23))
    seed23 = _write(tmp_path / "seed23.postflight.json", _postflight(summary23, template=0.8))

    report = build_phase2at_multiseed_smoke_report(
        postflight_jsons=[seed13, seed17, seed23]
    )

    assert report["passed"] is False
    assert report["checks"]["patch_template_slot_min_met"] is False
