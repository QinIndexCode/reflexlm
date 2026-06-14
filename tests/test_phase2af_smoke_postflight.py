import json
from pathlib import Path

from reflexlm.cli.audit_phase2af_smoke_postflight import (
    build_phase2af_smoke_postflight,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pretrain(*, source: float = 0.40, identity: float = 0.90) -> dict:
    metric = lambda value: {"correct": int(value * 30), "total": 30, "accuracy": value}
    return {
        "artifact_family": "phase2af_hardened_structural_sidecar_pretrain_gate",
        "passed": True,
        "split_metrics": {
            "val": {
                "raw_source_overlap": metric(source),
                "identity_text_ablated_source_overlap": metric(source),
                "runtime_identity_heuristic": metric(identity),
            }
        },
    }


def _summary(
    tmp_path: Path,
    *,
    val_accuracy: float = 1.0,
    source: float = 0.40,
    identity_pairwise: bool = False,
    train_examples: int = 30,
    val_examples: int = 30,
) -> dict:
    adapter = tmp_path / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "native_heads.pt").write_bytes(b"heads")
    (adapter / "head_config.json").write_text("{}", encoding="utf-8")
    return {
        "adapter_output_dir": str(adapter),
        "train_examples": train_examples,
        "val_examples": val_examples,
        "use_pairwise_command_reranker": identity_pairwise,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "effective_split_hashes": {
            "phase2c_head_train": "train-hash",
            "phase2c_head_val": "val-hash",
        },
        "config": {"device": "cuda"},
        "slot_intent_distribution": {
            "val": {"command_intents": {"other": val_examples}}
        },
        "source_overlap_command_slot_baseline": {
            "val": {"accuracy": source, "total": val_examples, "correct": int(source * val_examples)}
        },
        "pairwise_candidate_encoding": {
            "train": {"pairwise_scored_candidates": 0 if not identity_pairwise else 4},
            "val": {"pairwise_scored_candidates": 0 if not identity_pairwise else 4},
        },
        "history": [
            {
                "val_metrics": {
                    "command_slot_accuracy": val_accuracy,
                    "command_slot_count": val_examples,
                }
            }
        ],
    }


def test_phase2af_smoke_postflight_accepts_smoke_but_blocks_package_scale(
    tmp_path: Path,
) -> None:
    pretrain = _write(tmp_path / "pretrain.json", _pretrain())
    summary = _write(tmp_path / "summary.json", _summary(tmp_path))
    head_manifest = _write(
        tmp_path / "head_manifest.json",
        {"dataset_family": "phase2s_public_repair_head_dataset"},
    )

    report = build_phase2af_smoke_postflight(
        pretrain_gate_json=pretrain,
        training_summary_json=summary,
        head_manifest_json=head_manifest,
    )

    assert report["passed"] is True
    assert report["ready_for_full_train"] is False
    assert report["ready_for_package"] is False
    assert report["metrics"]["model_minus_source_overlap_accuracy"] == 0.6
    assert report["metrics"]["model_minus_runtime_identity_accuracy"] == 0.09999999999999998
    assert "expand_or_collect_nonsealed_phase2af_rows_before_full_train" in report["blocked_actions"]


def test_phase2af_smoke_postflight_rejects_identity_heuristic_tie(
    tmp_path: Path,
) -> None:
    pretrain = _write(tmp_path / "pretrain.json", _pretrain(identity=1.0))
    summary = _write(tmp_path / "summary.json", _summary(tmp_path, val_accuracy=1.0))

    report = build_phase2af_smoke_postflight(
        pretrain_gate_json=pretrain,
        training_summary_json=summary,
    )

    assert report["passed"] is False
    assert report["checks"]["runtime_identity_not_sufficient"] is False
    assert report["checks"]["model_beats_runtime_identity"] is False
    assert "do_not_claim_phase2af_hardened_structural_sidecar_mechanism" in report["blocked_actions"]


def test_phase2af_smoke_postflight_rejects_pairwise_confounded_run(
    tmp_path: Path,
) -> None:
    pretrain = _write(tmp_path / "pretrain.json", _pretrain())
    summary = _write(
        tmp_path / "summary.json",
        _summary(tmp_path, identity_pairwise=True),
    )

    report = build_phase2af_smoke_postflight(
        pretrain_gate_json=pretrain,
        training_summary_json=summary,
    )

    assert report["passed"] is False
    assert report["checks"]["pairwise_disabled"] is False
    assert report["checks"]["pairwise_encoded_candidates_zero"] is False


def test_phase2af_smoke_postflight_allows_full_train_only_when_rows_scale(
    tmp_path: Path,
) -> None:
    pretrain = _write(tmp_path / "pretrain.json", _pretrain())
    summary = _write(
        tmp_path / "summary.json",
        _summary(tmp_path, train_examples=128, val_examples=64),
    )

    report = build_phase2af_smoke_postflight(
        pretrain_gate_json=pretrain,
        training_summary_json=summary,
        min_val_rows=30,
    )

    assert report["passed"] is True
    assert report["ready_for_full_train"] is True
    assert report["allowed_next_action"] == "run_phase2af_full_nonsealed_training_only"
