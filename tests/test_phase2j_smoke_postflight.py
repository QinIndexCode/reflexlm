import json
from pathlib import Path

from reflexlm.cli.audit_phase2j_smoke_postflight import build_phase2j_smoke_postflight


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(
    *,
    val_accuracy: float = 0.90,
    source_overlap_accuracy: float = 0.80,
    action_accuracy: float = 1.0,
    command_intent_accuracy: float = 1.0,
) -> dict:
    return {
        "train_examples": 128,
        "val_examples": 192,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "use_pairwise_command_reranker": False,
        "effective_split_hashes": {
            "phase2c_head_train": "a" * 64,
            "phase2c_head_val": "b" * 64,
        },
        "config": {
            "max_train_records": 128,
            "max_val_records": 192,
            "latent_fusion": "additive",
        },
        "head_config": {
            "nsi_latent_dim": 30,
            "latent_fusion": "additive",
            "use_pairwise_command_reranker": False,
        },
        "source_overlap_command_slot_baseline": {
            "val": {"total": 64, "accuracy": source_overlap_accuracy}
        },
        "history": [
            {
                "train_elapsed_seconds": 100.0,
                "train_steps_per_second": 1.0,
                "train_pairwise_encoded_candidates": 0,
                "val_metrics": {
                    "elapsed_seconds": 20.0,
                    "action_accuracy": action_accuracy,
                    "action_count": 192,
                    "command_intent_accuracy": command_intent_accuracy,
                    "command_intent_count": 64,
                    "command_slot_accuracy": val_accuracy,
                    "command_slot_count": 64,
                    "pairwise_encoded_candidates": 0.0,
                },
            }
        ],
        "run_manifest": {
            "finished_at_utc": "2026-05-20T00:00:00+00:00",
            "duration_seconds": 120.0,
        },
    }


def _data_health() -> dict:
    return {
        "passed": True,
        "effective_split_hashes": {
            "phase2j_head_train": "c" * 64,
            "phase2j_head_val": "b" * 64,
        },
    }


def test_phase2j_smoke_postflight_allows_full_only_when_model_beats_source_overlap(
    tmp_path: Path,
) -> None:
    summary = _write(tmp_path / "summary.json", _summary())
    data_health = _write(tmp_path / "data_health.json", _data_health())
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})

    report = build_phase2j_smoke_postflight(
        training_summary_json=summary,
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
        min_mechanism_delta_vs_source_overlap=0.01,
    )

    assert report["passed"] is True
    assert report["smoke_viable"] is True
    assert report["mechanism_increment_supported"] is True
    assert report["ready_for_full_training"] is True
    assert report["checks"]["val_action_gate_passed"] is True


def test_phase2j_smoke_postflight_blocks_full_when_model_only_matches_source_overlap(
    tmp_path: Path,
) -> None:
    summary = _write(
        tmp_path / "summary.json",
        _summary(val_accuracy=0.921875, source_overlap_accuracy=0.921875),
    )
    data_health = _write(tmp_path / "data_health.json", _data_health())
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})

    report = build_phase2j_smoke_postflight(
        training_summary_json=summary,
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
        min_mechanism_delta_vs_source_overlap=0.01,
    )

    assert report["passed"] is False
    assert report["smoke_viable"] is True
    assert report["mechanism_increment_supported"] is False
    assert "do_not_start_full_training_without_beating_source_overlap_baseline" in report[
        "blocked_actions"
    ]


def test_phase2j_smoke_postflight_blocks_full_when_action_gate_fails(
    tmp_path: Path,
) -> None:
    summary = _write(
        tmp_path / "summary.json",
        _summary(
            val_accuracy=1.0,
            source_overlap_accuracy=0.1666666667,
            action_accuracy=0.6666666667,
        ),
    )
    data_health = _write(tmp_path / "data_health.json", _data_health())
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})

    report = build_phase2j_smoke_postflight(
        training_summary_json=summary,
        data_health_json=data_health,
        pretrain_gate_json=pretrain,
    )

    assert report["passed"] is False
    assert report["smoke_viable"] is False
    assert report["mechanism_increment_supported"] is True
    assert report["checks"]["val_action_gate_passed"] is False
    assert "do_not_start_full_training_until_action_gate_passes" in report["blocked_actions"]


def test_phase2j_smoke_postflight_rejects_pairwise_confounded_smoke(tmp_path: Path) -> None:
    payload = _summary()
    payload["use_pairwise_command_reranker"] = True
    payload["head_config"]["use_pairwise_command_reranker"] = True
    payload["history"][0]["train_pairwise_encoded_candidates"] = 12
    summary = _write(tmp_path / "summary.json", payload)

    report = build_phase2j_smoke_postflight(training_summary_json=summary)

    assert report["smoke_viable"] is False
    assert report["checks"]["pairwise_disabled_for_phase2j_isolation"] is False
