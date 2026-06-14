import json
from pathlib import Path

from reflexlm.cli.audit_phase2j_full_postflight import build_phase2j_full_postflight
from reflexlm.llm.candidate_features import CANDIDATE_FEATURE_DIM


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(
    *,
    val_accuracy: float = 0.90,
    source_overlap_accuracy: float = 0.20,
    action_accuracy: float = 1.0,
    command_intent_accuracy: float = 1.0,
    feature_dim: int = CANDIDATE_FEATURE_DIM,
) -> dict:
    return {
        "train_examples": 576,
        "val_examples": 192,
        "command_candidate_feature_dim": feature_dim,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "use_pairwise_command_reranker": False,
        "effective_split_hashes": {
            "phase2c_head_train": "a" * 64,
            "phase2c_head_val": "b" * 64,
        },
        "config": {
            "max_train_records": 1024,
            "max_val_records": 512,
            "latent_fusion": "additive",
        },
        "head_config": {
            "nsi_latent_dim": 30,
            "latent_fusion": "additive",
            "command_candidate_feature_dim": feature_dim,
            "use_pairwise_command_reranker": False,
        },
        "source_overlap_command_slot_baseline": {
            "val": {"total": 64, "accuracy": source_overlap_accuracy}
        },
        "history": [
            {
                "train_elapsed_seconds": 500.0,
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
            "duration_seconds": 520.0,
        },
    }


def test_phase2j_full_postflight_allows_package_after_full_mechanism_gate(
    tmp_path: Path,
) -> None:
    summary = _write(tmp_path / "summary.json", _summary())
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "effective_split_hashes": {
                "phase2j_head_train": "a" * 64,
                "phase2j_head_val": "b" * 64,
            },
        },
    )
    smoke = _write(tmp_path / "smoke.json", {"passed": True})

    report = build_phase2j_full_postflight(
        training_summary_json=summary,
        data_health_json=data_health,
        smoke_postflight_json=smoke,
    )

    assert report["passed"] is True
    assert report["ready_for_package"] is True
    assert report["checks"]["val_action_gate_passed"] is True
    assert report["checks"]["slot_aware_candidate_features_recorded"] is True


def test_phase2j_full_postflight_rejects_legacy_candidate_feature_dim(
    tmp_path: Path,
) -> None:
    summary = _write(tmp_path / "summary.json", _summary(feature_dim=CANDIDATE_FEATURE_DIM - 1))
    smoke = _write(tmp_path / "smoke.json", {"passed": True})

    report = build_phase2j_full_postflight(
        training_summary_json=summary,
        smoke_postflight_json=smoke,
    )

    assert report["passed"] is False
    assert report["checks"]["slot_aware_candidate_features_recorded"] is False
    assert "do_not_package_without_slot_aware_candidate_feature_evidence" in report[
        "blocked_actions"
    ]


def test_phase2j_full_postflight_blocks_package_without_delta(
    tmp_path: Path,
) -> None:
    summary = _write(
        tmp_path / "summary.json",
        _summary(val_accuracy=0.80, source_overlap_accuracy=0.80),
    )
    smoke = _write(tmp_path / "smoke.json", {"passed": True})

    report = build_phase2j_full_postflight(
        training_summary_json=summary,
        smoke_postflight_json=smoke,
    )

    assert report["passed"] is False
    assert report["checks"]["model_beats_source_overlap_baseline"] is False
    assert "do_not_package_without_phase2j_mechanism_increment" in report["blocked_actions"]


def test_phase2j_full_postflight_blocks_package_when_action_gate_fails(
    tmp_path: Path,
) -> None:
    summary = _write(
        tmp_path / "summary.json",
        _summary(
            val_accuracy=1.0,
            source_overlap_accuracy=0.20,
            action_accuracy=0.6666666667,
        ),
    )
    smoke = _write(tmp_path / "smoke.json", {"passed": True})

    report = build_phase2j_full_postflight(
        training_summary_json=summary,
        smoke_postflight_json=smoke,
    )

    assert report["passed"] is False
    assert report["checks"]["val_action_gate_passed"] is False
    assert "do_not_package_until_action_gate_passes" in report["blocked_actions"]
