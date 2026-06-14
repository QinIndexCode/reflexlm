import json
from pathlib import Path

from reflexlm.cli.audit_phase2av_smoke_postflight import audit_phase2av_smoke_postflight


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _launch(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "launch.json",
        {"passed": True, "ready_to_start_phase2av_smoke_training": True},
    )


def _summary(*, command_slot: float = 0.9, source: float = 0.5) -> dict:
    return {
        "train_examples": 14,
        "val_examples": 8,
        "config_hash": "abc",
        "open_repair_heads_enabled": True,
        "learned_patch_descriptor_heads": {"enabled": True},
        "use_pairwise_command_reranker": False,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "source_overlap_command_slot_baseline": {
            "val": {"accuracy": source, "total": 8, "correct": int(source * 8)}
        },
        "pairwise_candidate_encoding": {
            "val": {"max_valid_candidates_per_row": 4}
        },
        "history": [
            {
                "val_metrics": {
                    "command_slot_accuracy": command_slot,
                    "patch_operation_accuracy": 0.9,
                    "patch_template_slot_accuracy": 0.9,
                    "patch_target_file_slot_accuracy": 1.0,
                    "patch_operation_count": 8.0,
                    "patch_template_slot_count": 8.0,
                    "slot_confusion": {
                        "command_slot": {"0": {"0": 4}, "1": {"1": 4}},
                        "patch_operation": {"1": {"1": 4}, "2": {"2": 4}},
                    },
                }
            }
        ],
    }


def test_phase2av_smoke_postflight_accepts_nonsealed_descriptor_smoke(
    tmp_path: Path,
) -> None:
    report = audit_phase2av_smoke_postflight(
        training_summary_json=_write(tmp_path / "summary.json", _summary()),
        launch_gate_json=_launch(tmp_path),
    )

    assert report["passed"] is True
    assert (
        "phase2av_nonsealed_smoke_supports_bounded_descriptor_runtime_learning"
        in report["supported_claims"]
    )


def test_phase2av_smoke_postflight_rejects_source_overlap_tie_and_slot0_collapse(
    tmp_path: Path,
) -> None:
    summary = _summary(command_slot=0.625, source=0.625)
    summary["history"][0]["val_metrics"]["patch_operation_accuracy"] = 0.75
    summary["history"][0]["val_metrics"]["patch_template_slot_accuracy"] = 0.75
    summary["history"][0]["val_metrics"]["slot_confusion"] = {
        "command_slot": {"0": {"0": 5}, "1": {"0": 3}},
        "patch_operation": {"1": {"2": 2}, "2": {"2": 6}},
    }
    report = audit_phase2av_smoke_postflight(
        training_summary_json=_write(tmp_path / "summary.json", summary),
        launch_gate_json=_launch(tmp_path),
    )

    assert report["passed"] is False
    assert "model_does_not_beat_source_overlap" in report["failure_modes"]
    assert "command_slot_majority_slot0_collapse" in report["failure_modes"]
    assert "patch_operation_majority_insert_import_collapse" in report["failure_modes"]
    assert "do_not_start_phase2av_full_training" in report["blocked_actions"]
