import json
from pathlib import Path

from reflexlm.cli.audit_phase2y_open_repair_smoke_postflight import (
    audit_phase2y_open_repair_smoke_postflight,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(*, val_accuracy: float = 0.90, source_accuracy: float = 0.70) -> dict:
    return {
        "train_examples": 96,
        "val_examples": 64,
        "open_repair_heads_enabled": True,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "source_overlap_command_slot_baseline": {
            "val": {"accuracy": source_accuracy}
        },
        "history": [
            {
                "train_loss": 1.0,
                "train_elapsed_seconds": 10,
                "val_metrics": {
                    "loss": 0.2,
                    "command_slot_accuracy": val_accuracy,
                },
            }
        ],
    }


def test_phase2y_smoke_postflight_accepts_head_control_smoke(tmp_path: Path) -> None:
    report = audit_phase2y_open_repair_smoke_postflight(
        pretrain_gate_json=_write(tmp_path / "pretrain.json", {"passed": True}),
        training_summary_json=_write(tmp_path / "summary.json", _summary()),
        runtime_capability_audit_json=_write(tmp_path / "runtime.json", {"passed": True}),
    )

    assert report["passed"] is True
    assert report["ready_for_phase2y_execution_runner_development"] is True
    assert report["ready_for_open_ended_claim"] is False


def test_phase2y_smoke_postflight_rejects_source_overlap_tie(tmp_path: Path) -> None:
    report = audit_phase2y_open_repair_smoke_postflight(
        pretrain_gate_json=_write(tmp_path / "pretrain.json", {"passed": True}),
        training_summary_json=_write(
            tmp_path / "summary.json", _summary(val_accuracy=0.90, source_accuracy=0.85)
        ),
        runtime_capability_audit_json=_write(tmp_path / "runtime.json", {"passed": True}),
    )

    assert report["passed"] is False
    assert report["checks"]["model_beats_source_overlap"] is False


def test_phase2y_smoke_postflight_rejects_missing_runtime_capability(tmp_path: Path) -> None:
    report = audit_phase2y_open_repair_smoke_postflight(
        pretrain_gate_json=_write(tmp_path / "pretrain.json", {"passed": True}),
        training_summary_json=_write(tmp_path / "summary.json", _summary()),
        runtime_capability_audit_json=_write(tmp_path / "runtime.json", {"passed": False}),
    )

    assert report["passed"] is False
    assert report["checks"]["runtime_capability_audit_passed"] is False
