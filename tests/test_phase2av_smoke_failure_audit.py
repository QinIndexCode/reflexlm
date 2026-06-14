import json
from pathlib import Path

from reflexlm.cli.build_phase2av_smoke_failure_audit import (
    build_phase2av_smoke_failure_audit,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2av_failure_audit_classifies_descriptor_head_gap(tmp_path: Path) -> None:
    postflight = _write(
        tmp_path / "postflight.json",
        {
            "passed": False,
            "metrics": {
                "command_slot_accuracy": 0.7,
                "source_overlap_accuracy": 0.4,
                "model_minus_source_overlap_accuracy": 0.3,
                "patch_operation_accuracy": 0.5,
                "patch_template_slot_accuracy": 0.45,
                "patch_target_file_slot_accuracy": 1.0,
            },
            "failure_modes": ["patch_operation_accuracy_below_gate"],
        },
    )
    summary = _write(
        tmp_path / "summary.json",
        {
            "open_repair_training_contract": {"sealed_feedback_used": False},
            "history": [
                {
                    "epoch": 1,
                    "first_train_loss": 15.0,
                    "train_loss": 7.0,
                    "val_metrics": {
                        "command_slot_accuracy": 0.6,
                        "patch_operation_accuracy": 0.4,
                        "patch_template_slot_accuracy": 0.4,
                    },
                },
                {
                    "epoch": 2,
                    "train_loss": 2.0,
                    "val_metrics": {
                        "command_slot_accuracy": 0.75,
                        "patch_operation_accuracy": 0.5,
                        "patch_template_slot_accuracy": 0.45,
                    },
                },
            ],
        },
    )
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})
    pool = _write(tmp_path / "pool.json", {"passed": True})

    report = build_phase2av_smoke_failure_audit(
        postflight_json=postflight,
        training_summary_json=summary,
        pretrain_gate_json=pretrain,
        pool_gap_json=pool,
    )

    assert report["preconditions"]["pretrain_gate_passed"] is True
    assert "nonzero_descriptor_runtime_signal_above_source_overlap" in report[
        "issue_classification"
    ]
    assert "descriptor_operation_template_heads_underfit_or_confused" in report[
        "issue_classification"
    ]
    assert "do_not_package_phase2av" in report["blocked_actions"]
    assert "epoch_making_architecture" in report["unsupported_claims"]


def test_phase2av_failure_audit_flags_val_regression(tmp_path: Path) -> None:
    postflight = _write(
        tmp_path / "postflight.json",
        {
            "passed": False,
            "metrics": {
                "command_slot_accuracy": 0.6,
                "source_overlap_accuracy": 0.5,
                "model_minus_source_overlap_accuracy": 0.1,
                "patch_operation_accuracy": 0.5,
                "patch_template_slot_accuracy": 0.5,
                "patch_target_file_slot_accuracy": 1.0,
            },
        },
    )
    summary = _write(
        tmp_path / "summary.json",
        {
            "open_repair_training_contract": {"sealed_feedback_used": False},
            "history": [
                {
                    "epoch": 1,
                    "first_train_loss": 10.0,
                    "train_loss": 5.0,
                    "val_metrics": {"command_slot_accuracy": 0.7},
                },
                {
                    "epoch": 2,
                    "train_loss": 1.0,
                    "val_metrics": {"command_slot_accuracy": 0.6},
                },
            ],
        },
    )
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})
    pool = _write(tmp_path / "pool.json", {"passed": True})

    report = build_phase2av_smoke_failure_audit(
        postflight_json=postflight,
        training_summary_json=summary,
        pretrain_gate_json=pretrain,
        pool_gap_json=pool,
    )

    assert "late_epoch_overfit_or_val_regression" in report["issue_classification"]
