import json
from pathlib import Path

from reflexlm.cli.audit_phase2at_smoke_postflight import build_phase2at_smoke_postflight


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _data_health(passed: bool = True) -> dict:
    return {"passed": passed}


def _summary() -> dict:
    return {
        "effective_split_hashes": {"phase2c_head_train": "a", "phase2c_head_val": "b"},
        "learned_patch_descriptor_heads": {"enabled": True},
    }


def _eval_report(template_accuracy: float = 1.0) -> dict:
    return {
        "eval_split": "val",
        "eval_rows_hash": "abc",
        "open_repair_heads_enabled": True,
        "use_pairwise_command_reranker": False,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "head_config": {"open_repair_heads_enabled": True},
        "source_overlap_command_slot_baseline": {"val": {"accuracy": 0.5}},
        "eval_metrics": {
            "command_slot_accuracy": 1.0,
            "patch_operation_accuracy": 1.0,
            "patch_operation_count": 32.0,
            "patch_target_file_slot_accuracy": 1.0,
            "patch_target_file_slot_count": 32.0,
            "patch_template_slot_accuracy": template_accuracy,
            "patch_template_slot_count": 32.0,
        },
    }


def test_phase2at_smoke_postflight_accepts_descriptor_signal(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    summary = tmp_path / "summary.json"
    eval_report = tmp_path / "eval.json"
    _write_json(data, _data_health())
    _write_json(summary, _summary())
    _write_json(eval_report, _eval_report())

    report = build_phase2at_smoke_postflight(
        data_health_json=data,
        training_summary_json=summary,
        eval_json=eval_report,
    )

    assert report["passed"] is True
    assert report["checks"]["model_beats_source_overlap"] is True
    assert report["supported_claims"] == [
        "phase2at_nonsealed_smoke_supports_bounded_descriptor_learning"
    ]


def test_phase2at_smoke_postflight_rejects_weak_template_head(tmp_path: Path) -> None:
    data = tmp_path / "data.json"
    summary = tmp_path / "summary.json"
    eval_report = tmp_path / "eval.json"
    _write_json(data, _data_health())
    _write_json(summary, _summary())
    _write_json(eval_report, _eval_report(template_accuracy=0.47))

    report = build_phase2at_smoke_postflight(
        data_health_json=data,
        training_summary_json=summary,
        eval_json=eval_report,
    )

    assert report["passed"] is False
    assert report["checks"]["patch_template_slot_gate"] is False
    assert "do_not_start_phase2at_full_training_from_this_smoke" in report["blocked_actions"]
