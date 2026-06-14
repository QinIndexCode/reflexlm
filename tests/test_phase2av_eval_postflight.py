import json
from pathlib import Path

from reflexlm.cli.audit_phase2av_eval_postflight import audit_phase2av_eval_postflight


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(*, command_slot: float = 0.9, source: float = 0.4) -> dict:
    return {
        "eval_split": "holdout",
        "eval_examples": 10,
        "eval_rows_hash": "abc",
        "open_repair_heads_enabled": True,
        "use_pairwise_command_reranker": False,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "source_overlap_command_slot_baseline": {
            "holdout": {"accuracy": source, "total": 10, "correct": int(source * 10)}
        },
        "pairwise_candidate_encoding": {
            "holdout": {"max_valid_candidates_per_row": 4}
        },
        "eval_metrics": {
            "command_slot_accuracy": command_slot,
            "patch_operation_accuracy": 1.0,
            "patch_template_slot_accuracy": 1.0,
            "patch_target_file_slot_accuracy": 1.0,
            "patch_operation_count": 10.0,
            "patch_template_slot_count": 10.0,
            "slot_confusion": {
                "command_slot": {"0": {"0": 5}, "1": {"1": 5}},
                "patch_operation": {"1": {"1": 5}, "2": {"2": 5}},
            },
        },
    }


def test_phase2av_eval_postflight_accepts_holdout_delta(tmp_path: Path) -> None:
    report = audit_phase2av_eval_postflight(
        eval_summary_json=_write(tmp_path / "eval.json", _summary()),
        eval_split="holdout",
    )

    assert report["passed"] is True
    assert (
        "phase2av_nonsealed_holdout_supports_bounded_descriptor_runtime_learning"
        in report["supported_claims"]
    )
    assert report["ready_for_phase2av_full_training"] is False


def test_phase2av_eval_postflight_rejects_source_overlap_tie(tmp_path: Path) -> None:
    report = audit_phase2av_eval_postflight(
        eval_summary_json=_write(tmp_path / "eval.json", _summary(command_slot=0.6, source=0.6)),
        eval_split="holdout",
    )

    assert report["passed"] is False
    assert "model_does_not_beat_source_overlap" in report["failure_modes"]
    assert "do_not_start_phase2av_full_training" in report["blocked_actions"]
