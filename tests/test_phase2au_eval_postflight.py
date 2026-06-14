import json
from pathlib import Path

from reflexlm.cli.audit_phase2au_eval_postflight import audit_phase2au_eval_postflight


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(*, accuracy: float = 1.0, source_overlap: float = 0.55, delta_rows: int = 3) -> dict:
    records = [
        {"source_overlap_correct": False, "command_slot_correct": True}
        for _ in range(delta_rows)
    ]
    return {
        "eval_split": "holdout",
        "eval_examples": 20,
        "eval_rows_hash": "a" * 64,
        "training_summary_config_hash": "cfg",
        "low_level_qwen_calls_target": 0,
        "eval_metrics": {
            "command_slot_accuracy": accuracy,
            "patch_operation_count": 20,
            "patch_template_slot_count": 20,
        },
        "source_overlap_command_slot_baseline": {
            "holdout": {"accuracy": source_overlap}
        },
        "pairwise_candidate_encoding": {
            "holdout": {"max_valid_candidates_per_row": 4}
        },
        "prediction_records": records,
    }


def test_phase2au_eval_postflight_accepts_holdout_delta(tmp_path: Path) -> None:
    summary = _write(tmp_path / "summary.json", _summary())
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})

    report = audit_phase2au_eval_postflight(
        eval_summary_json=summary,
        pretrain_gate_json=pretrain,
    )

    assert report["passed"] is True
    assert report["checks"]["model_minus_source_overlap_gate"] is True
    assert report["metrics"]["model_minus_source_overlap"] == 0.44999999999999996
    assert report["metrics"]["source_overlap_delta_rows"] == 3


def test_phase2au_eval_postflight_rejects_insufficient_delta(tmp_path: Path) -> None:
    summary = _write(tmp_path / "summary.json", _summary(accuracy=0.7, source_overlap=0.65))
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})

    report = audit_phase2au_eval_postflight(
        eval_summary_json=summary,
        pretrain_gate_json=pretrain,
    )

    assert report["passed"] is False
    assert report["checks"]["eval_accuracy_gate"] is False
    assert report["checks"]["model_minus_source_overlap_gate"] is False
    assert "do_not_claim_phase2au_runtime_delta" in report["blocked_actions"]


def test_phase2au_eval_postflight_rejects_no_source_overlap_delta_rows(tmp_path: Path) -> None:
    summary = _write(tmp_path / "summary.json", _summary(delta_rows=0))
    pretrain = _write(tmp_path / "pretrain.json", {"passed": True})

    report = audit_phase2au_eval_postflight(
        eval_summary_json=summary,
        pretrain_gate_json=pretrain,
    )

    assert report["passed"] is False
    assert report["checks"]["source_overlap_delta_rows_present"] is False
