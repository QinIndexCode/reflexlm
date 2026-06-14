import json
from pathlib import Path

from reflexlm.cli.audit_phase2al_postflight import audit_phase2al_postflight


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pretrain(source_accuracy: float = 0.4, identity_accuracy: float = 1.0) -> dict:
    return {
        "passed": True,
        "claim_bearing_natural_trace_evidence": False,
        "split_metrics": {
            "val": {
                "identity_text_ablated_source_overlap": {
                    "accuracy": source_accuracy,
                    "correct": int(source_accuracy * 10),
                    "total": 10,
                },
                "runtime_identity_heuristic": {
                    "accuracy": identity_accuracy,
                    "correct": int(identity_accuracy * 10),
                    "total": 10,
                },
            }
        },
    }


def _summary(model_accuracy: float = 1.0, source_accuracy: float = 0.4) -> dict:
    return {
        "device": "cuda:0",
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "use_pairwise_command_reranker": False,
        "source_overlap_command_slot_baseline": {
            "val": {"accuracy": source_accuracy, "correct": int(source_accuracy * 10), "total": 10}
        },
        "history": [
            {
                "val_metrics": {
                    "command_slot_accuracy": model_accuracy,
                    "command_slot_count": 32,
                }
            }
        ],
    }


def test_phase2al_postflight_accepts_model_over_non_ceiling_source_overlap(
    tmp_path: Path,
) -> None:
    pretrain = _write_json(tmp_path / "pretrain.json", _pretrain())
    summary = _write_json(tmp_path / "summary.json", _summary())

    report = audit_phase2al_postflight(
        pretrain_gate_json=pretrain,
        summary_json=summary,
        split="val",
        output_json=tmp_path / "out.json",
    )

    assert report["passed"] is True
    assert report["metrics"]["model_minus_source_overlap_accuracy"] == 0.6
    assert report["checks"]["model_near_runtime_identity_upper_bound"] is True


def test_phase2al_postflight_rejects_source_overlap_ceiling(tmp_path: Path) -> None:
    pretrain = _write_json(tmp_path / "pretrain.json", _pretrain(source_accuracy=1.0))
    summary = _write_json(tmp_path / "summary.json", _summary(source_accuracy=1.0))

    report = audit_phase2al_postflight(
        pretrain_gate_json=pretrain,
        summary_json=summary,
        split="val",
    )

    assert report["passed"] is False
    assert report["checks"]["source_overlap_not_ceiling"] is False
    assert "do_not_package_phase2al_for_sealed_eval" in report["blocked_actions"]


def test_phase2al_postflight_rejects_model_that_only_matches_source_overlap(
    tmp_path: Path,
) -> None:
    pretrain = _write_json(tmp_path / "pretrain.json", _pretrain(source_accuracy=0.6))
    summary = _write_json(tmp_path / "summary.json", _summary(model_accuracy=0.6, source_accuracy=0.6))

    report = audit_phase2al_postflight(
        pretrain_gate_json=pretrain,
        summary_json=summary,
        split="val",
    )

    assert report["passed"] is False
    assert report["checks"]["model_beats_source_overlap"] is False
