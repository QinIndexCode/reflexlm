import json
from pathlib import Path

from reflexlm.cli.audit_phase2af_holdout_postflight import (
    build_phase2af_holdout_postflight,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metric(accuracy: float, *, total: int = 100) -> dict:
    return {"correct": int(accuracy * total), "total": total, "accuracy": accuracy}


def _pretrain(*, source: float = 0.45, identity: float = 0.70) -> dict:
    return {
        "passed": True,
        "split_metrics": {
            "holdout": {
                "raw_source_overlap": _metric(source),
                "identity_text_ablated_source_overlap": _metric(source),
                "runtime_identity_heuristic": _metric(identity),
            }
        },
    }


def _eval_summary(tmp_path: Path, *, accuracy: float = 0.90, source: float = 0.45) -> dict:
    adapter = tmp_path / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "native_heads.pt").write_bytes(b"heads")
    (adapter / "head_config.json").write_text("{}", encoding="utf-8")
    return {
        "adapter_output_dir": str(adapter),
        "eval_split": "holdout",
        "use_pairwise_command_reranker": False,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "config": {"device": "cuda"},
        "effective_split_hashes": {"phase2c_head_holdout": "holdout-hash"},
        "source_overlap_command_slot_baseline": {
            "holdout": _metric(source),
        },
        "pairwise_candidate_encoding": {
            "holdout": {"pairwise_scored_candidates": 0},
        },
        "eval_metrics": {
            "command_slot_accuracy": accuracy,
            "command_slot_count": 100,
        },
    }


def test_phase2af_holdout_postflight_accepts_nonsealed_holdout_signal(tmp_path: Path) -> None:
    pretrain = _write(tmp_path / "pretrain.json", _pretrain())
    summary = _write(tmp_path / "summary.json", _eval_summary(tmp_path))

    report = build_phase2af_holdout_postflight(
        pretrain_gate_json=pretrain,
        eval_summary_json=summary,
    )

    assert report["passed"] is True
    assert report["ready_for_package"] is False
    assert report["allowed_next_action"] == "run_phase2af_ablation_controls_before_package"
    assert report["metrics"]["model_minus_source_overlap_accuracy"] == 0.45
    assert report["metrics"]["model_minus_runtime_identity_accuracy"] == 0.20000000000000007


def test_phase2af_holdout_postflight_rejects_identity_tie(tmp_path: Path) -> None:
    pretrain = _write(tmp_path / "pretrain.json", _pretrain(identity=0.90))
    summary = _write(tmp_path / "summary.json", _eval_summary(tmp_path, accuracy=0.90))

    report = build_phase2af_holdout_postflight(
        pretrain_gate_json=pretrain,
        eval_summary_json=summary,
    )

    assert report["passed"] is False
    assert report["checks"]["model_beats_runtime_identity"] is False
    assert "do_not_package_phase2af" in report["blocked_actions"]


def test_phase2af_holdout_postflight_rejects_split_mismatch(tmp_path: Path) -> None:
    pretrain = _write(tmp_path / "pretrain.json", _pretrain())
    payload = _eval_summary(tmp_path)
    payload["eval_split"] = "val"
    summary = _write(tmp_path / "summary.json", payload)

    report = build_phase2af_holdout_postflight(
        pretrain_gate_json=pretrain,
        eval_summary_json=summary,
    )

    assert report["passed"] is False
    assert report["checks"]["eval_split_matches"] is False
