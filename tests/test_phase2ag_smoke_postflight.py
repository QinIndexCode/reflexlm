import json
from pathlib import Path

from reflexlm.cli.audit_phase2ag_smoke_postflight import (
    build_phase2ag_smoke_postflight,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _audit() -> dict:
    return {"artifact_family": "phase2ag_verifiable_candidate_sidecar_audit", "passed": True}


def _split_manifest() -> dict:
    return {
        "artifact_family": "phase2ag_verifiable_candidate_sidecar_split",
        "passed": True,
        "checks": {
            "repo_disjoint": True,
            "train_covers_val_and_holdout_slots": True,
        },
    }


def _head_manifest() -> dict:
    return {
        "source_data_health_passed": True,
        "source_pretrain_gate_passed": True,
        "splits": {
            "train": {"sha256": "train-hash"},
            "val": {"sha256": "val-hash"},
        },
    }


def _holdout_head_manifest() -> dict:
    return {
        "source_data_health_passed": True,
        "source_pretrain_gate_passed": True,
        "splits": {
            "train": {"sha256": "train-hash"},
            "val": {"sha256": "holdout-hash"},
        },
    }


def _summary(tmp_path: Path, *, val_accuracy: float = 1.0) -> dict:
    adapter = tmp_path / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "native_heads.pt").write_bytes(b"heads")
    (adapter / "head_config.json").write_text("{}", encoding="utf-8")
    return {
        "adapter_output_dir": str(adapter),
        "train_examples": 71,
        "val_examples": 56,
        "use_pairwise_command_reranker": False,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "effective_split_hashes": {
            "phase2c_head_train": "train-hash",
            "phase2c_head_val": "val-hash",
        },
        "config": {"device": "cuda"},
        "source_overlap_command_slot_baseline": {
            "val": {"accuracy": 0.70, "total": 56, "correct": 39}
        },
        "pairwise_candidate_encoding": {
            "train": {"pairwise_scored_candidates": 0},
            "val": {"pairwise_scored_candidates": 0},
        },
        "history": [
            {
                "val_metrics": {
                    "command_slot_accuracy": val_accuracy,
                    "command_slot_count": 56,
                }
            }
        ],
    }


def _holdout_eval(*, holdout_accuracy: float = 1.0) -> dict:
    return {
        "eval_examples": 58,
        "eval_rows_hash": "holdout-hash",
        "effective_split_hashes": {"phase2c_head_holdout": "holdout-hash"},
        "use_pairwise_command_reranker": False,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "device": "cuda:0",
        "source_overlap_command_slot_baseline": {
            "holdout": {"accuracy": 0.65, "total": 58, "correct": 38}
        },
        "pairwise_candidate_encoding": {
            "holdout": {"pairwise_scored_candidates": 0}
        },
        "eval_metrics": {
            "command_slot_accuracy": holdout_accuracy,
            "command_slot_count": 58,
        },
    }


def _guarded_holdout_eval() -> dict:
    payload = _holdout_eval()
    payload["effective_split_hashes"] = {
        "phase2c_head_holdout_guarded": "holdout-hash"
    }
    payload["source_overlap_command_slot_baseline"] = {
        "holdout_guarded": {"accuracy": 0.65, "total": 58, "correct": 38}
    }
    payload["pairwise_candidate_encoding"] = {
        "holdout_guarded": {"pairwise_scored_candidates": 0}
    }
    return payload


def _base_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "split_manifest_json": _write(tmp_path / "split.json", _split_manifest()),
        "train_audit_json": _write(tmp_path / "train_audit.json", _audit()),
        "val_audit_json": _write(tmp_path / "val_audit.json", _audit()),
        "holdout_audit_json": _write(tmp_path / "holdout_audit.json", _audit()),
        "head_manifest_json": _write(tmp_path / "head.json", _head_manifest()),
        "holdout_head_manifest_json": _write(
            tmp_path / "holdout_head.json", _holdout_head_manifest()
        ),
        "training_summary_json": _write(tmp_path / "summary.json", _summary(tmp_path)),
        "holdout_eval_json": _write(tmp_path / "holdout_eval.json", _holdout_eval()),
    }


def test_phase2ag_smoke_postflight_accepts_smoke_but_blocks_claims(
    tmp_path: Path,
) -> None:
    report = build_phase2ag_smoke_postflight(**_base_paths(tmp_path))

    assert report["passed"] is True
    assert report["ready_for_package"] is False
    assert report["ready_for_sealed_eval"] is False
    assert report["claim_bearing_mechanism_evidence"] is False
    assert report["metrics"]["holdout_model_minus_source_overlap_accuracy"] == 0.35
    assert "do_not_claim_epoch_making_architecture" in report["blocked_actions"]


def test_phase2ag_smoke_postflight_rejects_low_holdout_delta(tmp_path: Path) -> None:
    paths = _base_paths(tmp_path)
    paths["holdout_eval_json"] = _write(
        tmp_path / "holdout_eval.json", _holdout_eval(holdout_accuracy=0.76)
    )

    report = build_phase2ag_smoke_postflight(**paths)

    assert report["passed"] is False
    assert report["checks"]["holdout_model_beats_source_overlap"] is False
    assert "do_not_scale_phase2ag_training" in report["blocked_actions"]


def test_phase2ag_smoke_postflight_accepts_guarded_holdout_split_name(
    tmp_path: Path,
) -> None:
    paths = _base_paths(tmp_path)
    paths["holdout_eval_json"] = _write(
        tmp_path / "holdout_eval.json", _guarded_holdout_eval()
    )

    report = build_phase2ag_smoke_postflight(**paths)

    assert report["passed"] is True
    assert report["checks"]["holdout_hash_matches_eval_rows"] is True
    assert report["checks"]["holdout_source_overlap_present"] is True


def test_phase2ag_smoke_postflight_can_make_val_delta_non_claim_gate(
    tmp_path: Path,
) -> None:
    paths = _base_paths(tmp_path)
    summary = _summary(tmp_path)
    summary["source_overlap_command_slot_baseline"]["val"]["accuracy"] = 0.90
    paths["training_summary_json"] = _write(tmp_path / "summary.json", summary)

    strict_report = build_phase2ag_smoke_postflight(**paths)
    fullscale_report = build_phase2ag_smoke_postflight(
        **paths, require_val_model_minus_source_overlap=False
    )

    assert strict_report["passed"] is False
    assert strict_report["checks"]["val_model_beats_source_overlap"] is False
    assert fullscale_report["passed"] is True
    assert fullscale_report["thresholds"]["require_val_model_minus_source_overlap"] is False


def test_phase2ag_smoke_postflight_rejects_hash_drift(tmp_path: Path) -> None:
    paths = _base_paths(tmp_path)
    bad_holdout = _holdout_eval()
    bad_holdout["eval_rows_hash"] = "drift"
    paths["holdout_eval_json"] = _write(tmp_path / "holdout_eval.json", bad_holdout)

    report = build_phase2ag_smoke_postflight(**paths)

    assert report["passed"] is False
    assert report["checks"]["holdout_hash_matches_eval_rows"] is False


def test_phase2ag_smoke_postflight_rejects_failed_candidate_audit(
    tmp_path: Path,
) -> None:
    paths = _base_paths(tmp_path)
    paths["holdout_audit_json"] = _write(
        tmp_path / "holdout_audit.json",
        {"artifact_family": "phase2ag_verifiable_candidate_sidecar_audit", "passed": False},
    )

    report = build_phase2ag_smoke_postflight(**paths)

    assert report["passed"] is False
    assert report["checks"]["holdout_audit_passed"] is False
