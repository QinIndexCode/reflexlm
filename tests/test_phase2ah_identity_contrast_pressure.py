import json
from pathlib import Path

from reflexlm.cli.build_phase2ah_identity_contrast_pressure import (
    build_phase2ah_identity_contrast_pressure,
    phase2ah_identity_contrast_row,
)
from reflexlm.cli.audit_phase2ah_identity_contrast_postflight import (
    build_phase2ah_identity_contrast_postflight,
)
from reflexlm.cli.build_phase2af_hardened_structural_sidecar_split import (
    _row_candidate,
    _shortcut_key,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(index: int) -> dict:
    candidates = [
        {
            "repair_action": f"repair_action_{index}_{slot}",
            "intent": "apply_patch_and_rerun_tests",
            "edit_scope": f"pkg/module_{slot}.py",
            "target_symbol": f"symbol_{index}_{slot}",
            "verification_command": "python -m pytest -q <generated_repair_test> --maxfail=1",
        }
        for slot in range(3)
    ]
    return {
        "trace_id": f"val:repo_{index}:{index}",
        "split": "val",
        "source_kind": "public_repo",
        "repo_id": f"repo_{index % 3}",
        "repo_url_or_origin": f"https://example.invalid/repo_{index % 3}.git",
        "current_visible_text": "public runtime repair evidence without oracle markers",
        "runtime_visible_evidence": {
            "changed_files": ["pkg/module_1.py"],
            "traceback_symbols": ["unhelpful_symbol"],
            "watched_files": ["tests/test_generated.py"],
            "pytest_before_patch": {"stdout_excerpt": "AssertionError"},
        },
        "repair_candidates": candidates,
        "expected_repair_action": candidates[1]["repair_action"],
        "expected_repair_result": {"test_target": "phase2s_repair_tests/test_case.py"},
        "normalization": {"sealed_feedback_absent": True},
    }


def test_phase2ah_identity_contrast_row_creates_source_correct_identity_wrong() -> None:
    converted = phase2ah_identity_contrast_row(_row(0))

    assert converted is not None
    candidate = _row_candidate(converted, require_tie_residual_feasible=True)
    assert candidate is not None
    assert _shortcut_key(candidate) == (1, 0)
    assert converted["sealed_feedback_used"] is False
    assert converted["claim_boundary"].startswith("nonsealed_adversarial")


def test_phase2ah_identity_contrast_builder_writes_manifest(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(6)]
    source = _write_jsonl(tmp_path / "rows.jsonl", rows)

    manifest = build_phase2ah_identity_contrast_pressure(
        train_jsonl=source,
        val_jsonl=source,
        holdout_jsonl=source,
        output_dir=tmp_path / "out",
        manifest_json=tmp_path / "manifest.json",
        min_val_rows=4,
        min_holdout_rows=4,
    )

    assert manifest["passed"] is True
    assert manifest["claim_bearing_natural_trace_evidence"] is False
    assert manifest["bucket_counts"]["val"] == {"source_1_identity_0": 6}
    assert (tmp_path / "out" / "train.jsonl").exists()
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[
        "allowed_next_action"
    ] == "build_phase2ah_head_dataset_and_smoke_train"


def _postflight_paths(tmp_path: Path) -> dict[str, Path]:
    adapter = tmp_path / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "native_heads.pt").write_bytes(b"heads")
    (adapter / "head_config.json").write_text("{}", encoding="utf-8")
    split_manifest = {
        "passed": True,
        "claim_bearing_natural_trace_evidence": False,
        "checks": {"sealed_feedback_absent": True},
        "split_counts": {"train": 42, "val": 43, "holdout": 66},
        "bucket_counts": {
            "val": {"source_1_identity_0": 43},
            "holdout": {"source_1_identity_0": 66},
        },
        "unsupported_claims": [
            "sealed_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
    }
    head_manifest = {
        "splits": {
            "train": {"sha256": "train-head-hash"},
            "val": {"sha256": "val-head-hash"},
        }
    }
    holdout_head_manifest = {"splits": {"val": {"sha256": "holdout-head-hash"}}}
    summary = {
        "adapter_output_dir": str(adapter),
        "train_examples": 42,
        "val_examples": 43,
        "use_pairwise_command_reranker": False,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "effective_split_hashes": {
            "phase2c_head_train": "train-head-hash",
            "phase2c_head_val": "val-head-hash",
        },
        "pairwise_candidate_encoding": {
            "train": {"pairwise_scored_candidates": 0},
            "val": {"pairwise_scored_candidates": 0},
        },
        "config": {"device": "cuda"},
        "history": [
            {
                "val_metrics": {
                    "command_slot_accuracy": 0.91,
                    "command_slot_count": 43,
                }
            }
        ],
    }
    holdout_eval = {
        "eval_examples": 66,
        "eval_rows_hash": "holdout-head-hash",
        "effective_split_hashes": {
            "phase2c_head_phase2ah_holdout": "holdout-head-hash"
        },
        "use_pairwise_command_reranker": False,
        "no_json_motor_target": True,
        "low_level_qwen_calls_target": 0,
        "device": "cuda:0",
        "pairwise_candidate_encoding": {
            "phase2ah_holdout": {"pairwise_scored_candidates": 0}
        },
        "source_overlap_command_slot_baseline": {
            "phase2ah_holdout": {"accuracy": 1.0}
        },
        "eval_metrics": {
            "command_slot_accuracy": 0.87,
            "command_slot_count": 66,
        },
    }
    old_eval = dict(holdout_eval)
    old_eval["eval_metrics"] = {
        "command_slot_accuracy": 0.16,
        "command_slot_count": 66,
    }
    return {
        "split_manifest_json": _write(tmp_path / "split_manifest.json", split_manifest),
        "head_manifest_json": _write(tmp_path / "head_manifest.json", head_manifest),
        "holdout_head_manifest_json": _write(
            tmp_path / "holdout_head_manifest.json", holdout_head_manifest
        ),
        "training_summary_json": _write(tmp_path / "summary.json", summary),
        "holdout_eval_json": _write(tmp_path / "holdout_eval.json", holdout_eval),
        "old_adapter_holdout_eval_json": _write(tmp_path / "old_eval.json", old_eval),
    }


def test_phase2ah_postflight_accepts_adversarial_pressure_but_blocks_claims(
    tmp_path: Path,
) -> None:
    report = build_phase2ah_identity_contrast_postflight(**_postflight_paths(tmp_path))

    assert report["passed"] is True
    assert report["adversarial_identity_contrast_pressure_evidence"] is True
    assert report["claim_bearing_mechanism_evidence"] is False
    assert report["ready_for_package"] is False
    assert report["metrics"]["holdout_model_minus_source_overlap_accuracy"] == -0.13
    assert "do_not_claim_epoch_making_architecture" in report["blocked_actions"]


def test_phase2ah_postflight_rejects_old_adapter_that_also_solves_pressure(
    tmp_path: Path,
) -> None:
    paths = _postflight_paths(tmp_path)
    old_eval = json.loads(paths["old_adapter_holdout_eval_json"].read_text(encoding="utf-8"))
    old_eval["eval_metrics"]["command_slot_accuracy"] = 0.75
    paths["old_adapter_holdout_eval_json"] = _write(tmp_path / "old_eval.json", old_eval)

    report = build_phase2ah_identity_contrast_postflight(**paths)

    assert report["passed"] is False
    assert report["checks"]["old_adapter_degraded_by_wrong_identity"] is False


def test_phase2ah_postflight_rejects_non_ceiling_source_overlap(
    tmp_path: Path,
) -> None:
    paths = _postflight_paths(tmp_path)
    holdout = json.loads(paths["holdout_eval_json"].read_text(encoding="utf-8"))
    holdout["source_overlap_command_slot_baseline"]["phase2ah_holdout"]["accuracy"] = 0.8
    paths["holdout_eval_json"] = _write(tmp_path / "holdout_eval.json", holdout)

    report = build_phase2ah_identity_contrast_postflight(**paths)

    assert report["passed"] is False
    assert report["checks"]["source_overlap_ceiling_expected_for_pressure_split"] is False
