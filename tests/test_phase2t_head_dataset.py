import json
from pathlib import Path

import pytest

from reflexlm.cli.build_phase2t_head_dataset import (
    build_phase2t_head_dataset,
    phase2t_repair_trace_to_head_row,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(trace_id: str = "train:repo:phase2t:0") -> dict:
    return {
        "phase": "Phase2T",
        "trace_id": trace_id,
        "split": "train",
        "source_kind": "public_repo",
        "trace_construction_mode": "phase2t_dynamic_public_repo_repair_loop_trace",
        "repo_id": "repo",
        "repo_url_or_origin": "https://github.com/example/repo.git",
        "commit_hash": "1" * 40,
        "current_visible_text": "Visible repair-loop task.",
        "runtime_visible_evidence": {
            "traceback_symbols": ["module.alpha"],
            "changed_files": ["module.py"],
            "watched_files": ["tests/test_module.py"],
            "failing_test_target": "tests/test_module.py",
        },
        "repair_candidates": [
            {
                "repair_action": "repair_alpha",
                "intent": "apply_patch_and_rerun_tests",
                "edit_scope": "module.py",
                "target_symbol": "module.alpha",
                "verification_command": "python -m pytest -q tests/test_module.py",
                "description": "repair alpha",
            },
            {
                "repair_action": "repair_bravo",
                "intent": "apply_patch_and_rerun_tests",
                "edit_scope": "other.py",
                "target_symbol": "other.bravo",
                "verification_command": "python -m pytest -q tests/test_other.py",
                "description": "repair bravo",
            },
        ],
        "expected_repair_action": "repair_alpha",
        "repair_loop_episode": {"loop_schema": "phase2t_repair_loop_v1"},
        "trace_hash": trace_id,
    }


def test_phase2t_trace_to_head_row_records_phase2t_prompt_and_source_trace() -> None:
    head_row = phase2t_repair_trace_to_head_row(_row())

    assert head_row["prompt_style"] == "phase2t_dynamic_repair_head_v1"
    assert "Phase2T dynamic public repair-loop native-head state input" in head_row[
        "state_prompt"
    ]
    assert "Phase2T repair-loop constraints" in head_row["state_prompt"]
    assert "phase2t_dynamic_public_repair_loop" in head_row["runtime_overrides"]
    assert head_row["source_trace"]["phase"] == "Phase2T"
    assert head_row["source_trace"]["repair_loop_schema"] == "phase2t_repair_loop_v1"
    assert head_row["command"] == "repair_alpha"
    assert head_row["command_slot"] == 0


def test_phase2t_trace_to_head_row_rejects_non_phase2t_rows() -> None:
    row = _row()
    row["phase"] = "Phase2S"

    with pytest.raises(ValueError, match="expected Phase2T"):
        phase2t_repair_trace_to_head_row(row)


def test_phase2t_head_dataset_records_phase2t_hashes_and_manifest(tmp_path: Path) -> None:
    train = _write_jsonl(tmp_path / "train.raw.jsonl", [_row("train:repo:phase2t:0")])
    val_row = _row("val:repo:phase2t:0")
    val_row["split"] = "val"
    val = _write_jsonl(tmp_path / "val.raw.jsonl", [val_row])
    holdout_row = _row("holdout:repo:phase2t:0")
    holdout_row["split"] = "holdout"
    holdout = _write_jsonl(tmp_path / "holdout.raw.jsonl", [holdout_row])
    data_health = _write(
        tmp_path / "data_health.json",
        {
            "passed": True,
            "effective_split_hashes": {
                "phase2t_train": "train_hash",
                "phase2t_val": "val_hash",
                "phase2t_holdout": "holdout_hash",
            },
        },
    )
    pretrain_gate = _write(tmp_path / "pretrain_gate.json", {"passed": True})

    manifest = build_phase2t_head_dataset(
        train_jsonl=train,
        val_jsonl=val,
        holdout_jsonl=holdout,
        output_dir=tmp_path / "heads",
        data_health_json=data_health,
        pretrain_gate_json=pretrain_gate,
    )
    train_head = json.loads((tmp_path / "heads" / "train.jsonl").read_text().splitlines()[0])

    assert manifest["dataset_family"] == "phase2t_dynamic_repair_head_dataset"
    assert manifest["source_data_health_passed"] is True
    assert manifest["source_pretrain_gate_passed"] is True
    assert manifest["command_identity_margin_gate_passed"] is True
    assert manifest["command_identity_diagnostics"]["train"]["zero_margin_rows"] == 0
    assert manifest["command_identity_diagnostics"]["holdout"]["zero_margin_rows"] == 0
    assert manifest["effective_split_hashes"]["phase2t_train"] == "train_hash"
    assert manifest["splits"]["train"]["rows"] == 1
    assert manifest["splits"]["holdout"]["rows"] == 1
    assert (tmp_path / "heads" / "holdout.jsonl").exists()
    assert train_head["prompt_style"] == "phase2t_dynamic_repair_head_v1"
