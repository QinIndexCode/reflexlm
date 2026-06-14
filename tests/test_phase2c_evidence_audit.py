import json
from pathlib import Path

from reflexlm.cli.audit_phase2c_evidence import build_phase2c_evidence_audit


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_phase2c_evidence_audit_tracks_debug_command_overlap_and_low_level_calls(
    tmp_path: Path,
) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    _write_jsonl(
        run_path / "trace_rows.jsonl",
        [
            {
                "task_type": "blocking_input_detection",
                "policy_debug": {"action_source": "low_level_nsi"},
                "action": {"type": "WAIT"},
                "oracle_action": {"type": "WAIT"},
                "correct": True,
            },
            {
                "task_type": "test_failure_reflex",
                "policy_debug": {"action_source": "native_head_cortex"},
                "action": {"type": "RUN_COMMAND", "command": "pytest --snapshot-update"},
                "oracle_action": {"type": "RUN_COMMAND", "command": "pytest --snapshot-update"},
                "correct": True,
            },
        ],
    )
    eval_json = _write_json(
        tmp_path / "eval.json",
        {
            "run_path": str(run_path),
            "policy": {"policy_family": "phase2c_native_heads", "json_text_target": False},
            "metrics": {
                "aggregate": {
                    "task_completion_rate": {"mean": 1.0},
                    "reaction_latency_ms": {"mean": 5.0},
                    "model_calls": {"mean": 0.2},
                    "token_equivalent_cost": {"mean": 50.0},
                },
                "per_task": {
                    "blocking_input_detection": {
                        "metrics": {"task_completion_rate": {"mean": 1.0}}
                    },
                    "test_failure_reflex": {
                        "metrics": {"task_completion_rate": {"mean": 1.0}}
                    },
                },
            },
        },
    )
    gate_json = _write_json(tmp_path / "gate.json", {"passed": True, "checks": {}})
    manifest_json = _write_json(
        tmp_path / "manifest.json",
        {"coverage_audit": {"passed": False}, "leakage_audit": {"passed": True}},
    )
    train_jsonl = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "head_scope": "debug_cortex",
                "action_type": "RUN_COMMAND",
                "command": "python -m pip install -r requirements.txt",
                "command_intent": "dependency_install",
            }
        ],
    )
    val_jsonl = _write_jsonl(tmp_path / "val.jsonl", [])
    test_jsonl = _write_jsonl(
        tmp_path / "test.jsonl",
        [
            {
                "head_scope": "debug_cortex",
                "action_type": "RUN_COMMAND",
                "command": "pytest --snapshot-update",
                "command_intent": "snapshot_update",
            }
        ],
    )

    report = build_phase2c_evidence_audit(
        eval_json=eval_json,
        gate_json=gate_json,
        dataset_manifest_json=manifest_json,
        train_head_jsonl=train_jsonl,
        val_head_jsonl=val_jsonl,
        test_head_jsonl=test_jsonl,
    )

    assert report["evidence_status"]["fixed_split_claim_supported"] is True
    assert report["evidence_status"]["external_validity_complete"] is False
    assert report["debug_command_generalization"]["train_test_exact_overlap"] == []
    assert report["trace_audit"]["low_level_cortex_calls"] == 0
    assert "build a validation split" in report["next_required"][0]


def test_phase2c_evidence_audit_flags_exact_debug_command_overlap(tmp_path: Path) -> None:
    eval_json = _write_json(
        tmp_path / "eval.json",
        {
            "metrics": {
                "aggregate": {"task_completion_rate": {"mean": 1.0}},
                "per_task": {"test_failure_reflex": {"metrics": {"task_completion_rate": {"mean": 1.0}}}},
            }
        },
    )
    gate_json = _write_json(tmp_path / "gate.json", {"passed": True})
    train_jsonl = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {
                "head_scope": "debug_cortex",
                "action_type": "RUN_COMMAND",
                "command": "pytest --snapshot-update",
                "command_intent": "snapshot_update",
            }
        ],
    )
    test_jsonl = _write_jsonl(
        tmp_path / "test.jsonl",
        [
            {
                "head_scope": "debug_cortex",
                "action_type": "RUN_COMMAND",
                "command": "pytest --snapshot-update",
                "command_intent": "snapshot_update",
            }
        ],
    )

    report = build_phase2c_evidence_audit(
        eval_json=eval_json,
        gate_json=gate_json,
        train_head_jsonl=train_jsonl,
        val_head_jsonl=tmp_path / "missing-val.jsonl",
        test_head_jsonl=test_jsonl,
    )

    assert report["evidence_status"]["debug_train_test_exact_command_overlap_zero"] is False
    assert report["evidence_status"]["fixed_split_claim_supported"] is False
