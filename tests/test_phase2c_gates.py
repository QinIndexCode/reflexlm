import json
from pathlib import Path

from reflexlm.cli.check_phase2c_gates import build_phase2c_gate_report


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2c_gate_passes_native_head_contract(tmp_path: Path) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "trace_rows.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_type": "blocking_input_detection",
                        "policy_debug": {"action_source": "low_level_nsi"},
                    }
                ),
                json.dumps(
                    {
                        "task_type": "test_failure_reflex",
                        "policy_debug": {"action_source": "native_head_cortex"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    eval_json = _write(
        tmp_path / "eval.json",
        {
            "run_path": str(run_path),
            "policy": {"policy_family": "phase2c_native_heads", "json_text_target": False},
            "policy_stats": {"parse_failures": 0},
            "metrics": {
                "aggregate": {
                    "task_completion_rate": {"mean": 0.96},
                    "dangerous_action_block_rate": {"mean": 1.0},
                    "stale_state_action_rate": {"mean": 0.0},
                    "reaction_latency_ms": {"mean": 10.0},
                }
            },
        },
    )
    train_json = _write(tmp_path / "train.json", {"json_text_target": False})
    manifest_json = _write(
        tmp_path / "manifest.json",
        {"json_text_target": False, "leakage_audit": {"passed": True}},
    )
    reference_json = _write(
        tmp_path / "reference.json",
        {"metrics": {"aggregate": {"reaction_latency_ms": {"mean": 0.5}}}},
    )

    report = build_phase2c_gate_report(
        eval_json=eval_json,
        train_summary_json=train_json,
        dataset_manifest_json=manifest_json,
        reference_eval_json=reference_json,
    )

    assert report["passed"] is True
    assert report["checks"]["low_level_no_qwen_calls"] is True
    assert report["checks"]["debug_cortex_invoked"] is True


def test_phase2c_gate_rejects_low_level_qwen_calls(tmp_path: Path) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "trace_rows.jsonl").write_text(
        json.dumps(
            {
                "task_type": "external_file_change_reflex",
                "policy_debug": {"action_source": "native_head_cortex"},
            }
        ),
        encoding="utf-8",
    )
    eval_json = _write(
        tmp_path / "eval.json",
        {
            "run_path": str(run_path),
            "policy": {"policy_family": "phase2c_native_heads", "json_text_target": False},
            "policy_stats": {"parse_failures": 0},
            "metrics": {
                "aggregate": {
                    "task_completion_rate": {"mean": 1.0},
                    "dangerous_action_block_rate": {"mean": 1.0},
                    "stale_state_action_rate": {"mean": 0.0},
                    "reaction_latency_ms": {"mean": 1.0},
                }
            },
        },
    )

    report = build_phase2c_gate_report(eval_json=eval_json)

    assert report["passed"] is False
    assert report["checks"]["low_level_no_qwen_calls"] is False


def test_phase2c_gate_reports_coverage_audit_as_evidence_warning(tmp_path: Path) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "trace_rows.jsonl").write_text(
        json.dumps(
            {
                "task_type": "test_failure_reflex",
                "policy_debug": {"action_source": "native_head_cortex"},
            }
        ),
        encoding="utf-8",
    )
    eval_json = _write(
        tmp_path / "eval.json",
        {
            "run_path": str(run_path),
            "policy": {"policy_family": "phase2c_native_heads", "json_text_target": False},
            "policy_stats": {"parse_failures": 0},
            "metrics": {
                "aggregate": {
                    "task_completion_rate": {"mean": 1.0},
                    "dangerous_action_block_rate": {"mean": 1.0},
                    "stale_state_action_rate": {"mean": 0.0},
                    "reaction_latency_ms": {"mean": 1.0},
                }
            },
        },
    )
    manifest_json = _write(
        tmp_path / "manifest.json",
        {
            "json_text_target": False,
            "leakage_audit": {"passed": True},
            "coverage_audit": {
                "passed": False,
                "val_missing_test_pairs": ["debug_cortex/RUN_COMMAND"],
            },
        },
    )

    report = build_phase2c_gate_report(eval_json=eval_json, dataset_manifest_json=manifest_json)

    assert report["passed"] is True
    assert report["coverage_audit"]["passed"] is False
    assert "coverage_audit_failed" in report["evidence_warnings"]
