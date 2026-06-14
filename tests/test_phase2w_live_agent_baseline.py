import json
from pathlib import Path

from reflexlm.cli.audit_phase2w_live_agent_baseline import (
    audit_phase2w_live_agent_baseline,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _config(kind: str = "live_tool_agent") -> dict:
    return {
        "run_id": "phase2w-live-agent-baseline-test",
        "baseline_kind": kind,
        "model_or_provider": "baseline-model",
        "tool_budget": 30,
        "context_policy": "bounded",
        "retry_policy": "single_retry",
        "edit_permissions": "sandbox_only",
        "stop_rule": "tests_pass_or_budget_exhausted",
        "cost_or_command_budget": {"max_commands": 30},
        "task_source_manifest_sha256": "a" * 64,
        "executor_commit": "b" * 40,
        "sandbox_image_or_env": "local-test-sandbox",
        "trace_archive_uri": "file://trace-archive",
    }


def _rows(count: int = 64) -> list[dict]:
    return [
        {
            "task_id": f"task-{index}",
            "task_success": index % 2 == 0,
            "patch_correctness": index % 2 == 0,
            "test_pass_rate": 1.0 if index % 2 == 0 else 0.0,
            "stop_condition_correctness": True,
            "false_completion": False,
            "unauthorized_write_count": 0,
            "commands_used": 6,
            "elapsed_seconds": 12.0,
            "transcript_sha256": "c" * 64,
            "patch_sha256": "d" * 64,
            "pre_test_log_sha256": "e" * 64,
            "post_test_log_sha256": "f" * 64,
        }
        for index in range(count)
    ]


def test_phase2w_live_agent_baseline_accepts_real_measured_rows(tmp_path: Path) -> None:
    report = audit_phase2w_live_agent_baseline(
        config_json=_write(tmp_path / "config.json", _config()),
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", _rows()),
    )
    assert report["passed"] is True
    assert report["baseline_kind"] == "live_tool_agent"
    assert report["metrics"]["task_success"] == 0.5
    assert report["checks"]["config_provenance_complete"] is True
    assert report["checks"]["config_hashes_valid"] is True
    assert report["checks"]["row_provenance_hashes_present"] is True
    assert report["checks"]["row_provenance_hashes_valid"] is True


def test_phase2w_live_agent_baseline_rejects_static_declared_baseline(tmp_path: Path) -> None:
    report = audit_phase2w_live_agent_baseline(
        config_json=_write(tmp_path / "config.json", _config(kind="static_overlap")),
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", _rows()),
    )
    assert report["passed"] is False
    assert "do_not_use_static_or_declared_baseline_as_phase2w_live_agent" in report[
        "blocked_actions"
    ]


def test_phase2w_live_agent_baseline_rejects_empty_results(tmp_path: Path) -> None:
    report = audit_phase2w_live_agent_baseline(
        config_json=_write(tmp_path / "config.json", _config()),
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", []),
    )
    assert report["passed"] is False
    assert report["checks"]["rows_minimum_met"] is False


def test_phase2w_live_agent_baseline_rejects_missing_provenance(tmp_path: Path) -> None:
    config = _config()
    del config["trace_archive_uri"]
    rows = _rows()
    del rows[0]["transcript_sha256"]
    report = audit_phase2w_live_agent_baseline(
        config_json=_write(tmp_path / "config.json", config),
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", rows),
    )
    assert report["passed"] is False
    assert report["checks"]["config_complete"] is False
    assert report["checks"]["row_provenance_hashes_present"] is False


def test_phase2w_live_agent_baseline_rejects_invalid_hash_formats(tmp_path: Path) -> None:
    config = _config()
    config["task_source_manifest_sha256"] = "not-a-hash"
    rows = _rows()
    rows[0]["patch_sha256"] = "not-a-hash"
    report = audit_phase2w_live_agent_baseline(
        config_json=_write(tmp_path / "config.json", config),
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", rows),
    )
    assert report["passed"] is False
    assert report["checks"]["config_hashes_valid"] is False
    assert report["checks"]["row_provenance_hashes_valid"] is False
