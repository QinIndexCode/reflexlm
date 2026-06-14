import json
from pathlib import Path

from reflexlm.cli.audit_phase2x_open_repair_execution_results import (
    audit_phase2x_open_repair_execution_results,
)


HASH = "a" * 64


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )
    return path


def _result_row(**overrides: object) -> dict:
    payload = {
        "task_id": "task-1",
        "repo_origin": "https://github.com/example/repo",
        "repo_commit": "b" * 40,
        "result_source": "phase2x_package_runtime_execution",
        "native_policy_label": "phase2x_open_repair_smoke",
        "policy_package_manifest_sha256": HASH,
        "patch_source": "package_runtime_patch_proposal",
        "policy_open_repair_outputs": {
            "patch_proposal": 1,
            "bounded_edit_scope": 1,
            "rollback_safety": 1,
            "stop_condition": 0,
            "progress_monitor": 1,
            "verification_state": 1,
        },
        "patch_proposal": "diff --git a/a.py b/a.py",
        "patch_sha256": HASH,
        "selected_tests": ["python -m pytest tests/test_a.py -q"],
        "pre_test_log_sha256": HASH,
        "post_test_log_sha256": HASH,
        "rollback_safety_decision": "safe",
        "verification_state": "passed",
        "progress_monitor_trace": [{"event": "test_started"}, {"event": "test_passed"}],
        "stop_condition": "verification_passed",
        "elapsed_seconds": 1.25,
        "transcript_sha256": HASH,
        "oracle_trace_used": False,
        "sealed_feedback_used": False,
        "success": True,
    }
    payload.update(overrides)
    return payload


def test_phase2x_execution_results_accepts_row_level_provenance(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_execution_results(
        training_readiness_json=_write_json(tmp_path / "readiness.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime.json", {"passed": True}),
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", [_result_row()]),
    )

    assert report["passed"] is True
    assert report["success_rate"] == 1.0


def test_phase2x_execution_results_rejects_missing_hash_provenance(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_execution_results(
        training_readiness_json=_write_json(tmp_path / "readiness.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime.json", {"passed": True}),
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", [_result_row(patch_sha256="bad")]),
    )

    assert report["passed"] is False
    assert report["checks"]["hash_fields_valid"] is False
    assert "do_not_use_phase2x_results_as_real_execution_evidence" in report["blocked_actions"]


def test_phase2x_execution_results_rejects_when_training_readiness_failed(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_execution_results(
        training_readiness_json=_write_json(tmp_path / "readiness.json", {"passed": False}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime.json", {"passed": True}),
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", [_result_row()]),
    )

    assert report["passed"] is False
    assert report["checks"]["training_readiness_passed"] is False


def test_phase2x_execution_results_rejects_when_runtime_capability_failed(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_execution_results(
        training_readiness_json=_write_json(tmp_path / "readiness.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime.json", {"passed": False}),
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", [_result_row()]),
    )

    assert report["passed"] is False
    assert report["checks"]["runtime_capability_audit_passed"] is False


def test_phase2x_execution_results_rejects_oracle_trace_as_policy_result(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_execution_results(
        training_readiness_json=_write_json(tmp_path / "readiness.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime.json", {"passed": True}),
        results_jsonl=_write_jsonl(
            tmp_path / "results.jsonl",
            [
                _result_row(
                    result_source="phase2s_oracle_repair_trace",
                    patch_source="collector_oracle_patch",
                    oracle_trace_used=True,
                )
            ],
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["non_oracle_policy_execution"] is False
    assert "do_not_substitute_oracle_trace_for_policy_generated_patch" in report["blocked_actions"]


def test_phase2x_execution_results_rejects_success_without_patch_head_authorization(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_execution_results(
        training_readiness_json=_write_json(tmp_path / "readiness.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime.json", {"passed": True}),
        results_jsonl=_write_jsonl(
            tmp_path / "results.jsonl",
            [_result_row(policy_open_repair_outputs={"patch_proposal": 0, "bounded_edit_scope": 1})],
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["policy_control_authorized_patch"] is False


def test_phase2x_execution_results_accepts_policy_no_patch_as_failed_row(tmp_path: Path) -> None:
    rows = [
        _result_row(),
        _result_row(
            task_id="task-2",
            patch_source="package_runtime_no_patch_authorized",
            policy_open_repair_outputs={"patch_proposal": 0, "bounded_edit_scope": 1},
            patch_proposal="NO_PATCH_AUTHORIZED_BY_OPEN_REPAIR_HEADS",
            success=False,
            verification_state="failed",
        ),
    ]
    report = audit_phase2x_open_repair_execution_results(
        training_readiness_json=_write_json(tmp_path / "readiness.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime.json", {"passed": True}),
        results_jsonl=_write_jsonl(tmp_path / "results.jsonl", rows),
        min_rows=2,
        min_success_rate=0.5,
    )

    assert report["passed"] is True
    assert report["success_rate"] == 0.5
    assert report["checks"]["non_oracle_policy_execution"] is True
