import json
from pathlib import Path

from reflexlm.cli.audit_phase2y_open_repair_generalization_gap import (
    audit_phase2y_open_repair_generalization_gap,
)


HASH = "a" * 64


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int, **overrides: object) -> dict:
    payload = {
        "task_id": f"task-{index}",
        "repo_origin": f"https://github.com/example/repo-{index % 4}.git",
        "repo_commit": "b" * 40,
        "result_source": "phase2x_package_runtime_execution",
        "native_policy_label": "phase2x",
        "policy_package_manifest_sha256": HASH,
        "patch_source": "package_runtime_patch_proposal",
        "patch_generator": "bounded_assertion_literal_patch_v1",
        "policy_open_repair_outputs": {
            "patch_proposal": 1,
            "bounded_edit_scope": 1,
            "rollback_safety": 1,
            "stop_condition": 0,
            "progress_monitor": 2,
            "verification_state": 2,
        },
        "patch_proposal": "diff --git a/a.py b/a.py",
        "patch_sha256": HASH,
        "selected_tests": ["python -m pytest -q <generated_repair_test> --maxfail=1"],
        "pre_test_log_sha256": HASH,
        "post_test_log_sha256": HASH,
        "rollback_safety_decision": "not_required_after_verified_pass",
        "verification_state": "passed",
        "progress_monitor_trace": [{"event": "post_test_finished"}],
        "stop_condition": "verification_passed",
        "elapsed_seconds": 1.0,
        "transcript_sha256": HASH,
        "oracle_trace_used": False,
        "sealed_feedback_used": False,
        "success": True,
    }
    payload.update(overrides)
    return payload


def test_phase2y_gap_audit_supports_bounded_but_blocks_open_claim(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(128)]

    report = audit_phase2y_open_repair_generalization_gap(
        execution_results_jsonl=_write_jsonl(tmp_path / "results.jsonl", rows),
        execution_audit_json=_write_json(tmp_path / "audit.json", {"passed": True}),
    )

    assert report["bounded_execution_supported"] is True
    assert report["open_ended_claim_ready"] is False
    assert report["checks"]["non_literal_patch_present"] is False
    assert report["checks"]["multi_test_selection_present"] is False
    assert report["checks"]["rollback_required_path_present"] is False
    assert report["checks"]["no_edit_control_present"] is False
    assert report["checks"]["not_generated_test_only"] is False
    assert "do_not_claim_open_ended_debugging_generalization_from_phase2x" in report["blocked_actions"]


def test_phase2y_gap_audit_accepts_stronger_open_repair_mix(tmp_path: Path) -> None:
    rows = []
    for index in range(128):
        if index % 16 == 0:
            rows.append(
                _row(
                    index,
                    patch_source="package_runtime_no_patch_authorized",
                    patch_generator="no_edit_control_v1",
                    policy_open_repair_outputs={"patch_proposal": 0, "bounded_edit_scope": 1},
                    patch_proposal="NO_PATCH_AUTHORIZED_BY_OPEN_REPAIR_HEADS",
                    selected_tests=["python -m pytest tests/test_existing.py -q"],
                )
            )
        elif index % 10 == 0:
            rows.append(
                _row(
                    index,
                    patch_generator="bounded_symbolic_patch_v1",
                    selected_tests=[
                        "python -m pytest tests/test_a.py -q",
                        "python -m pytest tests/test_b.py -q",
                    ],
                )
            )
        elif index % 15 == 0:
            rows.append(
                _row(
                    index,
                    patch_generator="bounded_symbolic_patch_v1",
                    rollback_safety_decision="rollback_required_after_failed_patch",
                    progress_monitor_trace=[
                        {"event": "rollback_started"},
                        {"event": "rollback_finished"},
                        {"event": "post_test_finished"},
                    ],
                    selected_tests=["python -m pytest tests/test_repair.py -q"],
                )
            )
        else:
            rows.append(
                _row(
                    index,
                    patch_generator="bounded_symbolic_patch_v1",
                    selected_tests=["python -m pytest tests/test_repair.py -q"],
                )
            )

    report = audit_phase2y_open_repair_generalization_gap(
        execution_results_jsonl=_write_jsonl(tmp_path / "results.jsonl", rows),
        execution_audit_json=_write_json(tmp_path / "audit.json", {"passed": True}),
    )

    assert report["passed"] is True
    assert report["open_ended_claim_ready"] is True
    assert report["checks"]["non_literal_patch_present"] is True
    assert report["checks"]["multi_test_selection_present"] is True
    assert report["checks"]["rollback_required_path_present"] is True
    assert report["checks"]["no_edit_control_present"] is True
    assert report["checks"]["not_generated_test_only"] is True


def test_phase2y_gap_audit_rejects_oracle_or_sealed_rows(tmp_path: Path) -> None:
    rows = [_row(index) for index in range(128)]
    rows[0]["oracle_trace_used"] = True
    rows[1]["sealed_feedback_used"] = True

    report = audit_phase2y_open_repair_generalization_gap(
        execution_results_jsonl=_write_jsonl(tmp_path / "results.jsonl", rows),
        execution_audit_json=_write_json(tmp_path / "audit.json", {"passed": True}),
    )

    assert report["bounded_execution_supported"] is False
    assert report["checks"]["non_oracle_non_sealed"] is False
