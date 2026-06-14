import json
from pathlib import Path

from reflexlm.cli.audit_phase2x_open_repair_training_readiness import (
    audit_phase2x_open_repair_training_readiness,
)


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


def _ready_manifest(tmp_path: Path, *, ready: bool = True) -> Path:
    return _write_json(
        tmp_path / "head" / "manifest.json",
        {"full_repair_control_training_ready": ready},
    )


def _row(**overrides: object) -> dict:
    payload = {
        "state_prompt": "open repair visible state",
        "patch_proposal_label": 1,
        "test_selection_slot": 0,
        "rollback_safety_label": 1,
        "stop_condition_label": 0,
        "bounded_edit_scope_label": 1,
        "progress_monitor_label": 2,
        "verification_state_label": 1,
    }
    payload.update(overrides)
    return payload


def test_phase2x_training_readiness_accepts_runtime_and_label_coverage(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_training_readiness(
        task_manifest_audit_json=_write_json(tmp_path / "task_audit.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime_audit.json", {"passed": True}),
        train_jsonl=_write_jsonl(tmp_path / "train.jsonl", [_row()]),
        val_jsonl=_write_jsonl(tmp_path / "val.jsonl", [_row()]),
        head_dataset_manifest_json=_ready_manifest(tmp_path),
    )

    assert report["passed"] is True
    assert report["blocked_actions"] == []


def test_phase2x_training_readiness_rejects_missing_open_repair_labels(tmp_path: Path) -> None:
    row = _row(patch_proposal_label=-100)
    report = audit_phase2x_open_repair_training_readiness(
        task_manifest_audit_json=_write_json(tmp_path / "task_audit.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime_audit.json", {"passed": True}),
        train_jsonl=_write_jsonl(tmp_path / "train.jsonl", [row]),
        val_jsonl=_write_jsonl(tmp_path / "val.jsonl", [_row()]),
        head_dataset_manifest_json=_ready_manifest(tmp_path),
    )

    assert report["passed"] is False
    assert report["checks"]["train_open_repair_labels_covered"] is False
    assert "do_not_start_phase2x_open_repair_training" in report["blocked_actions"]


def test_phase2x_training_readiness_records_runtime_capability_without_training_cycle(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_training_readiness(
        task_manifest_audit_json=_write_json(tmp_path / "task_audit.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime_audit.json", {"passed": False}),
        train_jsonl=_write_jsonl(tmp_path / "train.jsonl", [_row()]),
        val_jsonl=_write_jsonl(tmp_path / "val.jsonl", [_row()]),
        head_dataset_manifest_json=_ready_manifest(tmp_path),
    )

    assert report["passed"] is True
    assert report["checks"]["runtime_capability_audit_recorded"] is True
    assert report["post_training_package_requirements"]["runtime_capability_audit_passed"] is False


def test_phase2x_training_readiness_allows_false_sealed_feedback_flag(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_training_readiness(
        task_manifest_audit_json=_write_json(tmp_path / "task_audit.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime_audit.json", {"passed": True}),
        train_jsonl=_write_jsonl(tmp_path / "train.jsonl", [_row(sealed_feedback_used=False)]),
        val_jsonl=_write_jsonl(tmp_path / "val.jsonl", [_row(sealed_feedback_used=False)]),
        head_dataset_manifest_json=_ready_manifest(tmp_path),
    )

    assert report["passed"] is True
    assert report["checks"]["no_forbidden_markers_in_train"] is True


def test_phase2x_training_readiness_rejects_initial_control_only_manifest(tmp_path: Path) -> None:
    report = audit_phase2x_open_repair_training_readiness(
        task_manifest_audit_json=_write_json(tmp_path / "task_audit.json", {"passed": True}),
        runtime_capability_audit_json=_write_json(tmp_path / "runtime_audit.json", {"passed": True}),
        train_jsonl=_write_jsonl(tmp_path / "train.jsonl", [_row()]),
        val_jsonl=_write_jsonl(tmp_path / "val.jsonl", [_row()]),
        head_dataset_manifest_json=_ready_manifest(tmp_path, ready=False),
    )

    assert report["passed"] is False
    assert report["checks"]["full_repair_control_label_scope"] is False
