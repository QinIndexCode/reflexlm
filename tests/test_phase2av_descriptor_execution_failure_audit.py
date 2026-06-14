import json
from pathlib import Path

from reflexlm.cli.build_phase2av_descriptor_execution_failure_audit import (
    build_phase2av_descriptor_execution_failure_audit,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int, *, selected: bool = True, success: bool = True) -> dict:
    split = "train" if index % 2 else "holdout"
    return {
        "trace_id": f"phase2av:holdout:{index:05d}",
        "repo_origin": "https://github.com/psf/black.git",
        "patch_candidate_selected_correctly": selected,
        "success": success,
        "rollback_failure_restored": True,
        "unauthorized_write_count": 0,
        "false_completion": False,
        "verification_state": "passed" if success else "failed",
        "stop_condition": "verification_passed"
        if success
        else "verification_failed_stop",
        "selected_repair_action": "structural_repair_a",
        "artifact_paths": {
            "source_patch_artifact": f"artifacts/datasets/x/artifacts/{split}/repo/row_{index:05d}/patch.diff"
        },
    }


def test_phase2av_failure_audit_detects_runtime_bottleneck(tmp_path: Path) -> None:
    full = [_row(i, selected=True, success=i < 15) for i in range(20)]
    control = [_row(i, selected=i < 8, success=i < 6) for i in range(20)]

    report = build_phase2av_descriptor_execution_failure_audit(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", full),
        control_execution_jsonl=_write_jsonl(tmp_path / "control.jsonl", control),
        execution_gate_json=_write_json(tmp_path / "gate.json", {"passed": False}),
    )

    assert report["passed"] is False
    assert report["diagnosis"] == "runtime_symbolic_patch_execution_bottleneck"
    assert "selected_candidate_runtime_execution_below_gate" in report["failure_modes"]
    assert "holdout_execution_uses_mixed_source_artifact_splits" in report[
        "failure_modes"
    ]
    assert report["metrics"]["full_selection_accuracy"] == 1.0
    assert report["metrics"]["full_success_rate"] == 0.75
    assert "do_not_package_phase2av" in report["blocked_actions"]
    assert report["checks"]["holdout_source_artifact_split_clean"] is False
    assert report["failure_breakdown"]["source_artifact_split_outcomes"]["holdout"][
        "rows"
    ] == 10


def test_phase2av_failure_audit_detects_selection_bottleneck(tmp_path: Path) -> None:
    full = [
        _row(i, selected=i < 10, success=i < 10)
        for i in range(20)
    ]

    report = build_phase2av_descriptor_execution_failure_audit(
        full_execution_jsonl=_write_jsonl(tmp_path / "full.jsonl", full),
    )

    assert report["passed"] is False
    assert report["diagnosis"] == "candidate_selection_bottleneck"
    assert "candidate_selection_accuracy_below_gate" in report["failure_modes"]
