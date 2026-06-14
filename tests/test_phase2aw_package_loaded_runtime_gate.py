import json
from pathlib import Path

from reflexlm.cli.audit_phase2aw_package_loaded_runtime_gate import (
    audit_phase2aw_package_loaded_runtime_gate,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _full() -> dict:
    return {
        "artifact_family": "phase2aw_package_loaded_descriptor_execution_runner",
        "policy_loaded": True,
        "success_rate": 0.92,
        "patch_candidate_selection_accuracy": 1.0,
        "claim_boundary": "phase2aw_package_loaded_bounded_descriptor_execution_not_freeform_patch_generation",
    }


def _control() -> dict:
    return {
        "artifact_family": "phase2av_descriptor_selected_execution_runner",
        "selection_mode": "source_overlap",
        "success_rate": 0.41,
        "selection_accuracy": 0.44,
    }


def _postpackage() -> dict:
    return {
        "passed": True,
        "ready_for_bounded_package_loaded_runtime_eval": True,
        "ready_for_sealed_eval": False,
    }


def test_phase2aw_package_loaded_runtime_gate_accepts_nonsealed_package_delta(
    tmp_path: Path,
) -> None:
    report = audit_phase2aw_package_loaded_runtime_gate(
        package_loaded_summary_json=_write(tmp_path / "full.json", _full()),
        source_overlap_summary_json=_write(tmp_path / "control.json", _control()),
        postpackage_gate_json=_write(tmp_path / "postpackage.json", _postpackage()),
    )

    assert report["passed"] is True
    assert report["ready_for_sealed_eval_gate_design"] is True
    assert report["ready_for_sealed_eval"] is False
    assert (
        "phase2aw_package_loaded_bounded_nonsealed_runtime_delta_supported"
        in report["supported_claims"]
    )


def test_phase2aw_package_loaded_runtime_gate_rejects_summary_replay_full(
    tmp_path: Path,
) -> None:
    full = _full()
    full["artifact_family"] = "phase2av_descriptor_selected_execution_runner"
    full["policy_loaded"] = False

    report = audit_phase2aw_package_loaded_runtime_gate(
        package_loaded_summary_json=_write(tmp_path / "full.json", full),
        source_overlap_summary_json=_write(tmp_path / "control.json", _control()),
        postpackage_gate_json=_write(tmp_path / "postpackage.json", _postpackage()),
    )

    assert report["passed"] is False
    assert report["checks"]["full_summary_is_package_loaded_runner"] is False
    assert report["checks"]["full_policy_loaded"] is False


def test_phase2aw_package_loaded_runtime_gate_rejects_control_ceiling(
    tmp_path: Path,
) -> None:
    control = _control()
    control["success_rate"] = 0.9

    report = audit_phase2aw_package_loaded_runtime_gate(
        package_loaded_summary_json=_write(tmp_path / "full.json", _full()),
        source_overlap_summary_json=_write(tmp_path / "control.json", control),
        postpackage_gate_json=_write(tmp_path / "postpackage.json", _postpackage()),
    )

    assert report["passed"] is False
    assert report["checks"]["source_overlap_control_not_ceiling"] is False
    assert report["checks"]["full_minus_control_success_rate_min"] is False


def test_phase2aw_package_loaded_runtime_gate_rejects_insufficient_delta(
    tmp_path: Path,
) -> None:
    full = _full()
    full["success_rate"] = 0.52

    report = audit_phase2aw_package_loaded_runtime_gate(
        package_loaded_summary_json=_write(tmp_path / "full.json", full),
        source_overlap_summary_json=_write(tmp_path / "control.json", _control()),
        postpackage_gate_json=_write(tmp_path / "postpackage.json", _postpackage()),
    )

    assert report["passed"] is False
    assert report["checks"]["full_success_rate_min"] is False
    assert report["checks"]["full_minus_control_success_rate_min"] is False
