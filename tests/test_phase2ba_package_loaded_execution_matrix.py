import json
from pathlib import Path

from reflexlm.cli.audit_phase2ba_package_loaded_execution_matrix import (
    audit_phase2ba_package_loaded_execution_matrix,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _package_gate() -> dict:
    return {
        "passed": True,
        "ready_for_phase2az_packaged_adapter_runtime_smoke": True,
        "ready_for_epoch_making_architecture_claim": False,
    }


def _package_execution(*, policy: str = "package_loaded_native_head") -> dict:
    return {
        "selection_policy": policy,
        "rows": 6,
        "slot_selection_accuracy": 1.0,
        "execution_attempts": 6,
        "success_rate": 0.8333333333333334,
        "attempt_success_rate": 0.8333333333333334,
        "package_policy_loaded_rows": 6 if policy == "package_loaded_native_head" else 0,
        "package_head_record_visible_state_rows": 6
        if policy == "package_loaded_native_head"
        else 0,
        "package_qwen_called_rows": 6 if policy == "package_loaded_native_head" else 0,
        "package_low_level_debug_receptor_observed_rows": 6
        if policy == "package_loaded_native_head"
        else 0,
        "package_open_repair_authorized_rows": 6
        if policy == "package_loaded_native_head"
        else 0,
        "recorded_patch_artifact_used_rows": 0,
        "recorded_patch_artifact_used_for_fault_injection_rows": 6,
        "claim_bearing_execution_evidence_rows": 6,
        "freeform_patch_generation_rows": 0,
        "sealed_feedback_used_rows": 0,
        "model_prediction_records_present_rows": 0 if policy == "package_loaded_native_head" else 6,
    }


def _wrong_cache(*, attempts: int = 0) -> dict:
    return {
        "selection_policy": "wrong_cache",
        "rows": 6,
        "execution_attempts": attempts,
        "success_rate": 0.0,
    }


def test_phase2ba_audit_accepts_package_loaded_matrix_but_blocks_epoch(
    tmp_path: Path,
) -> None:
    report = audit_phase2ba_package_loaded_execution_matrix(
        package_gate_json=_write_json(tmp_path / "package_gate.json", _package_gate()),
        package_execution_summary_json=_write_json(
            tmp_path / "package_execution.json",
            _package_execution(),
        ),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _wrong_cache()),
    )

    assert report["passed"] is True
    assert report["ready_for_phase2ba_package_loaded_runtime_matrix"] is True
    assert report["ready_for_phase2ax_package"] is False
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_phase2ba_audit_rejects_prediction_record_replay_as_package_loaded(
    tmp_path: Path,
) -> None:
    report = audit_phase2ba_package_loaded_execution_matrix(
        package_gate_json=_write_json(tmp_path / "package_gate.json", _package_gate()),
        package_execution_summary_json=_write_json(
            tmp_path / "package_execution.json",
            _package_execution(policy="model_prediction_records"),
        ),
        wrong_cache_summary_json=_write_json(tmp_path / "wrong.json", _wrong_cache()),
    )

    assert report["passed"] is False
    assert report["checks"]["execution_policy_is_package_loaded_native_head"] is False
    assert report["checks"]["package_policy_loaded_for_all_rows"] is False
    assert report["checks"]["model_prediction_json_not_used_as_selector"] is False


def test_phase2ba_audit_rejects_wrong_cache_execution_attempts(tmp_path: Path) -> None:
    report = audit_phase2ba_package_loaded_execution_matrix(
        package_gate_json=_write_json(tmp_path / "package_gate.json", _package_gate()),
        package_execution_summary_json=_write_json(
            tmp_path / "package_execution.json",
            _package_execution(),
        ),
        wrong_cache_summary_json=_write_json(
            tmp_path / "wrong.json",
            _wrong_cache(attempts=1),
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["wrong_cache_control_blocks_execution"] is False
