import json
from pathlib import Path

from reflexlm.cli.audit_phase2bd_nsi_reference_mechanism_matrix import (
    audit_phase2bd_nsi_reference_mechanism_matrix,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(mode: str, *, slot_accuracy: float, success_rate: float) -> dict:
    rows = 12
    override_rows = rows if mode == "runtime_visible_override" else 0
    attempts = int(rows * slot_accuracy)
    return {
        "rows": rows,
        "slot_selection_accuracy": slot_accuracy,
        "success_rate": success_rate,
        "attempt_success_rate": 1.0,
        "execution_attempts": attempts,
        "package_policy_loaded_rows": rows,
        "package_qwen_called_rows": rows,
        "package_nsi_reference_mode": mode,
        "package_nsi_reference_override_rows": override_rows,
        "freeform_patch_generation_rows": 0,
        "sealed_feedback_used_rows": 0,
    }


def test_phase2bd_audit_identifies_low_level_nsi_gap(tmp_path: Path) -> None:
    report = audit_phase2bd_nsi_reference_mechanism_matrix(
        runtime_visible_summary_json=_write(
            tmp_path / "full.json",
            _summary("runtime_visible_override", slot_accuracy=1.0, success_rate=1.0),
        ),
        low_level_only_summary_json=_write(
            tmp_path / "low.json",
            _summary("low_level_only", slot_accuracy=0.5, success_rate=0.5),
        ),
        wrong_cache_summary_json=_write(
            tmp_path / "wrong.json", {"success_rate": 0.0, "execution_attempts": 0}
        ),
        package_execution_audit_json=_write(tmp_path / "audit.json", {"passed": True}),
        sealed_transfer_report_json=_write(
            tmp_path / "sealed.json",
            {"passed": True, "ready_for_epoch_making_architecture_claim": False},
        ),
    )

    assert report["passed"] is True
    assert report["ready_for_runtime_visible_synaptic_signal_claim"] is True
    assert report["ready_for_low_level_nsi_natural_perception_claim"] is False
    assert report["metrics"]["runtime_visible_minus_low_level_success_rate"] == 0.5
    assert report["next_required_experiment"]["phase"] == (
        "phase2be_learned_low_level_runtime_receptor_latent"
    )


def test_phase2bd_audit_rejects_hidden_override_in_low_level_control(
    tmp_path: Path,
) -> None:
    low = _summary("low_level_only", slot_accuracy=0.5, success_rate=0.5)
    low["package_nsi_reference_override_rows"] = 1
    report = audit_phase2bd_nsi_reference_mechanism_matrix(
        runtime_visible_summary_json=_write(
            tmp_path / "full.json",
            _summary("runtime_visible_override", slot_accuracy=1.0, success_rate=1.0),
        ),
        low_level_only_summary_json=_write(tmp_path / "low.json", low),
        wrong_cache_summary_json=_write(
            tmp_path / "wrong.json", {"success_rate": 0.0, "execution_attempts": 0}
        ),
        package_execution_audit_json=_write(tmp_path / "audit.json", {"passed": True}),
        sealed_transfer_report_json=_write(
            tmp_path / "sealed.json",
            {"passed": True, "ready_for_epoch_making_architecture_claim": False},
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["low_level_only_uses_no_override"] is False
