import json
from pathlib import Path

from reflexlm.cli.audit_phase2bf_natural_receptor_format_shift import (
    audit_phase2bf_natural_receptor_format_shift,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(label: str) -> dict:
    return {
        "rows": 12,
        "success_rate": 1.0,
        "slot_selection_accuracy": 1.0,
        "attempt_success_rate": 1.0,
        "execution_attempts": 12,
        "package_nsi_reference_mode": "low_level_only",
        "package_nsi_reference_override_rows": 0,
        "package_runtime_evidence_label": label,
        "freeform_patch_generation_rows": 0,
        "sealed_feedback_used_rows": 0,
    }


def test_phase2bf_audit_accepts_bounded_format_shift(tmp_path: Path) -> None:
    report = audit_phase2bf_natural_receptor_format_shift(
        phase2be_audit_json=_write(tmp_path / "phase2be.json", {"passed": True}),
        standard_summary_json=_write(
            tmp_path / "standard.json", _summary("Prior runtime evidence")
        ),
        shifted_summary_json=_write(
            tmp_path / "shifted.json", _summary("Runtime-visible repair evidence")
        ),
        wrong_cache_summary_json=_write(
            tmp_path / "wrong.json", {"success_rate": 0.0, "execution_attempts": 0}
        ),
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_receptor_format_shift_claim"] is True
    assert report["ready_for_open_ended_format_invariance_claim"] is False


def test_phase2bf_audit_rejects_identical_labels(tmp_path: Path) -> None:
    report = audit_phase2bf_natural_receptor_format_shift(
        phase2be_audit_json=_write(tmp_path / "phase2be.json", {"passed": True}),
        standard_summary_json=_write(
            tmp_path / "standard.json", _summary("Prior runtime evidence")
        ),
        shifted_summary_json=_write(
            tmp_path / "shifted.json", _summary("Prior runtime evidence")
        ),
        wrong_cache_summary_json=_write(
            tmp_path / "wrong.json", {"success_rate": 0.0, "execution_attempts": 0}
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["evidence_labels_are_distinct"] is False
