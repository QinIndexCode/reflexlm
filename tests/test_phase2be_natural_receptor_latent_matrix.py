import json
from pathlib import Path

from reflexlm.cli.audit_phase2be_natural_receptor_latent_matrix import (
    audit_phase2be_natural_receptor_latent_matrix,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _summary(mode: str, success: float, overrides: int) -> dict:
    return {
        "rows": 12,
        "slot_selection_accuracy": success,
        "success_rate": success,
        "attempt_success_rate": 1.0,
        "package_policy_loaded_rows": 12,
        "package_qwen_called_rows": 12,
        "package_nsi_reference_mode": mode,
        "package_nsi_reference_override_rows": overrides,
        "freeform_patch_generation_rows": 0,
        "sealed_feedback_used_rows": 0,
    }


def test_phase2be_audit_accepts_low_level_receptor_gap_closure(tmp_path: Path) -> None:
    report = audit_phase2be_natural_receptor_latent_matrix(
        runtime_visible_summary_json=_write(
            tmp_path / "reference.json",
            _summary("runtime_visible_override", 1.0, 12),
        ),
        pre_receptor_low_level_summary_json=_write(
            tmp_path / "before.json", _summary("low_level_only", 0.5, 0)
        ),
        post_receptor_low_level_summary_json=_write(
            tmp_path / "after.json", _summary("low_level_only", 1.0, 0)
        ),
        wrong_cache_summary_json=_write(
            tmp_path / "wrong.json", {"success_rate": 0.0, "execution_attempts": 0}
        ),
        sealed_transfer_report_json=_write(
            tmp_path / "sealed.json",
            {"passed": True, "ready_for_epoch_making_architecture_claim": False},
        ),
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_low_level_nsi_natural_perception_claim"] is True
    assert report["metrics"]["low_level_success_improvement"] == 0.5
    assert report["metrics"]["runtime_visible_minus_post_receptor_success"] == 0.0


def test_phase2be_audit_rejects_post_receptor_override(tmp_path: Path) -> None:
    report = audit_phase2be_natural_receptor_latent_matrix(
        runtime_visible_summary_json=_write(
            tmp_path / "reference.json",
            _summary("runtime_visible_override", 1.0, 12),
        ),
        pre_receptor_low_level_summary_json=_write(
            tmp_path / "before.json", _summary("low_level_only", 0.5, 0)
        ),
        post_receptor_low_level_summary_json=_write(
            tmp_path / "after.json", _summary("low_level_only", 1.0, 1)
        ),
        wrong_cache_summary_json=_write(
            tmp_path / "wrong.json", {"success_rate": 0.0, "execution_attempts": 0}
        ),
        sealed_transfer_report_json=_write(
            tmp_path / "sealed.json",
            {"passed": True, "ready_for_epoch_making_architecture_claim": False},
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["before_and_after_use_no_override"] is False
