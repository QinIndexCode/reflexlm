import json
from pathlib import Path

from reflexlm.cli.build_phase2ag_evidence_sufficiency_report import (
    build_phase2ag_evidence_sufficiency_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _postflight() -> dict:
    return {
        "passed": True,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "claim_bearing_mechanism_evidence": False,
        "thresholds": {"require_val_model_minus_source_overlap": False},
        "metrics": {
            "val_command_slot_accuracy": 1.0,
            "holdout_command_slot_accuracy": 1.0,
            "val_source_overlap_accuracy": 0.875,
            "holdout_source_overlap_accuracy": 0.67,
            "val_model_minus_source_overlap_accuracy": 0.125,
            "holdout_model_minus_source_overlap_accuracy": 0.33,
        },
    }


def _controls() -> dict:
    return {
        "passed": True,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "claim_bearing_mechanism_evidence": False,
        "metrics": {
            "full_minus_sidecar_erased": 0.33,
            "full_minus_wrong_sidecar": 0.34,
            "sidecar_erased_accuracy": 0.67,
            "wrong_sidecar_accuracy": 0.66,
            "row_count": 58,
        },
    }


def test_phase2ag_sufficiency_report_keeps_bounded_sidecar_boundary(
    tmp_path: Path,
) -> None:
    report = build_phase2ag_evidence_sufficiency_report(
        postflight_json=_write(tmp_path / "postflight.json", _postflight()),
        sidecar_control_postflight_json=_write(tmp_path / "controls.json", _controls()),
    )

    assert report["passed"] is True
    assert (
        report["claim_scope"]
        == "phase2ag_bounded_runtime_visible_sidecar_dependency_evidence"
    )
    assert "epoch_making_architecture" in report["unsupported_claims"]
    assert "do_not_package_phase2ag_from_this_report" in report["blocked_actions"]
    assert report["caveats"]


def test_phase2ag_sufficiency_report_rejects_missing_control_delta(
    tmp_path: Path,
) -> None:
    controls = _controls()
    controls["metrics"]["full_minus_wrong_sidecar"] = 0.0

    report = build_phase2ag_evidence_sufficiency_report(
        postflight_json=_write(tmp_path / "postflight.json", _postflight()),
        sidecar_control_postflight_json=_write(tmp_path / "controls.json", controls),
    )

    assert report["passed"] is False
    assert report["checks"]["wrong_sidecar_guarded_degrades"] is False
