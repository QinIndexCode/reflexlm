import json
from pathlib import Path

from reflexlm.cli.audit_phase2de_compact_evidence_rollup import (
    REPORT_SPECS,
    audit_phase2de_compact_evidence_rollup,
    validate_phase2de_compact_evidence_rollup,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _bounded_flag_for_phase(phase_id: str) -> str:
    return f"ready_for_bounded_{phase_id}_claim"


def _phase_report(phase_id: str, *, overstated: bool = False) -> dict:
    return {
        "artifact_family": f"{phase_id}_artifact",
        "passed": True,
        _bounded_flag_for_phase(phase_id): True,
        "ready_for_general_shell_autonomy_claim": False,
        "ready_for_general_runtime_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": overstated,
        "checks": {"local_check": True},
        "metrics": {
            "runtime_count": 3,
            "seed_count": 3,
            "perturbation_count": 3,
            "control_count": 10,
            "negative_control_count": 9,
            "negative_controls_failed": 9,
            "order_count": 5,
        },
    }


def _report_dir_fixture(tmp_path: Path) -> Path:
    report_dir = tmp_path / "reports"
    for spec in REPORT_SPECS:
        _write(
            report_dir / spec["filename"],
            _phase_report(spec["phase_id"]),
        )
    return report_dir


def test_phase2de_accepts_complete_bounded_rollup(tmp_path: Path) -> None:
    report_dir = _report_dir_fixture(tmp_path)
    report = audit_phase2de_compact_evidence_rollup(
        phase2dd_report_json=report_dir / REPORT_SPECS[-1]["filename"],
        output_report_json=tmp_path / "phase2de.json",
    )
    validation = validate_phase2de_compact_evidence_rollup(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["metrics"]["phase_count"] == len(REPORT_SPECS)
    assert report["metrics"]["negative_control_phase_count"] >= 7
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_phase2de_rejects_missing_phase_report(tmp_path: Path) -> None:
    report_dir = _report_dir_fixture(tmp_path)
    (report_dir / REPORT_SPECS[3]["filename"]).unlink()
    report = audit_phase2de_compact_evidence_rollup(
        phase2dd_report_json=report_dir / REPORT_SPECS[-1]["filename"],
        output_report_json=tmp_path / "phase2de.json",
    )
    validation = validate_phase2de_compact_evidence_rollup(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["all_phase_reports_readable"] is False
    assert validation["checks"]["all_phase_reports_readable"] is False


def test_phase2de_rejects_phase_overstated_epoch_claim(tmp_path: Path) -> None:
    report_dir = _report_dir_fixture(tmp_path)
    _write(
        report_dir / REPORT_SPECS[5]["filename"],
        _phase_report(REPORT_SPECS[5]["phase_id"], overstated=True),
    )
    report = audit_phase2de_compact_evidence_rollup(
        phase2dd_report_json=report_dir / REPORT_SPECS[-1]["filename"],
        output_report_json=tmp_path / "phase2de.json",
    )
    validation = validate_phase2de_compact_evidence_rollup(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["all_phase_claims_are_bounded"] is False
    assert validation["checks"]["all_phase_claims_are_bounded"] is False


def test_phase2de_validation_rejects_top_level_overstated_epoch_claim(
    tmp_path: Path,
) -> None:
    report_dir = _report_dir_fixture(tmp_path)
    report = audit_phase2de_compact_evidence_rollup(
        phase2dd_report_json=report_dir / REPORT_SPECS[-1]["filename"],
        output_report_json=tmp_path / "phase2de.json",
    )
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2de_compact_evidence_rollup(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
