from pathlib import Path

from reflexlm.cli.audit_phase2fl_replay_bundle_summary_manifest_negative_controls import (
    audit_phase2fl_replay_bundle_summary_manifest_negative_controls,
)
from reflexlm.cli.audit_phase2fm_replay_bundle_summary_manifest_cross_directory_replay import (
    audit_phase2fm_replay_bundle_summary_manifest_cross_directory_replay,
    validate_phase2fm_replay_bundle_summary_manifest_cross_directory_replay,
)
from test_phase2fk_replay_bundle_summary_reproducibility_manifest import (
    _phase2fk_report,
)


def _phase2fl_fixture(tmp_path: Path) -> Path:
    _phase2fk_report(tmp_path)
    phase2fl = audit_phase2fl_replay_bundle_summary_manifest_negative_controls(
        phase2fk_report_json=tmp_path / "phase2fk.json",
        output_dir=tmp_path / "fl_controls",
        output_report_json=tmp_path / "phase2fl.json",
    )
    assert phase2fl["passed"] is True
    return tmp_path / "phase2fl.json"


def _phase2fm_report(tmp_path: Path) -> dict:
    report = audit_phase2fm_replay_bundle_summary_manifest_cross_directory_replay(
        phase2fl_report_json=_phase2fl_fixture(tmp_path),
        output_dir=tmp_path / "fm_replay",
        output_report_json=tmp_path / "phase2fm.json",
    )
    assert report["passed"] is True
    return report


def test_phase2fm_accepts_manifest_cross_directory_replay(tmp_path: Path) -> None:
    report = _phase2fm_report(tmp_path)
    validation = validate_phase2fm_replay_bundle_summary_manifest_cross_directory_replay(
        report
    )

    assert validation["passed"] is True
    assert report["metrics"]["replayed_source_report_count"] == 4
    assert report["metrics"]["replayed_bundle_artifact_count"] == report["metrics"][
        "replayed_bundle_artifact_hash_match_count"
    ]
    assert report["metrics"]["replayed_control_artifact_count"] == report[
        "source_summary"
    ]["phase2fk_control_artifact_count"]
    assert report["metrics"]["phase2fl_negative_control_count"] == report["metrics"][
        "phase2fl_negative_controls_failed"
    ]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2fn_replay_bundle_summary_manifest_cross_directory_negative_controls"
    )


def test_phase2fm_validation_rejects_tampered_replayed_artifact(
    tmp_path: Path,
) -> None:
    report = _phase2fm_report(tmp_path)
    artifact = next(Path(report["evidence"]["replay_dir"]).glob("bundle_artifacts/*"))
    artifact.write_text("tampered\n", encoding="utf-8")
    validation = validate_phase2fm_replay_bundle_summary_manifest_cross_directory_replay(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["replayed_phase2fk_validation_passed"] is False


def test_phase2fm_validation_rejects_missing_replayed_report(
    tmp_path: Path,
) -> None:
    report = _phase2fm_report(tmp_path)
    Path(report["evidence"]["replayed_phase2fk_report"]).unlink()
    validation = validate_phase2fm_replay_bundle_summary_manifest_cross_directory_replay(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["replayed_phase2fk_report_readable"] is False


def test_phase2fm_validation_rejects_control_count_mismatch(
    tmp_path: Path,
) -> None:
    report = _phase2fm_report(tmp_path)
    report["replay_summary"]["control_artifact_count"] = 0
    validation = validate_phase2fm_replay_bundle_summary_manifest_cross_directory_replay(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["replayed_control_artifact_count_preserved"] is False


def test_phase2fm_validation_rejects_non_distinct_replay_dir(
    tmp_path: Path,
) -> None:
    report = _phase2fm_report(tmp_path)
    report["replay_summary"]["replay_manifest_parent"] = report["replay_summary"][
        "source_manifest_parent"
    ]
    validation = validate_phase2fm_replay_bundle_summary_manifest_cross_directory_replay(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["replay_directory_is_distinct"] is False


def test_phase2fm_validation_rejects_top_level_epoch_claim(tmp_path: Path) -> None:
    report = _phase2fm_report(tmp_path)
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2fm_replay_bundle_summary_manifest_cross_directory_replay(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
