from pathlib import Path

from reflexlm.cli.audit_phase2eb_replay_bundle_portability_summary_negative_controls import (
    audit_phase2eb_replay_bundle_portability_summary_negative_controls,
)
from reflexlm.cli.audit_phase2ec_replay_bundle_summary_cross_directory_replay import (
    audit_phase2ec_replay_bundle_summary_cross_directory_replay,
    validate_phase2ec_replay_bundle_summary_cross_directory_replay,
)
from test_phase2ea_replay_bundle_portability_summary import _phase2ea_report


def _phase2eb_fixture(tmp_path: Path) -> Path:
    _phase2ea_report(tmp_path)
    phase2eb = audit_phase2eb_replay_bundle_portability_summary_negative_controls(
        phase2ea_report_json=tmp_path / "phase2ea.json",
        output_dir=tmp_path / "eb_controls",
        output_report_json=tmp_path / "phase2eb.json",
    )
    assert phase2eb["passed"] is True
    return tmp_path / "phase2eb.json"


def _phase2ec_report(tmp_path: Path) -> dict:
    report = audit_phase2ec_replay_bundle_summary_cross_directory_replay(
        phase2eb_report_json=_phase2eb_fixture(tmp_path),
        output_dir=tmp_path / "ec_replay",
        output_report_json=tmp_path / "phase2ec.json",
    )
    assert report["passed"] is True
    return report


def test_phase2ec_accepts_replay_bundle_summary_cross_directory_replay(
    tmp_path: Path,
) -> None:
    report = _phase2ec_report(tmp_path)
    validation = validate_phase2ec_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is True
    assert report["metrics"]["replayed_summary_row_count"] >= 6
    assert report["metrics"]["replayed_control_count"] == report["replay_summary"][
        "source_control_count"
    ]
    assert report["metrics"]["phase2eb_negative_control_count"] == report["metrics"][
        "phase2eb_negative_controls_failed"
    ]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2ed_replay_bundle_summary_replay_negative_controls"
    )


def test_phase2ec_validation_rejects_tampered_replayed_markdown(
    tmp_path: Path,
) -> None:
    report = _phase2ec_report(tmp_path)
    Path(report["evidence"]["replayed_markdown"]).write_text(
        "tampered\n",
        encoding="utf-8",
    )
    validation = validate_phase2ec_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["replayed_markdown_hash_matches_source"] is False


def test_phase2ec_validation_rejects_missing_replayed_report(
    tmp_path: Path,
) -> None:
    report = _phase2ec_report(tmp_path)
    Path(report["evidence"]["replayed_phase2ea_report"]).unlink()
    validation = validate_phase2ec_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["replayed_phase2ea_report_readable"] is False


def test_phase2ec_validation_rejects_missing_copied_control_file(
    tmp_path: Path,
) -> None:
    report = _phase2ec_report(tmp_path)
    first_control = report["replayed_control_results"][0]
    Path(first_control["replayed_validation_report_json"]).unlink()
    validation = validate_phase2ec_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["copied_control_files_complete"] is False


def test_phase2ec_validation_rejects_control_count_mismatch(
    tmp_path: Path,
) -> None:
    report = _phase2ec_report(tmp_path)
    report["replay_summary"]["replayed_control_count"] = 0
    validation = validate_phase2ec_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["control_results_count_preserved"] is False


def test_phase2ec_validation_rejects_non_distinct_replay_dir(
    tmp_path: Path,
) -> None:
    report = _phase2ec_report(tmp_path)
    report["replay_summary"]["replay_markdown_parent"] = report["replay_summary"][
        "source_markdown_parent"
    ]
    validation = validate_phase2ec_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["replay_directory_is_distinct"] is False


def test_phase2ec_validation_rejects_top_level_epoch_claim(
    tmp_path: Path,
) -> None:
    report = _phase2ec_report(tmp_path)
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2ec_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
