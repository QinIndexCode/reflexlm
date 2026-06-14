from pathlib import Path

from reflexlm.cli.audit_phase2el_replay_bundle_summary_portability_summary_negative_controls import (
    audit_phase2el_replay_bundle_summary_portability_summary_negative_controls,
)
from reflexlm.cli.audit_phase2em_replay_bundle_summary_cross_directory_replay import (
    audit_phase2em_replay_bundle_summary_cross_directory_replay,
    validate_phase2em_replay_bundle_summary_cross_directory_replay,
)
from test_phase2ek_replay_bundle_summary_portability_summary import _phase2ek_report


def _phase2el_fixture(tmp_path: Path) -> Path:
    _phase2ek_report(tmp_path)
    phase2el = audit_phase2el_replay_bundle_summary_portability_summary_negative_controls(
        phase2ek_report_json=tmp_path / "phase2ek.json",
        output_dir=tmp_path / "el_controls",
        output_report_json=tmp_path / "phase2el.json",
    )
    assert phase2el["passed"] is True
    return tmp_path / "phase2el.json"


def _phase2em_report(tmp_path: Path) -> dict:
    report = audit_phase2em_replay_bundle_summary_cross_directory_replay(
        phase2el_report_json=_phase2el_fixture(tmp_path),
        output_dir=tmp_path / "em_replay",
        output_report_json=tmp_path / "phase2em.json",
    )
    assert report["passed"] is True
    return report


def test_phase2em_accepts_summary_cross_directory_replay(tmp_path: Path) -> None:
    report = _phase2em_report(tmp_path)
    validation = validate_phase2em_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is True
    assert report["metrics"]["replayed_summary_row_count"] >= 7
    assert report["metrics"]["source_markdown_bytes"] == report["metrics"][
        "replayed_markdown_bytes"
    ]
    assert report["metrics"]["phase2el_negative_control_count"] == report["metrics"][
        "phase2el_negative_controls_failed"
    ]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2en_replay_bundle_summary_replay_negative_controls"
    )


def test_phase2em_validation_rejects_tampered_replayed_markdown(
    tmp_path: Path,
) -> None:
    report = _phase2em_report(tmp_path)
    Path(report["evidence"]["replayed_markdown"]).write_text(
        "tampered\n",
        encoding="utf-8",
    )
    validation = validate_phase2em_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["replayed_markdown_hash_matches_source"] is False


def test_phase2em_validation_rejects_missing_replayed_report(
    tmp_path: Path,
) -> None:
    report = _phase2em_report(tmp_path)
    Path(report["evidence"]["replayed_phase2ek_report"]).unlink()
    validation = validate_phase2em_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["replayed_phase2ek_report_readable"] is False


def test_phase2em_validation_rejects_control_count_mismatch(
    tmp_path: Path,
) -> None:
    report = _phase2em_report(tmp_path)
    report["replay_summary"]["replayed_control_count"] = 0
    validation = validate_phase2em_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["control_results_count_preserved"] is False


def test_phase2em_validation_rejects_non_distinct_replay_dir(tmp_path: Path) -> None:
    report = _phase2em_report(tmp_path)
    report["replay_summary"]["replay_markdown_parent"] = report["replay_summary"][
        "source_markdown_parent"
    ]
    validation = validate_phase2em_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["replay_directory_is_distinct"] is False


def test_phase2em_validation_rejects_top_level_epoch_claim(tmp_path: Path) -> None:
    report = _phase2em_report(tmp_path)
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2em_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
