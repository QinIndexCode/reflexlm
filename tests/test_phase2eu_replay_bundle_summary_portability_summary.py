from pathlib import Path

from reflexlm.cli.audit_phase2et_replay_bundle_summary_manifest_cross_directory_negative_controls import (
    audit_phase2et_replay_bundle_summary_manifest_cross_directory_negative_controls,
)
from reflexlm.cli.audit_phase2eu_replay_bundle_summary_portability_summary import (
    REQUIRED_DIMENSIONS,
    SUMMARY_COLUMNS,
    audit_phase2eu_replay_bundle_summary_portability_summary,
    validate_phase2eu_replay_bundle_summary_portability_summary,
)
from test_phase2es_replay_bundle_summary_manifest_cross_directory_replay import (
    _phase2es_report,
)


def _phase2et_fixture(tmp_path: Path) -> Path:
    _phase2es_report(tmp_path)
    phase2et = audit_phase2et_replay_bundle_summary_manifest_cross_directory_negative_controls(
        phase2es_report_json=tmp_path / "phase2es.json",
        output_dir=tmp_path / "et_controls",
        output_report_json=tmp_path / "phase2et.json",
    )
    assert phase2et["passed"] is True
    return tmp_path / "phase2et.json"


def _phase2eu_report(tmp_path: Path) -> dict:
    report = audit_phase2eu_replay_bundle_summary_portability_summary(
        phase2et_report_json=_phase2et_fixture(tmp_path),
        output_report_json=tmp_path / "phase2eu.json",
        output_markdown=tmp_path / "phase2eu.md",
    )
    assert report["passed"] is True
    return report


def test_phase2eu_accepts_replay_bundle_summary_portability_summary(
    tmp_path: Path,
) -> None:
    report = _phase2eu_report(tmp_path)
    validation = validate_phase2eu_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is True
    assert report["summary_table"]["columns"] == list(SUMMARY_COLUMNS)
    assert report["metrics"]["summary_row_count"] >= len(REQUIRED_DIMENSIONS)
    assert report["metrics"]["replayed_bundle_artifact_count"] == report["metrics"][
        "replayed_bundle_artifact_hash_match_count"
    ]
    assert report["metrics"]["phase2er_negative_control_count"] == report["metrics"][
        "phase2er_negative_controls_failed"
    ]
    assert report["metrics"]["phase2et_negative_control_count"] == report["metrics"][
        "phase2et_negative_controls_failed"
    ]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2ev_replay_bundle_summary_portability_summary_negative_controls"
    )


def test_phase2eu_validation_rejects_missing_markdown(tmp_path: Path) -> None:
    report = _phase2eu_report(tmp_path)
    Path(report["evidence"]["portability_summary_markdown"]).unlink()
    validation = validate_phase2eu_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is False
    assert validation["checks"]["markdown_summary_readable"] is False


def test_phase2eu_validation_rejects_missing_dimension_row(tmp_path: Path) -> None:
    report = _phase2eu_report(tmp_path)
    report["summary_table"]["rows"] = [
        row
        for row in report["summary_table"]["rows"]
        if row["Dimension"] != REQUIRED_DIMENSIONS[0]
    ]
    validation = validate_phase2eu_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is False
    assert validation["checks"]["summary_dimensions_complete"] is False


def test_phase2eu_validation_rejects_markdown_missing_dimension(
    tmp_path: Path,
) -> None:
    report = _phase2eu_report(tmp_path)
    markdown = Path(report["evidence"]["portability_summary_markdown"])
    first_dimension = report["summary_table"]["rows"][0]["Dimension"]
    markdown.write_text(
        markdown.read_text(encoding="utf-8").replace(first_dimension, "missing"),
        encoding="utf-8",
    )
    validation = validate_phase2eu_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is False
    assert validation["checks"]["markdown_contains_all_dimensions"] is False


def test_phase2eu_validation_rejects_missing_boundary(tmp_path: Path) -> None:
    report = _phase2eu_report(tmp_path)
    markdown = Path(report["evidence"]["portability_summary_markdown"])
    markdown.write_text("Phase2EU summary\n", encoding="utf-8")
    validation = validate_phase2eu_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is False
    assert validation["checks"]["markdown_contains_bounded_boundary"] is False


def test_phase2eu_validation_rejects_phase2er_controls_incomplete(
    tmp_path: Path,
) -> None:
    report = _phase2eu_report(tmp_path)
    report["source_summary"]["phase2er_negative_controls_failed"] = 0
    validation = validate_phase2eu_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is False
    assert validation["checks"]["source_phase2er_negative_controls_complete"] is False


def test_phase2eu_validation_rejects_phase2et_controls_incomplete(
    tmp_path: Path,
) -> None:
    report = _phase2eu_report(tmp_path)
    report["source_summary"]["phase2et_negative_controls_failed"] = 0
    validation = validate_phase2eu_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is False
    assert validation["checks"]["source_phase2et_negative_controls_complete"] is False


def test_phase2eu_validation_rejects_source_validation_failure(
    tmp_path: Path,
) -> None:
    report = _phase2eu_report(tmp_path)
    report["source_summary"]["phase2es_validation_passed"] = False
    validation = validate_phase2eu_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is False
    assert validation["checks"]["source_phase2es_validation_passed"] is False


def test_phase2eu_validation_rejects_metrics_mismatch(tmp_path: Path) -> None:
    report = _phase2eu_report(tmp_path)
    report["metrics"]["replayed_bundle_artifact_hash_match_count"] = 0
    validation = validate_phase2eu_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is False
    assert validation["checks"]["metrics_match_source"] is False


def test_phase2eu_validation_rejects_top_level_epoch_claim(tmp_path: Path) -> None:
    report = _phase2eu_report(tmp_path)
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2eu_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
