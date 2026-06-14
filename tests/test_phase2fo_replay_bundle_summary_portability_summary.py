from copy import deepcopy
from pathlib import Path

from reflexlm.cli.audit_phase2fn_replay_bundle_summary_manifest_cross_directory_negative_controls import (
    audit_phase2fn_replay_bundle_summary_manifest_cross_directory_negative_controls,
)
from reflexlm.cli.audit_phase2fo_replay_bundle_summary_portability_summary import (
    REQUIRED_DIMENSIONS,
    SUMMARY_COLUMNS,
    audit_phase2fo_replay_bundle_summary_portability_summary,
    validate_phase2fo_replay_bundle_summary_portability_summary,
)
from test_phase2fm_replay_bundle_summary_manifest_cross_directory_replay import (
    _phase2fm_report,
)


def _phase2fn_fixture(tmp_path: Path) -> Path:
    _phase2fm_report(tmp_path)
    phase2fn = audit_phase2fn_replay_bundle_summary_manifest_cross_directory_negative_controls(
        phase2fm_report_json=tmp_path / "phase2fm.json",
        output_dir=tmp_path / "fn_controls",
        output_report_json=tmp_path / "phase2fn.json",
    )
    assert phase2fn["passed"] is True
    return tmp_path / "phase2fn.json"


def _phase2fo_report(tmp_path: Path) -> dict:
    report = audit_phase2fo_replay_bundle_summary_portability_summary(
        phase2fn_report_json=_phase2fn_fixture(tmp_path),
        output_report_json=tmp_path / "phase2fo.json",
        output_markdown=tmp_path / "phase2fo.md",
    )
    assert report["passed"] is True
    return report


def test_phase2fo_accepts_replay_bundle_summary_portability_summary(
    tmp_path: Path,
) -> None:
    report = _phase2fo_report(tmp_path)
    validation = validate_phase2fo_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is True
    assert report["summary_table"]["columns"] == list(SUMMARY_COLUMNS)
    assert report["metrics"]["summary_row_count"] >= len(REQUIRED_DIMENSIONS)
    assert report["metrics"]["replayed_bundle_artifact_count"] == report["metrics"][
        "replayed_bundle_artifact_hash_match_count"
    ]
    assert report["metrics"]["phase2fl_negative_control_count"] == report["metrics"][
        "phase2fl_negative_controls_failed"
    ]
    assert report["metrics"]["phase2fn_negative_control_count"] == report["metrics"][
        "phase2fn_negative_controls_failed"
    ]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2fp_replay_bundle_summary_portability_summary_negative_controls"
    )


def test_phase2fo_validation_rejects_missing_markdown(tmp_path: Path) -> None:
    report = _isolated_report(_phase2fo_report(tmp_path), tmp_path, "missing_markdown")
    Path(report["evidence"]["portability_summary_markdown"]).unlink()
    validation = validate_phase2fo_replay_bundle_summary_portability_summary(report)

    assert validation["passed"] is False
    assert validation["checks"]["markdown_summary_readable"] is False


def _isolated_report(report: dict, tmp_path: Path, name: str) -> dict:
    isolated = deepcopy(report)
    source_markdown = Path(report["evidence"]["portability_summary_markdown"])
    target_markdown = tmp_path / f"{name}.md"
    target_markdown.write_text(source_markdown.read_text(encoding="utf-8"), encoding="utf-8")
    isolated["evidence"]["portability_summary_markdown"] = str(target_markdown)
    return isolated


def test_phase2fo_validation_rejects_negative_matrix(tmp_path: Path) -> None:
    base_report = _phase2fo_report(tmp_path)

    report = _isolated_report(base_report, tmp_path, "missing_dimension_row")
    report["summary_table"]["rows"] = [
        row
        for row in report["summary_table"]["rows"]
        if row["Dimension"] != REQUIRED_DIMENSIONS[0]
    ]
    validation = validate_phase2fo_replay_bundle_summary_portability_summary(report)
    assert validation["passed"] is False
    assert validation["checks"]["summary_dimensions_complete"] is False

    report = _isolated_report(base_report, tmp_path, "markdown_missing_dimension")
    markdown = Path(report["evidence"]["portability_summary_markdown"])
    first_dimension = report["summary_table"]["rows"][0]["Dimension"]
    markdown.write_text(
        markdown.read_text(encoding="utf-8").replace(first_dimension, "missing"),
        encoding="utf-8",
    )
    validation = validate_phase2fo_replay_bundle_summary_portability_summary(report)
    assert validation["passed"] is False
    assert validation["checks"]["markdown_contains_all_dimensions"] is False

    report = _isolated_report(base_report, tmp_path, "missing_boundary")
    markdown = Path(report["evidence"]["portability_summary_markdown"])
    markdown.write_text("Phase2FO summary\n", encoding="utf-8")
    validation = validate_phase2fo_replay_bundle_summary_portability_summary(report)
    assert validation["passed"] is False
    assert validation["checks"]["markdown_contains_bounded_boundary"] is False

    report = _isolated_report(base_report, tmp_path, "phase2fl_incomplete")
    report["source_summary"]["phase2fl_negative_controls_failed"] = 0
    validation = validate_phase2fo_replay_bundle_summary_portability_summary(report)
    assert validation["passed"] is False
    assert validation["checks"]["source_phase2fl_negative_controls_complete"] is False

    report = _isolated_report(base_report, tmp_path, "phase2fn_incomplete")
    report["source_summary"]["phase2fn_negative_controls_failed"] = 0
    validation = validate_phase2fo_replay_bundle_summary_portability_summary(report)
    assert validation["passed"] is False
    assert validation["checks"]["source_phase2fn_negative_controls_complete"] is False

    report = _isolated_report(base_report, tmp_path, "source_validation_failure")
    report["source_summary"]["phase2fm_validation_passed"] = False
    validation = validate_phase2fo_replay_bundle_summary_portability_summary(report)
    assert validation["passed"] is False
    assert validation["checks"]["source_phase2fm_validation_passed"] is False

    report = _isolated_report(base_report, tmp_path, "metrics_mismatch")
    report["metrics"]["replayed_bundle_artifact_hash_match_count"] = 0
    validation = validate_phase2fo_replay_bundle_summary_portability_summary(report)
    assert validation["passed"] is False
    assert validation["checks"]["metrics_match_source"] is False

    report = _isolated_report(base_report, tmp_path, "epoch_claim")
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2fo_replay_bundle_summary_portability_summary(report)
    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
