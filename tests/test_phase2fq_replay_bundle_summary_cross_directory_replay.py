from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any

from reflexlm.cli.audit_phase2fp_replay_bundle_summary_portability_summary_negative_controls import (
    audit_phase2fp_replay_bundle_summary_portability_summary_negative_controls,
)
from reflexlm.cli.audit_phase2fq_replay_bundle_summary_cross_directory_replay import (
    audit_phase2fq_replay_bundle_summary_cross_directory_replay,
    validate_phase2fq_replay_bundle_summary_cross_directory_replay,
)
from test_phase2fo_replay_bundle_summary_portability_summary import _phase2fo_report


def _phase2fp_fixture(tmp_path: Path) -> Path:
    _phase2fo_report(tmp_path)
    phase2fp = audit_phase2fp_replay_bundle_summary_portability_summary_negative_controls(
        phase2fo_report_json=tmp_path / "phase2fo.json",
        output_dir=tmp_path / "fp_controls",
        output_report_json=tmp_path / "phase2fp.json",
    )
    assert phase2fp["passed"] is True
    return tmp_path / "phase2fp.json"


def _phase2fq_report(tmp_path: Path) -> dict:
    report = audit_phase2fq_replay_bundle_summary_cross_directory_replay(
        phase2fp_report_json=_phase2fp_fixture(tmp_path),
        output_dir=tmp_path / "fq_replay",
        output_report_json=tmp_path / "phase2fq.json",
    )
    assert report["passed"] is True
    return report


def test_phase2fq_accepts_summary_cross_directory_replay(tmp_path: Path) -> None:
    report = _phase2fq_report(tmp_path)
    validation = validate_phase2fq_replay_bundle_summary_cross_directory_replay(report)

    assert validation["passed"] is True
    assert report["metrics"]["replayed_summary_row_count"] >= 7
    assert report["metrics"]["source_markdown_bytes"] == report["metrics"][
        "replayed_markdown_bytes"
    ]
    assert report["metrics"]["phase2fp_negative_control_count"] == report["metrics"][
        "phase2fp_negative_controls_failed"
    ]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2fr_replay_bundle_summary_replay_negative_controls"
    )


def test_phase2fq_validation_rejects_negative_matrix(tmp_path: Path) -> None:
    base_report = _phase2fq_report(tmp_path)

    report = _isolated_replay_report(base_report, tmp_path, "tampered_markdown")
    Path(report["evidence"]["replayed_markdown"]).write_text(
        "tampered\n",
        encoding="utf-8",
    )
    validation = validate_phase2fq_replay_bundle_summary_cross_directory_replay(report)
    assert validation["passed"] is False
    assert validation["checks"]["replayed_markdown_hash_matches_source"] is False

    report = _isolated_replay_report(base_report, tmp_path, "missing_report")
    Path(report["evidence"]["replayed_phase2fo_report"]).unlink()
    validation = validate_phase2fq_replay_bundle_summary_cross_directory_replay(report)
    assert validation["passed"] is False
    assert validation["checks"]["replayed_phase2fo_report_readable"] is False

    report = _isolated_replay_report(base_report, tmp_path, "count_mismatch")
    report["replay_summary"]["replayed_control_count"] = 0
    validation = validate_phase2fq_replay_bundle_summary_cross_directory_replay(report)
    assert validation["passed"] is False
    assert validation["checks"]["control_results_count_preserved"] is False

    report = _isolated_replay_report(base_report, tmp_path, "same_dir")
    report["replay_summary"]["replay_markdown_parent"] = report["replay_summary"][
        "source_markdown_parent"
    ]
    validation = validate_phase2fq_replay_bundle_summary_cross_directory_replay(report)
    assert validation["passed"] is False
    assert validation["checks"]["replay_directory_is_distinct"] is False

    report = _isolated_replay_report(base_report, tmp_path, "epoch_claim")
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2fq_replay_bundle_summary_cross_directory_replay(report)
    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False


def _rewrite_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, value)
    if isinstance(value, list):
        return [_rewrite_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, path_map) for key, item in value.items()}
    return value


def _isolated_replay_report(report: dict, tmp_path: Path, name: str) -> dict:
    isolated = deepcopy(report)
    source_replay_dir = Path(report["evidence"]["replay_dir"])
    target_replay_dir = tmp_path / f"{name}_replay"
    if target_replay_dir.exists():
        shutil.rmtree(target_replay_dir)
    shutil.copytree(source_replay_dir, target_replay_dir)
    path_map = {str(source_replay_dir): str(target_replay_dir)}
    for source_path in source_replay_dir.rglob("*"):
        if source_path.is_file():
            path_map[str(source_path)] = str(
                target_replay_dir / source_path.relative_to(source_replay_dir)
            )
    return _rewrite_paths(isolated, path_map)
