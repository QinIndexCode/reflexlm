from pathlib import Path

from reflexlm.cli.audit_phase2ex_replay_bundle_summary_replay_negative_controls import (
    audit_phase2ex_replay_bundle_summary_replay_negative_controls,
)
from test_phase2ew_replay_bundle_summary_cross_directory_replay import _phase2ew_report


def _phase2ew_fixture(tmp_path: Path) -> Path:
    _phase2ew_report(tmp_path)
    return tmp_path / "phase2ew.json"


def test_phase2ex_rejects_replay_bundle_summary_replay_negative_controls(
    tmp_path: Path,
) -> None:
    phase2ew = _phase2ew_fixture(tmp_path)
    source_replay_dir = tmp_path / "ew_replay"
    source_replay_files = sorted(
        path.relative_to(source_replay_dir).as_posix()
        for path in source_replay_dir.rglob("*")
        if path.is_file()
    )

    report = audit_phase2ex_replay_bundle_summary_replay_negative_controls(
        phase2ew_report_json=phase2ew,
        output_dir=tmp_path / "ex_controls",
        output_report_json=tmp_path / "phase2ex.json",
    )

    assert report["passed"] is True
    assert report["checks"]["positive_control_still_passes"] is True
    assert report["checks"]["all_negative_controls_failed"] is True
    assert report["metrics"]["negative_control_count"] >= 17
    assert report["metrics"]["negative_controls_failed"] == report["metrics"][
        "negative_control_count"
    ]
    assert all(
        row["expected_failed_checks_observed"]
        for row in report["control_results"]
    )
    assert (
        sorted(
            path.relative_to(source_replay_dir).as_posix()
            for path in source_replay_dir.rglob("*")
            if path.is_file()
        )
        == source_replay_files
    )
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2ey_replay_bundle_summary_replay_bundle"
    )
