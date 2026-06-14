from pathlib import Path

from reflexlm.cli.audit_phase2fz_replay_bundle_summary_portability_summary_negative_controls import (
    audit_phase2fz_replay_bundle_summary_portability_summary_negative_controls,
)
from test_phase2fy_replay_bundle_summary_portability_summary import _phase2fy_report


def _phase2fy_fixture(tmp_path: Path) -> Path:
    _phase2fy_report(tmp_path)
    return tmp_path / "phase2fy.json"


def test_phase2fz_rejects_replay_bundle_summary_portability_summary_negative_controls(
    tmp_path: Path,
) -> None:
    report = audit_phase2fz_replay_bundle_summary_portability_summary_negative_controls(
        phase2fy_report_json=_phase2fy_fixture(tmp_path),
        output_dir=tmp_path / "fz_controls",
        output_report_json=tmp_path / "phase2fz.json",
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
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2ga_replay_bundle_summary_cross_directory_replay"
    )
