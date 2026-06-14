from pathlib import Path

from reflexlm.cli.audit_phase2eb_replay_bundle_portability_summary_negative_controls import (
    audit_phase2eb_replay_bundle_portability_summary_negative_controls,
)
from test_phase2ea_replay_bundle_portability_summary import _phase2ea_report


def _phase2ea_fixture(tmp_path: Path) -> Path:
    _phase2ea_report(tmp_path)
    return tmp_path / "phase2ea.json"


def test_phase2eb_rejects_replay_bundle_portability_summary_negative_controls(
    tmp_path: Path,
) -> None:
    report = audit_phase2eb_replay_bundle_portability_summary_negative_controls(
        phase2ea_report_json=_phase2ea_fixture(tmp_path),
        output_dir=tmp_path / "eb_controls",
        output_report_json=tmp_path / "phase2eb.json",
    )

    assert report["passed"] is True
    assert report["checks"]["positive_control_still_passes"] is True
    assert report["checks"]["all_negative_controls_failed"] is True
    assert report["metrics"]["negative_control_count"] >= 13
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
        == "phase2ec_replay_bundle_summary_cross_directory_replay"
    )
