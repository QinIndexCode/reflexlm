from pathlib import Path

from reflexlm.cli.audit_phase2fd_replay_bundle_summary_manifest_cross_directory_negative_controls import (
    audit_phase2fd_replay_bundle_summary_manifest_cross_directory_negative_controls,
)
from test_phase2fc_replay_bundle_summary_manifest_cross_directory_replay import (
    _phase2fc_report,
)


def _phase2fc_fixture(tmp_path: Path) -> Path:
    _phase2fc_report(tmp_path)
    return tmp_path / "phase2fc.json"


def test_phase2fd_rejects_manifest_cross_directory_negative_controls(
    tmp_path: Path,
) -> None:
    phase2fc = _phase2fc_fixture(tmp_path)
    source_report_dir = tmp_path / "fc_replay"
    source_replay_files = sorted(
        path.relative_to(source_report_dir).as_posix()
        for path in source_report_dir.rglob("*")
        if path.is_file()
    )

    report = audit_phase2fd_replay_bundle_summary_manifest_cross_directory_negative_controls(
        phase2fc_report_json=phase2fc,
        output_dir=tmp_path / "fd_controls",
        output_report_json=tmp_path / "phase2fd.json",
    )

    assert report["passed"] is True
    assert report["checks"]["positive_control_still_passes"] is True
    assert report["checks"]["all_negative_controls_failed"] is True
    assert report["metrics"]["negative_control_count"] >= 15
    assert report["metrics"]["negative_controls_failed"] == report["metrics"][
        "negative_control_count"
    ]
    assert all(
        row["expected_failed_checks_observed"]
        for row in report["control_results"]
    )
    assert (
        sorted(
            path.relative_to(source_report_dir).as_posix()
            for path in source_report_dir.rglob("*")
            if path.is_file()
        )
        == source_replay_files
    )
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2fe_replay_bundle_summary_portability_summary"
    )
