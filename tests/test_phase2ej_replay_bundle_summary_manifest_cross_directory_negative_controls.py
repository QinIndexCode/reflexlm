from pathlib import Path

from reflexlm.cli.audit_phase2ej_replay_bundle_summary_manifest_cross_directory_negative_controls import (
    audit_phase2ej_replay_bundle_summary_manifest_cross_directory_negative_controls,
)
from test_phase2ei_replay_bundle_summary_manifest_cross_directory_replay import (
    _phase2ei_report,
)


def _phase2ei_fixture(tmp_path: Path) -> Path:
    _phase2ei_report(tmp_path)
    return tmp_path / "phase2ei.json"


def test_phase2ej_rejects_manifest_cross_directory_negative_controls(
    tmp_path: Path,
) -> None:
    phase2ei = _phase2ei_fixture(tmp_path)
    source_report_dir = tmp_path / "ei_replay"
    source_replay_files = sorted(
        path.relative_to(source_report_dir).as_posix()
        for path in source_report_dir.rglob("*")
        if path.is_file()
    )

    report = audit_phase2ej_replay_bundle_summary_manifest_cross_directory_negative_controls(
        phase2ei_report_json=phase2ei,
        output_dir=tmp_path / "ej_controls",
        output_report_json=tmp_path / "phase2ej.json",
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
        == "phase2ek_replay_bundle_summary_portability_summary"
    )
