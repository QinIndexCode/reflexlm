from pathlib import Path

from reflexlm.cli.audit_phase2fl_replay_bundle_summary_manifest_negative_controls import (
    audit_phase2fl_replay_bundle_summary_manifest_negative_controls,
)
from test_phase2fk_replay_bundle_summary_reproducibility_manifest import (
    _phase2fk_report,
)


def _phase2fk_fixture(tmp_path: Path) -> Path:
    _phase2fk_report(tmp_path)
    return tmp_path / "phase2fk.json"


def test_phase2fl_rejects_replay_bundle_summary_manifest_negative_controls(
    tmp_path: Path,
) -> None:
    phase2fk = _phase2fk_fixture(tmp_path)
    source_manifest = tmp_path / "phase2fk_manifest.json"
    source_manifest_text = source_manifest.read_text(encoding="utf-8")

    report = audit_phase2fl_replay_bundle_summary_manifest_negative_controls(
        phase2fk_report_json=phase2fk,
        output_dir=tmp_path / "fl_controls",
        output_report_json=tmp_path / "phase2fl.json",
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
    assert source_manifest.read_text(encoding="utf-8") == source_manifest_text
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2fm_replay_bundle_summary_manifest_cross_directory_replay"
    )
