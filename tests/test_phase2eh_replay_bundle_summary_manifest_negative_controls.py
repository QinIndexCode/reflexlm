import json
from pathlib import Path

from reflexlm.cli.audit_phase2eh_replay_bundle_summary_manifest_negative_controls import (
    audit_phase2eh_replay_bundle_summary_manifest_negative_controls,
)
from test_phase2eg_replay_bundle_summary_reproducibility_manifest import _phase2eg_report


def _phase2eg_fixture(tmp_path: Path) -> Path:
    _phase2eg_report(tmp_path)
    return tmp_path / "phase2eg.json"


def test_phase2eh_rejects_replay_bundle_summary_manifest_negative_controls(
    tmp_path: Path,
) -> None:
    phase2eg = _phase2eg_fixture(tmp_path)
    source_report = json.loads(phase2eg.read_text(encoding="utf-8"))
    source_manifest = Path(source_report["evidence"]["reproducibility_manifest"])
    source_manifest_text = source_manifest.read_text(encoding="utf-8")

    report = audit_phase2eh_replay_bundle_summary_manifest_negative_controls(
        phase2eg_report_json=phase2eg,
        output_dir=tmp_path / "eh_controls",
        output_report_json=tmp_path / "phase2eh.json",
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
        == "phase2ei_replay_bundle_summary_manifest_cross_directory_replay"
    )
