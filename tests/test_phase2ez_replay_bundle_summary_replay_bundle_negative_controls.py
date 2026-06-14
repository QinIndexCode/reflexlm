import json
from pathlib import Path

from reflexlm.cli.audit_phase2ez_replay_bundle_summary_replay_bundle_negative_controls import (
    audit_phase2ez_replay_bundle_summary_replay_bundle_negative_controls,
)
from test_phase2ey_replay_bundle_summary_replay_bundle import _phase2ey_report


def _phase2ey_fixture(tmp_path: Path) -> Path:
    _phase2ey_report(tmp_path)
    return tmp_path / "phase2ey.json"


def test_phase2ez_rejects_replay_bundle_summary_replay_bundle_negative_controls(
    tmp_path: Path,
) -> None:
    phase2ey = _phase2ey_fixture(tmp_path)
    source_report = json.loads(phase2ey.read_text(encoding="utf-8"))
    source_manifest = Path(source_report["evidence"]["bundle_manifest"])
    source_manifest_text = source_manifest.read_text(encoding="utf-8")

    report = audit_phase2ez_replay_bundle_summary_replay_bundle_negative_controls(
        phase2ey_report_json=phase2ey,
        output_dir=tmp_path / "ez_controls",
        output_report_json=tmp_path / "phase2ez.json",
    )

    assert report["passed"] is True
    assert report["checks"]["positive_control_still_passes"] is True
    assert report["checks"]["all_negative_controls_failed"] is True
    assert report["metrics"]["negative_control_count"] >= 16
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
        == "phase2fa_replay_bundle_summary_reproducibility_manifest"
    )
