import json
from pathlib import Path

from reflexlm.cli.audit_phase2ex_replay_bundle_summary_replay_negative_controls import (
    audit_phase2ex_replay_bundle_summary_replay_negative_controls,
)
from reflexlm.cli.audit_phase2ey_replay_bundle_summary_replay_bundle import (
    REQUIRED_ARTIFACT_ROLES,
    audit_phase2ey_replay_bundle_summary_replay_bundle,
    validate_phase2ey_replay_bundle_summary_replay_bundle,
)
from test_phase2ew_replay_bundle_summary_cross_directory_replay import _phase2ew_report


def _phase2ex_fixture(tmp_path: Path) -> Path:
    _phase2ew_report(tmp_path)
    phase2ex = audit_phase2ex_replay_bundle_summary_replay_negative_controls(
        phase2ew_report_json=tmp_path / "phase2ew.json",
        output_dir=tmp_path / "ex_controls",
        output_report_json=tmp_path / "phase2ex.json",
    )
    assert phase2ex["passed"] is True
    return tmp_path / "phase2ex.json"


def _phase2ey_report(tmp_path: Path) -> dict:
    report = audit_phase2ey_replay_bundle_summary_replay_bundle(
        phase2ex_report_json=_phase2ex_fixture(tmp_path),
        output_dir=tmp_path / "ey_bundle",
        output_report_json=tmp_path / "phase2ey.json",
    )
    assert report["passed"] is True
    return report


def test_phase2ey_accepts_replay_bundle_summary_replay_bundle(tmp_path: Path) -> None:
    report = _phase2ey_report(tmp_path)
    validation = validate_phase2ey_replay_bundle_summary_replay_bundle(report)
    manifest = json.loads(
        Path(report["evidence"]["bundle_manifest"]).read_text(encoding="utf-8")
    )

    assert validation["passed"] is True
    assert set(REQUIRED_ARTIFACT_ROLES).issubset(
        {item["role"] for item in manifest["artifacts"]}
    )
    assert report["metrics"]["copied_control_artifact_count"] == report["metrics"][
        "phase2ew_replayed_control_count"
    ] * 2
    assert report["metrics"]["phase2ex_negative_controls_failed"] == report[
        "metrics"
    ]["phase2ex_negative_control_count"]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2ez_replay_bundle_summary_replay_bundle_negative_controls"
    )


def test_phase2ey_validation_rejects_tampered_artifact(tmp_path: Path) -> None:
    report = _phase2ey_report(tmp_path)
    manifest = json.loads(
        Path(report["evidence"]["bundle_manifest"]).read_text(encoding="utf-8")
    )
    markdown = next(
        item
        for item in manifest["artifacts"]
        if item["role"] == "replayed_bundle_summary_markdown"
    )
    Path(markdown["path"]).write_text("tampered\n", encoding="utf-8")
    validation = validate_phase2ey_replay_bundle_summary_replay_bundle(report)

    assert validation["passed"] is False
    assert validation["checks"]["all_manifest_hashes_match"] is False


def test_phase2ey_validation_rejects_missing_artifact_role(tmp_path: Path) -> None:
    report = _phase2ey_report(tmp_path)
    manifest_path = Path(report["evidence"]["bundle_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        item
        for item in manifest["artifacts"]
        if item["role"] != "replayed_phase2eu_validation"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_phase2ey_replay_bundle_summary_replay_bundle(report)

    assert validation["passed"] is False
    assert validation["checks"]["manifest_roles_complete"] is False


def test_phase2ey_validation_rejects_negative_controls_incomplete(
    tmp_path: Path,
) -> None:
    report = _phase2ey_report(tmp_path)
    report["source_summary"]["phase2ex_negative_controls_failed"] = 0
    validation = validate_phase2ey_replay_bundle_summary_replay_bundle(report)

    assert validation["passed"] is False
    assert validation["checks"]["source_negative_controls_complete"] is False


def test_phase2ey_validation_rejects_top_level_epoch_claim(tmp_path: Path) -> None:
    report = _phase2ey_report(tmp_path)
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2ey_replay_bundle_summary_replay_bundle(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
