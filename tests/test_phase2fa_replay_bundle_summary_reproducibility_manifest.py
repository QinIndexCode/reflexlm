import json
from pathlib import Path

from reflexlm.cli.audit_phase2ez_replay_bundle_summary_replay_bundle_negative_controls import (
    audit_phase2ez_replay_bundle_summary_replay_bundle_negative_controls,
)
from reflexlm.cli.audit_phase2fa_replay_bundle_summary_reproducibility_manifest import (
    REQUIRED_REPRODUCTION_STEPS,
    REQUIRED_SOURCE_REPORT_ROLES,
    audit_phase2fa_replay_bundle_summary_reproducibility_manifest,
    validate_phase2fa_replay_bundle_summary_reproducibility_manifest,
)
from test_phase2ey_replay_bundle_summary_replay_bundle import _phase2ey_report


def _phase2ez_fixture(tmp_path: Path) -> Path:
    _phase2ey_report(tmp_path)
    phase2ez = audit_phase2ez_replay_bundle_summary_replay_bundle_negative_controls(
        phase2ey_report_json=tmp_path / "phase2ey.json",
        output_dir=tmp_path / "ez_controls",
        output_report_json=tmp_path / "phase2ez.json",
    )
    assert phase2ez["passed"] is True
    return tmp_path / "phase2ez.json"


def _phase2fa_report(tmp_path: Path) -> dict:
    report = audit_phase2fa_replay_bundle_summary_reproducibility_manifest(
        phase2ez_report_json=_phase2ez_fixture(tmp_path),
        output_manifest_json=tmp_path / "phase2fa_manifest.json",
        output_report_json=tmp_path / "phase2fa.json",
    )
    assert report["passed"] is True
    return report


def test_phase2fa_accepts_reproducibility_manifest(tmp_path: Path) -> None:
    report = _phase2fa_report(tmp_path)
    validation = validate_phase2fa_replay_bundle_summary_reproducibility_manifest(report)
    manifest = json.loads(
        Path(report["evidence"]["reproducibility_manifest"]).read_text(
            encoding="utf-8"
        )
    )

    assert validation["passed"] is True
    assert set(REQUIRED_SOURCE_REPORT_ROLES).issubset(
        {item["role"] for item in manifest["source_reports"]}
    )
    assert set(REQUIRED_REPRODUCTION_STEPS).issubset(
        {item["step_id"] for item in manifest["reproduction_steps"]}
    )
    assert report["metrics"]["bundle_artifact_count"] == report["metrics"][
        "bundle_artifact_hash_match_count"
    ]
    assert report["metrics"]["phase2ez_negative_control_count"] == report["metrics"][
        "phase2ez_negative_controls_failed"
    ]
    assert report["ready_for_epoch_making_architecture_claim"] is False
    assert (
        report["next_required_experiment"]
        == "phase2fb_replay_bundle_summary_manifest_negative_controls"
    )


def test_phase2fa_validation_rejects_tampered_bundle_artifact(
    tmp_path: Path,
) -> None:
    report = _phase2fa_report(tmp_path)
    manifest = json.loads(
        Path(report["evidence"]["reproducibility_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    first_artifact = manifest["bundle_artifacts"][0]
    Path(first_artifact["path"]).write_text("tampered\n", encoding="utf-8")
    validation = validate_phase2fa_replay_bundle_summary_reproducibility_manifest(report)

    assert validation["passed"] is False
    assert validation["checks"]["bundle_artifact_hashes_match"] is False


def test_phase2fa_validation_rejects_missing_source_report_role(
    tmp_path: Path,
) -> None:
    report = _phase2fa_report(tmp_path)
    manifest_path = Path(report["evidence"]["reproducibility_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_reports"] = [
        item
        for item in manifest["source_reports"]
        if item["role"] != REQUIRED_SOURCE_REPORT_ROLES[0]
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_phase2fa_replay_bundle_summary_reproducibility_manifest(report)

    assert validation["passed"] is False
    assert validation["checks"]["source_report_roles_complete"] is False


def test_phase2fa_validation_rejects_missing_reproduction_step(
    tmp_path: Path,
) -> None:
    report = _phase2fa_report(tmp_path)
    manifest_path = Path(report["evidence"]["reproducibility_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reproduction_steps"] = [
        item
        for item in manifest["reproduction_steps"]
        if item["step_id"] != REQUIRED_REPRODUCTION_STEPS[0]
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_phase2fa_replay_bundle_summary_reproducibility_manifest(report)

    assert validation["passed"] is False
    assert validation["checks"]["reproduction_steps_complete"] is False


def test_phase2fa_validation_rejects_control_artifact_count_mismatch(
    tmp_path: Path,
) -> None:
    report = _phase2fa_report(tmp_path)
    report["source_summary"]["phase2ey_copied_control_artifact_count"] = 0
    validation = validate_phase2fa_replay_bundle_summary_reproducibility_manifest(report)

    assert validation["passed"] is False
    assert validation["checks"]["control_artifact_count_preserved"] is False


def test_phase2fa_validation_rejects_top_level_epoch_claim(tmp_path: Path) -> None:
    report = _phase2fa_report(tmp_path)
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2fa_replay_bundle_summary_reproducibility_manifest(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
