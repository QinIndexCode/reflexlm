import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_publication_bundle import (
    POSITIVE_REPORT_SPECS,
)
from reflexlm.cli.audit_phase2homeostasis_publication_reproducibility_manifest import (
    REQUIRED_REPRODUCTION_STEPS,
    REQUIRED_SOURCE_REPORT_ROLES,
    REQUIRED_STATE_ARTIFACT_ROLES,
    audit_phase2homeostasis_publication_reproducibility_manifest,
    validate_phase2homeostasis_publication_reproducibility_manifest,
)
from reflexlm.cli.audit_phase2homeostasis_reproducibility_manifest_replay import (
    audit_phase2homeostasis_reproducibility_manifest_replay,
    validate_phase2homeostasis_reproducibility_manifest_replay,
)


def _write(path: Path, payload: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _source_report(path: Path, role: str) -> Path:
    return _write(
        path,
        {
            "artifact_family": role,
            "passed": True,
            "ready_for_bounded_fixture_claim": True,
            "ready_for_epoch_making_architecture_claim": False,
        },
    )


def _bundle_fixture(tmp_path: Path) -> Path:
    positive_rows = []
    for spec in POSITIVE_REPORT_SPECS:
        report_path = _source_report(tmp_path / f"{spec['role']}.json", spec["role"])
        positive_rows.append(
            {
                "role": spec["role"],
                "report_json": str(report_path),
                "passed": True,
                "bounded_claim_ok": True,
                "artifact_family": spec["family"],
            }
        )
    state_rows = []
    for index, role in enumerate(REQUIRED_STATE_ARTIFACT_ROLES):
        state_path = _write(
            tmp_path / f"{role}.json",
            {
                "role": role,
                "state": {"index": index},
                "authenticator": {"algorithm": "hmac-sha256"},
            },
        )
        state_rows.append(
            {
                "state_json": str(state_path),
                "authenticator_algorithm": "hmac-sha256",
            }
        )
    negative_path = _write(
        tmp_path / "hmac_missing_key_negative_control.json",
        {"artifact_family": "phase2homeostasis_persistent_state_chain", "passed": False},
    )
    markdown_path = _write(tmp_path / "bundle.md", "# bundle\n")
    return _write(
        tmp_path / "bundle.json",
        {
            "artifact_family": "phase2homeostasis_publication_bundle",
            "passed": True,
            "metrics": {
                "positive_report_count": len(positive_rows),
                "state_artifact_count": len(state_rows),
                "negative_control_count": 1,
            },
            "positive_evidence": positive_rows,
            "state_artifacts": state_rows,
            "negative_controls": [
                {
                    "role": "hmac_missing_key_negative_control",
                    "report_json": str(negative_path),
                    "expected_failure_observed": True,
                }
            ],
            "evidence": {"output_markdown": str(markdown_path)},
        },
    )


def _manifest_report(tmp_path: Path) -> dict:
    report = audit_phase2homeostasis_publication_reproducibility_manifest(
        bundle_report_json=_bundle_fixture(tmp_path),
        output_manifest_json=tmp_path / "manifest.json",
        output_report_json=tmp_path / "manifest_report.json",
    )
    assert report["passed"] is True
    return report


def test_homeostasis_reproducibility_manifest_accepts_bundle(
    tmp_path: Path,
) -> None:
    report = _manifest_report(tmp_path)
    validation = validate_phase2homeostasis_publication_reproducibility_manifest(
        report
    )

    assert validation["passed"] is True
    assert report["metrics"]["source_report_count"] == len(
        REQUIRED_SOURCE_REPORT_ROLES
    )
    assert report["metrics"]["state_artifact_count"] == len(
        REQUIRED_STATE_ARTIFACT_ROLES
    )
    assert report["metrics"]["reproduction_step_count"] == len(
        REQUIRED_REPRODUCTION_STEPS
    )
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_homeostasis_reproducibility_manifest_validation_rejects_missing_state(
    tmp_path: Path,
) -> None:
    report = _manifest_report(tmp_path)
    manifest = json.loads(
        Path(report["evidence"]["reproducibility_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    Path(manifest["state_artifacts"][0]["path"]).unlink()

    validation = validate_phase2homeostasis_publication_reproducibility_manifest(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["state_artifacts_exist"] is False


def test_homeostasis_reproducibility_manifest_validation_rejects_tampered_source(
    tmp_path: Path,
) -> None:
    report = _manifest_report(tmp_path)
    manifest = json.loads(
        Path(report["evidence"]["reproducibility_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    Path(manifest["source_reports"][1]["path"]).write_text(
        "tampered\n",
        encoding="utf-8",
    )

    validation = validate_phase2homeostasis_publication_reproducibility_manifest(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["source_report_hashes_match"] is False


def test_homeostasis_reproducibility_manifest_validation_rejects_missing_step(
    tmp_path: Path,
) -> None:
    report = _manifest_report(tmp_path)
    manifest_path = Path(report["evidence"]["reproducibility_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reproduction_steps"] = manifest["reproduction_steps"][:-1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validation = validate_phase2homeostasis_publication_reproducibility_manifest(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["reproduction_steps_complete"] is False


def test_homeostasis_reproducibility_manifest_validation_rejects_epoch_claim(
    tmp_path: Path,
) -> None:
    report = _manifest_report(tmp_path)
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2homeostasis_publication_reproducibility_manifest(
        report
    )

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False


def test_homeostasis_reproducibility_manifest_replay_accepts_distinct_copy(
    tmp_path: Path,
) -> None:
    manifest_report = _manifest_report(tmp_path)
    replay = audit_phase2homeostasis_reproducibility_manifest_replay(
        reproducibility_report_json=tmp_path / "manifest_report.json",
        output_dir=tmp_path / "replay",
        output_report_json=tmp_path / "replay_report.json",
    )
    validation = validate_phase2homeostasis_reproducibility_manifest_replay(
        replay
    )

    assert manifest_report["passed"] is True
    assert replay["passed"] is True
    assert validation["passed"] is True
    assert replay["metrics"]["replayed_state_artifact_count"] == len(
        REQUIRED_STATE_ARTIFACT_ROLES
    )
    assert replay["ready_for_epoch_making_architecture_claim"] is False


def test_homeostasis_reproducibility_manifest_replay_rejects_tampered_copy(
    tmp_path: Path,
) -> None:
    _manifest_report(tmp_path)
    replay = audit_phase2homeostasis_reproducibility_manifest_replay(
        reproducibility_report_json=tmp_path / "manifest_report.json",
        output_dir=tmp_path / "replay",
        output_report_json=tmp_path / "replay_report.json",
    )
    replayed_report = json.loads(
        Path(replay["evidence"]["replayed_reproducibility_report"]).read_text(
            encoding="utf-8"
        )
    )
    replayed_manifest = json.loads(
        Path(replayed_report["evidence"]["reproducibility_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    Path(replayed_manifest["state_artifacts"][0]["path"]).write_text(
        "tampered\n",
        encoding="utf-8",
    )
    validation = validate_phase2homeostasis_reproducibility_manifest_replay(
        replay
    )

    assert validation["passed"] is False
    assert validation["checks"]["replayed_manifest_validation_passed"] is False


def test_homeostasis_reproducibility_manifest_replay_rejects_non_distinct_dir(
    tmp_path: Path,
) -> None:
    _manifest_report(tmp_path)
    replay = audit_phase2homeostasis_reproducibility_manifest_replay(
        reproducibility_report_json=tmp_path / "manifest_report.json",
        output_dir=tmp_path / "replay",
        output_report_json=tmp_path / "replay_report.json",
    )
    replay["replay_summary"]["replay_manifest_parent"] = replay["replay_summary"][
        "source_manifest_parent"
    ]
    validation = validate_phase2homeostasis_reproducibility_manifest_replay(
        replay
    )

    assert validation["passed"] is False
    assert validation["checks"]["replay_directory_is_distinct"] is False
