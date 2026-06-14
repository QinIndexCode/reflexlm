import json
import hashlib
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_fresh_rerun_from_manifest import (
    audit_phase2homeostasis_fresh_rerun_from_manifest,
    validate_phase2homeostasis_fresh_rerun_from_manifest,
)
from reflexlm.cli.audit_phase2homeostasis_publication_bundle import (
    POSITIVE_REPORT_SPECS,
)
from reflexlm.cli.audit_phase2homeostasis_publication_reproducibility_manifest import (
    REQUIRED_REPRODUCTION_STEPS,
    REQUIRED_SOURCE_REPORT_ROLES,
    REQUIRED_STATE_ARTIFACT_ROLES,
)


def _write(path: Path, payload: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(role: str, path: Path) -> dict:
    return {
        "role": role,
        "path": str(path),
        "content_type": "application/json",
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_manifest_report(tmp_path: Path) -> Path:
    source_reports = []
    for role in [
        "phase2homeostasis_publication_bundle",
        *(spec["role"] for spec in POSITIVE_REPORT_SPECS),
        "hmac_missing_key_negative_control",
    ]:
        report_path = _write(tmp_path / "source" / f"{role}.json", {"role": role})
        source_reports.append({"role": role, "path": str(report_path)})
    state_artifacts = []
    for role in REQUIRED_STATE_ARTIFACT_ROLES:
        state_path = _write(tmp_path / "source" / f"{role}.json", {"role": role})
        state_artifacts.append({"role": role, "path": str(state_path)})
    manifest_path = _write(
        tmp_path / "source" / "manifest.json",
        {
            "source_reports": source_reports,
            "state_artifacts": state_artifacts,
        },
    )
    return _write(
        tmp_path / "source" / "repro_report.json",
        {
            "passed": True,
            "evidence": {"reproducibility_manifest": str(manifest_path)},
        },
    )


def _fresh_bundle(tmp_path: Path, *, reuse_source_path: Path | None = None) -> Path:
    positive_rows = []
    for spec in POSITIVE_REPORT_SPECS:
        report_path = (
            reuse_source_path
            if reuse_source_path is not None and spec["role"] == POSITIVE_REPORT_SPECS[0]["role"]
            else _write(tmp_path / "fresh" / f"{spec['role']}.json", {"role": spec["role"]})
        )
        positive_rows.append(
            {
                "role": spec["role"],
                "report_json": str(report_path),
                "readable": True,
                "passed": True,
                "artifact_family": spec["family"],
                "expected_family": spec["family"],
                "family_matches_expected": True,
                "bounded_claim_ok": True,
            }
        )
    state_rows = []
    for role in REQUIRED_STATE_ARTIFACT_ROLES:
        state_path = _write(tmp_path / "fresh" / f"{role}.json", {"role": role})
        state_rows.append(
            {
                "state_json": str(state_path),
                "readable": True,
                "schema_valid": True,
                "bounded_state_keys_only": True,
                "authenticator_algorithm": "hmac-sha256",
                "key_fingerprint_sha256": "abc",
                "integrity_valid": True,
            }
        )
    negative_path = _write(
        tmp_path / "fresh" / "hmac_missing_key_negative_control.json",
        {"passed": False},
    )
    bundle_path = tmp_path / "fresh" / "bundle.json"
    return _write(
        bundle_path,
        {
            "artifact_family": "phase2homeostasis_publication_bundle",
            "passed": True,
            "ready_for_bounded_homeostasis_publication_bundle_claim": True,
            "ready_for_epoch_making_architecture_claim": False,
            "checks": {"ok": True},
            "metrics": {
                "positive_report_count": len(positive_rows),
                "state_artifact_count": len(state_rows),
                "negative_control_count": 1,
                "hmac_state_artifact_count": len(state_rows),
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
            "forbidden_secret_scan": {"enabled": True, "matches": []},
            "evidence": {
                "phase2homeostasis_publication_bundle_report_json": str(bundle_path)
            },
        },
    )


def _fresh_repro_report(tmp_path: Path) -> Path:
    source_entries = []
    for role in REQUIRED_SOURCE_REPORT_ROLES:
        source_entries.append(
            _entry(role, _write(tmp_path / "fresh_repro" / f"{role}.json", {"role": role}))
        )
    state_entries = []
    for role in REQUIRED_STATE_ARTIFACT_ROLES:
        state_entries.append(
            _entry(role, _write(tmp_path / "fresh_repro" / f"{role}.json", {"role": role}))
        )
    supporting = [
        _entry(
            "publication_bundle_markdown_table",
            _write(tmp_path / "fresh_repro" / "bundle.md", "# bundle\n"),
        )
    ]
    steps = [
        {
            "step_id": step_id,
            "module": f"reflexlm.cli.{step_id}",
            "outputs": {"output_report_json": str(tmp_path / "fresh_repro" / f"{step_id}.json")},
        }
        for step_id in REQUIRED_REPRODUCTION_STEPS
    ]
    manifest = _write(
        tmp_path / "fresh" / "manifest.json",
        {
            "source_reports": source_entries,
            "state_artifacts": state_entries,
            "supporting_artifacts": supporting,
            "reproduction_steps": steps,
            "claim_boundary": (
                "bounded fresh rerun manifest; does not support unbounded memory, "
                "open-ended native perception, production autonomy, or epoch-making architecture"
            ),
        },
    )
    return _write(
        tmp_path / "fresh" / "repro_report.json",
        {
            "artifact_family": "phase2homeostasis_publication_reproducibility_manifest",
            "passed": True,
            "ready_for_bounded_homeostasis_reproducibility_manifest_claim": True,
            "ready_for_epoch_making_architecture_claim": False,
            "checks": {"ok": True},
            "metrics": {"reproduction_step_count": 9},
            "source_summary": {"bundle_passed": True},
            "evidence": {"reproducibility_manifest": str(manifest)},
        },
    )


def _fresh_replay_report(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "fresh" / "replay_report.json",
        {
            "artifact_family": "phase2homeostasis_reproducibility_manifest_replay",
            "passed": True,
            "ready_for_bounded_homeostasis_manifest_replay_claim": True,
            "ready_for_epoch_making_architecture_claim": False,
            "checks": {"replay_directory_is_distinct": True},
            "source_summary": {"source_manifest_report_passed": True},
            "replay_summary": {
                "source_manifest_parent": str(tmp_path / "fresh_source"),
                "replay_manifest_parent": str(tmp_path / "fresh_replay"),
                "source_report_roles": [
                    "phase2homeostasis_publication_bundle",
                    *(spec["role"] for spec in POSITIVE_REPORT_SPECS),
                    "hmac_missing_key_negative_control",
                ],
                "state_artifact_roles": list(REQUIRED_STATE_ARTIFACT_ROLES),
                "reproduction_step_ids": [
                    "run_hmac_generation1_py313",
                    "run_hmac_generation2_py313",
                    "run_hmac_generation2_py312",
                    "audit_hmac_chain_py313",
                    "audit_hmac_chain_py313_to_py312",
                    "audit_hmac_cross_runtime_dynamics",
                    "audit_hmac_phase2cj_invariance",
                    "audit_hmac_no_key_negative_control",
                    "audit_hmac_publication_bundle",
                ],
            },
            "evidence": {"replayed_reproducibility_report": str(tmp_path / "fresh" / "repro_report.json")},
        },
    )


def _audit(tmp_path: Path, *, reuse_source_path: bool = False) -> dict:
    source = _source_manifest_report(tmp_path)
    source_manifest = json.loads(
        Path(json.loads(source.read_text(encoding="utf-8"))["evidence"]["reproducibility_manifest"]).read_text(
            encoding="utf-8"
        )
    )
    reused = Path(source_manifest["source_reports"][1]["path"]) if reuse_source_path else None
    return audit_phase2homeostasis_fresh_rerun_from_manifest(
        source_reproducibility_report_json=source,
        fresh_bundle_report_json=_fresh_bundle(tmp_path, reuse_source_path=reused),
        fresh_reproducibility_report_json=_fresh_repro_report(tmp_path),
        fresh_manifest_replay_report_json=_fresh_replay_report(tmp_path),
        output_report_json=tmp_path / "fresh_audit.json",
    )


def test_fresh_rerun_accepts_distinct_passed_evidence(tmp_path: Path) -> None:
    report = _audit(tmp_path)
    validation = validate_phase2homeostasis_fresh_rerun_from_manifest(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["ready_for_epoch_making_architecture_claim"] is False


def test_fresh_rerun_rejects_reused_source_report_path(tmp_path: Path) -> None:
    report = _audit(tmp_path, reuse_source_path=True)
    validation = validate_phase2homeostasis_fresh_rerun_from_manifest(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert report["checks"]["fresh_source_report_paths_distinct"] is False


def test_fresh_rerun_validation_rejects_epoch_claim(tmp_path: Path) -> None:
    report = _audit(tmp_path)
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2homeostasis_fresh_rerun_from_manifest(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
