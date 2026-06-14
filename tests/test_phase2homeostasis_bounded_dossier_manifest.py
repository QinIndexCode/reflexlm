import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_bounded_dossier_manifest import (
    REQUIRED_REPRODUCTION_STEPS,
    REQUIRED_SOURCE_REPORT_ROLES,
    REQUIRED_STATE_ARTIFACT_ROLES,
    audit_phase2homeostasis_bounded_dossier_manifest,
    validate_phase2homeostasis_bounded_dossier_manifest,
)
from reflexlm.cli.audit_phase2homeostasis_bounded_publication_dossier import (
    CORE_REPORT_SPECS,
)


def _write(path: Path, payload: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _dossier_report(tmp_path: Path) -> Path:
    core_rows = []
    for spec in CORE_REPORT_SPECS:
        report_path = _write(
            tmp_path / "source" / f"{spec['role']}.json",
            {"role": spec["role"], "artifact_family": spec["family"]},
        )
        core_rows.append(
            {
                "role": spec["role"],
                "report_json": str(report_path),
                "readable": True,
                "passed": True,
                "artifact_family": spec["family"],
                "expected_family": spec["family"],
                "family_matches_expected": True,
                "bounded_claim_ok": True,
                "compact_metrics": {},
            }
        )
    limit_path = _write(
        tmp_path / "source" / "cross-runtime-limit.json",
        {"artifact_family": "phase2homeostasis_cross_runtime_dynamics"},
    )
    negative_path = _write(
        tmp_path / "source" / "no-key-negative.json",
        {"artifact_family": "phase2homeostasis_persistent_state_chain"},
    )
    markdown_path = _write(tmp_path / "source" / "dossier.md", "# dossier\n")
    state_rows = []
    for index, role in enumerate(REQUIRED_STATE_ARTIFACT_ROLES):
        state_path = _write(tmp_path / "source" / f"{role}.json", {"role": role})
        state_rows.append(
            {
                "state_json": str(state_path),
                "readable": True,
                "schema_valid": True,
                "bounded_state_keys_only": True,
                "authenticator_algorithm": "hmac-sha256",
                "key_fingerprint_sha256": f"fingerprint-{index}",
                "integrity_valid": True,
            }
        )
    checks = {
        "core_positive_reports_passed_and_bounded": True,
        "state_artifacts_hmac_bounded_and_valid": True,
        "missing_key_negative_control_failed_closed": True,
        "exact_cross_runtime_limitation_recorded": True,
        "fresh_limitations_report_links_cross_runtime_failure": True,
        "forbidden_secret_scan_clean": True,
    }
    return _write(
        tmp_path / "source" / "dossier.json",
        {
            "artifact_family": "phase2homeostasis_bounded_publication_dossier",
            "passed": True,
            "ready_for_bounded_homeostasis_publication_dossier_claim": True,
            "ready_for_exact_cross_runtime_homeostatic_dynamics_claim": False,
            "ready_for_unbounded_long_term_memory_claim": False,
            "ready_for_general_runtime_interpreter_invariance_claim": False,
            "ready_for_open_ended_native_perception_claim": False,
            "ready_for_production_autonomy_claim": False,
            "ready_for_epoch_making_architecture_claim": False,
            "checks": checks,
            "metrics": {
                "core_positive_report_count": len(core_rows),
                "state_artifact_count": len(state_rows),
                "negative_control_count": 1,
                "limitation_evidence_count": 1,
            },
            "core_positive_evidence": core_rows,
            "state_artifacts": state_rows,
            "negative_controls": [
                {
                    "role": "hmac_missing_key_negative_control",
                    "report_json": str(negative_path),
                    "readable": True,
                    "passed": False,
                    "expected_failure_observed": True,
                }
            ],
            "limitation_evidence": [
                {
                    "role": "exact_cross_runtime_homeostatic_dynamics_limit",
                    "report_json": str(limit_path),
                    "readable": True,
                    "passed": False,
                    "expected_failure_observed": True,
                    "bounded_to_internal_dynamics": True,
                }
            ],
            "forbidden_secret_scan": {"enabled": True, "matches": []},
            "claim_boundary": (
                "bounded evidence; does not support exact cross-runtime internal "
                "homeostatic microdynamics or epoch-making architecture claims"
            ),
            "evidence": {"output_markdown": str(markdown_path)},
        },
    )


def test_bounded_dossier_manifest_accepts_complete_dossier(
    tmp_path: Path,
) -> None:
    report = audit_phase2homeostasis_bounded_dossier_manifest(
        dossier_report_json=_dossier_report(tmp_path),
        output_manifest_json=tmp_path / "manifest.json",
        output_report_json=tmp_path / "manifest-report.json",
    )
    validation = validate_phase2homeostasis_bounded_dossier_manifest(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["metrics"]["source_report_count"] == len(REQUIRED_SOURCE_REPORT_ROLES)
    assert report["metrics"]["state_artifact_count"] == len(REQUIRED_STATE_ARTIFACT_ROLES)
    assert report["metrics"]["reproduction_step_count"] == len(
        REQUIRED_REPRODUCTION_STEPS
    )


def test_bounded_dossier_manifest_validation_rejects_tampered_source_hash(
    tmp_path: Path,
) -> None:
    report = audit_phase2homeostasis_bounded_dossier_manifest(
        dossier_report_json=_dossier_report(tmp_path),
        output_manifest_json=tmp_path / "manifest.json",
        output_report_json=tmp_path / "manifest-report.json",
    )
    manifest = json.loads(Path(report["evidence"]["reproducibility_manifest"]).read_text())
    first_source = Path(manifest["source_reports"][1]["path"])
    first_source.write_text("tampered", encoding="utf-8")

    validation = validate_phase2homeostasis_bounded_dossier_manifest(report)

    assert validation["passed"] is False
    assert validation["checks"]["source_report_hashes_match"] is False


def test_bounded_dossier_manifest_validation_rejects_epoch_claim(
    tmp_path: Path,
) -> None:
    report = audit_phase2homeostasis_bounded_dossier_manifest(
        dossier_report_json=_dossier_report(tmp_path),
        output_manifest_json=tmp_path / "manifest.json",
        output_report_json=tmp_path / "manifest-report.json",
    )
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2homeostasis_bounded_dossier_manifest(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
