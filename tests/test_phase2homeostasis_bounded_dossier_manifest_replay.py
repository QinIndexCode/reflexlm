import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_bounded_dossier_manifest import (
    REQUIRED_STATE_ARTIFACT_ROLES,
    audit_phase2homeostasis_bounded_dossier_manifest,
)
from reflexlm.cli.audit_phase2homeostasis_bounded_dossier_manifest_replay import (
    audit_phase2homeostasis_bounded_dossier_manifest_replay,
    validate_phase2homeostasis_bounded_dossier_manifest_replay,
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
            {"role": spec["role"]},
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
    limit_path = _write(tmp_path / "source" / "limit.json", {"limit": True})
    negative_path = _write(tmp_path / "source" / "negative.json", {"failed": True})
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
            "checks": {
                "core_positive_reports_passed_and_bounded": True,
                "state_artifacts_hmac_bounded_and_valid": True,
                "missing_key_negative_control_failed_closed": True,
                "exact_cross_runtime_limitation_recorded": True,
                "fresh_limitations_report_links_cross_runtime_failure": True,
                "forbidden_secret_scan_clean": True,
            },
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


def test_bounded_dossier_manifest_replay_accepts_distinct_copy(
    tmp_path: Path,
) -> None:
    manifest_report = tmp_path / "manifest-report.json"
    audit_phase2homeostasis_bounded_dossier_manifest(
        dossier_report_json=_dossier_report(tmp_path),
        output_manifest_json=tmp_path / "manifest.json",
        output_report_json=manifest_report,
    )
    report = audit_phase2homeostasis_bounded_dossier_manifest_replay(
        reproducibility_report_json=manifest_report,
        output_dir=tmp_path / "replay",
        output_report_json=tmp_path / "replay-report.json",
    )
    validation = validate_phase2homeostasis_bounded_dossier_manifest_replay(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["checks"]["replay_directory_is_distinct"] is True


def test_bounded_dossier_manifest_replay_validation_rejects_epoch_claim(
    tmp_path: Path,
) -> None:
    manifest_report = tmp_path / "manifest-report.json"
    audit_phase2homeostasis_bounded_dossier_manifest(
        dossier_report_json=_dossier_report(tmp_path),
        output_manifest_json=tmp_path / "manifest.json",
        output_report_json=manifest_report,
    )
    report = audit_phase2homeostasis_bounded_dossier_manifest_replay(
        reproducibility_report_json=manifest_report,
        output_dir=tmp_path / "replay",
        output_report_json=tmp_path / "replay-report.json",
    )
    report["ready_for_epoch_making_architecture_claim"] = True
    validation = validate_phase2homeostasis_bounded_dossier_manifest_replay(report)

    assert validation["passed"] is False
    assert validation["checks"]["top_level_ready_claim_is_bounded"] is False
