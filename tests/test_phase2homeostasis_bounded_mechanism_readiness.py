import json
from pathlib import Path

from reflexlm.cli.audit_phase2homeostasis_bounded_dossier_manifest import (
    audit_phase2homeostasis_bounded_dossier_manifest,
)
from reflexlm.cli.audit_phase2homeostasis_bounded_dossier_manifest_replay import (
    audit_phase2homeostasis_bounded_dossier_manifest_replay,
)
from reflexlm.cli.audit_phase2homeostasis_bounded_mechanism_readiness import (
    audit_phase2homeostasis_bounded_mechanism_readiness,
    validate_phase2homeostasis_bounded_mechanism_readiness,
)
from test_phase2homeostasis_bounded_dossier_manifest_replay import _dossier_report


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _limitations_report(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "limitations.json",
        {
            "artifact_family": "phase2homeostasis_fresh_rerun_limitations",
            "passed": True,
            "ready_for_bounded_homeostasis_fresh_behavioral_claim": True,
            "ready_for_exact_cross_runtime_homeostatic_dynamics_claim": False,
            "ready_for_unbounded_long_term_memory_claim": False,
            "ready_for_general_runtime_interpreter_invariance_claim": False,
            "ready_for_open_ended_native_perception_claim": False,
            "ready_for_production_autonomy_claim": False,
            "ready_for_epoch_making_architecture_claim": False,
            "checks": {
                "all_fresh_runtime_reports_passed": True,
                "fresh_runtime_metrics_match": True,
                "fresh_hmac_chains_passed": True,
                "fresh_hmac_chains_preserve_side_effect_trace": True,
                "fresh_phase2cj_passed": True,
                "fresh_cross_runtime_exact_dynamics_not_claimed": True,
                "fresh_cross_runtime_side_effect_traces_match": True,
                "fresh_threshold_drift_bounded_for_limitation": True,
            },
            "metrics": {
                "threshold_drift_limit": 0.01,
                "maximum_active_threshold_delta": 0.008,
                "side_effect_trace_rows": 18,
            },
            "limitations": {
                "exact_cross_runtime_homeostatic_dynamics": (
                    "not_supported_by_this_fresh_rerun"
                )
            },
        },
    )


def _cross_runtime_report(tmp_path: Path, *, passed: bool = False) -> Path:
    return _write(
        tmp_path / "cross-runtime.json",
        {
            "artifact_family": "phase2homeostasis_cross_runtime_dynamics",
            "passed": passed,
            "ready_for_bounded_cross_runtime_homeostatic_dynamics_claim": False,
            "ready_for_general_runtime_interpreter_invariance_claim": False,
            "ready_for_open_ended_native_perception_claim": False,
            "ready_for_production_autonomy_claim": False,
            "ready_for_epoch_making_architecture_claim": False,
            "checks": {
                "canonical_runtime_passed": True,
                "alternate_runtime_passed": True,
                "same_core_completion_metrics": True,
                "discrete_homeostatic_dynamics_match": False,
                "active_threshold_deltas_within_tolerance": False,
                "wake_reason_counts_match": False,
                "runtime_normalized_executable_action_traces_match": False,
            },
            "metrics": {"maximum_active_threshold_delta": 0.008},
        },
    )


def _validated_dossier_stack(tmp_path: Path) -> tuple[Path, Path, Path]:
    dossier = _dossier_report(tmp_path)
    manifest_report = tmp_path / "manifest-report.json"
    audit_phase2homeostasis_bounded_dossier_manifest(
        dossier_report_json=dossier,
        output_manifest_json=tmp_path / "manifest.json",
        output_report_json=manifest_report,
    )
    replay_report = tmp_path / "replay-report.json"
    audit_phase2homeostasis_bounded_dossier_manifest_replay(
        reproducibility_report_json=manifest_report,
        output_dir=tmp_path / "replay",
        output_report_json=replay_report,
    )
    return dossier, manifest_report, replay_report


def test_bounded_mechanism_readiness_accepts_validated_limited_stack(
    tmp_path: Path,
) -> None:
    dossier, manifest, replay = _validated_dossier_stack(tmp_path)
    report = audit_phase2homeostasis_bounded_mechanism_readiness(
        bounded_dossier_report_json=dossier,
        bounded_manifest_report_json=manifest,
        bounded_manifest_replay_report_json=replay,
        fresh_rerun_limitations_report_json=_limitations_report(tmp_path),
        cross_runtime_dynamics_report_json=_cross_runtime_report(tmp_path),
        output_report_json=tmp_path / "readiness.json",
    )
    validation = validate_phase2homeostasis_bounded_mechanism_readiness(report)

    assert report["passed"] is True
    assert validation["passed"] is True
    assert report["ready_for_bounded_nsi_mechanism_evidence_argument"] is True
    assert report["ready_for_submission_without_external_machine_replay"] is False


def test_bounded_mechanism_readiness_rejects_missing_exact_failure(
    tmp_path: Path,
) -> None:
    dossier, manifest, replay = _validated_dossier_stack(tmp_path)
    report = audit_phase2homeostasis_bounded_mechanism_readiness(
        bounded_dossier_report_json=dossier,
        bounded_manifest_report_json=manifest,
        bounded_manifest_replay_report_json=replay,
        fresh_rerun_limitations_report_json=_limitations_report(tmp_path),
        cross_runtime_dynamics_report_json=_cross_runtime_report(tmp_path, passed=True),
        output_report_json=tmp_path / "readiness.json",
    )
    validation = validate_phase2homeostasis_bounded_mechanism_readiness(report)

    assert report["passed"] is False
    assert validation["passed"] is False
    assert (
        report["checks"]["exact_cross_runtime_failure_recorded_not_claimed"] is False
    )
