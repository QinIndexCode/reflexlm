from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2homeostasis_bounded_dossier_manifest import (
    validate_phase2homeostasis_bounded_dossier_manifest,
)
from reflexlm.cli.audit_phase2homeostasis_bounded_dossier_manifest_replay import (
    validate_phase2homeostasis_bounded_dossier_manifest_replay,
)
from reflexlm.cli.audit_phase2homeostasis_bounded_publication_dossier import (
    validate_phase2homeostasis_bounded_publication_dossier,
)
from reflexlm.cli.audit_phase2homeostasis_fresh_rerun_limitations import (
    validate_phase2homeostasis_fresh_rerun_limitations,
)
from reflexlm.cli.audit_phase2homeostasis_publication_bundle import (
    _read_json,
    _write_json,
)


OVERCLAIM_READY_FLAGS: tuple[str, ...] = (
    "ready_for_exact_cross_runtime_homeostatic_dynamics_claim",
    "ready_for_unbounded_long_term_memory_claim",
    "ready_for_general_runtime_interpreter_invariance_claim",
    "ready_for_open_ended_native_perception_claim",
    "ready_for_production_autonomy_claim",
    "ready_for_epoch_making_architecture_claim",
)


def _artifact_ok(report: dict[str, Any], validation: dict[str, Any]) -> bool:
    return report.get("passed") is True and validation.get("passed") is True


def _overclaim_closed(report: dict[str, Any]) -> bool:
    return all(report.get(flag) is not True for flag in OVERCLAIM_READY_FLAGS)


def validate_phase2homeostasis_bounded_mechanism_readiness(
    report: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "artifact_family_matches": (
            report.get("artifact_family")
            == "phase2homeostasis_bounded_mechanism_readiness"
        ),
        "top_level_passed": report.get("passed") is True,
        "top_level_ready_claim_is_bounded": (
            report.get("ready_for_bounded_nsi_mechanism_evidence_argument") is True
            and report.get("ready_for_exact_cross_runtime_homeostatic_dynamics_claim")
            is False
            and all(report.get(flag) is not True for flag in OVERCLAIM_READY_FLAGS)
        ),
        "all_recorded_checks_true": bool(report.get("checks"))
        and all(value is True for value in report.get("checks", {}).values()),
        "dossier_manifest_and_replay_validated": (
            report.get("source_summary", {}).get("bounded_dossier_validated") is True
            and report.get("source_summary", {}).get("bounded_manifest_validated")
            is True
            and report.get("source_summary", {}).get("bounded_manifest_replay_validated")
            is True
        ),
        "limitation_and_negative_evidence_validated": (
            report.get("source_summary", {}).get("fresh_limitations_validated") is True
            and report.get("source_summary", {}).get("exact_cross_runtime_failure_recorded")
            is True
        ),
        "claim_boundary_is_explicit": (
            "does not support exact cross-runtime internal homeostatic microdynamics"
            in report.get("claim_boundary", "")
            and "does not support unbounded" in report.get("claim_boundary", "")
            and "epoch-making architecture" in report.get("claim_boundary", "")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": report.get("metrics", {}),
    }


def audit_phase2homeostasis_bounded_mechanism_readiness(
    *,
    bounded_dossier_report_json: str | Path,
    bounded_manifest_report_json: str | Path,
    bounded_manifest_replay_report_json: str | Path,
    fresh_rerun_limitations_report_json: str | Path,
    cross_runtime_dynamics_report_json: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    dossier = _read_json(bounded_dossier_report_json)
    manifest = _read_json(bounded_manifest_report_json)
    replay = _read_json(bounded_manifest_replay_report_json)
    limitations = _read_json(fresh_rerun_limitations_report_json)
    cross_runtime = _read_json(cross_runtime_dynamics_report_json)
    dossier_validation = validate_phase2homeostasis_bounded_publication_dossier(
        dossier
    )
    manifest_validation = validate_phase2homeostasis_bounded_dossier_manifest(
        manifest
    )
    replay_validation = validate_phase2homeostasis_bounded_dossier_manifest_replay(
        replay
    )
    limitations_validation = validate_phase2homeostasis_fresh_rerun_limitations(
        limitations
    )
    cross_checks = cross_runtime.get("checks", {})
    exact_failure_recorded = (
        cross_runtime.get("artifact_family") == "phase2homeostasis_cross_runtime_dynamics"
        and cross_runtime.get("passed") is False
        and cross_checks.get("canonical_runtime_passed") is True
        and cross_checks.get("alternate_runtime_passed") is True
        and cross_checks.get("same_core_completion_metrics") is True
        and any(
            cross_checks.get(name) is False
            for name in (
                "discrete_homeostatic_dynamics_match",
                "active_threshold_deltas_within_tolerance",
                "wake_reason_counts_match",
                "runtime_normalized_executable_action_traces_match",
            )
        )
    )
    checks = {
        "bounded_dossier_passed_and_validates": _artifact_ok(
            dossier,
            dossier_validation,
        ),
        "bounded_manifest_passed_and_validates": _artifact_ok(
            manifest,
            manifest_validation,
        ),
        "bounded_manifest_replay_passed_and_validates": _artifact_ok(
            replay,
            replay_validation,
        ),
        "fresh_limitations_passed_and_validates": _artifact_ok(
            limitations,
            limitations_validation,
        ),
        "exact_cross_runtime_failure_recorded_not_claimed": (
            exact_failure_recorded
            and cross_runtime.get(
                "ready_for_bounded_cross_runtime_homeostatic_dynamics_claim"
            )
            is False
        ),
        "all_source_overclaims_closed": all(
            _overclaim_closed(report)
            for report in (dossier, manifest, replay, limitations, cross_runtime)
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2homeostasis_bounded_mechanism_readiness",
        "passed": passed,
        "ready_for_bounded_nsi_mechanism_evidence_argument": passed,
        "ready_for_exact_cross_runtime_homeostatic_dynamics_claim": False,
        "ready_for_unbounded_long_term_memory_claim": False,
        "ready_for_general_runtime_interpreter_invariance_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "ready_for_submission_without_external_machine_replay": False,
        "checks": checks,
        "metrics": {
            "bounded_dossier_core_positive_report_count": dossier.get(
                "metrics", {}
            ).get("core_positive_report_count"),
            "bounded_manifest_source_report_count": manifest.get("metrics", {}).get(
                "source_report_count"
            ),
            "bounded_manifest_replay_source_report_count": replay.get(
                "metrics", {}
            ).get("replayed_source_report_count"),
            "maximum_active_threshold_delta": cross_runtime.get("metrics", {}).get(
                "maximum_active_threshold_delta"
            ),
            "fresh_limitation_threshold_drift_limit": limitations.get(
                "metrics", {}
            ).get("threshold_drift_limit"),
            "fresh_limitation_side_effect_trace_rows": limitations.get(
                "metrics", {}
            ).get("side_effect_trace_rows"),
        },
        "source_summary": {
            "bounded_dossier_validated": _artifact_ok(dossier, dossier_validation),
            "bounded_manifest_validated": _artifact_ok(manifest, manifest_validation),
            "bounded_manifest_replay_validated": _artifact_ok(
                replay,
                replay_validation,
            ),
            "fresh_limitations_validated": _artifact_ok(
                limitations,
                limitations_validation,
            ),
            "exact_cross_runtime_failure_recorded": exact_failure_recorded,
        },
        "claim_boundary": (
            "The current evidence is ready for a bounded NSI/homeostatic mechanism "
            "argument: HMAC-authenticated persistent state transfers across fresh "
            "bounded runtime executions, task completion is preserved, missing-key "
            "state verification fails closed, limitation evidence is explicit, and "
            "the manifest replays by hash in a distinct directory. It does not "
            "support exact cross-runtime internal homeostatic microdynamics, does "
            "not support unbounded or semantic long-term memory, and does not "
            "support free-form shell autonomy, open-ended native perception, "
            "production autonomy, or epoch-making architecture."
        ),
        "supported_claims": [
            (
                "bounded NSI/homeostatic mechanism evidence argument for "
                "HMAC-authenticated persistent-state transfer under controlled "
                "runtime tasks, with explicit limitation evidence"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "exact cross-runtime homeostatic microdynamics",
            "unbounded or semantic long-term memory",
            "free-form shell autonomy",
            "general runtime interpreter invariance",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
            "submission without external machine replay",
        ],
        "remaining_publication_risks": [
            "external-machine replay has not yet been run in this evidence gate",
            "exact internal homeostatic microdynamics drift across Python runtimes",
            "paper text must preserve the bounded claim boundary",
        ],
        "next_required_experiment": (
            "external_machine_replay_or_manuscript_claim_boundary_update"
            if passed
            else "repair_bounded_mechanism_readiness_gate"
        ),
        "evidence": {
            "bounded_dossier_report_json": str(bounded_dossier_report_json),
            "bounded_manifest_report_json": str(bounded_manifest_report_json),
            "bounded_manifest_replay_report_json": str(
                bounded_manifest_replay_report_json
            ),
            "fresh_rerun_limitations_report_json": str(
                fresh_rerun_limitations_report_json
            ),
            "cross_runtime_dynamics_report_json": str(cross_runtime_dynamics_report_json),
        },
    }
    _write_json(output_report_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate bounded Phase2Homeostasis mechanism readiness evidence."
    )
    parser.add_argument("--bounded-dossier-report-json", required=True)
    parser.add_argument("--bounded-manifest-report-json", required=True)
    parser.add_argument("--bounded-manifest-replay-report-json", required=True)
    parser.add_argument("--fresh-rerun-limitations-report-json", required=True)
    parser.add_argument("--cross-runtime-dynamics-report-json", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2homeostasis_bounded_mechanism_readiness(
        bounded_dossier_report_json=args.bounded_dossier_report_json,
        bounded_manifest_report_json=args.bounded_manifest_report_json,
        bounded_manifest_replay_report_json=args.bounded_manifest_replay_report_json,
        fresh_rerun_limitations_report_json=args.fresh_rerun_limitations_report_json,
        cross_runtime_dynamics_report_json=args.cross_runtime_dynamics_report_json,
        output_report_json=args.output_report_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
