import json
from pathlib import Path

from reflexlm.cli.build_phase2_architecture_claim_boundary_report import (
    build_phase2_architecture_claim_boundary_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _phase2aa_report(*, retry_delta: float = 0.0, retry_gate: bool = False) -> dict:
    return {
        "passed": False,
        "checks": {
            "data_health_passed": True,
            "runtime_artifacts_resolved": True,
            "full_runtime_reports_present": True,
            "full_beats_fixed_no_policy_control": True,
            "full_beats_no_nsi_control": True,
            "cross_model_requirement_met": True,
            "multiseed_requirement_met": True,
            "all_full_delta_gates_passed": True,
            "all_full_runs_policy_loaded": True,
            "model_count_minimum_met": True,
            "independent_seed_count_minimum_met": True,
            "full_minus_control_delta_met": True,
            "full_minus_no_nsi_delta_met": True,
            "full_minus_retry_control_delta_met_if_required": retry_gate,
        },
        "metrics": {
            "full_minus_control_success_rate_min": 0.60,
            "full_minus_no_nsi_success_rate_min": 0.60,
            "full_minus_retry_control_success_rate_min": retry_delta,
        },
    }


def _phase2ag_report(*, blocked: bool = True) -> dict:
    return {
        "passed": True,
        "metrics": {
            "holdout_model_minus_source_overlap_accuracy": 0.32,
            "full_minus_sidecar_erased": 0.32,
            "full_minus_wrong_sidecar": 0.34,
        },
        "blocked_actions": [
            "do_not_package_phase2ag_from_this_report",
            "do_not_run_sealed_phase2ag_from_this_report",
        ]
        if blocked
        else [],
    }


def _phase2am_report() -> dict:
    return {
        "passed": True,
        "natural_repo_disjoint_sidecar_dependency_reproduced": True,
        "claim_bearing_mechanism_evidence": True,
        "metrics": {
            "observed_min_full_minus_source_overlap": 0.46,
            "observed_min_full_minus_sidecar_erased": 0.47,
            "observed_min_full_minus_wrong_sidecar": 0.58,
        },
    }


def _phase2ap_report(*, pure: bool = False) -> dict:
    return {
        "passed": True,
        "stable_bounded_sidecar_control_supported": True,
        "strict_pure_sidecar_claim_ready": pure,
        "metrics": {},
    }


def _phase2z_report() -> dict:
    return {
        "passed": True,
        "supported_claims": [
            "bounded_runtime_symbolic_text_membership_patch_proposal_holdout24_supported"
        ],
        "metrics": {
            "runtime_control_success_rate": 1.0,
            "symbolic_patch_success_rate": 1.0,
        },
    }


def _phase2ae_report() -> dict:
    return {
        "passed": True,
        "checks": {
            "provenance_audit_passed": True,
            "structural_sidecar_holdout_solves": True,
            "stripped_identity_holdout_does_not_solve": True,
            "full_success_rate_gate": True,
            "full_beats_policyless_budget": True,
            "erased_structural_counterfactual_fails": True,
            "wrong_structural_counterfactual_fails": True,
        },
        "metrics": {
            "full_success_rate": 1.0,
            "full_minus_policyless_slot0_budget2": 1.0,
            "full_minus_erased_structural": 1.0,
            "full_minus_wrong_structural": 1.0,
        },
    }


def _phase2ae_freeze(*, runtime_identity_accuracy: float = 1.0) -> dict:
    return {
        "frozen": True,
        "metrics": {
            "runtime_identity_heuristic_accuracy": runtime_identity_accuracy,
        },
    }


def _phase2at_report(*, passed: bool = False, delta: float = 0.0) -> dict:
    return {
        "passed": passed,
        "supported_claims": ["phase2at_package_runtime_delta_supported"]
        if passed
        else [],
        "metrics": {
            "package_runtime_delta_metrics": {
                "full_minus_control_success_rate": delta,
            }
        },
    }


def _phase2au_report(*, passed: bool) -> dict:
    return {
        "passed": passed,
        "artifact_family": "phase2au_policy_required_runtime_task_gate",
    }


def _phase2au_pretrain_report(*, passed: bool) -> dict:
    return {
        "passed": passed,
        "artifact_family": "phase2au_policy_required_pretrain_gate",
    }


def _phase2au_runtime_delta_report(*, passed: bool, delta: float = 0.45) -> dict:
    return {
        "passed": passed,
        "artifact_family": "phase2au_policy_required_runtime_delta_gate",
        "supported_claims": ["phase2au_bounded_package_runtime_delta_supported"]
        if passed
        else [],
        "metrics": {"full_minus_control_success_rate": delta},
    }


def _phase2au_patch_delta_report(*, passed: bool, delta: float = 0.45) -> dict:
    return {
        "passed": passed,
        "artifact_family": "phase2au_patch_execution_delta_gate",
        "supported_claims": [
            "phase2au_bounded_recorded_patch_candidate_execution_delta_supported"
        ]
        if passed
        else [],
        "metrics": {"full_minus_control_success_rate": delta},
    }


def _phase2au_descriptor_observability_report(
    *, passed: bool, pair_count: int = 2
) -> dict:
    return {
        "passed": passed,
        "artifact_family": "phase2au_descriptor_runtime_observability_audit",
        "supported_claims": ["phase2au_runtime_descriptor_outputs_present_and_diverse"]
        if passed
        else [],
        "metrics": {"operation_template_pair_count": pair_count},
    }


def test_architecture_boundary_allows_bounded_but_blocks_strong_claim(
    tmp_path: Path,
) -> None:
    report = build_phase2_architecture_claim_boundary_report(
        phase2aa_runtime_replication_json=_write(
            tmp_path / "phase2aa.json", _phase2aa_report()
        ),
        phase2ag_evidence_sufficiency_json=_write(
            tmp_path / "phase2ag.json", _phase2ag_report()
        ),
    )

    assert report["passed"] is True
    assert report["bounded_mechanism_claim_ready"] is True
    assert report["strong_architecture_claim_ready"] is False
    assert "identity_first_verification_retry_control_ties_or_is_unbeaten" in report[
        "claim_upgrade_blockers"
    ]
    assert "epoch_making_architecture" in report["unsupported_claims"]


def test_architecture_boundary_requires_sidecar_control_delta(tmp_path: Path) -> None:
    phase2ag = _phase2ag_report()
    phase2ag["metrics"]["full_minus_wrong_sidecar"] = 0.0

    report = build_phase2_architecture_claim_boundary_report(
        phase2aa_runtime_replication_json=_write(
            tmp_path / "phase2aa.json", _phase2aa_report(retry_delta=0.2)
        ),
        phase2ag_evidence_sufficiency_json=_write(
            tmp_path / "phase2ag.json", phase2ag
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["sidecar_erased_and_wrong_controls_degrade"] is False
    assert report["supported_claims"] == []


def test_architecture_boundary_strong_claim_requires_unblocked_package_path(
    tmp_path: Path,
) -> None:
    report = build_phase2_architecture_claim_boundary_report(
        phase2aa_runtime_replication_json=_write(
            tmp_path / "phase2aa.json",
            _phase2aa_report(retry_delta=0.2, retry_gate=True),
        ),
        phase2ag_evidence_sufficiency_json=_write(
            tmp_path / "phase2ag.json", _phase2ag_report(blocked=False)
        ),
    )

    assert report["passed"] is True
    assert report["strong_architecture_claim_ready"] is True
    assert report["claim_upgrade_blockers"] == []


def test_architecture_boundary_records_natural_sidecar_but_blocks_pure_causality(
    tmp_path: Path,
) -> None:
    report = build_phase2_architecture_claim_boundary_report(
        phase2aa_runtime_replication_json=_write(
            tmp_path / "phase2aa.json",
            _phase2aa_report(retry_delta=0.2, retry_gate=True),
        ),
        phase2ag_evidence_sufficiency_json=_write(
            tmp_path / "phase2ag.json", _phase2ag_report()
        ),
        phase2am_reproduction_json=_write(
            tmp_path / "phase2am.json", _phase2am_report()
        ),
        phase2ap_control_synthesis_json=_write(
            tmp_path / "phase2ap.json", _phase2ap_report(pure=False)
        ),
        phase2z_phase2aq_evidence_json=_write(
            tmp_path / "phase2z.json", _phase2z_report()
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["natural_sidecar_dependency_reproduced"] is True
    assert report["checks"]["stable_sidecar_controls_supported"] is True
    assert report["checks"]["strict_pure_sidecar_causality_ready"] is False
    assert "strict_pure_sidecar_causality_not_ready" in report[
        "claim_upgrade_blockers"
    ]
    assert "natural repo-disjoint sidecar dependency replicated across models/seeds" in report[
        "supported_claims"
    ]
    assert "bounded symbolic patch proposal runtime on public-repo holdout24" in report[
        "supported_claims"
    ]


def test_architecture_boundary_accepts_package_sidecar_but_keeps_heuristic_blocker(
    tmp_path: Path,
) -> None:
    report = build_phase2_architecture_claim_boundary_report(
        phase2aa_runtime_replication_json=_write(
            tmp_path / "phase2aa.json",
            _phase2aa_report(retry_delta=0.2, retry_gate=True),
        ),
        phase2ag_evidence_sufficiency_json=_write(
            tmp_path / "phase2ag.json", _phase2ag_report(blocked=True)
        ),
        phase2ae_structural_sidecar_comparison_json=_write(
            tmp_path / "phase2ae.json", _phase2ae_report()
        ),
        phase2ae_freeze_manifest_json=_write(
            tmp_path / "phase2ae_freeze.json", _phase2ae_freeze()
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["package_level_structural_sidecar_runtime_supported"] is True
    assert report["checks"]["package_sidecar_path_blocked"] is False
    assert report["strong_architecture_claim_ready"] is False
    assert "sidecar_dependency_evidence_is_nonpackage_nonsealed_only" not in report[
        "claim_upgrade_blockers"
    ]
    assert "learned_head_does_not_beat_runtime_identity_heuristic" in report[
        "claim_upgrade_blockers"
    ]
    assert (
        "package-level structural sidecar runtime closes residual budget-pressure gap"
        in report["supported_claims"]
    )
    assert (
        "package-level runtime execution for sidecar dependency with nonzero feasible controls"
        not in report["next_required_evidence"]
    )


def test_architecture_boundary_records_phase2at_runtime_delta_failure(
    tmp_path: Path,
) -> None:
    report = build_phase2_architecture_claim_boundary_report(
        phase2aa_runtime_replication_json=_write(
            tmp_path / "phase2aa.json", _phase2aa_report()
        ),
        phase2ag_evidence_sufficiency_json=_write(
            tmp_path / "phase2ag.json", _phase2ag_report()
        ),
        phase2at_evidence_sufficiency_json=_write(
            tmp_path / "phase2at.json", _phase2at_report(passed=False, delta=0.0)
        ),
        phase2au_task_gate_json=_write(
            tmp_path / "phase2au.json", _phase2au_report(passed=False)
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["learned_bounded_patch_package_delta_supported"] is False
    assert report["metrics"]["phase2at_package_runtime_full_minus_control"] == 0.0
    assert "phase2at_package_runtime_delta_not_supported" in report[
        "claim_upgrade_blockers"
    ]
    assert "phase2au_policy_required_runtime_task_gate_not_ready" in report[
        "claim_upgrade_blockers"
    ]
    assert "phase2au_policy_required_non_parser_oracle_task_gate" in report[
        "next_required_evidence"
    ]
    assert (
        "learned_bounded_patch_descriptor_generation_runtime_delta"
        in report["next_required_evidence"]
    )


def test_architecture_boundary_records_phase2au_pretrain_readiness(
    tmp_path: Path,
) -> None:
    report = build_phase2_architecture_claim_boundary_report(
        phase2aa_runtime_replication_json=_write(
            tmp_path / "phase2aa.json", _phase2aa_report()
        ),
        phase2ag_evidence_sufficiency_json=_write(
            tmp_path / "phase2ag.json", _phase2ag_report()
        ),
        phase2at_evidence_sufficiency_json=_write(
            tmp_path / "phase2at.json", _phase2at_report(passed=False, delta=0.0)
        ),
        phase2au_task_gate_json=_write(
            tmp_path / "phase2au.json", _phase2au_report(passed=True)
        ),
        phase2au_pretrain_gate_json=_write(
            tmp_path / "phase2au_pretrain.json", _phase2au_pretrain_report(passed=True)
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["policy_required_runtime_task_gate_ready"] is True
    assert report["checks"]["policy_required_runtime_pretrain_ready"] is True
    assert report["metrics"]["phase2au_policy_required_pretrain_ready"] == 1.0
    assert "phase2au_policy_required_runtime_task_gate_not_ready" not in report[
        "claim_upgrade_blockers"
    ]
    assert "phase2au_policy_required_pretrain_gate_not_ready" not in report[
        "claim_upgrade_blockers"
    ]
    assert "phase2au_policy_required_non_parser_oracle_task_gate" not in report[
        "next_required_evidence"
    ]
    assert (
        "learned_bounded_patch_descriptor_generation_runtime_delta"
        in report["next_required_evidence"]
    )


def test_architecture_boundary_records_phase2au_runtime_and_patch_delta_without_upgrade(
    tmp_path: Path,
) -> None:
    report = build_phase2_architecture_claim_boundary_report(
        phase2aa_runtime_replication_json=_write(
            tmp_path / "phase2aa.json", _phase2aa_report()
        ),
        phase2ag_evidence_sufficiency_json=_write(
            tmp_path / "phase2ag.json", _phase2ag_report()
        ),
        phase2at_evidence_sufficiency_json=_write(
            tmp_path / "phase2at.json", _phase2at_report(passed=False, delta=0.0)
        ),
        phase2au_task_gate_json=_write(
            tmp_path / "phase2au.json", _phase2au_report(passed=True)
        ),
        phase2au_pretrain_gate_json=_write(
            tmp_path / "phase2au_pretrain.json", _phase2au_pretrain_report(passed=True)
        ),
        phase2au_runtime_delta_gate_json=_write(
            tmp_path / "phase2au_runtime.json",
            _phase2au_runtime_delta_report(passed=True),
        ),
        phase2au_patch_execution_delta_gate_json=_write(
            tmp_path / "phase2au_patch.json",
            _phase2au_patch_delta_report(passed=True),
        ),
    )

    assert report["passed"] is True
    assert report["strong_architecture_claim_ready"] is False
    assert report["checks"]["policy_required_runtime_delta_supported"] is True
    assert report["checks"]["policy_required_recorded_patch_execution_supported"] is True
    assert report["metrics"]["phase2au_policy_required_runtime_delta"] == 0.45
    assert report["metrics"]["phase2au_recorded_patch_execution_delta"] == 0.45
    assert (
        "Phase2AU policy-required package runtime delta over fixed no-policy control"
        in report["supported_claims"]
    )
    assert (
        "Phase2AU bounded recorded patch-candidate execution delta"
        in report["supported_claims"]
    )
    assert "phase2au_recorded_patch_execution_delta_not_supported" not in report[
        "claim_upgrade_blockers"
    ]
    assert (
        "learned_bounded_patch_descriptor_generation_runtime_delta"
        in report["next_required_evidence"]
    )
    assert "phase2at_package_runtime_delta_not_supported" in report[
        "claim_upgrade_blockers"
    ]


def test_architecture_boundary_records_phase2au_descriptor_diversity_blocker(
    tmp_path: Path,
) -> None:
    report = build_phase2_architecture_claim_boundary_report(
        phase2aa_runtime_replication_json=_write(
            tmp_path / "phase2aa.json", _phase2aa_report()
        ),
        phase2ag_evidence_sufficiency_json=_write(
            tmp_path / "phase2ag.json", _phase2ag_report()
        ),
        phase2at_evidence_sufficiency_json=_write(
            tmp_path / "phase2at.json", _phase2at_report(passed=False, delta=0.0)
        ),
        phase2au_task_gate_json=_write(
            tmp_path / "phase2au.json", _phase2au_report(passed=True)
        ),
        phase2au_pretrain_gate_json=_write(
            tmp_path / "phase2au_pretrain.json", _phase2au_pretrain_report(passed=True)
        ),
        phase2au_runtime_delta_gate_json=_write(
            tmp_path / "phase2au_runtime.json",
            _phase2au_runtime_delta_report(passed=True),
        ),
        phase2au_patch_execution_delta_gate_json=_write(
            tmp_path / "phase2au_patch.json",
            _phase2au_patch_delta_report(passed=True),
        ),
        phase2au_descriptor_observability_json=_write(
            tmp_path / "phase2au_descriptor.json",
            _phase2au_descriptor_observability_report(passed=False, pair_count=1),
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["policy_required_descriptor_runtime_diverse"] is False
    assert report["metrics"]["phase2au_descriptor_operation_template_pair_count"] == 1
    assert "phase2au_descriptor_runtime_lacks_operation_template_diversity" in report[
        "claim_upgrade_blockers"
    ]
    assert "phase2au_graded_descriptor_runtime_task_family" in report[
        "next_required_evidence"
    ]
