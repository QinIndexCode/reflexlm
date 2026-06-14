from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _metric(report: dict[str, Any], key: str) -> float | None:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _check(report: dict[str, Any], key: str) -> bool:
    checks = report.get("checks")
    return isinstance(checks, dict) and checks.get(key) is True


def build_phase2_architecture_claim_boundary_report(
    *,
    phase2aa_runtime_replication_json: str | Path,
    phase2ag_evidence_sufficiency_json: str | Path,
    phase2am_reproduction_json: str | Path | None = None,
    phase2ap_control_synthesis_json: str | Path | None = None,
    phase2z_phase2aq_evidence_json: str | Path | None = None,
    phase2ae_structural_sidecar_comparison_json: str | Path | None = None,
    phase2ae_freeze_manifest_json: str | Path | None = None,
    phase2at_evidence_sufficiency_json: str | Path | None = None,
    phase2au_task_gate_json: str | Path | None = None,
    phase2au_pretrain_gate_json: str | Path | None = None,
    phase2au_runtime_delta_gate_json: str | Path | None = None,
    phase2au_patch_execution_delta_gate_json: str | Path | None = None,
    phase2au_descriptor_observability_json: str | Path | None = None,
    require_retry_control_delta: bool = True,
) -> dict[str, Any]:
    phase2aa = _read_json(phase2aa_runtime_replication_json)
    phase2ag = _read_json(phase2ag_evidence_sufficiency_json)
    phase2am = _read_json(phase2am_reproduction_json) if phase2am_reproduction_json else {}
    phase2ap = _read_json(phase2ap_control_synthesis_json) if phase2ap_control_synthesis_json else {}
    phase2z = _read_json(phase2z_phase2aq_evidence_json) if phase2z_phase2aq_evidence_json else {}
    phase2ae = (
        _read_json(phase2ae_structural_sidecar_comparison_json)
        if phase2ae_structural_sidecar_comparison_json
        else {}
    )
    phase2ae_freeze = (
        _read_json(phase2ae_freeze_manifest_json) if phase2ae_freeze_manifest_json else {}
    )
    phase2at = (
        _read_json(phase2at_evidence_sufficiency_json)
        if phase2at_evidence_sufficiency_json
        else {}
    )
    phase2au = _read_json(phase2au_task_gate_json) if phase2au_task_gate_json else {}
    phase2au_pretrain = (
        _read_json(phase2au_pretrain_gate_json) if phase2au_pretrain_gate_json else {}
    )
    phase2au_runtime_delta = (
        _read_json(phase2au_runtime_delta_gate_json)
        if phase2au_runtime_delta_gate_json
        else {}
    )
    phase2au_patch_delta = (
        _read_json(phase2au_patch_execution_delta_gate_json)
        if phase2au_patch_execution_delta_gate_json
        else {}
    )
    phase2au_descriptor_observability = (
        _read_json(phase2au_descriptor_observability_json)
        if phase2au_descriptor_observability_json
        else {}
    )

    aa_fixed_delta = _metric(phase2aa, "full_minus_control_success_rate_min")
    aa_no_nsi_delta = _metric(phase2aa, "full_minus_no_nsi_success_rate_min")
    aa_retry_delta = _metric(phase2aa, "full_minus_retry_control_success_rate_min")
    aa_retry_required_met = _check(
        phase2aa, "full_minus_retry_control_delta_met_if_required"
    )
    ag_holdout_delta = _metric(phase2ag, "holdout_model_minus_source_overlap_accuracy")
    ag_erased_delta = _metric(phase2ag, "full_minus_sidecar_erased")
    ag_wrong_delta = _metric(phase2ag, "full_minus_wrong_sidecar")
    am_source_delta = _metric(phase2am, "observed_min_full_minus_source_overlap")
    am_erased_delta = _metric(phase2am, "observed_min_full_minus_sidecar_erased")
    am_wrong_delta = _metric(phase2am, "observed_min_full_minus_wrong_sidecar")
    ap_stable_sidecar = phase2ap.get("stable_bounded_sidecar_control_supported") is True
    ap_pure_sidecar = phase2ap.get("strict_pure_sidecar_claim_ready") is True
    z_runtime_success = _metric(phase2z, "runtime_control_success_rate")
    aq_symbolic_success = _metric(phase2z, "symbolic_patch_success_rate")
    ae_full_success = _metric(phase2ae, "full_success_rate")
    ae_policyless_delta = _metric(phase2ae, "full_minus_policyless_slot0_budget2")
    ae_erased_delta = _metric(phase2ae, "full_minus_erased_structural")
    ae_wrong_delta = _metric(phase2ae, "full_minus_wrong_structural")
    ae_runtime_identity_heuristic = None
    freeze_metrics = phase2ae_freeze.get("metrics")
    if isinstance(freeze_metrics, dict):
        value = freeze_metrics.get("runtime_identity_heuristic_accuracy")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            ae_runtime_identity_heuristic = float(value)
    at_package_delta = None
    at_metrics = phase2at.get("metrics")
    if isinstance(at_metrics, dict):
        package_delta_metrics = at_metrics.get("package_runtime_delta_metrics")
        if isinstance(package_delta_metrics, dict):
            value = package_delta_metrics.get("full_minus_control_success_rate")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                at_package_delta = float(value)
    au_runtime_delta_value = _metric(
        phase2au_runtime_delta, "full_minus_control_success_rate"
    )
    au_patch_delta_value = _metric(
        phase2au_patch_delta, "full_minus_control_success_rate"
    )
    au_descriptor_pair_count = _metric(
        phase2au_descriptor_observability, "operation_template_pair_count"
    )

    bounded_candidate_selection = (
        phase2aa.get("passed") is True
        or (
            _check(phase2aa, "data_health_passed")
            and _check(phase2aa, "runtime_artifacts_resolved")
            and _check(phase2aa, "full_beats_fixed_no_policy_control")
            and _check(phase2aa, "full_beats_no_nsi_control")
            and _check(phase2aa, "cross_model_requirement_met")
            and _check(phase2aa, "multiseed_requirement_met")
        )
        or (
            _check(phase2aa, "data_health_passed")
            and _check(phase2aa, "runtime_artifacts_resolved")
            and _check(phase2aa, "all_full_delta_gates_passed")
            and _check(phase2aa, "all_full_runs_policy_loaded")
            and _check(phase2aa, "model_count_minimum_met")
            and _check(phase2aa, "independent_seed_count_minimum_met")
            and _check(phase2aa, "full_minus_control_delta_met")
            and _check(phase2aa, "full_minus_no_nsi_delta_met")
        )
    )
    sidecar_dependency = phase2ag.get("passed") is True
    natural_sidecar_reproduced = (
        phase2am.get("passed") is True
        and phase2am.get("natural_repo_disjoint_sidecar_dependency_reproduced") is True
        and phase2am.get("claim_bearing_mechanism_evidence") is True
    )
    bounded_symbolic_patch_runtime_supported = (
        phase2z.get("passed") is True
        and "bounded_runtime_symbolic_text_membership_patch_proposal_holdout24_supported"
        in phase2z.get("supported_claims", [])
    )
    learned_bounded_patch_package_delta_supported = (
        phase2at.get("passed") is True
        and "phase2at_package_runtime_delta_supported"
        in phase2at.get("supported_claims", [])
    )
    policy_required_runtime_task_gate_ready = phase2au.get("passed") is True
    policy_required_runtime_pretrain_ready = phase2au_pretrain.get("passed") is True
    policy_required_runtime_delta_supported = (
        phase2au_runtime_delta.get("passed") is True
        and "phase2au_bounded_package_runtime_delta_supported"
        in phase2au_runtime_delta.get("supported_claims", [])
    )
    policy_required_recorded_patch_execution_supported = (
        phase2au_patch_delta.get("passed") is True
        and "phase2au_bounded_recorded_patch_candidate_execution_delta_supported"
        in phase2au_patch_delta.get("supported_claims", [])
    )
    policy_required_descriptor_runtime_diverse = (
        phase2au_descriptor_observability.get("passed") is True
        and "phase2au_runtime_descriptor_outputs_present_and_diverse"
        in phase2au_descriptor_observability.get("supported_claims", [])
    )
    package_level_structural_sidecar_runtime_supported = (
        phase2ae.get("passed") is True
        and _check(phase2ae, "provenance_audit_passed")
        and _check(phase2ae, "structural_sidecar_holdout_solves")
        and _check(phase2ae, "stripped_identity_holdout_does_not_solve")
        and _check(phase2ae, "full_success_rate_gate")
        and _check(phase2ae, "full_beats_policyless_budget")
        and _check(phase2ae, "erased_structural_counterfactual_fails")
        and _check(phase2ae, "wrong_structural_counterfactual_fails")
    )
    learned_head_beats_runtime_identity_heuristic_if_measured = (
        ae_runtime_identity_heuristic is None
        or (
            ae_full_success is not None
            and ae_full_success > ae_runtime_identity_heuristic
        )
    )
    strict_retry_gap_closed = (
        not require_retry_control_delta
        or aa_retry_required_met
        or (
            isinstance(aa_retry_delta, float)
            and aa_retry_delta > 0.0
        )
    )
    package_or_sealed_blocked = (
        "do_not_package_phase2ag_from_this_report" in phase2ag.get("blocked_actions", [])
        or "do_not_run_sealed_phase2ag_from_this_report"
        in phase2ag.get("blocked_actions", [])
    )
    package_sidecar_path_blocked = (
        package_or_sealed_blocked and not package_level_structural_sidecar_runtime_supported
    )

    checks = {
        "bounded_candidate_selection_replicated": bounded_candidate_selection,
        "candidate_identity_or_nsi_delta_positive": isinstance(aa_fixed_delta, float)
        and aa_fixed_delta > 0.0
        and isinstance(aa_no_nsi_delta, float)
        and aa_no_nsi_delta > 0.0,
        "strict_retry_control_gap_closed": strict_retry_gap_closed,
        "runtime_visible_sidecar_dependency_supported": sidecar_dependency,
        "natural_sidecar_dependency_reproduced": natural_sidecar_reproduced,
        "stable_sidecar_controls_supported": ap_stable_sidecar,
        "strict_pure_sidecar_causality_ready": ap_pure_sidecar,
        "bounded_symbolic_patch_runtime_supported": bounded_symbolic_patch_runtime_supported,
        "learned_bounded_patch_package_delta_supported": learned_bounded_patch_package_delta_supported,
        "policy_required_runtime_task_gate_ready": policy_required_runtime_task_gate_ready,
        "policy_required_runtime_pretrain_ready": policy_required_runtime_pretrain_ready,
        "policy_required_runtime_delta_supported": policy_required_runtime_delta_supported,
        "policy_required_recorded_patch_execution_supported": policy_required_recorded_patch_execution_supported,
        "policy_required_descriptor_runtime_diverse": policy_required_descriptor_runtime_diverse,
        "package_level_structural_sidecar_runtime_supported": package_level_structural_sidecar_runtime_supported,
        "learned_head_beats_runtime_identity_heuristic_if_measured": learned_head_beats_runtime_identity_heuristic_if_measured,
        "sidecar_erased_and_wrong_controls_degrade": isinstance(ag_erased_delta, float)
        and ag_erased_delta > 0.0
        and isinstance(ag_wrong_delta, float)
        and ag_wrong_delta > 0.0,
        "phase2ag_package_and_sealed_blocked": package_or_sealed_blocked,
        "package_sidecar_path_blocked": package_sidecar_path_blocked,
    }

    strong_claim_ready = (
        checks["bounded_candidate_selection_replicated"]
        and checks["candidate_identity_or_nsi_delta_positive"]
        and checks["strict_retry_control_gap_closed"]
        and checks["runtime_visible_sidecar_dependency_supported"]
        and (
            not phase2am_reproduction_json
            or checks["natural_sidecar_dependency_reproduced"]
        )
        and (
            not phase2ap_control_synthesis_json
            or checks["stable_sidecar_controls_supported"]
        )
        and (
            not phase2z_phase2aq_evidence_json
            or checks["bounded_symbolic_patch_runtime_supported"]
        )
        and (
            not phase2ae_structural_sidecar_comparison_json
            or checks["package_level_structural_sidecar_runtime_supported"]
        )
        and (
            not phase2at_evidence_sufficiency_json
            or checks["learned_bounded_patch_package_delta_supported"]
        )
        and (
            not phase2ap_control_synthesis_json
            or checks["strict_pure_sidecar_causality_ready"]
        )
        and checks["sidecar_erased_and_wrong_controls_degrade"]
        and not checks["package_sidecar_path_blocked"]
        and checks["learned_head_beats_runtime_identity_heuristic_if_measured"]
    )
    bounded_claim_ready = (
        checks["bounded_candidate_selection_replicated"]
        and checks["candidate_identity_or_nsi_delta_positive"]
        and checks["runtime_visible_sidecar_dependency_supported"]
        and (
            not phase2am_reproduction_json
            or checks["natural_sidecar_dependency_reproduced"]
        )
        and (
            not phase2ap_control_synthesis_json
            or checks["stable_sidecar_controls_supported"]
        )
        and (
            not phase2z_phase2aq_evidence_json
            or checks["bounded_symbolic_patch_runtime_supported"]
        )
        and (
            not phase2ae_structural_sidecar_comparison_json
            or checks["package_level_structural_sidecar_runtime_supported"]
        )
        and checks["sidecar_erased_and_wrong_controls_degrade"]
    )

    blockers: list[str] = []
    if not checks["strict_retry_control_gap_closed"]:
        blockers.append("identity_first_verification_retry_control_ties_or_is_unbeaten")
    if checks["package_sidecar_path_blocked"]:
        blockers.append("sidecar_dependency_evidence_is_nonpackage_nonsealed_only")
    if (
        phase2ae_structural_sidecar_comparison_json
        and checks["package_level_structural_sidecar_runtime_supported"]
        and not checks["learned_head_beats_runtime_identity_heuristic_if_measured"]
    ):
        blockers.append("learned_head_does_not_beat_runtime_identity_heuristic")
    if not checks["runtime_visible_sidecar_dependency_supported"]:
        blockers.append("runtime_visible_sidecar_dependency_not_supported")
    if phase2am_reproduction_json and not checks["natural_sidecar_dependency_reproduced"]:
        blockers.append("natural_sidecar_reproduction_not_supported")
    if phase2ap_control_synthesis_json and not checks["stable_sidecar_controls_supported"]:
        blockers.append("stable_sidecar_control_synthesis_not_supported")
    if phase2ap_control_synthesis_json and not checks["strict_pure_sidecar_causality_ready"]:
        blockers.append("strict_pure_sidecar_causality_not_ready")
    if (
        phase2z_phase2aq_evidence_json
        and not checks["bounded_symbolic_patch_runtime_supported"]
    ):
        blockers.append("bounded_symbolic_patch_runtime_not_supported")
    if (
        phase2ae_structural_sidecar_comparison_json
        and not checks["package_level_structural_sidecar_runtime_supported"]
    ):
        blockers.append("package_level_structural_sidecar_runtime_not_supported")
    if (
        phase2at_evidence_sufficiency_json
        and not checks["learned_bounded_patch_package_delta_supported"]
    ):
        blockers.append("phase2at_package_runtime_delta_not_supported")
    if phase2au_task_gate_json and not checks["policy_required_runtime_task_gate_ready"]:
        blockers.append("phase2au_policy_required_runtime_task_gate_not_ready")
    if phase2au_pretrain_gate_json and not checks["policy_required_runtime_pretrain_ready"]:
        blockers.append("phase2au_policy_required_pretrain_gate_not_ready")
    if (
        phase2au_runtime_delta_gate_json
        and not checks["policy_required_runtime_delta_supported"]
    ):
        blockers.append("phase2au_policy_required_runtime_delta_not_supported")
    if (
        phase2au_patch_execution_delta_gate_json
        and not checks["policy_required_recorded_patch_execution_supported"]
    ):
        blockers.append("phase2au_recorded_patch_execution_delta_not_supported")
    if (
        phase2au_descriptor_observability_json
        and not checks["policy_required_descriptor_runtime_diverse"]
    ):
        blockers.append("phase2au_descriptor_runtime_lacks_operation_template_diversity")
    if not checks["bounded_candidate_selection_replicated"]:
        blockers.append("candidate_selection_replication_incomplete")

    supported_claims: list[str] = []
    if bounded_claim_ready:
        supported_claims.extend(
            [
                "bounded candidate-selection runtime delta replicated across models/seeds",
                "runtime-visible sidecar dependency on non-sealed repo-disjoint holdout",
            ]
        )
        if phase2am_reproduction_json:
            supported_claims.append(
                "natural repo-disjoint sidecar dependency replicated across models/seeds"
            )
        if phase2ap_control_synthesis_json:
            supported_claims.append(
                "stable bounded sidecar control under neutralization/permutation"
            )
        if phase2z_phase2aq_evidence_json:
            supported_claims.append(
                "bounded symbolic patch proposal runtime on public-repo holdout24"
            )
        if phase2ae_structural_sidecar_comparison_json:
            supported_claims.append(
                "package-level structural sidecar runtime closes residual budget-pressure gap"
            )
        if phase2at_evidence_sufficiency_json and checks[
            "learned_bounded_patch_package_delta_supported"
        ]:
            supported_claims.append(
                "learned bounded patch package runtime delta over no-policy symbolic control"
            )
        if checks["policy_required_runtime_delta_supported"]:
            supported_claims.append(
                "Phase2AU policy-required package runtime delta over fixed no-policy control"
            )
        if checks["policy_required_recorded_patch_execution_supported"]:
            supported_claims.append(
                "Phase2AU bounded recorded patch-candidate execution delta"
            )

    return {
        "artifact_family": "phase2_architecture_claim_boundary_report",
        "passed": bounded_claim_ready,
        "strong_architecture_claim_ready": strong_claim_ready,
        "bounded_mechanism_claim_ready": bounded_claim_ready,
        "checks": checks,
        "metrics": {
            "phase2aa_full_minus_fixed_no_policy": aa_fixed_delta,
            "phase2aa_full_minus_no_nsi": aa_no_nsi_delta,
            "phase2aa_full_minus_retry_control": aa_retry_delta,
            "phase2ag_holdout_model_minus_source_overlap": ag_holdout_delta,
            "phase2ag_full_minus_sidecar_erased": ag_erased_delta,
            "phase2ag_full_minus_wrong_sidecar": ag_wrong_delta,
            "phase2am_min_full_minus_source_overlap": am_source_delta,
            "phase2am_min_full_minus_sidecar_erased": am_erased_delta,
            "phase2am_min_full_minus_wrong_sidecar": am_wrong_delta,
            "phase2z_runtime_control_success_rate": z_runtime_success,
            "phase2aq_symbolic_patch_success_rate": aq_symbolic_success,
            "phase2ae_full_success_rate": ae_full_success,
            "phase2ae_full_minus_policyless_slot0_budget2": ae_policyless_delta,
            "phase2ae_full_minus_erased_structural": ae_erased_delta,
            "phase2ae_full_minus_wrong_structural": ae_wrong_delta,
            "phase2ae_runtime_identity_heuristic_accuracy": ae_runtime_identity_heuristic,
            "phase2at_package_runtime_full_minus_control": at_package_delta,
            "phase2au_policy_required_runtime_task_gate_ready": 1.0
            if policy_required_runtime_task_gate_ready
            else 0.0
            if phase2au_task_gate_json
            else None,
            "phase2au_policy_required_pretrain_ready": 1.0
            if policy_required_runtime_pretrain_ready
            else 0.0
            if phase2au_pretrain_gate_json
            else None,
            "phase2au_policy_required_runtime_delta": au_runtime_delta_value,
            "phase2au_recorded_patch_execution_delta": au_patch_delta_value,
            "phase2au_descriptor_operation_template_pair_count": au_descriptor_pair_count,
        },
        "supported_claims": supported_claims,
        "unsupported_claims": [
            "epoch_making_architecture",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "sealed_cross_model_transfer",
            "freeform_model_generated_patch_repair",
            "learned_head_superiority_over_runtime_identity_heuristic",
        ],
        "claim_upgrade_blockers": blockers,
        "next_required_evidence": [
            "nonsealed_task_family_where_identity_first_retry_control_does_not_tie_full",
            *(
                []
                if checks["package_level_structural_sidecar_runtime_supported"]
                else [
                    "package-level runtime execution for sidecar dependency with nonzero feasible controls"
                ]
            ),
            *(
                []
                if checks["learned_bounded_patch_package_delta_supported"]
                else [
                    *(
                        []
                        if checks["policy_required_runtime_task_gate_ready"]
                        else [
                            "phase2au_policy_required_non_parser_oracle_task_gate"
                        ]
                    ),
                    *(
                        []
                        if (
                            not phase2au_pretrain_gate_json
                            or checks["policy_required_runtime_pretrain_ready"]
                        )
                        else ["phase2au_policy_required_pretrain_gate"]
                    ),
                    *(
                        []
                        if (
                            not phase2au_runtime_delta_gate_json
                            or checks["policy_required_runtime_delta_supported"]
                        )
                        else ["phase2au_policy_required_runtime_delta_gate"]
                    ),
                    *(
                        []
                        if (
                            not phase2au_patch_execution_delta_gate_json
                            or checks[
                                "policy_required_recorded_patch_execution_supported"
                            ]
                        )
                        else ["phase2au_recorded_patch_execution_delta_gate"]
                    ),
                    *(
                        []
                        if (
                            not phase2au_descriptor_observability_json
                            or checks["policy_required_descriptor_runtime_diverse"]
                        )
                        else ["phase2au_graded_descriptor_runtime_task_family"]
                    ),
                    "learned_bounded_patch_descriptor_generation_runtime_delta",
                ]
            ),
            "multi-seed and cross-model reproduction for the sidecar-dependent task family",
        ],
        "inputs": {
            "phase2aa_runtime_replication_json": str(Path(phase2aa_runtime_replication_json)),
            "phase2ag_evidence_sufficiency_json": str(Path(phase2ag_evidence_sufficiency_json)),
            "phase2am_reproduction_json": str(Path(phase2am_reproduction_json))
            if phase2am_reproduction_json
            else None,
            "phase2ap_control_synthesis_json": str(Path(phase2ap_control_synthesis_json))
            if phase2ap_control_synthesis_json
            else None,
            "phase2z_phase2aq_evidence_json": str(Path(phase2z_phase2aq_evidence_json))
            if phase2z_phase2aq_evidence_json
            else None,
            "phase2ae_structural_sidecar_comparison_json": str(
                Path(phase2ae_structural_sidecar_comparison_json)
            )
            if phase2ae_structural_sidecar_comparison_json
            else None,
            "phase2ae_freeze_manifest_json": str(Path(phase2ae_freeze_manifest_json))
            if phase2ae_freeze_manifest_json
            else None,
            "phase2at_evidence_sufficiency_json": str(Path(phase2at_evidence_sufficiency_json))
            if phase2at_evidence_sufficiency_json
            else None,
            "phase2au_task_gate_json": str(Path(phase2au_task_gate_json))
            if phase2au_task_gate_json
            else None,
            "phase2au_pretrain_gate_json": str(Path(phase2au_pretrain_gate_json))
            if phase2au_pretrain_gate_json
            else None,
            "phase2au_runtime_delta_gate_json": str(Path(phase2au_runtime_delta_gate_json))
            if phase2au_runtime_delta_gate_json
            else None,
            "phase2au_patch_execution_delta_gate_json": str(
                Path(phase2au_patch_execution_delta_gate_json)
            )
            if phase2au_patch_execution_delta_gate_json
            else None,
            "phase2au_descriptor_observability_json": str(
                Path(phase2au_descriptor_observability_json)
            )
            if phase2au_descriptor_observability_json
            else None,
            "require_retry_control_delta": require_retry_control_delta,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a conservative Phase2 architecture claim-boundary report."
    )
    parser.add_argument("--phase2aa-runtime-replication-json", required=True)
    parser.add_argument("--phase2ag-evidence-sufficiency-json", required=True)
    parser.add_argument("--phase2am-reproduction-json")
    parser.add_argument("--phase2ap-control-synthesis-json")
    parser.add_argument("--phase2z-phase2aq-evidence-json")
    parser.add_argument("--phase2ae-structural-sidecar-comparison-json")
    parser.add_argument("--phase2ae-freeze-manifest-json")
    parser.add_argument("--phase2at-evidence-sufficiency-json")
    parser.add_argument("--phase2au-task-gate-json")
    parser.add_argument("--phase2au-pretrain-gate-json")
    parser.add_argument("--phase2au-runtime-delta-gate-json")
    parser.add_argument("--phase2au-patch-execution-delta-gate-json")
    parser.add_argument("--phase2au-descriptor-observability-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-require-retry-control-delta", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2_architecture_claim_boundary_report(
        phase2aa_runtime_replication_json=args.phase2aa_runtime_replication_json,
        phase2ag_evidence_sufficiency_json=args.phase2ag_evidence_sufficiency_json,
        phase2am_reproduction_json=args.phase2am_reproduction_json,
        phase2ap_control_synthesis_json=args.phase2ap_control_synthesis_json,
        phase2z_phase2aq_evidence_json=args.phase2z_phase2aq_evidence_json,
        phase2ae_structural_sidecar_comparison_json=args.phase2ae_structural_sidecar_comparison_json,
        phase2ae_freeze_manifest_json=args.phase2ae_freeze_manifest_json,
        phase2at_evidence_sufficiency_json=args.phase2at_evidence_sufficiency_json,
        phase2au_task_gate_json=args.phase2au_task_gate_json,
        phase2au_pretrain_gate_json=args.phase2au_pretrain_gate_json,
        phase2au_runtime_delta_gate_json=args.phase2au_runtime_delta_gate_json,
        phase2au_patch_execution_delta_gate_json=args.phase2au_patch_execution_delta_gate_json,
        phase2au_descriptor_observability_json=args.phase2au_descriptor_observability_json,
        require_retry_control_delta=not args.no_require_retry_control_delta,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
