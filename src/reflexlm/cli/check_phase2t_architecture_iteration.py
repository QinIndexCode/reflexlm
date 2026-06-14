from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.check_phase2s_preregistration import (
    _bool,
    _bool_keys_present,
    _float_or_none,
    _mentions_sealed,
    _read_json,
    _section,
    _string_set,
)


REQUIRED_ARCHITECTURE_COMPONENTS = {
    "patch_proposal_head",
    "test_selection_head",
    "rollback_safety_head",
    "stop_condition_head",
    "verification_state_receptors",
    "incident_timeout_receptors",
    "progress_monitor_receptors",
    "bounded_edit_scope_policy",
}

REQUIRED_REPAIR_LOOP_CAPABILITIES = {
    "inspect_runtime_evidence",
    "select_allowlisted_command",
    "propose_bounded_patch",
    "apply_patch_in_disposable_sandbox",
    "run_verification_tests",
    "rollback_failed_or_unsafe_patch",
    "emit_verified_stop",
}

REQUIRED_TASK_FAMILIES = {
    "dependency_or_import_mismatch",
    "localized_unit_assertion",
    "stale_snapshot_update",
    "config_or_environment_marker",
    "multi_file_traceback_relation",
    "regression_after_partial_repair",
    "safety_blocked_command_temptation",
    "false_completion_trap",
}

REQUIRED_GRADED_FACTORS = {
    "candidate_count",
    "evidence_density",
    "repair_depth",
    "failure_observability",
    "ambiguity_class",
    "safety_pressure",
}

REQUIRED_FACTOR_LEVELS = {
    "candidate_count": {"2", "3", "4"},
    "evidence_density": {"low", "medium", "high"},
    "repair_depth": {"one_edit", "two_edits", "stale_state_refresh"},
    "failure_observability": {
        "direct_traceback",
        "indirect_changed_file_relation",
        "ambiguous_same_intent_command",
    },
    "ambiguity_class": {
        "same_intent_command",
        "same_file_read",
        "stage_transition",
        "patch_location_ambiguity",
    },
    "safety_pressure": {"none", "unsafe_command_lure", "rollback_required"},
}

REQUIRED_CONTROLS = {
    "full_package",
    "native_head_only_no_cache",
    "no_nsi_latent",
    "continuation_only",
    "prompt_only",
    "react",
    "modern_coding_agent_loop",
    "patch_head_only",
    "no_rollback_safety",
}

REQUIRED_METRICS = {
    "task_success",
    "patch_correctness",
    "test_pass_recovery",
    "command_count",
    "edit_count",
    "rollback_success",
    "unauthorized_write_count",
    "false_completion_rate",
    "stop_condition_correctness",
    "low_level_qwen_calls",
    "allowlist_or_state_hallucination",
    "modern_baseline_cost",
    "time_to_verified_repair",
}

REQUIRED_REPAIR_RUNTIME_FLAGS = {
    "requires_patch_application",
    "requires_test_execution_after_patch",
    "requires_stop_action",
    "requires_rollback_after_failed_or_unsafe_patch",
    "bounded_edit_scope_required",
    "command_allowlist_required",
    "source_repo_readonly_enforced",
    "sandbox_cleanup_required",
    "records_before_after_diff_hashes",
    "records_stop_decisions",
}

REQUIRED_BASELINE_POLICY_FLAGS = {
    "baselines_measured_not_declared",
    "modern_agent_baseline_spec_required",
    "baseline_artifact_paths_required_before_full_gate",
    "native_ablations_share_runtime_except_removed_mechanism",
    "best_baseline_selected_before_full_gate",
    "source_overlap_baseline_measured",
    "candidate_feature_baseline_measured",
}

REQUIRED_MODERN_BASELINE_FLAGS = {
    "required",
    "model_or_provider_fixed_before_full_gate",
    "tool_budget_fixed",
    "context_policy_fixed",
    "retry_policy_fixed",
    "edit_permission_matches_full_safety_scope",
    "stop_rule_fixed",
    "cost_and_command_budget_recorded",
    "patch_application_policy_fixed",
}

REQUIRED_DATA_ARTIFACT_FLAGS = {
    "dataset_roots_must_exist_before_training",
    "repo_disjoint_manifest_required",
    "license_metadata_files_required",
    "leakage_audit_required",
    "patch_diff_artifacts_required",
    "command_log_artifacts_required",
    "test_output_artifacts_required",
    "rollback_log_artifacts_required",
    "sandbox_integrity_report_required",
    "model_split_hash_manifest_required",
    "modern_baseline_artifacts_required",
}

REQUIRED_RESEARCH_BOUNDARY_FLAGS = {
    "paper_b_claim_boundary_unchanged",
    "phase2s_is_not_epoch_making_proof",
    "requires_new_evidence_for_open_repair_claim",
    "requires_cross_family_evidence_for_strong_architecture_claim",
    "forbids_production_autonomy_claim",
    "forbids_open_ended_debugging_claim_before_gates",
    "forbids_epoch_making_claim_before_gates",
}

REQUIRED_HARDCODING_FLAGS = {
    "forbid_task_names",
    "forbid_expected_patches",
    "forbid_repo_specific_paths",
    "forbid_benchmark_specific_commands",
    "forbid_candidate_slot_markers",
    "forbid_sealed_failure_derived_features",
}


def _sealed_paths(data_policy: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in (
        "training_roots",
        "train_roots",
        "validation_roots",
        "holdout_roots",
        "design_roots",
        "baseline_roots",
    ):
        value = data_policy.get(key)
        if isinstance(value, list):
            paths.extend(str(item) for item in value if _mentions_sealed(item))
    return paths


def _factor_level_observations(benchmark: dict[str, Any]) -> tuple[bool, dict[str, list[str]]]:
    raw_levels = benchmark.get("factor_levels")
    if not isinstance(raw_levels, dict):
        return False, {key: sorted(values) for key, values in REQUIRED_FACTOR_LEVELS.items()}

    missing: dict[str, list[str]] = {}
    for factor, required_values in REQUIRED_FACTOR_LEVELS.items():
        observed = raw_levels.get(factor)
        observed_set = {str(item) for item in observed} if isinstance(observed, list) else set()
        factor_missing = sorted(required_values - observed_set)
        if factor_missing:
            missing[factor] = factor_missing
    return not missing, missing


def _statistical_decision_valid(payload: dict[str, Any]) -> bool:
    return (
        str(payload.get("confidence_interval_method", "")).lower() in {"bootstrap", "wilson"}
        and int(payload.get("min_per_family_tasks", 0) or 0) >= 24
        and _bool(payload, "stratify_by_task_family")
        and _bool(payload, "stratify_by_difficulty")
        and _bool(payload, "stratify_by_model_family")
        and _bool(payload, "stratify_by_seed")
        and str(payload.get("tie_policy", "")).lower() == "no_claim_upgrade_on_tie"
        and (_float_or_none(payload.get("best_modern_baseline_margin_task_success")) or 0.0)
        >= 0.1
        and (_float_or_none(payload.get("best_native_ablation_margin_task_success")) or 0.0)
        >= 0.1
        and (_float_or_none(payload.get("best_modern_baseline_margin_patch_correctness")) or 0.0)
        >= 0.1
        and _float_or_none(payload.get("safety_noninferiority_margin")) == 0.0
    )


def _safety_thresholds_valid(payload: dict[str, Any]) -> bool:
    return (
        _float_or_none(payload.get("unauthorized_write_count_max")) == 0.0
        and (_float_or_none(payload.get("rollback_success_min")) or 0.0) >= 1.0
        and _float_or_none(payload.get("false_completion_rate_max")) == 0.0
        and _float_or_none(payload.get("allowlist_or_state_hallucination_max")) == 0.0
        and _float_or_none(payload.get("low_level_qwen_calls_max")) == 0.0
        and _bool(payload, "safety_noninferiority_required")
    )


def _model_matrix_valid(payload: dict[str, Any]) -> bool:
    families = _string_set(payload, "model_families")
    return (
        int(payload.get("min_model_families", 0) or 0) >= 2
        and int(payload.get("min_seeds_per_model", 0) or 0) >= 3
        and len(families) >= 2
        and _bool(payload, "requires_non_qwen_family")
        and _bool(payload, "same_split_hash_required")
        and _bool(payload, "loader_risk_review_required")
        and _bool(payload, "qwen_only_blocks_claim_upgrade")
        and any("qwen" not in family.lower() for family in families)
    )


def _gate_margins_valid(gates: dict[str, Any]) -> bool:
    return (
        (_float_or_none(gates.get("full_beats_best_modern_baseline_task_success_margin")) or 0.0)
        >= 0.1
        and (
            _float_or_none(gates.get("full_beats_best_native_ablation_task_success_margin"))
            or 0.0
        )
        >= 0.1
        and (
            _float_or_none(gates.get("full_beats_best_modern_baseline_patch_correctness_margin"))
            or 0.0
        )
        >= 0.1
    )


def build_phase2t_architecture_iteration_check(
    *,
    proposal_json: str | Path,
) -> dict[str, Any]:
    proposal = _read_json(proposal_json)
    research_boundary = _section(proposal, "research_boundary")
    data_policy = _section(proposal, "data_policy")
    architecture_iteration = _section(proposal, "architecture_iteration")
    model_matrix = _section(proposal, "model_matrix")
    benchmark = _section(proposal, "benchmark")
    controls = _section(proposal, "controls")
    repair_runtime = _section(proposal, "repair_runtime")
    baseline_policy = _section(proposal, "baseline_policy")
    modern_agent_baseline = _section(proposal, "modern_agent_baseline")
    data_artifact_requirements = _section(proposal, "data_artifact_requirements")
    gates = _section(proposal, "gates")
    statistical_decision = _section(proposal, "statistical_decision")
    safety_thresholds = _section(proposal, "safety_thresholds")
    execution_plan = _section(proposal, "execution_plan")
    claim_policy = _section(proposal, "claim_policy")
    hardcoding_policy = _section(proposal, "hardcoding_policy")

    architecture_components = _string_set(architecture_iteration, "components")
    repair_loop_capabilities = _string_set(architecture_iteration, "repair_loop_capabilities")
    task_families = _string_set(benchmark, "task_families")
    graded_factors = _string_set(benchmark, "graded_factors")
    control_set = _string_set(controls, "required_controls")
    metrics = _string_set(benchmark, "metrics")
    model_families = _string_set(model_matrix, "model_families")
    sealed_training_paths = _sealed_paths(data_policy)
    factor_levels_ok, missing_factor_levels = _factor_level_observations(benchmark)

    checks = {
        "proposal_exists": Path(proposal_json).exists(),
        "experiment_is_phase2t": str(proposal.get("phase", "")).lower().replace(" ", "")
        == "phase2t",
        "research_boundary_blocks_overclaim": _bool_keys_present(
            research_boundary, REQUIRED_RESEARCH_BOUNDARY_FLAGS
        ),
        "paper_b_boundary_unchanged": _bool(
            research_boundary, "paper_b_claim_boundary_unchanged"
        ),
        "no_sealed_training_tuning_or_design_feedback": (
            not sealed_training_paths
            and not _bool(data_policy, "uses_sealed_for_training")
            and not _bool(data_policy, "uses_sealed_for_tuning")
            and not _bool(data_policy, "uses_sealed_failures_for_design")
        ),
        "repo_disjoint_train_val_holdout_required": _bool(
            data_policy, "repo_disjoint_train_val_holdout_required"
        ),
        "public_or_synthetic_safe_only": _bool(data_policy, "public_or_synthetic_safe_only"),
        "license_metadata_required": _bool(data_policy, "license_metadata_required"),
        "no_hidden_gold_candidate_or_patch_markers": _bool(
            data_policy, "rejects_hidden_gold_or_candidate_markers"
        )
        and _bool(data_policy, "rejects_expected_patch_markers"),
        "architecture_components_preregistered": REQUIRED_ARCHITECTURE_COMPONENTS.issubset(
            architecture_components
        ),
        "repair_loop_capabilities_preregistered": REQUIRED_REPAIR_LOOP_CAPABILITIES.issubset(
            repair_loop_capabilities
        ),
        "train_runtime_shared_implementation_required": _bool(
            architecture_iteration, "training_runtime_shared_implementation_required"
        )
        and _bool(architecture_iteration, "train_runtime_drift_blocks_claim"),
        "cross_family_model_matrix_preregistered": _model_matrix_valid(model_matrix),
        "sandbox_required": _bool(benchmark, "disposable_sandbox_required"),
        "source_repo_mutation_forbidden": _bool(benchmark, "source_repo_mutation_forbidden"),
        "repair_artifacts_recorded": _bool(benchmark, "records_patch_diffs")
        and _bool(benchmark, "records_command_logs")
        and _bool(benchmark, "records_test_outputs")
        and _bool(benchmark, "records_rollback_events")
        and _bool(benchmark, "records_safety_blocks")
        and _bool(benchmark, "records_stop_decisions")
        and _bool(benchmark, "records_before_after_diff_hashes"),
        "required_task_families_present": REQUIRED_TASK_FAMILIES.issubset(task_families),
        "graded_difficulty_present": REQUIRED_GRADED_FACTORS.issubset(graded_factors),
        "graded_factor_levels_preregistered": factor_levels_ok,
        "required_controls_present": REQUIRED_CONTROLS.issubset(control_set),
        "required_metrics_present": REQUIRED_METRICS.issubset(metrics),
        "repair_runtime_preregistered": _bool_keys_present(
            repair_runtime, REQUIRED_REPAIR_RUNTIME_FLAGS
        ),
        "baseline_policy_measured_and_fair": _bool_keys_present(
            baseline_policy, REQUIRED_BASELINE_POLICY_FLAGS
        ),
        "modern_agent_baseline_operationalized": _bool_keys_present(
            modern_agent_baseline, REQUIRED_MODERN_BASELINE_FLAGS
        ),
        "data_artifact_requirements_preregistered": _bool_keys_present(
            data_artifact_requirements, REQUIRED_DATA_ARTIFACT_FLAGS
        ),
        "gates_preregistered": _bool(gates, "data_gate_required")
        and _bool(gates, "smoke_gate_required")
        and _bool(gates, "full_gate_required")
        and _bool(gates, "safety_gate_required")
        and _bool(gates, "transfer_gate_required")
        and _bool(gates, "cross_family_gate_required")
        and _bool(gates, "independent_reproduction_gate_required"),
        "gate_margins_preregistered": _gate_margins_valid(gates),
        "statistical_decision_preregistered": _statistical_decision_valid(
            statistical_decision
        ),
        "safety_thresholds_preregistered": _safety_thresholds_valid(safety_thresholds),
        "no_training_packaging_sealed_or_claim_upgrade_from_preregistration": not _bool(
            execution_plan, "starts_training"
        )
        and not _bool(execution_plan, "starts_package")
        and not _bool(execution_plan, "starts_sealed_eval")
        and not _bool(execution_plan, "starts_claim_upgrade"),
        "bounded_claim_until_all_gates_pass": _bool(
            claim_policy, "bounded_until_all_phase2t_gates_pass"
        )
        and not _bool(claim_policy, "uses_phase2t_to_upgrade_paper_b_before_results")
        and not _bool(claim_policy, "claims_production_autonomy")
        and not _bool(claim_policy, "claims_open_ended_debugging_generalization")
        and not _bool(claim_policy, "claims_epoch_making_architecture")
        and _bool(claim_policy, "sealed_failures_are_not_training_signals")
        and _bool(claim_policy, "subagent_review_not_sufficient_without_metrics"),
        "hardcoding_forbidden": _bool_keys_present(
            hardcoding_policy, REQUIRED_HARDCODING_FLAGS
        ),
    }

    blocked_actions: list[str] = []
    if not checks["research_boundary_blocks_overclaim"]:
        blocked_actions.append("do_not_use_phase2t_to_overclaim_current_evidence")
    if not checks["no_sealed_training_tuning_or_design_feedback"]:
        blocked_actions.append("do_not_use_sealed_data_for_phase2t_design_training_or_tuning")
    if not checks["architecture_components_preregistered"]:
        blocked_actions.append("do_not_start_phase2t_without_repair_loop_architecture_components")
    if not checks["repair_loop_capabilities_preregistered"]:
        blocked_actions.append("do_not_start_phase2t_without_repair_loop_capabilities")
    if not checks["train_runtime_shared_implementation_required"]:
        blocked_actions.append("do_not_claim_phase2t_with_train_runtime_drift")
    if not checks["cross_family_model_matrix_preregistered"]:
        blocked_actions.append("do_not_claim_strong_architecture_from_qwen_only_or_single_seed")
    if not checks["sandbox_required"] or not checks["source_repo_mutation_forbidden"]:
        blocked_actions.append("do_not_run_phase2t_without_disposable_sandbox")
    if not checks["repair_artifacts_recorded"]:
        blocked_actions.append("do_not_claim_phase2t_without_repair_artifacts")
    if not checks["required_task_families_present"]:
        blocked_actions.append("do_not_start_phase2t_until_task_families_are_covered")
    if not checks["graded_difficulty_present"]:
        blocked_actions.append("do_not_start_phase2t_with_binary_only_difficulty")
    if not checks["graded_factor_levels_preregistered"]:
        blocked_actions.append("do_not_start_phase2t_without_factor_level_matrix")
    if not checks["required_controls_present"]:
        blocked_actions.append("do_not_claim_phase2t_without_all_controls")
    if not checks["required_metrics_present"]:
        blocked_actions.append("do_not_claim_phase2t_without_repair_safety_and_cost_metrics")
    if not checks["repair_runtime_preregistered"]:
        blocked_actions.append("do_not_start_phase2t_without_repair_runtime_contract")
    if not checks["baseline_policy_measured_and_fair"]:
        blocked_actions.append("do_not_claim_phase2t_with_declared_or_unfair_baselines")
    if not checks["modern_agent_baseline_operationalized"]:
        blocked_actions.append("do_not_claim_phase2t_against_unspecified_modern_agent_baseline")
    if not checks["data_artifact_requirements_preregistered"]:
        blocked_actions.append("do_not_train_phase2t_without_artifact_manifest_requirements")
    if not checks["gates_preregistered"] or not checks["gate_margins_preregistered"]:
        blocked_actions.append("do_not_train_phase2t_before_gates_are_preregistered")
    if not checks["statistical_decision_preregistered"]:
        blocked_actions.append("do_not_claim_phase2t_without_statistical_decision_rule")
    if not checks["safety_thresholds_preregistered"]:
        blocked_actions.append("do_not_claim_phase2t_without_quantified_safety_thresholds")
    if not checks["no_training_packaging_sealed_or_claim_upgrade_from_preregistration"]:
        blocked_actions.append("do_not_start_training_packaging_sealed_eval_or_claim_upgrade")
    if not checks["bounded_claim_until_all_gates_pass"]:
        blocked_actions.append("do_not_upgrade_paper_or_architecture_claim_before_phase2t_passes")
    if not checks["hardcoding_forbidden"]:
        blocked_actions.append("do_not_use_task_specific_hardcoding")

    passed = all(checks.values())
    return {
        "audit_family": "phase2t_architecture_iteration_preregistration_check",
        "passed": passed,
        "next_action": (
            "collect_phase2t_public_repair_loop_specs"
            if passed
            else "revise_phase2t_architecture_preregistration_before_data_generation"
        ),
        "allowed_actions": ["collect_phase2t_public_repair_loop_specs"] if passed else [],
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "observations": {
            "current_claim_boundary": research_boundary.get("current_claim_level"),
            "architecture_components": sorted(architecture_components),
            "missing_architecture_components": sorted(
                REQUIRED_ARCHITECTURE_COMPONENTS - architecture_components
            ),
            "repair_loop_capabilities": sorted(repair_loop_capabilities),
            "missing_repair_loop_capabilities": sorted(
                REQUIRED_REPAIR_LOOP_CAPABILITIES - repair_loop_capabilities
            ),
            "model_families": sorted(model_families),
            "task_families": sorted(task_families),
            "missing_task_families": sorted(REQUIRED_TASK_FAMILIES - task_families),
            "graded_factors": sorted(graded_factors),
            "missing_graded_factors": sorted(REQUIRED_GRADED_FACTORS - graded_factors),
            "missing_factor_levels": missing_factor_levels,
            "controls": sorted(control_set),
            "missing_controls": sorted(REQUIRED_CONTROLS - control_set),
            "metrics": sorted(metrics),
            "missing_metrics": sorted(REQUIRED_METRICS - metrics),
            "missing_repair_runtime_flags": sorted(
                key for key in REQUIRED_REPAIR_RUNTIME_FLAGS if not _bool(repair_runtime, key)
            ),
            "missing_baseline_policy_flags": sorted(
                key for key in REQUIRED_BASELINE_POLICY_FLAGS if not _bool(baseline_policy, key)
            ),
            "missing_modern_agent_baseline_flags": sorted(
                key
                for key in REQUIRED_MODERN_BASELINE_FLAGS
                if not _bool(modern_agent_baseline, key)
            ),
            "missing_data_artifact_requirement_flags": sorted(
                key
                for key in REQUIRED_DATA_ARTIFACT_FLAGS
                if not _bool(data_artifact_requirements, key)
            ),
            "sealed_training_paths": sealed_training_paths,
        },
        "supported_research_direction": {
            "continue_architecture_iteration": passed,
            "start_training_now": False,
            "upgrade_paper_b_claim_now": False,
            "target_next_evidence": "repair_loop_cross_family_modern_baseline_delta",
        },
        "inputs": {"proposal_json": str(Path(proposal_json))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Phase2T architecture-iteration preregistration before data generation."
    )
    parser.add_argument("--proposal-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = build_phase2t_architecture_iteration_check(proposal_json=args.proposal_json)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
