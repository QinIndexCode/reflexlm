from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_TASK_FAMILIES = {
    "dependency_or_import_mismatch",
    "localized_unit_assertion",
    "stale_snapshot_update",
    "config_or_environment_marker",
    "multi_file_traceback_relation",
}

REQUIRED_GRADED_FACTORS = {
    "candidate_count",
    "evidence_density",
    "repair_depth",
    "failure_observability",
    "ambiguity_class",
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
    },
}

REQUIRED_CONTROLS = {
    "full_package",
    "native_head_only_no_cache",
    "no_nsi_latent",
    "continuation_only",
    "prompt_only",
    "react",
    "modern_coding_agent_loop",
}

REQUIRED_METRICS = {
    "task_success",
    "patch_correctness",
    "test_pass_rate",
    "command_count",
    "edit_count",
    "rollback_success",
    "unauthorized_write_count",
    "low_level_qwen_calls",
    "allowlist_or_state_hallucination",
    "stop_condition_correctness",
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
}

REQUIRED_BASELINE_POLICY_FLAGS = {
    "baselines_measured_not_declared",
    "modern_agent_baseline_spec_required",
    "baseline_artifact_paths_required_before_full_gate",
    "native_ablations_share_runtime_except_removed_mechanism",
    "best_baseline_selected_before_full_gate",
}

REQUIRED_CLAIM_UPGRADE_FLAGS = {
    "phase2s_pass_required_for_open_repair_claim",
    "external_public_holdout_required",
    "multi_seed_required",
    "multi_model_required",
    "independent_reproduction_required_for_epoch_claim",
    "subagent_review_unanimity_required_before_claim_upgrade",
    "subagent_review_not_sufficient_without_metrics",
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
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _bool(payload: dict[str, Any], key: str, *, default: bool = False) -> bool:
    value = payload.get(key, default)
    return value if isinstance(value, bool) else default


def _string_set(payload: dict[str, Any], key: str) -> set[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if str(item)}


def _bool_keys_present(payload: dict[str, Any], keys: set[str]) -> bool:
    return all(_bool(payload, key) for key in keys)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        and int(payload.get("min_per_family_tasks", 0) or 0) > 0
        and _bool(payload, "stratify_by_task_family")
        and _bool(payload, "stratify_by_difficulty")
        and str(payload.get("tie_policy", "")).lower() == "no_claim_upgrade_on_tie"
        and (_float_or_none(payload.get("best_baseline_margin_task_success")) or 0.0) > 0
        and (_float_or_none(payload.get("best_baseline_margin_patch_correctness")) or 0.0) > 0
        and _float_or_none(payload.get("safety_noninferiority_margin")) is not None
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


def _mentions_sealed(value: Any) -> bool:
    text = str(value).replace("\\", "/").lower()
    if "external_trace_v3" in text or "phase2i_external_trace" in text:
        return True
    tokens = [token for token in re.split(r"[/_.\-\s]+", text) if token]
    return "sealed" in tokens


def _sealed_paths(data_policy: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("training_roots", "train_roots", "validation_roots", "design_roots"):
        value = data_policy.get(key)
        if isinstance(value, list):
            paths.extend(str(item) for item in value if _mentions_sealed(item))
    return paths


def build_phase2s_preregistration_check(
    *,
    proposal_json: str | Path,
) -> dict[str, Any]:
    proposal = _read_json(proposal_json)
    data_policy = _section(proposal, "data_policy")
    benchmark = _section(proposal, "benchmark")
    controls = _section(proposal, "controls")
    repair_runtime = _section(proposal, "repair_runtime")
    baseline_policy = _section(proposal, "baseline_policy")
    modern_agent_baseline = _section(proposal, "modern_agent_baseline")
    data_artifact_requirements = _section(proposal, "data_artifact_requirements")
    gates = _section(proposal, "gates")
    statistical_decision = _section(proposal, "statistical_decision")
    safety_thresholds = _section(proposal, "safety_thresholds")
    claim_upgrade_policy = _section(proposal, "claim_upgrade_policy")
    execution_plan = _section(proposal, "execution_plan")
    claim_policy = _section(proposal, "claim_policy")
    hardcoding_policy = _section(proposal, "hardcoding_policy")

    task_families = _string_set(benchmark, "task_families")
    graded_factors = _string_set(benchmark, "graded_factors")
    control_set = _string_set(controls, "required_controls")
    metrics = _string_set(benchmark, "metrics")
    sealed_training_paths = _sealed_paths(data_policy)
    factor_levels_ok, missing_factor_levels = _factor_level_observations(benchmark)
    full_margin = _float_or_none(gates.get("full_beats_best_baseline_margin", 0.0))

    checks = {
        "proposal_exists": Path(proposal_json).exists(),
        "experiment_is_phase2s": str(proposal.get("phase", "")).lower().replace(" ", "")
        == "phase2s",
        "scientific_data_claim_not_inflated": _bool(
            claim_policy, "scientific_data_submission_unchanged"
        )
        and not _bool(claim_policy, "uses_phase2s_to_upgrade_data_descriptor_claim"),
        "no_sealed_training_tuning_or_design_feedback": (
            not sealed_training_paths
            and not _bool(data_policy, "uses_sealed_for_training")
            and not _bool(data_policy, "uses_sealed_for_tuning")
            and not _bool(data_policy, "uses_sealed_failures_for_design")
        ),
        "repo_disjoint_holdout_required": _bool(data_policy, "repo_disjoint_holdout_required"),
        "public_or_synthetic_safe_only": _bool(data_policy, "public_or_synthetic_safe_only"),
        "license_metadata_required": _bool(data_policy, "license_metadata_required"),
        "no_hidden_gold_or_candidate_markers": _bool(
            data_policy, "rejects_hidden_gold_or_candidate_markers"
        ),
        "sandbox_required": _bool(benchmark, "disposable_sandbox_required"),
        "source_repo_mutation_forbidden": _bool(benchmark, "source_repo_mutation_forbidden"),
        "patch_and_command_logs_recorded": _bool(benchmark, "records_patch_diffs")
        and _bool(benchmark, "records_command_logs")
        and _bool(benchmark, "records_test_outputs"),
        "rollback_and_safety_recorded": _bool(benchmark, "records_rollback_events")
        and _bool(benchmark, "records_safety_blocks"),
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
        "modern_agent_baseline_required": "modern_coding_agent_loop" in control_set,
        "gates_preregistered": _bool(gates, "data_gate_required")
        and _bool(gates, "smoke_gate_required")
        and _bool(gates, "full_gate_required")
        and _bool(gates, "safety_gate_required")
        and _bool(gates, "transfer_gate_required"),
        "full_margin_preregistered": full_margin is not None and full_margin > 0.0,
        "statistical_decision_preregistered": _statistical_decision_valid(
            statistical_decision
        ),
        "safety_thresholds_preregistered": _safety_thresholds_valid(safety_thresholds),
        "no_training_before_gates": not _bool(execution_plan, "starts_training")
        and not _bool(execution_plan, "starts_package")
        and not _bool(execution_plan, "starts_sealed_eval"),
        "bounded_claim_until_gates_pass": _bool(claim_policy, "bounded_until_gates_pass")
        and not _bool(claim_policy, "claims_production_autonomy")
        and not _bool(claim_policy, "claims_epoch_making_architecture"),
        "claim_upgrade_requires_reproduction_and_review": _bool_keys_present(
            claim_upgrade_policy, REQUIRED_CLAIM_UPGRADE_FLAGS
        ),
        "hardcoding_forbidden": _bool(hardcoding_policy, "forbid_task_names")
        and _bool(hardcoding_policy, "forbid_expected_patches")
        and _bool(hardcoding_policy, "forbid_repo_specific_paths")
        and _bool(hardcoding_policy, "forbid_benchmark_specific_commands"),
    }

    blocked_actions: list[str] = []
    if not checks["scientific_data_claim_not_inflated"]:
        blocked_actions.append("do_not_use_phase2s_to_inflate_scientific_data_submission")
    if not checks["no_sealed_training_tuning_or_design_feedback"]:
        blocked_actions.append("do_not_use_sealed_data_for_phase2s_design_training_or_tuning")
    if not checks["sandbox_required"] or not checks["source_repo_mutation_forbidden"]:
        blocked_actions.append("do_not_run_phase2s_without_disposable_sandbox")
    if not checks["required_task_families_present"]:
        blocked_actions.append("do_not_start_phase2s_until_task_families_are_covered")
    if not checks["graded_difficulty_present"]:
        blocked_actions.append("do_not_start_phase2s_with_binary_only_difficulty")
    if not checks["graded_factor_levels_preregistered"]:
        blocked_actions.append("do_not_start_phase2s_without_factor_level_matrix")
    if not checks["required_controls_present"]:
        blocked_actions.append("do_not_claim_phase2s_without_all_controls")
    if not checks["required_metrics_present"]:
        blocked_actions.append("do_not_claim_phase2s_without_repair_and_safety_metrics")
    if not checks["repair_runtime_preregistered"]:
        blocked_actions.append("do_not_start_phase2s_without_repair_runtime_contract")
    if not checks["baseline_policy_measured_and_fair"]:
        blocked_actions.append("do_not_claim_phase2s_with_declared_or_unfair_baselines")
    if not checks["modern_agent_baseline_operationalized"]:
        blocked_actions.append("do_not_claim_phase2s_against_unspecified_modern_agent_baseline")
    if not checks["data_artifact_requirements_preregistered"]:
        blocked_actions.append("do_not_train_phase2s_without_artifact_manifest_requirements")
    if not checks["gates_preregistered"] or not checks["full_margin_preregistered"]:
        blocked_actions.append("do_not_train_phase2s_before_gates_are_preregistered")
    if not checks["statistical_decision_preregistered"]:
        blocked_actions.append("do_not_claim_phase2s_without_statistical_decision_rule")
    if not checks["safety_thresholds_preregistered"]:
        blocked_actions.append("do_not_claim_phase2s_without_quantified_safety_thresholds")
    if not checks["no_training_before_gates"]:
        blocked_actions.append("do_not_start_training_packaging_or_sealed_eval_from_preregistration")
    if not checks["bounded_claim_until_gates_pass"]:
        blocked_actions.append("do_not_upgrade_architecture_claim_before_phase2s_passes")
    if not checks["claim_upgrade_requires_reproduction_and_review"]:
        blocked_actions.append("do_not_claim_epoch_making_architecture_without_reproduction_and_review")
    if not checks["hardcoding_forbidden"]:
        blocked_actions.append("do_not_use_task_specific_hardcoding")

    passed = all(checks.values())
    return {
        "audit_family": "phase2s_preregistration_check",
        "passed": passed,
        "next_action": (
            "build_phase2s_nonsealed_data_audit_smoke"
            if passed
            else "revise_phase2s_preregistration_before_data_generation"
        ),
        "allowed_actions": ["build_phase2s_nonsealed_data_audit_smoke"] if passed else [],
        "blocked_actions": sorted(set(blocked_actions)),
        "checks": checks,
        "observations": {
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
            "missing_claim_upgrade_flags": sorted(
                key for key in REQUIRED_CLAIM_UPGRADE_FLAGS if not _bool(claim_upgrade_policy, key)
            ),
            "full_beats_best_baseline_margin": full_margin,
            "sealed_training_paths": sealed_training_paths,
        },
        "inputs": {"proposal_json": str(Path(proposal_json))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Phase2S open-ended repair preregistration before data generation."
    )
    parser.add_argument("--proposal-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = build_phase2s_preregistration_check(proposal_json=args.proposal_json)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
