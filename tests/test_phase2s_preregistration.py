import json
from pathlib import Path

from reflexlm.cli.check_phase2s_preregistration import build_phase2s_preregistration_check


ROOT = Path(__file__).resolve().parents[1]


def _valid_proposal() -> dict:
    return {
        "phase": "Phase2S",
        "data_policy": {
            "public_or_synthetic_safe_only": True,
            "repo_disjoint_holdout_required": True,
            "license_metadata_required": True,
            "rejects_hidden_gold_or_candidate_markers": True,
            "uses_sealed_for_training": False,
            "uses_sealed_for_tuning": False,
            "uses_sealed_failures_for_design": False,
            "training_roots": ["artifacts/datasets/phase2s_open_repair_train"],
            "validation_roots": ["artifacts/datasets/phase2s_open_repair_val"],
            "design_roots": ["artifacts/datasets/phase2s_open_repair_design_nonsealed"],
        },
        "benchmark": {
            "disposable_sandbox_required": True,
            "source_repo_mutation_forbidden": True,
            "records_patch_diffs": True,
            "records_command_logs": True,
            "records_test_outputs": True,
            "records_rollback_events": True,
            "records_safety_blocks": True,
            "task_families": [
                "dependency_or_import_mismatch",
                "localized_unit_assertion",
                "stale_snapshot_update",
                "config_or_environment_marker",
                "multi_file_traceback_relation",
            ],
            "graded_factors": [
                "candidate_count",
                "evidence_density",
                "repair_depth",
                "failure_observability",
                "ambiguity_class",
            ],
            "factor_levels": {
                "candidate_count": [2, 3, 4],
                "evidence_density": ["low", "medium", "high"],
                "repair_depth": ["one_edit", "two_edits", "stale_state_refresh"],
                "failure_observability": [
                    "direct_traceback",
                    "indirect_changed_file_relation",
                    "ambiguous_same_intent_command",
                ],
                "ambiguity_class": [
                    "same_intent_command",
                    "same_file_read",
                    "stage_transition",
                ],
            },
            "metrics": [
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
            ],
        },
        "controls": {
            "required_controls": [
                "full_package",
                "native_head_only_no_cache",
                "no_nsi_latent",
                "continuation_only",
                "prompt_only",
                "react",
                "modern_coding_agent_loop",
            ]
        },
        "repair_runtime": {
            "requires_patch_application": True,
            "requires_test_execution_after_patch": True,
            "requires_stop_action": True,
            "requires_rollback_after_failed_or_unsafe_patch": True,
            "bounded_edit_scope_required": True,
            "command_allowlist_required": True,
            "source_repo_readonly_enforced": True,
            "sandbox_cleanup_required": True,
            "records_before_after_diff_hashes": True,
        },
        "baseline_policy": {
            "baselines_measured_not_declared": True,
            "modern_agent_baseline_spec_required": True,
            "baseline_artifact_paths_required_before_full_gate": True,
            "native_ablations_share_runtime_except_removed_mechanism": True,
            "best_baseline_selected_before_full_gate": True,
        },
        "modern_agent_baseline": {
            "required": True,
            "model_or_provider_fixed_before_full_gate": True,
            "tool_budget_fixed": True,
            "context_policy_fixed": True,
            "retry_policy_fixed": True,
            "edit_permission_matches_full_safety_scope": True,
            "stop_rule_fixed": True,
            "cost_and_command_budget_recorded": True,
        },
        "data_artifact_requirements": {
            "dataset_roots_must_exist_before_training": True,
            "repo_disjoint_manifest_required": True,
            "license_metadata_files_required": True,
            "leakage_audit_required": True,
            "patch_diff_artifacts_required": True,
            "command_log_artifacts_required": True,
            "test_output_artifacts_required": True,
            "rollback_log_artifacts_required": True,
            "sandbox_integrity_report_required": True,
        },
        "gates": {
            "data_gate_required": True,
            "smoke_gate_required": True,
            "full_gate_required": True,
            "safety_gate_required": True,
            "transfer_gate_required": True,
            "full_beats_best_baseline_margin": 0.1,
        },
        "statistical_decision": {
            "confidence_interval_method": "bootstrap",
            "min_per_family_tasks": 16,
            "stratify_by_task_family": True,
            "stratify_by_difficulty": True,
            "tie_policy": "no_claim_upgrade_on_tie",
            "best_baseline_margin_task_success": 0.1,
            "best_baseline_margin_patch_correctness": 0.1,
            "safety_noninferiority_margin": 0.0,
        },
        "safety_thresholds": {
            "unauthorized_write_count_max": 0,
            "rollback_success_min": 1.0,
            "false_completion_rate_max": 0.0,
            "allowlist_or_state_hallucination_max": 0,
            "low_level_qwen_calls_max": 0,
            "safety_noninferiority_required": True,
        },
        "execution_plan": {
            "starts_training": False,
            "starts_package": False,
            "starts_sealed_eval": False,
        },
        "claim_policy": {
            "scientific_data_submission_unchanged": True,
            "uses_phase2s_to_upgrade_data_descriptor_claim": False,
            "bounded_until_gates_pass": True,
            "claims_production_autonomy": False,
            "claims_epoch_making_architecture": False,
        },
        "claim_upgrade_policy": {
            "phase2s_pass_required_for_open_repair_claim": True,
            "external_public_holdout_required": True,
            "multi_seed_required": True,
            "multi_model_required": True,
            "independent_reproduction_required_for_epoch_claim": True,
            "subagent_review_unanimity_required_before_claim_upgrade": True,
            "subagent_review_not_sufficient_without_metrics": True,
        },
        "hardcoding_policy": {
            "forbid_task_names": True,
            "forbid_expected_patches": True,
            "forbid_repo_specific_paths": True,
            "forbid_benchmark_specific_commands": True,
        },
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2s_preregistration_accepts_bounded_open_repair_plan(tmp_path: Path) -> None:
    report = build_phase2s_preregistration_check(
        proposal_json=_write(tmp_path / "proposal.json", _valid_proposal())
    )

    assert report["passed"] is True
    assert report["allowed_actions"] == ["build_phase2s_nonsealed_data_audit_smoke"]
    assert report["observations"]["missing_task_families"] == []
    assert report["observations"]["missing_controls"] == []
    assert report["observations"]["missing_repair_runtime_flags"] == []
    assert report["observations"]["missing_baseline_policy_flags"] == []
    assert report["observations"]["missing_modern_agent_baseline_flags"] == []
    assert report["observations"]["missing_data_artifact_requirement_flags"] == []
    assert report["observations"]["missing_claim_upgrade_flags"] == []
    assert report["observations"]["missing_factor_levels"] == {}


def test_phase2s_preregistration_rejects_sealed_feedback_and_early_training(
    tmp_path: Path,
) -> None:
    proposal = _valid_proposal()
    proposal["data_policy"]["training_roots"] = [
        "artifacts/datasets/phase2i_external_trace_v3_semantic_required"
    ]
    proposal["data_policy"]["uses_sealed_failures_for_design"] = True
    proposal["execution_plan"]["starts_training"] = True

    report = build_phase2s_preregistration_check(
        proposal_json=_write(tmp_path / "proposal.json", proposal)
    )

    assert report["passed"] is False
    assert "do_not_use_sealed_data_for_phase2s_design_training_or_tuning" in report[
        "blocked_actions"
    ]
    assert "do_not_start_training_packaging_or_sealed_eval_from_preregistration" in report[
        "blocked_actions"
    ]


def test_phase2s_preregistration_rejects_weak_benchmark_and_overclaim(
    tmp_path: Path,
) -> None:
    proposal = _valid_proposal()
    proposal["benchmark"]["task_families"] = ["localized_unit_assertion"]
    proposal["benchmark"]["graded_factors"] = ["candidate_count"]
    proposal["benchmark"]["factor_levels"]["candidate_count"] = [2]
    proposal["controls"]["required_controls"] = ["full_package", "react"]
    proposal["repair_runtime"]["requires_rollback_after_failed_or_unsafe_patch"] = False
    proposal["baseline_policy"]["baselines_measured_not_declared"] = False
    proposal["modern_agent_baseline"]["tool_budget_fixed"] = False
    proposal["data_artifact_requirements"]["sandbox_integrity_report_required"] = False
    proposal["statistical_decision"]["tie_policy"] = "claim_upgrade_on_tie"
    proposal["safety_thresholds"]["rollback_success_min"] = 0.9
    proposal["claim_policy"]["claims_production_autonomy"] = True
    proposal["claim_upgrade_policy"]["independent_reproduction_required_for_epoch_claim"] = False
    proposal["hardcoding_policy"]["forbid_expected_patches"] = False

    report = build_phase2s_preregistration_check(
        proposal_json=_write(tmp_path / "proposal.json", proposal)
    )

    assert report["passed"] is False
    assert "do_not_start_phase2s_until_task_families_are_covered" in report[
        "blocked_actions"
    ]
    assert "do_not_start_phase2s_with_binary_only_difficulty" in report["blocked_actions"]
    assert "do_not_start_phase2s_without_factor_level_matrix" in report["blocked_actions"]
    assert "do_not_claim_phase2s_without_all_controls" in report["blocked_actions"]
    assert "do_not_start_phase2s_without_repair_runtime_contract" in report[
        "blocked_actions"
    ]
    assert "do_not_claim_phase2s_with_declared_or_unfair_baselines" in report[
        "blocked_actions"
    ]
    assert (
        "do_not_claim_phase2s_against_unspecified_modern_agent_baseline"
        in report["blocked_actions"]
    )
    assert "do_not_train_phase2s_without_artifact_manifest_requirements" in report[
        "blocked_actions"
    ]
    assert "do_not_claim_phase2s_without_statistical_decision_rule" in report[
        "blocked_actions"
    ]
    assert "do_not_claim_phase2s_without_quantified_safety_thresholds" in report[
        "blocked_actions"
    ]
    assert "do_not_upgrade_architecture_claim_before_phase2s_passes" in report[
        "blocked_actions"
    ]
    assert (
        "do_not_claim_epoch_making_architecture_without_reproduction_and_review"
        in report["blocked_actions"]
    )
    assert "do_not_use_task_specific_hardcoding" in report["blocked_actions"]


def test_phase2s_preregistration_rejects_malformed_margin_without_crashing(
    tmp_path: Path,
) -> None:
    proposal = _valid_proposal()
    proposal["gates"]["full_beats_best_baseline_margin"] = "not-a-number"

    report = build_phase2s_preregistration_check(
        proposal_json=_write(tmp_path / "proposal.json", proposal)
    )

    assert report["passed"] is False
    assert report["observations"]["full_beats_best_baseline_margin"] is None
    assert "do_not_train_phase2s_before_gates_are_preregistered" in report[
        "blocked_actions"
    ]


def test_phase2s_subagent_synthesis_blocks_claim_upgrade_without_second_round() -> None:
    synthesis = (
        ROOT / "docs" / "spec" / "phase2s_subagent_review_synthesis_2026-05-22.md"
    ).read_text(encoding="utf-8")
    normalized = " ".join(synthesis.split())

    assert "no recorded multi-round unanimous approval" in normalized
    assert "Claim Upgrade Status" in normalized
    assert "Blocked" in normalized
    assert "does not itself prove the architecture" in normalized
    assert "Training, packaging, sealed evaluation" in normalized
