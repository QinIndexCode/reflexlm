import copy
import json
from pathlib import Path

from reflexlm.cli.check_phase2t_architecture_iteration import (
    build_phase2t_architecture_iteration_check,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "docs" / "spec" / "phase2t_architecture_iteration_template.json"
SPEC = ROOT / "docs" / "spec" / "phase2t_architecture_iteration_preregistration.md"


def _valid_proposal() -> dict:
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2t_architecture_preregistration_accepts_strong_but_bounded_plan(
    tmp_path: Path,
) -> None:
    report = build_phase2t_architecture_iteration_check(
        proposal_json=_write(tmp_path / "proposal.json", _valid_proposal())
    )

    assert report["passed"] is True
    assert report["allowed_actions"] == ["collect_phase2t_public_repair_loop_specs"]
    assert report["supported_research_direction"] == {
        "continue_architecture_iteration": True,
        "start_training_now": False,
        "upgrade_paper_b_claim_now": False,
        "target_next_evidence": "repair_loop_cross_family_modern_baseline_delta",
    }
    assert report["observations"]["missing_architecture_components"] == []
    assert report["observations"]["missing_repair_loop_capabilities"] == []
    assert report["observations"]["missing_task_families"] == []
    assert report["observations"]["missing_controls"] == []
    assert report["observations"]["missing_metrics"] == []
    assert report["observations"]["missing_factor_levels"] == {}
    assert "llama_or_mistral_family_preregistered" in report["observations"]["model_families"]


def test_phase2t_rejects_sealed_feedback_and_early_training(tmp_path: Path) -> None:
    proposal = _valid_proposal()
    proposal["data_policy"]["training_roots"] = [
        "artifacts/datasets/phase2i_external_trace_v3_semantic_required"
    ]
    proposal["data_policy"]["uses_sealed_failures_for_design"] = True
    proposal["execution_plan"]["starts_training"] = True
    proposal["execution_plan"]["starts_claim_upgrade"] = True

    report = build_phase2t_architecture_iteration_check(
        proposal_json=_write(tmp_path / "proposal.json", proposal)
    )

    assert report["passed"] is False
    assert "do_not_use_sealed_data_for_phase2t_design_training_or_tuning" in report[
        "blocked_actions"
    ]
    assert "do_not_start_training_packaging_sealed_eval_or_claim_upgrade" in report[
        "blocked_actions"
    ]


def test_phase2t_rejects_qwen_only_or_single_seed_claim_matrix(tmp_path: Path) -> None:
    proposal = _valid_proposal()
    proposal["model_matrix"]["min_model_families"] = 1
    proposal["model_matrix"]["min_seeds_per_model"] = 1
    proposal["model_matrix"]["requires_non_qwen_family"] = False
    proposal["model_matrix"]["model_families"] = ["qwen2_5"]

    report = build_phase2t_architecture_iteration_check(
        proposal_json=_write(tmp_path / "proposal.json", proposal)
    )

    assert report["passed"] is False
    assert "do_not_claim_strong_architecture_from_qwen_only_or_single_seed" in report[
        "blocked_actions"
    ]


def test_phase2t_rejects_architecture_component_and_runtime_drift_gaps(
    tmp_path: Path,
) -> None:
    proposal = _valid_proposal()
    proposal["architecture_iteration"]["components"] = [
        "patch_proposal_head",
        "test_selection_head",
    ]
    proposal["architecture_iteration"]["repair_loop_capabilities"] = [
        "inspect_runtime_evidence"
    ]
    proposal["architecture_iteration"]["train_runtime_drift_blocks_claim"] = False
    proposal["repair_runtime"]["requires_rollback_after_failed_or_unsafe_patch"] = False

    report = build_phase2t_architecture_iteration_check(
        proposal_json=_write(tmp_path / "proposal.json", proposal)
    )

    assert report["passed"] is False
    assert (
        "do_not_start_phase2t_without_repair_loop_architecture_components"
        in report["blocked_actions"]
    )
    assert "do_not_start_phase2t_without_repair_loop_capabilities" in report[
        "blocked_actions"
    ]
    assert "do_not_claim_phase2t_with_train_runtime_drift" in report["blocked_actions"]
    assert "do_not_start_phase2t_without_repair_runtime_contract" in report[
        "blocked_actions"
    ]
    assert "rollback_safety_head" in report["observations"]["missing_architecture_components"]


def test_phase2t_rejects_weak_benchmark_baseline_safety_and_hardcoding(
    tmp_path: Path,
) -> None:
    proposal = _valid_proposal()
    proposal["benchmark"]["task_families"] = ["localized_unit_assertion"]
    proposal["benchmark"]["graded_factors"] = ["candidate_count"]
    proposal["benchmark"]["factor_levels"]["candidate_count"] = [2]
    proposal["benchmark"]["metrics"] = ["task_success"]
    proposal["controls"]["required_controls"] = ["full_package", "react"]
    proposal["baseline_policy"]["baselines_measured_not_declared"] = False
    proposal["modern_agent_baseline"]["tool_budget_fixed"] = False
    proposal["data_artifact_requirements"]["modern_baseline_artifacts_required"] = False
    proposal["gates"]["full_beats_best_modern_baseline_task_success_margin"] = 0.0
    proposal["statistical_decision"]["stratify_by_model_family"] = False
    proposal["safety_thresholds"]["rollback_success_min"] = 0.9
    proposal["claim_policy"]["claims_epoch_making_architecture"] = True
    proposal["hardcoding_policy"]["forbid_expected_patches"] = False

    report = build_phase2t_architecture_iteration_check(
        proposal_json=_write(tmp_path / "proposal.json", proposal)
    )

    assert report["passed"] is False
    assert "do_not_start_phase2t_until_task_families_are_covered" in report[
        "blocked_actions"
    ]
    assert "do_not_start_phase2t_with_binary_only_difficulty" in report["blocked_actions"]
    assert "do_not_start_phase2t_without_factor_level_matrix" in report["blocked_actions"]
    assert "do_not_claim_phase2t_without_all_controls" in report["blocked_actions"]
    assert "do_not_claim_phase2t_without_repair_safety_and_cost_metrics" in report[
        "blocked_actions"
    ]
    assert "do_not_claim_phase2t_with_declared_or_unfair_baselines" in report[
        "blocked_actions"
    ]
    assert (
        "do_not_claim_phase2t_against_unspecified_modern_agent_baseline"
        in report["blocked_actions"]
    )
    assert "do_not_train_phase2t_without_artifact_manifest_requirements" in report[
        "blocked_actions"
    ]
    assert "do_not_train_phase2t_before_gates_are_preregistered" in report[
        "blocked_actions"
    ]
    assert "do_not_claim_phase2t_without_statistical_decision_rule" in report[
        "blocked_actions"
    ]
    assert "do_not_claim_phase2t_without_quantified_safety_thresholds" in report[
        "blocked_actions"
    ]
    assert "do_not_upgrade_paper_or_architecture_claim_before_phase2t_passes" in report[
        "blocked_actions"
    ]
    assert "do_not_use_task_specific_hardcoding" in report["blocked_actions"]


def test_phase2t_spec_states_current_claim_remains_bounded() -> None:
    text = SPEC.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "current paper position stays bounded" in normalized.lower()
    assert "Sealed evaluations remain final evaluation artifacts only" in normalized
    assert "Training and runtime must share the same implementation" in normalized
    assert "No claim upgrade from Qwen-only evidence" in normalized
