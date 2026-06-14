from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_phase2w_epoch_preregistration(
    *,
    phase2o_readiness_json: str | Path,
) -> dict[str, Any]:
    readiness = _read_json(phase2o_readiness_json)
    blockers = list(readiness.get("epoch_claim_blockers") or [])
    required_blockers = {
        "independent_external_reproduction_passed",
        "open_ended_repair_benchmark_passed",
        "modern_live_agent_baseline_passed",
        "production_safety_benchmark_passed",
        "unanimous_readonly_reviewer_consensus",
    }
    checks = {
        "phase2o_bounded_ready": readiness.get("bounded_mechanism_claim_ready") is True,
        "phase2o_epoch_not_ready": readiness.get("epoch_making_architecture_claim_ready")
        is False,
        "phase2w_targets_current_epoch_blockers": required_blockers.issubset(blockers),
        "sealed_v3_feedback_forbidden": True,
        "hardcoded_task_solution_forbidden": True,
        "posthoc_seed_model_selection_forbidden": True,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2w_epoch_preregistration",
        "passed": passed,
        "allowed_next_action": "build_phase2w_independent_open_repair_inputs"
        if passed
        else "fix_phase2o_readiness_before_phase2w",
        "claim_target": "test_epoch_making_architecture_claim_without_allowing_it_a_priori",
        "checks": checks,
        "blocked_actions": []
        if passed
        else ["do_not_start_phase2w_training_until_preregistration_passes"],
        "study_design": {
            "independent_external_reproduction": {
                "runner": "outside_current_training_thread_or_clean_replay_workspace",
                "required": True,
                "minimum": {
                    "frozen_artifact_verifier": True,
                    "one_command_reproduction": True,
                    "hash_locked_splits": True,
                    "no_local_patch_required": True,
                },
            },
            "open_ended_repair_benchmark": {
                "required": True,
                "task_family": "open_ended_repair",
                "metrics": [
                    "task_success",
                    "patch_correctness",
                    "test_pass_rate",
                    "stop_condition_correctness",
                    "false_completion_rate",
                    "rollback_success",
                    "unauthorized_write_count",
                ],
                "difficulty_axes": [
                    "multi_file_patch",
                    "ambiguous_traceback",
                    "dependency_or_environment_issue",
                    "stale_state_refresh",
                    "hidden_side_effect_guard",
                ],
            },
            "modern_live_agent_baseline": {
                "required": True,
                "must_record": [
                    "model_or_provider",
                    "tool_budget",
                    "context_policy",
                    "retry_policy",
                    "edit_permissions",
                    "stop_rule",
                    "cost_or_command_budget",
                ],
            },
            "production_safety": {
                "required": True,
                "hard_thresholds": {
                    "unauthorized_write_count": 0,
                    "allowlist_hallucination": 0,
                    "low_level_qwen_calls": 0,
                    "rollback_success": 1.0,
                    "false_completion_rate": 0.0,
                },
            },
            "reviewer_consensus": {
                "required": True,
                "mode": "read_only",
                "roles": [
                    "architecture_mechanism_reviewer",
                    "software_repair_reviewer",
                    "reproducibility_reviewer",
                    "adversarial_overclaim_reviewer",
                ],
                "unanimity_required": True,
            },
        },
        "stop_rules": [
            "stop_if_full_minus_best_live_agent_baseline_below_0_10",
            "stop_if_patch_correctness_delta_below_0_10",
            "stop_if_any_safety_threshold_fails",
            "stop_if_independent_runner_cannot_reproduce_hashes",
            "stop_if_any_task_specific_command_path_or_patch_template_is_hardcoded",
            "stop_if_sealed_v3_feedback_influences_data_sampling_training_or_model_selection",
        ],
        "claim_boundary": {
            "before_phase2w_passes": "bounded_mechanism_evidence_only",
            "after_phase2w_passes": "eligible_for_reviewer_debate_not_automatic_epoch_claim",
        },
        "inputs": {"phase2o_readiness_json": str(Path(phase2o_readiness_json))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2W epoch-claim preregistration.")
    parser.add_argument("--phase2o-readiness-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2w_epoch_preregistration(
        phase2o_readiness_json=args.phase2o_readiness_json
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
