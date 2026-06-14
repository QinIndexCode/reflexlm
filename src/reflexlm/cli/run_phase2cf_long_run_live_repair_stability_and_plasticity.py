from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reflexlm.cli.run_phase2ay_counterfactual_slot_execution import (
    _candidate_actions,
    _execute_selected_row,
    _expected_slot,
    _head_records_by_episode,
    _prediction_key,
    _read_jsonl,
)
from reflexlm.cli.run_phase2cd_public_repo_post_verification_control import (
    FINISH_INTENT,
    _exit_code,
    _feedback_text,
    _select,
    _split_repo_disjoint,
    _state,
)
from reflexlm.cli.run_phase2ce_single_policy_live_patch_verify_stop_loop import (
    ContinuousLiveRepairPolicy,
    _artifact_log,
    _read_json,
    _train_verification_matcher,
    _verification_controls,
    _write_jsonl,
)
from reflexlm.llm.native_nervous_package import NativeNervousPolicyPackage
from reflexlm.models.semantic_matcher import HashedDualEncoderSemanticMatcher, RecencyWeightedSemanticMatcher


def _execution_rows(
    historical_execution_jsonl: str | Path,
    tasks_jsonl: str | Path,
) -> list[dict[str, Any]]:
    from reflexlm.cli.run_phase2cd_public_repo_post_verification_control import (
        _execution_rows as phase2cd_execution_rows,
    )

    return phase2cd_execution_rows(historical_execution_jsonl, tasks_jsonl)


def _test_python_for_row(
    row: dict[str, Any],
    *,
    test_python_map: dict[str, Any],
) -> tuple[str | None, str]:
    from reflexlm.cli.run_phase2ay_counterfactual_slot_execution import (
        _test_python_for_row as phase2ay_test_python_for_row,
    )

    return phase2ay_test_python_for_row(
        row,
        default_test_python=None,
        test_python_map=test_python_map,
    )


def _row_key(row: dict[str, Any]) -> str:
    return str(row.get("base_task_id") or row.get("task_id") or "")


def _stable_selection_rate(rows: list[dict[str, Any]]) -> float:
    grouped: dict[str, set[Any]] = defaultdict(set)
    for row in rows:
        grouped[_row_key(row)].add(row.get("selected_slot"))
    if not grouped:
        return 0.0
    return sum(len(slots) == 1 for slots in grouped.values()) / len(grouped)


def _build_phase2cf_report(
    *,
    live_rows: list[dict[str, Any]],
    cycles: int,
    train_rows: int,
    holdout_repos: set[str],
    tasks_jsonl: str | Path,
    head_records_jsonl: str | Path,
    output_jsonl: str | Path,
    policy_metadata: dict[str, Any],
    full_postflight_passed: bool,
    require_package_internal_verification: bool = False,
) -> dict[str, Any]:
    count = max(len(live_rows), 1)
    after_first = [row for row in live_rows if int(row.get("cycle_index", 0)) > 0]
    after_first_count = max(len(after_first), 1)
    metrics = {
        "rows": len(live_rows),
        "cycles": cycles,
        "repo_count": len({str(row.get("repo_origin")) for row in live_rows}),
        "patch_selection_accuracy": sum(row["selected_patch_correctly"] for row in live_rows)
        / count,
        "stable_selection_rate": _stable_selection_rate(live_rows),
        "live_patch_execution_success_rate": sum(
            row["live_patch_execution_success"] for row in live_rows
        )
        / count,
        "visible_finish_rate": sum(
            row.get("visible_control", {}).get("finish_selected") is True
            for row in live_rows
        )
        / count,
        "single_policy_lifecycle_rate": sum(
            row["single_policy_lifecycle"] for row in live_rows
        )
        / count,
        "plasticity_feedback_accepted_rate": sum(
            row.get("plasticity_feedback", {}).get("accepted") is True
            for row in live_rows
        )
        / count,
        "plasticity_memory_hit_rate_after_first_cycle": sum(
            row.get("plasticity_prediction", {}).get("memory_hit") is True
            for row in after_first
        )
        / after_first_count,
        "erased_finish_rate": sum(
            row.get("counterfactual_controls", {})
            .get("erased_post", {})
            .get("finish_selected")
            is True
            for row in live_rows
        )
        / count,
        "wrong_finish_rate": sum(
            row.get("counterfactual_controls", {})
            .get("wrong_post", {})
            .get("finish_selected")
            is True
            for row in live_rows
        )
        / count,
        "frozen_finish_rate": sum(
            row.get("counterfactual_controls", {})
            .get("frozen_pre", {})
            .get("finish_selected")
            is True
            for row in live_rows
        )
        / count,
        "lexical_finish_rate": sum(row["lexical_visible_finish"] for row in live_rows)
        / count,
        "no_prior_finish_rate": sum(row["no_prior_visible_finish"] for row in live_rows)
        / count,
        "package_internal_verification_rate": sum(
            row.get("visible_control", {}).get("verification_source")
            == "package_internal_verification_cortex"
            for row in live_rows
        )
        / count,
        "package_internal_counterfactual_verification_rate": sum(
            control.get("verification_source")
            == "package_internal_verification_cortex"
            for row in live_rows
            for control in row.get("counterfactual_controls", {}).values()
        )
        / max(
            sum(
                len(row.get("counterfactual_controls", {}))
                for row in live_rows
            ),
            1,
        ),
    }
    package_verification_packaged = (
        policy_metadata.get("package_policy", {}).get("verification_cortex_packaged")
        is True
    )
    checks = {
        "phase2ax_full_postflight_passed": full_postflight_passed,
        "minimum_live_holdout_repos_met": metrics["repo_count"] >= 3,
        "minimum_cycles_met": cycles >= 2,
        "minimum_live_rows_met": metrics["rows"] >= 12,
        "package_policy_loaded_once": policy_metadata.get("package_policy", {}).get(
            "package_family"
        )
        == "phase2d_native_nervous_package",
        "stable_patch_selection_across_cycles": metrics["stable_selection_rate"] == 1.0,
        "package_selected_patch_gate": metrics["patch_selection_accuracy"] >= 0.90,
        "live_patch_execution_success_gate": metrics[
            "live_patch_execution_success_rate"
        ]
        >= 0.90,
        "actual_pre_tests_failed": all(row["actual_pre_test_failed"] for row in live_rows),
        "actual_post_tests_passed": all(row["actual_post_test_passed"] for row in live_rows),
        "same_policy_lifecycle_selects_and_stops": metrics[
            "single_policy_lifecycle_rate"
        ]
        == 1.0,
        "visible_post_verification_finish_gate": metrics["visible_finish_rate"] >= 0.90,
        "plasticity_feedback_accepted": metrics["plasticity_feedback_accepted_rate"] >= 0.90,
        "plasticity_recall_after_first_cycle": metrics[
            "plasticity_memory_hit_rate_after_first_cycle"
        ]
        >= 0.90,
        "erased_feedback_reduces_finish": metrics["erased_finish_rate"]
        <= metrics["visible_finish_rate"] - 0.50,
        "wrong_feedback_reduces_finish": metrics["wrong_finish_rate"]
        <= metrics["visible_finish_rate"] - 0.50,
        "frozen_feedback_reduces_finish": metrics["frozen_finish_rate"]
        <= metrics["visible_finish_rate"] - 0.50,
        "natural_controller_outperforms_lexical": metrics["visible_finish_rate"]
        > metrics["lexical_finish_rate"],
        "natural_controller_outperforms_no_prior": metrics["visible_finish_rate"]
        > metrics["no_prior_finish_rate"],
        "recorded_patch_not_used_as_generated_evidence": not any(
            row["recorded_patch_artifact_used"] for row in live_rows
        ),
        "recorded_patch_only_used_for_fault_injection": all(
            row["recorded_patch_artifact_used_for_fault_injection"] for row in live_rows
        ),
        "no_freeform_patch_generation": not any(
            row["freeform_patch_generation"] for row in live_rows
        ),
        "no_sealed_feedback": not any(row["sealed_feedback_used"] for row in live_rows),
        "required_package_verification_cortex_packaged": (
            not require_package_internal_verification or package_verification_packaged
        ),
        "required_visible_verification_is_package_internal": (
            not require_package_internal_verification
            or metrics["package_internal_verification_rate"] == 1.0
        ),
        "required_counterfactual_verification_is_package_internal": (
            not require_package_internal_verification
            or metrics["package_internal_counterfactual_verification_rate"] == 1.0
        ),
    }
    passed = all(checks.values())
    unified_package_passed = (
        passed
        and package_verification_packaged
        and metrics["package_internal_verification_rate"] == 1.0
        and metrics["package_internal_counterfactual_verification_rate"] == 1.0
    )
    return {
        "artifact_family": (
            "phase2ch_unified_package_long_run_stability_and_plasticity"
            if require_package_internal_verification
            else "phase2cf_long_run_live_repair_stability_and_plasticity"
        ),
        "passed": passed,
        "ready_for_long_run_live_repair_stability_and_plasticity_claim": passed,
        "ready_for_unified_package_long_run_stability_and_plasticity_claim": (
            unified_package_passed
        ),
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "policy_metadata": policy_metadata,
        "dataset": {
            "historical_train_rows": train_rows,
            "holdout_repos": sorted(holdout_repos),
            "cycles": cycles,
            "tasks_jsonl": str(tasks_jsonl),
            "head_records_jsonl": str(head_records_jsonl),
        },
        "checks": checks,
        "metrics": metrics,
        "output_jsonl": str(output_jsonl),
        "supported_claims": [
            (
                "one unified multi-cortical package repeatedly selected bounded repair "
                "candidates, executed fresh public-repository patch/test loops, accepted "
                "verifier-gated plasticity feedback, recalled repeated environment "
                "patterns, and used only its packaged verification cortex to stop after "
                "visible post-test success"
                if unified_package_passed
                else "one loaded package policy repeatedly selected bounded repair "
                "candidates, executed fresh public-repository patch/test loops, accepted "
                "verifier-gated plasticity feedback, recalled repeated environment "
                "patterns, and stopped only after visible post-test success"
            )
        ]
        if passed
        else [],
        "unsupported_claims": [
            "arbitrary public-repository repair",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
            "differentiable neural plasticity",
        ],
        "next_required_experiment": (
            "phase2ci_repo_diverse_unseen_failure_family_under_unified_package"
            if passed
            else "repair_phase2cf_long_run_live_repair_stability_and_plasticity"
        ),
    }


def run_phase2cf_long_run_live_repair_stability_and_plasticity(
    *,
    historical_execution_jsonl: str | Path,
    tasks_jsonl: str | Path,
    head_records_jsonl: str | Path,
    full_postflight_json: str | Path,
    dataset_root: str | Path,
    clone_root: str | Path,
    package_path: str | Path,
    lexical_matcher_path: str | Path,
    cortex_model_path: str | Path,
    cortex_device: str,
    cortex_dtype: str,
    artifact_root: str | Path,
    output_jsonl: str | Path,
    output_report_json: str | Path,
    cycles: int = 2,
    max_base_rows: int = 6,
    timeout_seconds: int = 30,
    test_python_map_json: str | Path | None = None,
    package_device: str | None = None,
    package_quantization: str | None = None,
    package_model_load_strategy: str | None = None,
    package_offload_state_dict: bool | None = None,
    require_package_internal_verification: bool = False,
) -> dict[str, Any]:
    historical_rows = _execution_rows(historical_execution_jsonl, tasks_jsonl)
    train_rows, historical_holdout_rows = _split_repo_disjoint(historical_rows)
    holdout_repos = {row["repo_origin"] for row in historical_holdout_rows}
    tasks = [
        row
        for row in _read_jsonl(tasks_jsonl)
        if str(row.get("repo_origin") or "") in holdout_repos
    ][:max_base_rows]
    head_records = _head_records_by_episode(_read_jsonl(head_records_jsonl))
    full_postflight = _read_json(full_postflight_json)
    test_python_map = (
        _read_json(test_python_map_json) if test_python_map_json is not None else {}
    )
    lexical_base = HashedDualEncoderSemanticMatcher.load(lexical_matcher_path)
    lexical_base.lexical_residual_weight = 3.0
    lexical_matcher = RecencyWeightedSemanticMatcher(lexical_base, recency_decay=0.25)
    package_policy = NativeNervousPolicyPackage(
        package_path,
        device=package_device,
        quantization=package_quantization,
        model_load_strategy=package_model_load_strategy,
        offload_state_dict=package_offload_state_dict,
    )
    verification_matcher = (
        None
        if package_policy.verification_cortex is not None
        else _train_verification_matcher(
            train_rows=train_rows,
            cortex_model_path=cortex_model_path,
            cortex_device=cortex_device,
            cortex_dtype=cortex_dtype,
        )
    )
    workspace_root = Path.cwd().resolve()
    policy = ContinuousLiveRepairPolicy(
        package_policy=package_policy,
        verification_matcher=verification_matcher,
        workspace_root=workspace_root,
    )
    artifacts = Path(artifact_root)
    artifacts.mkdir(parents=True, exist_ok=True)
    live_rows = []
    for cycle_index in range(cycles):
        for task_index, row in enumerate(tasks):
            base_task_id = str(row.get("task_id") or f"phase2cf_{task_index:05d}")
            task_id = f"{base_task_id}:cycle_{cycle_index + 1:02d}"
            policy.reset_episode(task_id)
            reset_count_before_selection = policy.reset_calls
            selection = policy.select_patch(
                row=row,
                head_record=head_records.get(_prediction_key(row)),
                row_index=(cycle_index * len(tasks)) + task_index,
            )
            expected_slot = _expected_slot(row)
            selected_slot = selection.get("selected_slot")
            selected_correct = selected_slot == expected_slot and selected_slot is not None
            row_test_python, test_python_source = _test_python_for_row(
                row,
                test_python_map=test_python_map,
            )
            execution_payload: dict[str, Any] = {}
            executed: dict[str, Any] = {}
            if (
                full_postflight.get("passed") is True
                and selected_correct
                and selection.get("open_repair_authorized") is True
            ):
                execution_payload = _execute_selected_row(
                    row=row,
                    dataset_root=Path(dataset_root),
                    clone_root=Path(clone_root),
                    package_path=Path(package_path),
                    artifact_root=artifacts
                    / f"cycle_{cycle_index + 1:02d}"
                    / f"row_{task_index:05d}"
                    / "execution",
                    timeout_seconds=timeout_seconds,
                    test_python=row_test_python,
                )
                executed = execution_payload.get("row") or {}
            pre_log = _artifact_log(executed, "pre_test_log")
            post_log = _artifact_log(executed, "post_test_log")
            visible_control = (
                policy.decide_after_verification(
                    pre_log=pre_log,
                    post_log=post_log,
                    control="visible",
                    execute_action=True,
                )
                if pre_log and post_log
                else {}
            )
            plasticity_feedback = (
                package_policy.observe_feedback(
                    verified_success=_exit_code(post_log, default=1) == 0,
                    verifier="post_execution_verifier",
                )
                if pre_log and post_log
                else {}
            )
            counterfactuals = (
                _verification_controls(
                    matcher=verification_matcher,
                    package_policy=package_policy,
                    pre_log=pre_log,
                    post_log=post_log,
                    workspace_root=workspace_root,
                    episode_id=task_id,
                )
                if pre_log and post_log
                else {}
            )
            lexical_state = (
                _state(
                    frames=[_feedback_text(pre_log), _feedback_text(post_log)],
                    exit_code=_exit_code(post_log, default=0),
                    finish_correct=True,
                )
                if pre_log and post_log
                else None
            )
            lexical_selected = _select(lexical_matcher, lexical_state) if lexical_state else None
            no_prior_selected = _select(None, lexical_state) if lexical_state else None
            actions = _candidate_actions(row)
            live_rows.append(
                {
                    "task_id": task_id,
                    "base_task_id": base_task_id,
                    "cycle_index": cycle_index,
                    "repo_origin": row.get("repo_origin"),
                    "selected_slot": selected_slot,
                    "expected_slot": expected_slot,
                    "selected_repair_action": (
                        actions[selected_slot]
                        if isinstance(selected_slot, int) and 0 <= selected_slot < len(actions)
                        else None
                    ),
                    "selected_patch_correctly": selected_correct,
                    "open_repair_authorized": selection.get("open_repair_authorized") is True,
                    "package_qwen_called": selection.get("qwen_called") is True,
                    "live_execution_attempted": bool(executed),
                    "live_patch_execution_success": executed.get("success") is True,
                    "actual_pre_test_failed": _exit_code(pre_log, default=0) != 0,
                    "actual_post_test_passed": _exit_code(post_log, default=1) == 0,
                    "visible_control": visible_control,
                    "counterfactual_controls": counterfactuals,
                    "lexical_visible_finish": lexical_selected == FINISH_INTENT,
                    "no_prior_visible_finish": no_prior_selected == FINISH_INTENT,
                    "single_policy_lifecycle": policy.reset_calls == reset_count_before_selection
                    and policy.selection_calls >= 1
                    and policy.verification_calls >= 1,
                    "policy_final_phase": policy.phase,
                    "plasticity_prediction": (
                        selection.get("policy_outputs", {}).get("plasticity_prediction")
                        if selection
                        else {}
                    ),
                    "plasticity_feedback": plasticity_feedback,
                    "recorded_patch_artifact_used": executed.get(
                        "recorded_patch_artifact_used"
                    )
                    is True,
                    "recorded_patch_artifact_used_for_fault_injection": executed.get(
                        "recorded_patch_artifact_used_for_fault_injection"
                    )
                    is True,
                    "freeform_patch_generation": executed.get("freeform_patch_generation")
                    is True,
                    "sealed_feedback_used": executed.get("sealed_feedback_used") is True,
                    "test_python": row_test_python,
                    "test_python_source": test_python_source,
                    "artifact_paths": executed.get("artifact_paths") or {},
                    "phase2z_execution_summary": execution_payload.get("summary"),
                }
            )
    _write_jsonl(output_jsonl, live_rows)
    report = _build_phase2cf_report(
        live_rows=live_rows,
        cycles=cycles,
        train_rows=len(train_rows),
        holdout_repos=holdout_repos,
        tasks_jsonl=tasks_jsonl,
        head_records_jsonl=head_records_jsonl,
        output_jsonl=output_jsonl,
        policy_metadata=policy.metadata(),
        full_postflight_passed=full_postflight.get("passed") is True,
        require_package_internal_verification=require_package_internal_verification,
    )
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run long-horizon live repair stability and plasticity validation."
    )
    parser.add_argument("--historical-execution-jsonl", required=True)
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--head-records-jsonl", required=True)
    parser.add_argument("--full-postflight-json", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--clone-root", required=True)
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--package-device")
    parser.add_argument("--package-quantization")
    parser.add_argument("--package-model-load-strategy")
    parser.add_argument(
        "--package-offload-state-dict",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--require-package-internal-verification", action="store_true")
    parser.add_argument("--lexical-matcher-path", required=True)
    parser.add_argument("--cortex-model-path", required=True)
    parser.add_argument("--cortex-device", default="cpu")
    parser.add_argument("--cortex-dtype", default="auto")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--max-base-rows", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--test-python-map-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2cf_long_run_live_repair_stability_and_plasticity(
        historical_execution_jsonl=args.historical_execution_jsonl,
        tasks_jsonl=args.tasks_jsonl,
        head_records_jsonl=args.head_records_jsonl,
        full_postflight_json=args.full_postflight_json,
        dataset_root=args.dataset_root,
        clone_root=args.clone_root,
        package_path=args.package_path,
        lexical_matcher_path=args.lexical_matcher_path,
        cortex_model_path=args.cortex_model_path,
        cortex_device=args.cortex_device,
        cortex_dtype=args.cortex_dtype,
        artifact_root=args.artifact_root,
        output_jsonl=args.output_jsonl,
        output_report_json=args.output_report_json,
        cycles=args.cycles,
        max_base_rows=args.max_base_rows,
        timeout_seconds=args.timeout_seconds,
        test_python_map_json=args.test_python_map_json,
        package_device=args.package_device,
        package_quantization=args.package_quantization,
        package_model_load_strategy=args.package_model_load_strategy,
        package_offload_state_dict=args.package_offload_state_dict,
        require_package_internal_verification=args.require_package_internal_verification,
    )
    print(
        json.dumps(
            {
                "artifact_family": report["artifact_family"],
                "passed": report["passed"],
                "checks": report["checks"],
                "metrics": report["metrics"],
                "next_required_experiment": report["next_required_experiment"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
