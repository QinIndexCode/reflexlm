from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.run_phase2ay_counterfactual_slot_execution import (
    _candidate_actions,
    _execute_selected_row,
    _expected_slot,
    _head_records_by_episode,
    _prediction_key,
    _read_jsonl,
    _select_with_package_native_head_state,
    _test_python_for_row,
)
from reflexlm.cli.run_phase2cd_public_repo_post_verification_control import (
    CONTINUE_INTENT,
    FINISH_INTENT,
    _execute_control,
    _execution_rows,
    _exit_code,
    _feedback_text,
    _command_semantic_text,
    _select,
    _split_repo_disjoint,
    _state,
)
from reflexlm.llm.native_nervous_package import NativeNervousPolicyPackage
from reflexlm.models.semantic_matcher import (
    FrozenEncoderDualSemanticMatcher,
    HashedDualEncoderSemanticMatcher,
    RecencyWeightedSemanticMatcher,
)


class ContinuousLiveRepairPolicy:
    """Keep patch selection and verification control in one policy lifecycle."""

    def __init__(
        self,
        *,
        package_policy: Any,
        verification_matcher: Any | None,
        workspace_root: Path,
    ) -> None:
        self.package_policy = package_policy
        self.verification_matcher = verification_matcher
        self.workspace_root = workspace_root
        self.episode_id: str | None = None
        self.phase = "idle"
        self.selection_calls = 0
        self.verification_calls = 0
        self.reset_calls = 0
        self.trace: list[dict[str, Any]] = []

    def reset_episode(self, episode_id: str) -> None:
        if hasattr(self.package_policy, "reset"):
            self.package_policy.reset()
        self.episode_id = episode_id
        self.phase = "select_patch"
        self.reset_calls += 1
        self.trace = [{"phase": "reset", "episode_id": episode_id}]

    def select_patch(
        self,
        *,
        row: dict[str, Any],
        head_record: dict[str, Any] | None,
        row_index: int,
    ) -> dict[str, Any]:
        if self.phase != "select_patch" or self.episode_id is None:
            raise RuntimeError("continuous repair policy must reset before patch selection")
        selection = _select_with_package_native_head_state(
            policy=self.package_policy,
            row=row,
            head_record=head_record,
            row_index=row_index,
            nsi_reference_mode="runtime_visible_override",
            runtime_evidence_label="Prior runtime evidence",
            runtime_evidence_channel="prompt_text",
            runtime_evidence_control="normal",
        )
        self.selection_calls += 1
        self.phase = "await_verification"
        self.trace.append(
            {
                "phase": "patch_selected",
                "selected_slot": selection.get("selected_slot"),
                "open_repair_authorized": selection.get("open_repair_authorized") is True,
                "qwen_called": selection.get("qwen_called") is True,
            }
        )
        return selection

    def decide_after_verification(
        self,
        *,
        pre_log: dict[str, Any],
        post_log: dict[str, Any],
        control: str = "visible",
        execute_action: bool = True,
    ) -> dict[str, Any]:
        if self.phase != "await_verification" or self.episode_id is None:
            raise RuntimeError("verification control requires a selected patch in the same lifecycle")
        pre_text = _feedback_text(pre_log)
        post_text = _feedback_text(post_log)
        if control == "visible":
            frames = [pre_text, post_text]
            exit_code = _exit_code(post_log, default=0)
        elif control == "erased_post":
            frames = [pre_text]
            exit_code = 1
        elif control in {"wrong_post", "frozen_pre"}:
            frames = [pre_text, pre_text]
            exit_code = 1
        else:
            raise ValueError(f"unsupported Phase2CE verification control {control!r}")
        state = _state(frames=frames, exit_code=exit_code, finish_correct=True)
        package_decision = (
            self.package_policy.decide_verification(state)
            if callable(getattr(self.package_policy, "decide_verification", None))
            and getattr(self.package_policy, "verification_cortex", None) is not None
            else None
        )
        if isinstance(package_decision, dict):
            selected_slot = package_decision.get("selected_slot")
            order = [
                _command_semantic_text(command)
                for command in state.goal.command_allowlist
            ]
            selected = (
                order[selected_slot]
                if isinstance(selected_slot, int) and 0 <= selected_slot < len(order)
                else CONTINUE_INTENT
            )
        else:
            selected = _select(self.verification_matcher, state)
        execution = (
            _execute_control(
                selected=selected,
                state=state,
                workspace_root=self.workspace_root,
                episode_id=f"{self.episode_id}_{control}_control",
            )
            if execute_action
            else {}
        )
        self.verification_calls += 1
        if control == "visible":
            self.phase = "done" if selected == FINISH_INTENT else "replan"
        self.trace.append(
            {
                "phase": f"verification_{control}",
                "selected_intent": selected,
                "action_type": execution.get("action_type"),
            }
        )
        return {
            "control": control,
            "selected_intent": selected,
            "action_type": execution.get("action_type"),
            "finish_selected": selected == FINISH_INTENT,
            "continue_selected": selected == CONTINUE_INTENT,
            "shell_false": execution.get("shell_false") is True if execute_action else True,
            "verification_source": (
                "package_internal_verification_cortex"
                if isinstance(package_decision, dict)
                else "external_verification_matcher"
            ),
            "package_verification_decision": package_decision or {},
        }

    def metadata(self) -> dict[str, Any]:
        package_metadata = (
            self.package_policy.metadata()
            if callable(getattr(self.package_policy, "metadata", None))
            else {}
        )
        matcher_metadata = (
            self.verification_matcher.metadata()
            if callable(getattr(self.verification_matcher, "metadata", None))
            else {}
        )
        return {
            "policy_family": "continuous_live_repair_policy",
            "single_lifecycle": True,
            "phases": ["select_patch", "await_verification", "done_or_replan"],
            "package_policy": package_metadata,
            "verification_matcher": matcher_metadata,
            "verification_control_source": (
                "package_internal_verification_cortex"
                if getattr(self.package_policy, "verification_cortex", None) is not None
                else "external_verification_matcher"
            ),
        }


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _artifact_log(executed: dict[str, Any], key: str) -> dict[str, Any]:
    paths = executed.get("artifact_paths")
    path = Path(str(paths.get(key) or "")) if isinstance(paths, dict) else Path()
    return _read_json(path) if path.is_file() else {}


def _train_verification_matcher(
    *,
    train_rows: list[dict[str, Any]],
    cortex_model_path: str | Path,
    cortex_device: str,
    cortex_dtype: str,
) -> RecencyWeightedSemanticMatcher:
    base = FrozenEncoderDualSemanticMatcher.from_pretrained(
        cortex_model_path,
        device=cortex_device,
        dtype=cortex_dtype,
        seed=41,
    )
    base.fit(
        [
            *[(_feedback_text(row["pre"]), CONTINUE_INTENT) for row in train_rows],
            *[(_feedback_text(row["post"]), FINISH_INTENT) for row in train_rows],
        ]
    )
    return RecencyWeightedSemanticMatcher(base, recency_decay=0.25)


def _verification_controls(
    *,
    matcher: Any | None,
    pre_log: dict[str, Any],
    post_log: dict[str, Any],
    workspace_root: Path,
    episode_id: str,
    package_policy: Any | None = None,
) -> dict[str, dict[str, Any]]:
    controls = {}
    for control in ("erased_post", "wrong_post", "frozen_pre"):
        controller = ContinuousLiveRepairPolicy(
            package_policy=package_policy or object(),
            verification_matcher=matcher,
            workspace_root=workspace_root,
        )
        controller.episode_id = episode_id
        controller.phase = "await_verification"
        controls[control] = controller.decide_after_verification(
            pre_log=pre_log,
            post_log=post_log,
            control=control,
            execute_action=False,
        )
    return controls


def run_phase2ce_single_policy_live_patch_verify_stop_loop(
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
    timeout_seconds: int = 30,
    test_python_map_json: str | Path | None = None,
    package_device: str | None = None,
    package_quantization: str | None = None,
    package_model_load_strategy: str | None = None,
    package_offload_state_dict: bool | None = None,
) -> dict[str, Any]:
    historical_rows = _execution_rows(historical_execution_jsonl, tasks_jsonl)
    train_rows, historical_holdout_rows = _split_repo_disjoint(historical_rows)
    holdout_repos = {row["repo_origin"] for row in historical_holdout_rows}
    tasks = [
        row
        for row in _read_jsonl(tasks_jsonl)
        if str(row.get("repo_origin") or "") in holdout_repos
    ]
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
    for index, row in enumerate(tasks):
        task_id = str(row.get("task_id") or f"phase2ce_{index:05d}")
        policy.reset_episode(task_id)
        reset_count_before_selection = policy.reset_calls
        selection = policy.select_patch(
            row=row,
            head_record=head_records.get(_prediction_key(row)),
            row_index=index,
        )
        expected_slot = _expected_slot(row)
        selected_slot = selection.get("selected_slot")
        selected_correct = selected_slot == expected_slot and selected_slot is not None
        row_test_python, test_python_source = _test_python_for_row(
            row,
            default_test_python=None,
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
                artifact_root=artifacts / f"row_{index:05d}" / "execution",
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
        counterfactuals = (
            _verification_controls(
                matcher=verification_matcher,
                pre_log=pre_log,
                post_log=post_log,
                workspace_root=workspace_root,
                episode_id=task_id,
                package_policy=package_policy,
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
                "policy_trace": list(policy.trace),
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
    count = max(len(live_rows), 1)
    metrics = {
        "rows": len(live_rows),
        "repo_count": len({str(row.get("repo_origin")) for row in live_rows}),
        "patch_selection_accuracy": sum(row["selected_patch_correctly"] for row in live_rows)
        / count,
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
    }
    checks = {
        "phase2ax_full_postflight_passed": full_postflight.get("passed") is True,
        "minimum_live_holdout_repos_met": metrics["repo_count"] >= 3,
        "minimum_live_rows_met": metrics["rows"] >= 6,
        "package_policy_loaded_once": policy.metadata()["package_policy"].get(
            "package_family"
        )
        == "phase2d_native_nervous_package",
        "package_selected_patch_gate": metrics["patch_selection_accuracy"] >= 0.90,
        "live_patch_execution_success_gate": metrics[
            "live_patch_execution_success_rate"
        ]
        >= 0.80,
        "actual_pre_tests_failed": all(row["actual_pre_test_failed"] for row in live_rows),
        "actual_post_tests_passed": all(row["actual_post_test_passed"] for row in live_rows),
        "same_policy_lifecycle_selects_and_stops": metrics[
            "single_policy_lifecycle_rate"
        ]
        == 1.0,
        "visible_post_verification_finish_gate": metrics["visible_finish_rate"] >= 0.90,
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
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ce_single_policy_live_patch_verify_stop_loop",
        "passed": passed,
        "ready_for_single_policy_live_public_repo_patch_verify_stop_claim": passed,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "policy_metadata": policy.metadata(),
        "dataset": {
            "historical_train_rows": len(train_rows),
            "live_holdout_rows": len(tasks),
            "holdout_repos": sorted(holdout_repos),
            "tasks_jsonl": str(tasks_jsonl),
            "head_records_jsonl": str(head_records_jsonl),
        },
        "checks": checks,
        "metrics": metrics,
        "output_jsonl": str(output_jsonl),
        "supported_claims": [
            "one continuous composite policy lifecycle selected bounded package-loaded repair candidates, executed live public-repository symbolic patches, consumed newly produced pytest feedback, and selected DONE on repo-disjoint holdout repositories"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "arbitrary public-repository repair",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cf_long_run_live_repair_stability_and_plasticity"
            if passed
            else "repair_phase2ce_single_policy_live_patch_verify_stop_loop"
        ),
    }
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one continuous policy through live patch selection, execution, verification, and stop."
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
    parser.add_argument("--lexical-matcher-path", required=True)
    parser.add_argument("--cortex-model-path", required=True)
    parser.add_argument("--cortex-device", default="cpu")
    parser.add_argument("--cortex-dtype", default="auto")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--test-python-map-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2ce_single_policy_live_patch_verify_stop_loop(
        historical_execution_jsonl=args.historical_execution_jsonl,
        tasks_jsonl=args.tasks_jsonl,
        head_records_jsonl=args.head_records_jsonl,
        full_postflight_json=args.full_postflight_json,
        dataset_root=args.dataset_root,
        clone_root=args.clone_root,
        package_path=args.package_path,
        package_device=args.package_device,
        package_quantization=args.package_quantization,
        package_model_load_strategy=args.package_model_load_strategy,
        package_offload_state_dict=args.package_offload_state_dict,
        lexical_matcher_path=args.lexical_matcher_path,
        cortex_model_path=args.cortex_model_path,
        cortex_device=args.cortex_device,
        cortex_dtype=args.cortex_dtype,
        artifact_root=args.artifact_root,
        output_jsonl=args.output_jsonl,
        output_report_json=args.output_report_json,
        timeout_seconds=args.timeout_seconds,
        test_python_map_json=args.test_python_map_json,
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
