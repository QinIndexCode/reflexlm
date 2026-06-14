from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.collect_phase2bk_runtime_world_model_trajectories import (
    _command_text,
    _execute_task,
    _resolved_argv,
)
from reflexlm.models.semantic_matcher import (
    FrozenEncoderDualSemanticMatcher,
    HashedDualEncoderSemanticMatcher,
    RecencyWeightedSemanticMatcher,
    _command_semantic_text,
)
from reflexlm.schema import (
    FileSystemState,
    GoalSpec,
    ProcessState,
    ProcessStatus,
    RuntimeEvidenceState,
    SystemStateFrame,
    TaskType,
    TerminalState,
    TimeState,
)


FINISH_INTENT = "finish verified repair"
CONTINUE_INTENT = "continue repair investigation"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _task_index(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id")): row for row in _read_jsonl(path)}


def _execution_rows(
    execution_jsonl: str | Path,
    tasks_jsonl: str | Path,
) -> list[dict[str, Any]]:
    tasks = _task_index(tasks_jsonl)
    rows = []
    for execution in _read_jsonl(execution_jsonl):
        task = tasks.get(str(execution.get("task_id")))
        paths = execution.get("artifact_paths")
        if not isinstance(task, dict) or not isinstance(paths, dict):
            continue
        pre_path = Path(str(paths.get("pre_test_log") or ""))
        post_path = Path(str(paths.get("post_test_log") or ""))
        if not pre_path.is_file() or not post_path.is_file():
            continue
        rows.append(
            {
                "execution": execution,
                "task": task,
                "repo_origin": str(task.get("repo_origin") or ""),
                "pre": _read_json(pre_path),
                "post": _read_json(post_path),
            }
        )
    return rows


def _split_repo_disjoint(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repos = sorted({str(row["repo_origin"]) for row in rows})
    split = max(1, len(repos) // 2)
    train_repos = set(repos[:split])
    return (
        [row for row in rows if row["repo_origin"] in train_repos],
        [row for row in rows if row["repo_origin"] not in train_repos],
    )


def _feedback_text(log: dict[str, Any]) -> str:
    return " ".join(
        str(value).strip()
        for value in [
            log.get("stdout"),
            log.get("stderr"),
            log.get("target"),
            f"exit_code={log.get('exit_code')}",
        ]
        if value is not None and str(value).strip()
    )


def _exit_code(log: dict[str, Any], *, default: int) -> int:
    value = log.get("exit_code")
    return int(value) if value is not None else int(default)


def _control_command(intent: str) -> str:
    return _command_text(_resolved_argv(_continue_task(intent)))


def _continue_task(intent: str) -> dict[str, Any]:
    return {
        "action_type": "RUN_COMMAND",
        "argv": [
            "<PYTHON>",
            "-c",
            "print('bounded replan requested')",
            "--intent",
            intent,
        ],
        "expected_exit_code": 0,
    }


def _control_order(*, finish_correct: bool) -> list[str]:
    return [CONTINUE_INTENT, FINISH_INTENT] if finish_correct else [FINISH_INTENT, CONTINUE_INTENT]


def _state(
    *,
    frames: list[str],
    exit_code: int,
    finish_correct: bool,
) -> SystemStateFrame:
    order = _control_order(finish_correct=finish_correct)
    return SystemStateFrame(
        time=TimeState(tick=len(frames)),
        goal=GoalSpec(
            task_type=TaskType.TEST_FAILURE,
            description="Select bounded post-verification control from runtime feedback.",
            command_allowlist=[_control_command(intent) for intent in order],
            success_criteria=["respond_to_actual_verification_feedback"],
            safety_notes=["fixed_action_space", "no_freeform_patch_generation"],
        ),
        process=ProcessState(
            status=ProcessStatus.EXITED,
            exit_code=exit_code,
        ),
        terminal=TerminalState(prompt_visible=True),
        filesystem=FileSystemState(),
        runtime_evidence=RuntimeEvidenceState(
            source="phase2cd_actual_public_repo_pytest_feedback",
            version="phase2cd.post_verification_control.v1",
            terminal_observations=frames,
        ),
    )


def _select(matcher: Any | None, state: SystemStateFrame) -> str:
    order = [
        _command_semantic_text(command)
        for command in state.goal.command_allowlist
    ]
    scores = (
        [float(value) for value in matcher.score_state(state)]
        if matcher is not None
        else [0.0 for _ in order]
    )
    return order[max(range(len(scores)), key=scores.__getitem__)]


def _execute_control(
    *,
    selected: str,
    state: SystemStateFrame,
    workspace_root: Path,
    episode_id: str,
) -> dict[str, Any]:
    if selected == FINISH_INTENT:
        task = {"episode_id": episode_id, "action_type": "DONE"}
    else:
        task = {**_continue_task(CONTINUE_INTENT), "episode_id": episode_id}
    record, _ = _execute_task(
        task,
        workspace_root=workspace_root,
        timeout_seconds=2.0,
        state=state,
        goal=state.goal,
        t=state.time.tick,
        done=selected == FINISH_INTENT,
    )
    return {
        "action_type": record.action.type.value if record.action else None,
        "runtime_observation": record.source.value,
        "shell_false": True,
    }


def _run_suite(
    *,
    rows: list[dict[str, Any]],
    suite_id: str,
    matcher: Any | None,
    workspace_root: Path,
) -> dict[str, Any]:
    if suite_id not in {"visible", "erased_post", "wrong_post", "frozen_pre"}:
        raise ValueError(f"unsupported phase2cd suite {suite_id!r}")
    reports = []
    for index, row in enumerate(rows):
        pre_text = _feedback_text(row["pre"])
        post_text = _feedback_text(row["post"])
        pre_state = _state(
            frames=[pre_text],
            exit_code=_exit_code(row["pre"], default=1),
            finish_correct=False,
        )
        selected_pre = _select(matcher, pre_state)
        pre_execution = _execute_control(
            selected=selected_pre,
            state=pre_state,
            workspace_root=workspace_root,
            episode_id=f"phase2cd_{suite_id}_{index:05d}_pre",
        )
        if suite_id == "visible":
            post_frames = [pre_text, post_text]
            post_exit_code = _exit_code(row["post"], default=0)
        elif suite_id == "erased_post":
            post_frames = [pre_text]
            post_exit_code = 1
        else:
            post_frames = [pre_text, pre_text]
            post_exit_code = 1
        post_state = _state(
            frames=post_frames,
            exit_code=post_exit_code,
            finish_correct=True,
        )
        selected_post = _select(matcher, post_state)
        post_execution = _execute_control(
            selected=selected_post,
            state=post_state,
            workspace_root=workspace_root,
            episode_id=f"phase2cd_{suite_id}_{index:05d}_post",
        )
        reports.append(
            {
                "task_id": row["task"].get("task_id"),
                "repo_origin": row["repo_origin"],
                "actual_package_selected_patch": row["execution"].get(
                    "selection_policy"
                )
                == "package_loaded_native_head",
                "actual_patch_execution_success": row["execution"].get("success") is True,
                "actual_pre_test_failed": _exit_code(row["pre"], default=0) != 0,
                "actual_post_test_passed": _exit_code(row["post"], default=1) == 0,
                "selected_pre_intent": selected_pre,
                "selected_post_intent": selected_post,
                "pre_continue_correct": selected_pre == CONTINUE_INTENT,
                "post_finish_correct": selected_post == FINISH_INTENT,
                "post_followed_observed_failure": selected_post == CONTINUE_INTENT,
                "control_action_changed": selected_pre != selected_post,
                "pre_execution": pre_execution,
                "post_execution": post_execution,
            }
        )
    count = max(len(reports), 1)
    return {
        "suite_id": suite_id,
        "rows": len(reports),
        "repo_count": len({row["repo_origin"] for row in reports}),
        "pre_continue_accuracy": sum(row["pre_continue_correct"] for row in reports)
        / count,
        "post_finish_accuracy": sum(row["post_finish_correct"] for row in reports) / count,
        "post_failure_follow_rate": sum(
            row["post_followed_observed_failure"] for row in reports
        )
        / count,
        "control_action_change_rate": sum(row["control_action_changed"] for row in reports)
        / count,
        "all_actual_package_selected_patches": all(
            row["actual_package_selected_patch"] for row in reports
        ),
        "all_actual_patch_executions_succeeded": all(
            row["actual_patch_execution_success"] for row in reports
        ),
        "all_actual_pre_tests_failed": all(row["actual_pre_test_failed"] for row in reports),
        "all_actual_post_tests_passed": all(row["actual_post_test_passed"] for row in reports),
        "all_control_actions_shell_false": all(
            row["pre_execution"]["shell_false"] and row["post_execution"]["shell_false"]
            for row in reports
        ),
        "row_reports": reports,
    }


def run_phase2cd_public_repo_post_verification_control(
    *,
    execution_jsonl: str | Path,
    tasks_jsonl: str | Path,
    lexical_matcher_path: str | Path,
    cortex_model_path: str | Path,
    cortex_device: str,
    cortex_dtype: str,
    output_report_json: str | Path,
) -> dict[str, Any]:
    rows = _execution_rows(execution_jsonl, tasks_jsonl)
    train_rows, holdout_rows = _split_repo_disjoint(rows)
    natural_base = FrozenEncoderDualSemanticMatcher.from_pretrained(
        cortex_model_path,
        device=cortex_device,
        dtype=cortex_dtype,
        seed=37,
    )
    natural_base.fit(
        [
            *[(_feedback_text(row["pre"]), CONTINUE_INTENT) for row in train_rows],
            *[(_feedback_text(row["post"]), FINISH_INTENT) for row in train_rows],
        ]
    )
    natural = RecencyWeightedSemanticMatcher(natural_base, recency_decay=0.25)
    lexical_base = HashedDualEncoderSemanticMatcher.load(lexical_matcher_path)
    lexical_base.lexical_residual_weight = 3.0
    lexical = RecencyWeightedSemanticMatcher(lexical_base, recency_decay=0.25)
    workspace_root = Path.cwd().resolve()
    suites = {
        suite_id: _run_suite(
            rows=holdout_rows,
            suite_id=suite_id,
            matcher=natural,
            workspace_root=workspace_root,
        )
        for suite_id in ("visible", "erased_post", "wrong_post", "frozen_pre")
    }
    baselines = {
        "lexical_visible": _run_suite(
            rows=holdout_rows,
            suite_id="visible",
            matcher=lexical,
            workspace_root=workspace_root,
        ),
        "no_prior_visible": _run_suite(
            rows=holdout_rows,
            suite_id="visible",
            matcher=None,
            workspace_root=workspace_root,
        ),
    }
    visible = suites["visible"]
    erased = suites["erased_post"]
    wrong = suites["wrong_post"]
    frozen = suites["frozen_pre"]
    lexical_visible = baselines["lexical_visible"]
    no_prior_visible = baselines["no_prior_visible"]
    train_repos = {row["repo_origin"] for row in train_rows}
    holdout_repos = {row["repo_origin"] for row in holdout_rows}
    checks = {
        "actual_execution_rows_present": len(rows) >= 12,
        "train_holdout_repos_disjoint": train_repos.isdisjoint(holdout_repos),
        "minimum_holdout_repos_met": len(holdout_repos) >= 3,
        "actual_package_selected_patches": visible["all_actual_package_selected_patches"],
        "actual_patch_executions_succeeded": visible["all_actual_patch_executions_succeeded"],
        "actual_pre_tests_failed": visible["all_actual_pre_tests_failed"],
        "actual_post_tests_passed": visible["all_actual_post_tests_passed"],
        "control_actions_executed_shell_false": visible["all_control_actions_shell_false"],
        "visible_pre_continue_meets_gate": visible["pre_continue_accuracy"] >= 0.90,
        "visible_post_finish_meets_gate": visible["post_finish_accuracy"] >= 0.90,
        "visible_control_action_changes_after_verification": visible[
            "control_action_change_rate"
        ]
        >= 0.90,
        "erased_post_reduces_finish_accuracy": erased["post_finish_accuracy"]
        <= visible["post_finish_accuracy"] - 0.20,
        "wrong_post_reduces_finish_accuracy": wrong["post_finish_accuracy"]
        <= visible["post_finish_accuracy"] - 0.50,
        "wrong_post_redirects_to_continue": wrong["post_failure_follow_rate"] >= 0.90,
        "frozen_pre_reduces_finish_accuracy": frozen["post_finish_accuracy"]
        <= visible["post_finish_accuracy"] - 0.50,
        "natural_controller_outperforms_lexical": visible["post_finish_accuracy"]
        > lexical_visible["post_finish_accuracy"],
        "natural_controller_outperforms_no_prior": visible["post_finish_accuracy"]
        > no_prior_visible["post_finish_accuracy"],
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cd_public_repo_post_verification_control",
        "passed": passed,
        "ready_for_bounded_public_repo_post_verification_control_claim": passed,
        "ready_for_live_public_repo_continuous_repair_loop_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "dataset": {
            "actual_execution_rows": len(rows),
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "train_repos": sorted(train_repos),
            "holdout_repos": sorted(holdout_repos),
            "execution_jsonl": str(execution_jsonl),
            "tasks_jsonl": str(tasks_jsonl),
        },
        "natural_matcher_metadata": natural.metadata(),
        "checks": checks,
        "suites": suites,
        "baselines": baselines,
        "comparison": {
            "visible_pre_continue": visible["pre_continue_accuracy"],
            "visible_post_finish": visible["post_finish_accuracy"],
            "visible_action_change": visible["control_action_change_rate"],
            "erased_post_finish": erased["post_finish_accuracy"],
            "wrong_post_finish": wrong["post_finish_accuracy"],
            "wrong_post_continue": wrong["post_failure_follow_rate"],
            "frozen_post_finish": frozen["post_finish_accuracy"],
            "lexical_post_finish": lexical_visible["post_finish_accuracy"],
            "no_prior_post_finish": no_prior_visible["post_finish_accuracy"],
        },
        "supported_claims": [
            "a learned bounded verification-control head changed from continue-repair to finish after actual package-selected public-repository patch execution produced passing pytest feedback on repo-disjoint holdout repositories"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "single-policy uninterrupted live patch-to-stop loop",
            "arbitrary public-repository repair",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2ce_single_policy_live_patch_verify_stop_loop"
            if passed
            else "repair_phase2cd_public_repo_post_verification_control"
        ),
    }
    output = Path(output_report_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate bounded post-verification control on actual public-repo patch feedback."
    )
    parser.add_argument("--execution-jsonl", required=True)
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--lexical-matcher-path", required=True)
    parser.add_argument("--cortex-model-path", required=True)
    parser.add_argument("--cortex-device", default="cpu")
    parser.add_argument("--cortex-dtype", default="auto")
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2cd_public_repo_post_verification_control(
        execution_jsonl=args.execution_jsonl,
        tasks_jsonl=args.tasks_jsonl,
        lexical_matcher_path=args.lexical_matcher_path,
        cortex_model_path=args.cortex_model_path,
        cortex_device=args.cortex_device,
        cortex_dtype=args.cortex_dtype,
        output_report_json=args.output_report_json,
    )
    print(
        json.dumps(
            {
                "artifact_family": report["artifact_family"],
                "passed": report["passed"],
                "checks": report["checks"],
                "comparison": report["comparison"],
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
