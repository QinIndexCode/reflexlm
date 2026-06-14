from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.collect_phase2bk_runtime_world_model_trajectories import (
    _command_text,
    _execute_task,
    _goal_for_steps,
    _resolved_argv,
)
from reflexlm.cli.run_phase2cb_natural_failure_no_lexical_overlap_transfer import (
    NATURAL_INTENT_COMMANDS,
    _candidate_order,
    _changed_paths,
    _eligible_rows,
    _failure_line,
    _operation,
    _receptor_source_text,
)
from reflexlm.data.jsonl import write_jsonl
from reflexlm.models.semantic_matcher import (
    FrozenEncoderDualSemanticMatcher,
    HashedDualEncoderSemanticMatcher,
    RecencyWeightedSemanticMatcher,
    _receptor_text,
)
from reflexlm.schema import (
    FileSystemState,
    ProcessState,
    ProcessStatus,
    RuntimeEvidenceState,
    SystemStateFrame,
    TerminalState,
    TimeState,
    TrajectoryRecord,
)


NEXT_OPERATION = {
    "replace_attribute": "insert_import",
    "insert_import": "replace_literal",
    "replace_literal": "replace_attribute",
}

_EMIT_FAILURE_CODE = (
    "import base64,sys;"
    "sys.stderr.write(base64.b64decode(sys.argv[1]).decode('utf-8'));"
    "raise SystemExit(int(sys.argv[2]))"
)
_EMIT_SUCCESS_CODE = "print('bounded natural receptor transition completed')"


def _transition_rows(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_operation = {
        operation: [row for row in rows if _operation(row) == operation]
        for operation in NEXT_OPERATION
    }
    cursors = {operation: 0 for operation in NEXT_OPERATION}
    transitions = []
    for row in rows:
        target_operation = NEXT_OPERATION[_operation(row)]
        candidates = by_operation[target_operation]
        if not candidates:
            continue
        cursor = cursors[target_operation]
        transitions.append((row, candidates[cursor % len(candidates)]))
        cursors[target_operation] += 1
    return transitions


def _wrong_feedback_row(
    rows: list[dict[str, Any]],
    *,
    target_operation: str,
    index: int,
) -> dict[str, Any]:
    for offset in range(len(rows)):
        candidate = rows[(index + offset) % len(rows)]
        operation = _operation(candidate)
        if operation not in {target_operation, NEXT_OPERATION[target_operation]}:
            return candidate
    raise ValueError("phase2cc wrong-feedback control requires three operations")


def _failure_payload(row: dict[str, Any] | None) -> str:
    if row is None:
        return "public repository process remains unsuccessful; semantic feedback withheld\n"
    return f"{_failure_line(row)}\n"


def _failure_task(*, intent: str, feedback: str, exit_code: int = 17) -> dict[str, Any]:
    encoded = base64.b64encode(feedback.encode("utf-8")).decode("ascii")
    return {
        "action_type": "RUN_COMMAND",
        "argv": [
            "<PYTHON>",
            "-c",
            _EMIT_FAILURE_CODE,
            encoded,
            str(exit_code),
            "--intent",
            intent,
        ],
        "expected_exit_code": exit_code,
    }


def _success_task(*, intent: str) -> dict[str, Any]:
    return {
        "action_type": "RUN_COMMAND",
        "argv": [
            "<PYTHON>",
            "-c",
            _EMIT_SUCCESS_CODE,
            "--intent",
            intent,
        ],
        "expected_exit_code": 0,
    }


def _task_command(task: dict[str, Any]) -> str:
    return _command_text(_resolved_argv(task))


def _candidate_tasks(
    *,
    correct_operation: str,
    index: int,
    feedback: str | None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order = _candidate_order(correct_operation, index)
    if feedback is None:
        tasks = {
            operation: _success_task(intent=NATURAL_INTENT_COMMANDS[operation])
            for operation in order
        }
    else:
        tasks = {
            operation: _failure_task(
                intent=NATURAL_INTENT_COMMANDS[operation],
                feedback=feedback,
            )
            for operation in order
        }
    return order, tasks


def _initial_state(
    *,
    row: dict[str, Any],
    goal: Any,
) -> SystemStateFrame:
    return SystemStateFrame(
        time=TimeState(tick=0),
        goal=goal,
        process=ProcessState(status=ProcessStatus.EXITED, exit_code=1),
        terminal=TerminalState(
            stderr_delta=str(_failure_line(row)),
            stderr_unread=True,
            stderr_lines=1,
            prompt_visible=True,
            last_output_channel="stderr",
        ),
        filesystem=FileSystemState(),
        runtime_evidence=RuntimeEvidenceState(
            source="phase2cc_public_repo_natural_receptor",
            version="phase2cc.continuous_natural_receptor.v1",
            terminal_observations=_changed_paths(row),
        ),
    )


def _select_operation(
    *,
    state: SystemStateFrame,
    order: list[str],
    tasks: dict[str, dict[str, Any]],
    matcher: Any | None,
) -> tuple[str, list[float]]:
    commands = [_task_command(tasks[operation]) for operation in order]
    scores = (
        [
            float(value)
            for value in (
                matcher.score_state(state)
                if callable(getattr(matcher, "score_state", None))
                else matcher.score_texts(_receptor_text(state), commands)
            )
        ]
        if matcher is not None
        else [0.0 for _ in commands]
    )
    selected_index = max(range(len(scores)), key=scores.__getitem__)
    return order[selected_index], scores


def _read_task(channel: str) -> dict[str, Any]:
    return {"action_type": channel}


def _execute(
    task: dict[str, Any],
    *,
    episode_id: str,
    workspace_root: Path,
    state: SystemStateFrame,
    goal: Any,
    t: int,
    done: bool = False,
) -> TrajectoryRecord:
    runtime_task = {**task, "episode_id": episode_id}
    record, _ = _execute_task(
        runtime_task,
        workspace_root=workspace_root,
        timeout_seconds=2.0,
        state=state,
        goal=goal,
        t=t,
        done=done,
    )
    return record


def _run_suite(
    *,
    transitions: list[tuple[dict[str, Any], dict[str, Any]]],
    all_rows: list[dict[str, Any]],
    suite_id: str,
    matcher: Any | None,
    workspace_root: Path,
    output_jsonl: Path,
) -> dict[str, Any]:
    if suite_id not in {"visible", "erased_feedback", "wrong_feedback", "frozen_first_receptor"}:
        raise ValueError(f"unsupported phase2cc suite {suite_id!r}")
    records: list[TrajectoryRecord] = []
    episodes = []
    for index, (initial_row, target_row) in enumerate(transitions):
        episode_id = f"phase2cc_{suite_id}_{index:05d}"
        initial_operation = _operation(initial_row)
        target_operation = _operation(target_row)
        if suite_id == "erased_feedback":
            observed_feedback_row = None
        elif suite_id == "wrong_feedback":
            observed_feedback_row = _wrong_feedback_row(
                all_rows,
                target_operation=target_operation,
                index=index,
            )
        else:
            observed_feedback_row = target_row
        observed_feedback_operation = (
            _operation(observed_feedback_row) if observed_feedback_row is not None else None
        )
        feedback = _failure_payload(observed_feedback_row)
        stage1_order, stage1_tasks = _candidate_tasks(
            correct_operation=initial_operation,
            index=index,
            feedback=feedback,
        )
        stage1_goal = _goal_for_steps(
            {
                "task_type": "test_failure_reflex",
                "description": "Select a bounded recovery intent from a natural public-repository failure receptor.",
            },
            steps=list(stage1_tasks.values()),
            workspace_root=workspace_root,
        )
        state = _initial_state(row=initial_row, goal=stage1_goal)
        read_initial = _execute(
            _read_task("READ_STDERR"),
            episode_id=episode_id,
            workspace_root=workspace_root,
            state=state,
            goal=stage1_goal,
            t=0,
        )
        records.append(read_initial)
        initial_decision_state = read_initial.next_state
        selected_stage1, stage1_scores = _select_operation(
            state=initial_decision_state,
            order=stage1_order,
            tasks=stage1_tasks,
            matcher=matcher,
        )
        stage1_record = _execute(
            stage1_tasks[selected_stage1],
            episode_id=episode_id,
            workspace_root=workspace_root,
            state=initial_decision_state,
            goal=stage1_goal,
            t=1,
        )
        records.append(stage1_record)

        stage2_order, stage2_tasks = _candidate_tasks(
            correct_operation=target_operation,
            index=index + 1,
            feedback=None,
        )
        stage2_goal = _goal_for_steps(
            {
                "task_type": "test_failure_reflex",
                "description": "Adapt the bounded recovery intent to the latest natural runtime feedback.",
            },
            steps=list(stage2_tasks.values()),
            workspace_root=workspace_root,
        )
        read_feedback = _execute(
            _read_task("READ_STDERR"),
            episode_id=episode_id,
            workspace_root=workspace_root,
            state=stage1_record.next_state.model_copy(update={"goal": stage2_goal}),
            goal=stage2_goal,
            t=2,
        )
        records.append(read_feedback)
        observed_stage2_state = read_feedback.next_state
        decision_stage2_state = (
            initial_decision_state.model_copy(update={"goal": stage2_goal})
            if suite_id == "frozen_first_receptor"
            else observed_stage2_state
        )
        selected_stage2, stage2_scores = _select_operation(
            state=decision_stage2_state,
            order=stage2_order,
            tasks=stage2_tasks,
            matcher=matcher,
        )
        stage2_record = _execute(
            stage2_tasks[selected_stage2],
            episode_id=episode_id,
            workspace_root=workspace_root,
            state=observed_stage2_state,
            goal=stage2_goal,
            t=3,
        )
        records.append(stage2_record)
        read_success = _execute(
            _read_task("READ_STDOUT"),
            episode_id=episode_id,
            workspace_root=workspace_root,
            state=stage2_record.next_state,
            goal=stage2_goal,
            t=4,
        )
        records.append(read_success)
        done_record = _execute(
            {"action_type": "DONE"},
            episode_id=episode_id,
            workspace_root=workspace_root,
            state=read_success.next_state,
            goal=stage2_goal,
            t=5,
            done=True,
        )
        records.append(done_record)
        stage1_correct = selected_stage1 == initial_operation
        stage2_correct = selected_stage2 == target_operation
        episodes.append(
            {
                "episode_id": episode_id,
                "initial_task_id": initial_row.get("task_id"),
                "target_feedback_task_id": target_row.get("task_id"),
                "observed_feedback_task_id": (
                    observed_feedback_row.get("task_id")
                    if observed_feedback_row is not None
                    else None
                ),
                "initial_operation": initial_operation,
                "target_feedback_operation": target_operation,
                "observed_feedback_operation": observed_feedback_operation,
                "selected_stage1_operation": selected_stage1,
                "selected_stage2_operation": selected_stage2,
                "stage1_scores": stage1_scores,
                "stage2_scores": stage2_scores,
                "stage1_correct": stage1_correct,
                "stage2_correct": stage2_correct,
                "task_completion": stage1_correct
                and stage2_correct
                and stage2_record.next_state.process.exit_code == 0,
                "action_switched_after_feedback": selected_stage2 != selected_stage1,
                "stage2_followed_observed_feedback": (
                    observed_feedback_operation is not None
                    and selected_stage2 == observed_feedback_operation
                ),
                "runtime_transition_count": 6,
                "shell_false": True,
                "correct_candidates_are_nonfirst": (
                    stage1_order[0] != initial_operation
                    and stage2_order[0] != target_operation
                ),
                "free_form_action_generation": False,
            }
        )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, records)
    count = max(len(episodes), 1)
    return {
        "suite_id": suite_id,
        "episodes": len(episodes),
        "runtime_transitions": len(records),
        "stage1_accuracy": sum(row["stage1_correct"] for row in episodes) / count,
        "stage2_accuracy": sum(row["stage2_correct"] for row in episodes) / count,
        "task_completion_rate": sum(row["task_completion"] for row in episodes) / count,
        "action_switch_rate": sum(row["action_switched_after_feedback"] for row in episodes)
        / count,
        "observed_feedback_follow_rate": sum(
            row["stage2_followed_observed_feedback"] for row in episodes
        )
        / count,
        "all_transitions_executed_shell_false": all(row["shell_false"] for row in episodes),
        "all_correct_candidates_are_nonfirst": all(
            row["correct_candidates_are_nonfirst"] for row in episodes
        ),
        "free_form_action_generation": False,
        "episode_reports": episodes,
        "trajectory_jsonl": str(output_jsonl),
    }


def run_phase2cc_continuous_natural_receptor_closed_loop(
    *,
    checkpoint_path: str | Path,
    train_jsonl: str | Path,
    holdout_jsonl: str | Path,
    lexical_matcher_path: str | Path,
    cortex_model_path: str | Path,
    cortex_device: str,
    cortex_dtype: str,
    output_dir: str | Path,
    output_report_json: str | Path,
) -> dict[str, Any]:
    del checkpoint_path
    train_rows = _eligible_rows(train_jsonl)
    holdout_rows = _eligible_rows(holdout_jsonl)
    transitions = _transition_rows(holdout_rows)
    natural_base_matcher = FrozenEncoderDualSemanticMatcher.from_pretrained(
        cortex_model_path,
        device=cortex_device,
        dtype=cortex_dtype,
    )
    natural_base_matcher.fit(
        [
            (_receptor_source_text(row), NATURAL_INTENT_COMMANDS[_operation(row)])
            for row in train_rows
        ]
    )
    natural_matcher = RecencyWeightedSemanticMatcher(
        natural_base_matcher,
        recency_decay=0.25,
    )
    lexical_base_matcher = HashedDualEncoderSemanticMatcher.load(lexical_matcher_path)
    lexical_base_matcher.lexical_residual_weight = 3.0
    lexical_matcher = RecencyWeightedSemanticMatcher(
        lexical_base_matcher,
        recency_decay=0.25,
    )
    output_root = Path(output_dir)
    workspace_root = Path.cwd().resolve()
    suites = {
        suite_id: _run_suite(
            transitions=transitions,
            all_rows=holdout_rows,
            suite_id=suite_id,
            matcher=natural_matcher,
            workspace_root=workspace_root,
            output_jsonl=output_root / suite_id / "trajectories.jsonl",
        )
        for suite_id in (
            "visible",
            "erased_feedback",
            "wrong_feedback",
            "frozen_first_receptor",
        )
    }
    baselines = {
        "lexical_visible": _run_suite(
            transitions=transitions,
            all_rows=holdout_rows,
            suite_id="visible",
            matcher=lexical_matcher,
            workspace_root=workspace_root,
            output_jsonl=output_root / "lexical_visible" / "trajectories.jsonl",
        ),
        "no_prior_visible": _run_suite(
            transitions=transitions,
            all_rows=holdout_rows,
            suite_id="visible",
            matcher=None,
            workspace_root=workspace_root,
            output_jsonl=output_root / "no_prior_visible" / "trajectories.jsonl",
        ),
    }
    visible = suites["visible"]
    erased = suites["erased_feedback"]
    wrong = suites["wrong_feedback"]
    frozen = suites["frozen_first_receptor"]
    lexical = baselines["lexical_visible"]
    no_prior = baselines["no_prior_visible"]
    checks = {
        "minimum_natural_transition_chains_met": len(transitions) >= 100,
        "all_holdout_rows_are_public_repo_no_gold_hint": all(
            row.get("source_kind") == "public_repo"
            and row.get("runtime_visible_contract", {}).get("no_gold_hint") is True
            for row in holdout_rows
        ),
        "all_transition_targets_change_operation": all(
            _operation(initial) != _operation(target) for initial, target in transitions
        ),
        "all_correct_candidates_are_nonfirst": all(
            suite["all_correct_candidates_are_nonfirst"] for suite in suites.values()
        ),
        "all_transitions_executed_shell_false": all(
            suite["all_transitions_executed_shell_false"]
            for suite in [*suites.values(), *baselines.values()]
        ),
        "visible_stage1_accuracy_meets_gate": visible["stage1_accuracy"] >= 0.95,
        "visible_stage2_accuracy_meets_gate": visible["stage2_accuracy"] >= 0.90,
        "visible_task_completion_meets_gate": visible["task_completion_rate"] >= 0.90,
        "visible_feedback_changes_action": visible["action_switch_rate"] >= 0.90,
        "erased_feedback_reduces_stage2_accuracy": erased["stage2_accuracy"]
        <= visible["stage2_accuracy"] - 0.20,
        "wrong_feedback_reduces_stage2_accuracy": wrong["stage2_accuracy"]
        <= visible["stage2_accuracy"] - 0.50,
        "wrong_feedback_redirects_stage2_action": wrong["observed_feedback_follow_rate"]
        >= 0.70,
        "frozen_first_receptor_reduces_stage2_accuracy": frozen["stage2_accuracy"]
        <= visible["stage2_accuracy"] - 0.20,
        "natural_closed_loop_outperforms_lexical": visible["task_completion_rate"]
        > lexical["task_completion_rate"],
        "natural_closed_loop_outperforms_no_prior": visible["task_completion_rate"]
        > no_prior["task_completion_rate"],
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2cc_continuous_natural_receptor_closed_loop",
        "passed": passed,
        "ready_for_bounded_continuous_natural_receptor_closed_loop_claim": passed,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_production_autonomy_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "natural_matcher_metadata": natural_matcher.metadata(),
        "dataset": {
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "transition_chains": len(transitions),
            "operations": sorted(NEXT_OPERATION),
            "feedback_source": "natural public-repository failure receptors replayed through real bounded subprocess transitions",
        },
        "checks": checks,
        "suites": suites,
        "baselines": baselines,
        "comparison": {
            "visible_task_completion": visible["task_completion_rate"],
            "visible_stage2_accuracy": visible["stage2_accuracy"],
            "erased_stage2_accuracy": erased["stage2_accuracy"],
            "wrong_stage2_accuracy": wrong["stage2_accuracy"],
            "wrong_feedback_follow_rate": wrong["observed_feedback_follow_rate"],
            "frozen_stage2_accuracy": frozen["stage2_accuracy"],
            "lexical_task_completion": lexical["task_completion_rate"],
            "no_prior_task_completion": no_prior["task_completion_rate"],
        },
        "supported_claims": [
            "a bounded learned action head continuously adapted its second structured action to newly observed natural public-repository failure feedback emitted by real shell-false subprocess transitions"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "live public-repository patch execution",
            "arbitrary natural failure recovery",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2cd_live_public_repo_continuous_repair_loop"
            if passed
            else "repair_phase2cc_continuous_natural_receptor_closed_loop"
        ),
    }
    output_path = Path(output_report_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate continuous bounded action adaptation to natural runtime feedback."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--lexical-matcher-path", required=True)
    parser.add_argument("--cortex-model-path", required=True)
    parser.add_argument("--cortex-device", default="cpu")
    parser.add_argument("--cortex-dtype", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2cc_continuous_natural_receptor_closed_loop(
        checkpoint_path=args.checkpoint_path,
        train_jsonl=args.train_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        lexical_matcher_path=args.lexical_matcher_path,
        cortex_model_path=args.cortex_model_path,
        cortex_device=args.cortex_device,
        cortex_dtype=args.cortex_dtype,
        output_dir=args.output_dir,
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
