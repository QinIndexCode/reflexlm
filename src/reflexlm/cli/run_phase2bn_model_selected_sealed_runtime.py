from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

from reflexlm.cli.collect_phase2bk_runtime_world_model_trajectories import (
    SUPPORTED_ACTIONS,
    _command_text,
    _execute_task,
    _goal_for_steps,
    _initial_state,
    _resolved_argv,
)
from reflexlm.data.jsonl import write_jsonl
from reflexlm.eval import SequenceModelPolicy
from reflexlm.schema import (
    ActionDecision,
    ActionType,
    FileSystemState,
    ProcessState,
    RuntimeEvidenceState,
    SystemStateFrame,
    TerminalState,
    TrajectoryRecord,
)
from reflexlm.train import load_model_checkpoint


def _action_key(action: ActionDecision) -> tuple[str, str | None, str | None]:
    return action.type.value, action.command, action.file_target


def _step_action(step: dict[str, Any]) -> ActionDecision:
    action_type = ActionType(str(step["action_type"]))
    argv = _resolved_argv(step)
    return ActionDecision(
        type=action_type,
        command=_command_text(argv) if action_type == ActionType.RUN_COMMAND else None,
        file_target=(
            str(step["file_target"]).replace("\\", "/")
            if action_type == ActionType.READ_FILE
            else None
        ),
        confidence=1.0,
    )


def _allowed_step_map(
    steps: list[dict[str, Any]],
    *,
    ambient_observation_actions: list[str] | None = None,
) -> dict[tuple[str, str | None, str | None], dict[str, Any]]:
    allowed: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
    for step in steps:
        action = _step_action(step)
        allowed.setdefault(_action_key(action), step)
    safe_ambient_actions = {ActionType.READ_STDOUT, ActionType.READ_STDERR}
    for value in ambient_observation_actions or []:
        action_type = ActionType(value)
        if action_type not in safe_ambient_actions:
            raise ValueError(
                "ambient observation authorization is limited to read-only terminal "
                f"receptors, got: {action_type.value}"
            )
        action = ActionDecision(type=action_type, confidence=1.0)
        allowed.setdefault(
            _action_key(action),
            {
                "action_type": action_type.value,
                "_phase2bn_ambient_observation": True,
            },
        )
    return allowed


def _required_completion_actions(
    steps: list[dict[str, Any]],
) -> Counter[tuple[str, str | None, str | None]]:
    required_types = {
        ActionType.RUN_COMMAND,
        ActionType.READ_STDOUT,
        ActionType.READ_STDERR,
        ActionType.READ_FILE,
    }
    return Counter(
        _action_key(action)
        for action in (_step_action(step) for step in steps)
        if action.type in required_types
    )


def _completion_actions_satisfied(
    required: Counter[tuple[str, str | None, str | None]],
    executed: Counter[tuple[str, str | None, str | None]],
) -> bool:
    return all(executed[key] >= count for key, count in required.items())


def _failure_recovery_gate_status(
    *,
    failure_episode_count: int,
    failure_recovery_count: int,
    minimum_success_rate: float = 0.80,
) -> tuple[bool, bool, float]:
    if failure_episode_count == 0:
        return True, False, 1.0
    success_rate = failure_recovery_count / failure_episode_count
    return success_rate >= minimum_success_rate, True, success_rate


def _apply_initial_state_overrides(
    state: SystemStateFrame,
    episode: dict[str, Any],
) -> SystemStateFrame:
    """Apply bounded manifest-provided receptor state without changing the goal."""
    overrides = episode.get("initial_state")
    if overrides is None:
        return state
    if not isinstance(overrides, dict):
        raise ValueError("episode initial_state must be an object")
    allowed_models = {
        "process": ProcessState,
        "terminal": TerminalState,
        "filesystem": FileSystemState,
        "runtime_evidence": RuntimeEvidenceState,
    }
    unknown_domains = sorted(set(overrides) - set(allowed_models))
    if unknown_domains:
        raise ValueError(
            "episode initial_state contains unsupported domains: "
            + ", ".join(unknown_domains)
        )
    updates: dict[str, Any] = {}
    for domain, model_type in allowed_models.items():
        domain_override = overrides.get(domain)
        if domain_override is None:
            continue
        if not isinstance(domain_override, dict):
            raise ValueError(f"episode initial_state.{domain} must be an object")
        current = getattr(state, domain).model_dump(mode="python")
        updates[domain] = model_type.model_validate({**current, **domain_override})
    return state.model_copy(update=updates)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[max(0, min(index, len(ordered) - 1))]


def _episode_contract(
    episode: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    bool | None,
]:
    permissions = episode.get("permissions")
    if permissions is not None:
        if not isinstance(permissions, list) or not permissions:
            raise ValueError("episode permissions must be a non-empty list")
        completion_requirements = episode.get("completion_requirements")
        if not isinstance(completion_requirements, list) or not completion_requirements:
            raise ValueError(
                "contract episode requires non-empty completion_requirements"
            )
        expected_sequence = episode.get("expected_sequence", [])
        if not isinstance(expected_sequence, list):
            raise ValueError("expected_sequence must be a list when provided")
        requires_failure = episode.get("requires_failure")
        if requires_failure is not None and not isinstance(requires_failure, bool):
            raise ValueError("requires_failure must be boolean when provided")
        return permissions, completion_requirements, expected_sequence, requires_failure
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("each sealed runtime episode requires steps or permissions")
    return steps, steps, steps, None


def _task_for_selected_action(
    action: ActionDecision,
    *,
    allowed_steps: dict[tuple[str, str | None, str | None], dict[str, Any]],
    episode_id: str,
) -> dict[str, Any]:
    if action.type not in SUPPORTED_ACTIONS:
        raise ValueError(f"model selected unsupported sealed-runtime action: {action.type.value}")
    template = allowed_steps.get(_action_key(action))
    if template is None:
        raise ValueError(
            "model-selected action is outside the sealed manifest allowlist: "
            f"{_action_key(action)}"
        )
    task = dict(template)
    task["episode_id"] = episode_id
    task.pop("reward", None)
    return task


def run_phase2bn_model_selected_sealed_runtime(
    *,
    checkpoint_path: str | Path | None,
    manifest_json: str | Path,
    output_jsonl: str | Path,
    output_report_json: str | Path,
    timeout_seconds: float = 2.0,
    max_extra_steps: int = 3,
    policy_label: str = "phase2bn_model_selected_sealed_runtime",
    authorize_bounded_debug_cortex_recovery: bool = True,
    use_synaptic_motor_plan: bool = True,
    policy_instance: Any | None = None,
) -> dict[str, Any]:
    if policy_instance is None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required without policy_instance")
        model, vectorizer, checkpoint_payload = load_model_checkpoint(
            checkpoint_path,
            device="cpu",
        )
    else:
        model = None
        vectorizer = None
        checkpoint_payload = {}
    manifest_path = Path(manifest_json)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    workspace_root = Path(str(manifest["workspace_root"])).resolve()
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("sealed runtime manifest requires non-empty episodes")
    repetitions_per_episode = int(manifest.get("repetitions_per_episode", 1))

    all_records: list[TrajectoryRecord] = []
    episode_reports: list[dict[str, Any]] = []
    rejected_actions = 0
    executed_actions = 0
    ambient_observation_actions_executed = 0
    all_decision_latencies_ms: list[float] = []
    total_token_equivalent_cost = 0
    total_model_calls = 0
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("each sealed runtime episode must be an object")
        (
            permissions,
            completion_requirements,
            expected_sequence,
            explicit_requires_failure,
        ) = _episode_contract(episode)
        goal = _goal_for_steps(
            episode,
            steps=permissions,
            workspace_root=workspace_root,
        )
        ambient_observation_actions = list(
            dict.fromkeys(
                list(manifest.get("ambient_observation_actions", []))
                + list(episode.get("ambient_observation_actions", []))
            )
        )
        allowed_steps = _allowed_step_map(
            permissions,
            ambient_observation_actions=ambient_observation_actions,
        )
        expected_actions = [_step_action(step) for step in expected_sequence]
        required_completion_actions = _required_completion_actions(
            completion_requirements
        )
        repetitions = int(episode.get("repetitions", repetitions_per_episode))
        for repetition in range(repetitions):
            episode_id = (
                str(episode["episode_id"])
                if repetitions == 1
                else f"{episode['episode_id']}-sealed-{repetition:03d}"
            )
            state = _initial_state(
                goal=goal,
                watched_paths=goal.watched_paths,
                runtime_evidence=RuntimeEvidenceState(
                    source="phase2bn_model_selected_sealed_runtime",
                    version="phase2bn.model_selected_runtime.v1",
                    watched_files=goal.watched_paths,
                ),
            )
            state = _apply_initial_state_overrides(state, episode)
            if policy_instance is None:
                policy = SequenceModelPolicy(
                    model,
                    vectorizer,
                    policy_label=policy_label,
                    training_summary=checkpoint_payload.get("training_summary", {}),
                    authorize_bounded_debug_cortex_recovery=authorize_bounded_debug_cortex_recovery,
                    use_synaptic_motor_plan=use_synaptic_motor_plan,
                )
            else:
                policy = policy_instance
                reset_policy = getattr(policy, "reset", None)
                if callable(reset_policy):
                    reset_policy()
            selected_actions: list[dict[str, Any]] = []
            policy_debug_steps: list[dict[str, Any]] = []
            decision_latencies_ms: list[float] = []
            token_equivalent_cost = 0
            model_calls = 0
            expected_cursor = 0
            observed_failure = state.process.exit_code not in (None, 0)
            observed_timeout = state.process.interrupted
            observed_recovery_after_failure = False
            episode_rejected = False
            executed_action_counts: Counter[
                tuple[str, str | None, str | None]
            ] = Counter()
            unexpected_outcomes = 0
            episode_ambient_observation_actions = 0
            max_steps = int(
                episode.get(
                    "max_steps",
                    max(len(permissions), len(completion_requirements))
                    + max_extra_steps,
                )
            )
            if max_steps <= 0:
                raise ValueError("episode max_steps must be positive")
            for t in range(max_steps):
                before_token_cost = int(policy.stats.token_cost)
                before_model_calls = int(policy.stats.model_calls)
                decision_start = time.perf_counter()
                selected = policy.act(state)
                decision_latency_ms = (time.perf_counter() - decision_start) * 1000.0
                decision_latencies_ms.append(decision_latency_ms)
                all_decision_latencies_ms.append(decision_latency_ms)
                token_equivalent_cost += int(policy.stats.token_cost) - before_token_cost
                model_calls += int(policy.stats.model_calls) - before_model_calls
                selected_actions.append(selected.model_dump(mode="json"))
                policy_debug_steps.append(
                    {
                        **dict(policy.last_call),
                        "decision_latency_ms": round(decision_latency_ms, 6),
                        "token_equivalent_cost_delta": int(policy.stats.token_cost)
                        - before_token_cost,
                        "model_calls_delta": int(policy.stats.model_calls)
                        - before_model_calls,
                    }
                )
                if (
                    expected_cursor < len(expected_actions)
                    and _action_key(selected) == _action_key(expected_actions[expected_cursor])
                ):
                    expected_cursor += 1
                try:
                    task = _task_for_selected_action(
                        selected,
                        allowed_steps=allowed_steps,
                        episode_id=episode_id,
                    )
                except ValueError:
                    rejected_actions += 1
                    episode_rejected = True
                    break
                record, evidence = _execute_task(
                    task,
                    workspace_root=workspace_root,
                    timeout_seconds=timeout_seconds,
                    state=state,
                    goal=goal,
                    t=t,
                    done=selected.type == ActionType.DONE or t == max_steps - 1,
                )
                record = record.model_copy(
                    update={
                        "action": selected,
                        "reward": 1.0 if evidence["expected_outcome_matched"] else 0.0,
                    }
                )
                all_records.append(record)
                executed_actions += 1
                if task.get("_phase2bn_ambient_observation") is True:
                    episode_ambient_observation_actions += 1
                    ambient_observation_actions_executed += 1
                executed_action_counts[_action_key(selected)] += 1
                if not evidence["expected_outcome_matched"]:
                    unexpected_outcomes += 1
                exit_code = record.next_state.process.exit_code
                if exit_code not in (None, 0):
                    observed_failure = True
                if record.next_state.process.interrupted:
                    observed_timeout = True
                if (
                    observed_failure
                    and selected.type == ActionType.RUN_COMMAND
                    and exit_code == 0
                ):
                    observed_recovery_after_failure = True
                state = record.next_state
                if selected.type == ActionType.DONE:
                    break
            selected_done = bool(selected_actions) and selected_actions[-1]["type"] == ActionType.DONE.value
            strict_sequence_success = (
                expected_cursor == len(expected_actions) and selected_done
                if expected_actions
                else None
            )
            requires_failure = (
                explicit_requires_failure
                if explicit_requires_failure is not None
                else any(
                    int(permission.get("expected_exit_code", 0)) != 0
                    or permission.get("expected_timed_out") is True
                    for permission in permissions
                )
            )
            recovery_success = (
                observed_recovery_after_failure and selected_done
                if requires_failure
                else selected_done and not episode_rejected
            )
            completion_actions_satisfied = _completion_actions_satisfied(
                required_completion_actions,
                executed_action_counts,
            )
            task_completion_success = (
                completion_actions_satisfied
                and recovery_success
                and selected_done
                and not episode_rejected
                and unexpected_outcomes == 0
            )
            total_token_equivalent_cost += token_equivalent_cost
            total_model_calls += model_calls
            episode_reports.append(
                {
                    "episode_id": episode_id,
                    "base_episode_id": str(episode["episode_id"]),
                    "selected_actions": selected_actions,
                    "policy_debug_steps": policy_debug_steps,
                    "contract_mode": "permissions" if "permissions" in episode else "steps",
                    "expected_action_count": len(expected_actions),
                    "matched_expected_prefix": expected_cursor,
                    "strict_sequence_success": strict_sequence_success,
                    "required_completion_action_count": sum(
                        required_completion_actions.values()
                    ),
                    "completion_actions_satisfied": completion_actions_satisfied,
                    "unexpected_outcomes": unexpected_outcomes,
                    "ambient_observation_actions_executed": episode_ambient_observation_actions,
                    "task_completion_success": task_completion_success,
                    "requires_failure": requires_failure,
                    "observed_failure": observed_failure,
                    "observed_timeout": observed_timeout,
                    "observed_recovery_after_failure": observed_recovery_after_failure,
                    "recovery_success": recovery_success,
                    "selected_done": selected_done,
                    "rejected_action": episode_rejected,
                    "decision_count": len(decision_latencies_ms),
                    "mean_decision_latency_ms": round(
                        sum(decision_latencies_ms) / max(len(decision_latencies_ms), 1),
                        6,
                    ),
                    "p95_decision_latency_ms": round(
                        _percentile(decision_latencies_ms, 0.95),
                        6,
                    ),
                    "episode_policy_compute_latency_ms": round(
                        sum(decision_latencies_ms),
                        6,
                    ),
                    "token_equivalent_cost": token_equivalent_cost,
                    "model_calls": model_calls,
                }
            )

    write_jsonl(Path(output_jsonl), all_records)
    total_episodes = len(episode_reports)
    strict_sequence_rows = [
        row for row in episode_reports if row["strict_sequence_success"] is not None
    ]
    strict_successes = sum(row["strict_sequence_success"] for row in strict_sequence_rows)
    recovery_successes = sum(row["recovery_success"] for row in episode_reports)
    task_completion_successes = sum(
        row["task_completion_success"] for row in episode_reports
    )
    failure_episodes = [row for row in episode_reports if row["requires_failure"]]
    failure_recoveries = sum(row["recovery_success"] for row in failure_episodes)
    (
        failure_recovery_gate_passed,
        failure_recovery_gate_applicable,
        failure_recovery_success_rate,
    ) = _failure_recovery_gate_status(
        failure_episode_count=len(failure_episodes),
        failure_recovery_count=failure_recoveries,
    )
    checks = {
        "all_model_selected_actions_were_allowlisted": rejected_actions == 0,
        "model_selected_actions_were_actually_executed": executed_actions > 0,
        "all_task_completion_predicates_satisfied": task_completion_successes
        == total_episodes,
        "failure_recovery_success_rate_meets_gate": failure_recovery_gate_passed,
        "all_episodes_terminated_with_done": all(
            row["selected_done"] for row in episode_reports
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2bn_model_selected_sealed_runtime",
        "passed": passed,
        "ready_for_bounded_model_selected_real_execution_claim": passed,
        "ready_for_repo_disjoint_runtime_claim": False,
        "ready_for_open_ended_native_perception_claim": False,
        "ready_for_epoch_making_architecture_claim": False,
        "policy_configuration": {
            "policy_label": policy_label,
            "policy_class": type(policy).__name__,
            "policy_metadata": (
                policy.metadata() if callable(getattr(policy, "metadata", None)) else {}
            ),
            "authorize_bounded_debug_cortex_recovery": authorize_bounded_debug_cortex_recovery,
            "use_synaptic_motor_plan": use_synaptic_motor_plan,
            "checkpoint_path": str(Path(checkpoint_path)) if checkpoint_path is not None else None,
        },
        "checks": checks,
        "metrics": {
            "episodes": total_episodes,
            "executed_actions": executed_actions,
            "ambient_observation_actions_executed": ambient_observation_actions_executed,
            "rejected_actions": rejected_actions,
            "strict_sequence_successes": strict_successes,
            "strict_sequence_evaluated_episodes": len(strict_sequence_rows),
            "strict_sequence_success_rate": strict_successes
            / max(len(strict_sequence_rows), 1),
            "task_completion_successes": task_completion_successes,
            "task_completion_success_rate": task_completion_successes
            / max(total_episodes, 1),
            "recovery_successes": recovery_successes,
            "recovery_success_rate": recovery_successes / max(total_episodes, 1),
            "failure_episodes": len(failure_episodes),
            "failure_recoveries": failure_recoveries,
            "failure_recovery_gate_applicable": failure_recovery_gate_applicable,
            "failure_recovery_success_rate": failure_recovery_success_rate,
            "decision_count": len(all_decision_latencies_ms),
            "mean_decision_latency_ms": sum(all_decision_latencies_ms)
            / max(len(all_decision_latencies_ms), 1),
            "p50_decision_latency_ms": _percentile(all_decision_latencies_ms, 0.50),
            "p95_decision_latency_ms": _percentile(all_decision_latencies_ms, 0.95),
            "total_policy_compute_latency_ms": sum(all_decision_latencies_ms),
            "token_equivalent_cost": total_token_equivalent_cost,
            "mean_token_equivalent_cost_per_episode": total_token_equivalent_cost
            / max(total_episodes, 1),
            "model_calls": total_model_calls,
            "mean_model_calls_per_episode": total_model_calls
            / max(total_episodes, 1),
        },
        "episode_reports": episode_reports,
        "trajectory_jsonl": str(Path(output_jsonl)),
        "supported_claims": [
            "the bounded policy selected allowlisted structured actions that were actually executed in state-chained sealed runtime episodes"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "repo-disjoint runtime transfer",
            "open-ended native perception",
            "production autonomy",
            "epoch-making architecture",
        ],
        "next_required_experiment": (
            "phase2bo_repo_disjoint_model_selected_sealed_runtime"
            if passed
            else "repair_phase2bn_model_selected_sealed_runtime"
        ),
    }
    output_report = Path(output_report_json)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute model-selected allowlisted actions in a sealed continuous runtime."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-report-json", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=2.0)
    parser.add_argument("--max-extra-steps", type=int, default=3)
    parser.add_argument("--policy-label", default="phase2bn_model_selected_sealed_runtime")
    parser.add_argument("--disable-bounded-debug-cortex-recovery", action="store_true")
    parser.add_argument("--disable-synaptic-motor-plan", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = run_phase2bn_model_selected_sealed_runtime(
        checkpoint_path=args.checkpoint_path,
        manifest_json=args.manifest_json,
        output_jsonl=args.output_jsonl,
        output_report_json=args.output_report_json,
        timeout_seconds=args.timeout_seconds,
        max_extra_steps=args.max_extra_steps,
        policy_label=args.policy_label,
        authorize_bounded_debug_cortex_recovery=not args.disable_bounded_debug_cortex_recovery,
        use_synaptic_motor_plan=not args.disable_synaptic_motor_plan,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
