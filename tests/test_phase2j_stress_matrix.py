import json
from pathlib import Path

from reflexlm.cli.build_phase2j_stress_matrix import build_phase2j_stress_matrix
from reflexlm.data.tasks import build_env, rollout_env
from reflexlm.schema import ActionDecision, ActionType, TaskType


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _write_eval(tmp_path: Path, name: str, trace_rows: list[dict]) -> Path:
    run_path = tmp_path / "runs" / name
    _write_jsonl(run_path / "trace_rows.jsonl", trace_rows)
    payload = {"run_path": str(run_path), "policy": {"policy_label": name}}
    output = tmp_path / f"{name}.json"
    output.write_text(json.dumps(payload), encoding="utf-8")
    return output


def _trace_rows(
    records,
    *,
    fail_command: bool = False,
    fail_run_command_action: bool = False,
) -> list[dict]:
    rows = []
    for record in records:
        oracle = record.action.model_dump(mode="json")
        action = dict(oracle)
        correct = True
        reward = record.reward
        done = record.done
        if fail_command and record.action.type == ActionType.RUN_COMMAND:
            wrong_command = next(
                command
                for command in record.state.goal.command_allowlist
                if command != record.action.command
            )
            action = ActionDecision(
                type=ActionType.RUN_COMMAND,
                command=wrong_command,
                reason="wrong_command",
            ).model_dump(mode="json")
            correct = False
            reward = -1.0
            done = True
        elif fail_run_command_action and record.action.type == ActionType.RUN_COMMAND:
            action = ActionDecision(
                type=ActionType.READ_STDERR,
                reason="wrong_action_gate",
            ).model_dump(mode="json")
            correct = False
            reward = -1.0
            done = True
        rows.append(
            {
                "episode_id": record.episode_id,
                "task_type": record.goal.task_type.value,
                "oracle_action": oracle,
                "action": action,
                "correct": correct,
                "reward": reward,
                "done": done,
            }
        )
    return rows


def test_phase2j_stress_matrix_reports_graded_policy_scores(tmp_path: Path) -> None:
    records = rollout_env(
        build_env(TaskType.TEST_FAILURE, 0, profile="phase2j_source_overlap_hard_val")
    )
    dataset = _write_jsonl(
        tmp_path / "challenge.jsonl",
        [record.model_dump(mode="json") for record in records],
    )
    full_eval = _write_eval(tmp_path, "full", _trace_rows(records))
    no_nsi_eval = _write_eval(tmp_path, "no_nsi", _trace_rows(records, fail_command=True))

    report = build_phase2j_stress_matrix(
        dataset_jsonl=dataset,
        eval_jsons={"full": full_eval, "no_nsi": no_nsi_eval},
        min_source_overlap_hard_episodes=1,
    )

    assert report["matrix"]["full"]["overall"]["task_completion_rate"] == 1.0
    assert report["matrix"]["full"]["overall"]["run_command_action_accuracy"] == 1.0
    assert report["matrix"]["full"]["overall"]["command_slot_match_when_run_command"] == 1.0
    assert report["checks"]["full_run_command_action_gate_on_hard"] is True
    assert report["checks"]["full_command_slot_gate_on_hard"] is True
    assert report["matrix"]["no_nsi"]["overall"]["task_completion_rate"] == 0.0
    assert 0.0 < report["matrix"]["no_nsi"]["overall"]["oracle_step_accuracy"] < 1.0
    assert report["matrix"]["no_nsi"]["overall"]["run_command_action_accuracy"] == 1.0
    assert report["matrix"]["no_nsi"]["overall"]["command_decision_accuracy"] == 0.0
    assert report["matrix"]["no_nsi"]["overall"]["command_slot_match_when_run_command"] == 0.0
    assert report["dataset"]["source_overlap_hard_episodes"] >= 1


def test_phase2j_stress_matrix_rejects_action_gate_fail_before_slot_claim(
    tmp_path: Path,
) -> None:
    records = rollout_env(
        build_env(TaskType.TEST_FAILURE, 0, profile="phase2j_source_overlap_hard_val")
    )
    dataset = _write_jsonl(
        tmp_path / "challenge.jsonl",
        [record.model_dump(mode="json") for record in records],
    )
    full_eval = _write_eval(
        tmp_path,
        "full",
        _trace_rows(records, fail_run_command_action=True),
    )
    no_nsi_eval = _write_eval(tmp_path, "no_nsi", _trace_rows(records, fail_command=True))

    report = build_phase2j_stress_matrix(
        dataset_jsonl=dataset,
        eval_jsons={"full": full_eval, "no_nsi": no_nsi_eval},
        min_source_overlap_hard_episodes=1,
    )

    assert report["checks"]["full_run_command_action_gate_on_hard"] is False
    assert report["checks"]["full_command_slot_gate_on_hard"] is False
    assert report["matrix"]["full"]["overall"]["run_command_action_accuracy"] == 0.0
    assert report["matrix"]["full"]["overall"]["command_slot_match_when_run_command"] is None
