import json
from pathlib import Path

from reflexlm.cli.check_gain_gate import check_gate
from reflexlm.cli.summarize_gain_matrix import summarize_matrix
from reflexlm.task_scope import scope_for_task


def _metric(mean: float, count: int = 10) -> dict[str, object]:
    return {"mean": mean, "ci95": [mean, mean], "count": count}


def _payload(label: str, seed: int, per_task_completion: dict[str, float]) -> dict[str, object]:
    per_task = {}
    for task, completion in per_task_completion.items():
        per_task[task] = {
            "episode_count": 10,
            "metrics": {
                "task_completion_rate": _metric(completion),
                "reaction_latency_ms": _metric(1.0),
                "dangerous_action_block_rate": _metric(1.0)
                if task == "dangerous_action_interception"
                else None,
                "state_hallucination_rate": _metric(0.0),
                "stale_state_action_rate": _metric(0.0)
                if task == "external_file_change_reflex"
                else None,
                "recovery_success_rate": _metric(completion)
                if task
                in {"test_failure_reflex", "common_error_recovery_routine"}
                else None,
                "false_reflex_rate": _metric(1.0 - completion),
            },
        }
    return {
        "policy": {
            "policy_label": label,
            "training_summary": {"trainer_config": {"seed": seed}},
        },
        "metrics": {
            "aggregate": {
                "task_completion_rate": _metric(
                    sum(per_task_completion.values()) / len(per_task_completion)
                ),
                "reaction_latency_ms": _metric(1.0),
                "dangerous_action_block_rate": _metric(1.0),
                "state_hallucination_rate": _metric(0.0),
                "stale_state_action_rate": _metric(0.0),
                "recovery_success_rate": _metric(0.0),
                "false_reflex_rate": _metric(0.0),
            },
            "per_task": per_task,
        },
        "run_path": f"/tmp/{label}-{seed}",
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_scope_for_task_routes_semantic_debug_separately() -> None:
    assert scope_for_task("test_failure_reflex") == "debug_cortex"
    assert scope_for_task("process_hang_detection") == "reflex_layer"
    assert scope_for_task("common_error_recovery_routine") == "reflex_layer"


def test_reflex_layer_gate_excludes_debug_cortex_task(tmp_path: Path) -> None:
    flat = _payload(
        "flat",
        13,
        {
            "blocking_input_detection": 0.9,
            "process_hang_detection": 0.8,
            "dangerous_action_interception": 1.0,
            "external_file_change_reflex": 0.8,
            "common_error_recovery_routine": 0.8,
            "test_failure_reflex": 1.0,
        },
    )
    candidate = _payload(
        "nsi",
        13,
        {
            "blocking_input_detection": 1.0,
            "process_hang_detection": 1.0,
            "dangerous_action_interception": 1.0,
            "external_file_change_reflex": 1.0,
            "common_error_recovery_routine": 1.0,
            "test_failure_reflex": 0.0,
        },
    )
    flat_path = _write(tmp_path / "flat.json", flat)
    nsi_path = _write(tmp_path / "nsi.json", candidate)

    all_scope = check_gate(
        flat_path=flat_path,
        candidate_path=nsi_path,
        min_total_gain=0.10,
        min_hard_gain=0.15,
    )
    reflex_scope = check_gate(
        flat_path=flat_path,
        candidate_path=nsi_path,
        min_total_gain=0.10,
        min_hard_gain=0.15,
        task_scope="reflex_layer",
        hard_task_set="reflex_layer",
    )

    assert all_scope["metrics"]["candidate_total_completion"] < all_scope["metrics"]["flat_total_completion"]
    assert reflex_scope["passed"] is True
    assert "test_failure_reflex" not in reflex_scope["thresholds"]["scored_tasks"]
    assert reflex_scope["thresholds"]["hard_tasks"] == [
        "common_error_recovery_routine",
        "external_file_change_reflex",
        "process_hang_detection",
    ]


def test_reflex_layer_matrix_summary_records_scope(tmp_path: Path) -> None:
    eval_paths = [
        _write(
            tmp_path / "flat.json",
            _payload(
                "flat",
                13,
                {
                    "blocking_input_detection": 0.5,
                    "process_hang_detection": 0.2,
                    "dangerous_action_interception": 1.0,
                    "external_file_change_reflex": 1.0,
                    "common_error_recovery_routine": 0.0,
                    "test_failure_reflex": 1.0,
                },
            ),
        ),
        _write(
            tmp_path / "nsi.json",
            _payload(
                "nsi",
                13,
                {
                    "blocking_input_detection": 1.0,
                    "process_hang_detection": 1.0,
                    "dangerous_action_interception": 1.0,
                    "external_file_change_reflex": 1.0,
                    "common_error_recovery_routine": 0.5,
                    "test_failure_reflex": 0.0,
                },
            ),
        ),
    ]
    summary = summarize_matrix(
        eval_jsons=eval_paths,
        baseline_label="flat",
        min_total_gain=0.10,
        min_hard_gain=0.15,
        task_scope="reflex_layer",
        hard_task_set="reflex_layer",
    )

    assert summary["promotion_ready"] is True
    assert summary["thresholds"]["task_scope"] == "reflex_layer"
    assert summary["aggregate_by_label"]["nsi"]["mean_total_completion"] == 0.9
