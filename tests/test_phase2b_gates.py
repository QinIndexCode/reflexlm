import json
from pathlib import Path

from reflexlm.cli.check_phase2b_gates import check_phase2b_gates


def _metric(mean: float, count: int = 10) -> dict[str, float | int]:
    return {"mean": mean, "count": count}


def _eval_payload(
    *,
    label: str,
    completion: float,
    dangerous: float,
    calls: float,
    tokens: float,
    route_completion: float,
    low_latency: float = 1.0,
    low_calls: float = 1.0,
) -> dict[str, object]:
    per_task = {}
    for task_name in [
        "test_failure_reflex",
        "external_file_change_reflex",
        "common_error_recovery_routine",
    ]:
        per_task[task_name] = {
            "episode_count": 10,
            "metrics": {
                "task_completion_rate": _metric(route_completion),
                "reaction_latency_ms": _metric(20.0),
                "model_calls": _metric(calls),
            },
        }
    for task_name in [
        "blocking_input_detection",
        "process_hang_detection",
        "dangerous_action_interception",
    ]:
        per_task[task_name] = {
            "episode_count": 10,
            "metrics": {
                "task_completion_rate": _metric(1.0),
                "reaction_latency_ms": _metric(low_latency),
                "model_calls": _metric(low_calls),
            },
        }
    per_task["dangerous_action_interception"]["metrics"][
        "dangerous_action_block_rate"
    ] = _metric(dangerous)
    return {
        "policy": label,
        "episode_count": 60,
        "trace_count": 60,
        "metrics": {
            "aggregate": {
                "task_completion_rate": _metric(completion, 60),
                "dangerous_action_block_rate": _metric(dangerous, 10),
                "reaction_latency_ms": _metric(5.0, 60),
                "model_calls": _metric(calls, 60),
                "token_equivalent_cost": _metric(tokens, 60),
                "state_hallucination_rate": _metric(0.0, 60),
                "stale_state_action_rate": _metric(0.0, 60),
            },
            "per_task": per_task,
        },
    }


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _audit(path: Path, *, passed: bool = True) -> Path:
    return _write(
        path,
        {
            "passed": passed,
            "exact_memorization_clear": passed,
            "classic_train_val_overfit_clear": passed,
            "semantic_similarity_clear": passed,
            "semantic_similarity": {
                "semantic_nearest_neighbor": {
                    "mean": 0.2,
                    "p50": 0.2,
                    "p95": 0.3,
                    "max": 0.4,
                }
            },
            "warnings": [] if passed else [{"type": "high_semantic_nearest_neighbor_similarity"}],
        },
    )


def test_phase2b_gate_accepts_complete_unified_evidence(tmp_path: Path) -> None:
    prompt = _write(
        tmp_path / "prompt.json",
        _eval_payload(
            label="prompt",
            completion=0.30,
            dangerous=0.0,
            calls=2.0,
            tokens=500.0,
            route_completion=0.20,
        ),
    )
    react = _write(
        tmp_path / "react.json",
        _eval_payload(
            label="react",
            completion=0.55,
            dangerous=0.0,
            calls=2.2,
            tokens=520.0,
            route_completion=0.40,
        ),
    )
    reflex = _write(
        tmp_path / "reflex.json",
        _eval_payload(
            label="reflex",
            completion=0.80,
            dangerous=1.0,
            calls=1.0,
            tokens=60.0,
            route_completion=0.55,
        ),
    )
    unified = _write(
        tmp_path / "unified.json",
        _eval_payload(
            label="unified",
            completion=0.96,
            dangerous=1.0,
            calls=0.9,
            tokens=250.0,
            route_completion=0.72,
        ),
    )
    audit = _write(
        tmp_path / "generalization.json",
        {
            "passed": True,
            "overlap_with_train": {
                "test_prompt_overlap_rate": 0.0,
                "test_prompt_target_overlap_rate": 0.0,
                "test_episode_overlap_rate": 0.0,
                "test_scenario_overlap_rate": 0.0,
            },
            "hidden_leakage": {"hit_count": 0},
        },
    )
    overfit = _audit(tmp_path / "overfit.json")

    report = check_phase2b_gates(
        unified_eval_paths=[unified],
        prompt_only_eval_path=prompt,
        react_eval_path=react,
        reflex_eval_path=reflex,
        generalization_audit_path=audit,
        overfit_audit_path=overfit,
    )

    assert report["passed"] is True
    assert report["best_unified_label"] == "unified"
    checks = report["unified_assessments"][0]["checks"]
    assert checks["route_sensitive_gain_vs_reflex"]["passed"] is True
    assert checks["wide_model_call_reduction_vs_best_cost_baseline"]["passed"] is True


def test_phase2b_gate_rejects_incomplete_baseline_evidence(tmp_path: Path) -> None:
    unified = _write(
        tmp_path / "unified.json",
        _eval_payload(
            label="unified",
            completion=0.96,
            dangerous=1.0,
            calls=0.9,
            tokens=250.0,
            route_completion=0.72,
        ),
    )

    report = check_phase2b_gates(unified_eval_paths=[unified])

    assert report["passed"] is False
    assert report["complete_gate_evidence"] is False
    assert set(report["missing_inputs"]) == {
        "prompt_only_eval",
        "react_eval",
        "reflex_eval",
        "generalization_audit",
        "overfit_audit",
    }


def test_phase2b_gate_rejects_route_regression(tmp_path: Path) -> None:
    prompt = _write(
        tmp_path / "prompt.json",
        _eval_payload(
            label="prompt",
            completion=0.30,
            dangerous=0.0,
            calls=2.0,
            tokens=500.0,
            route_completion=0.20,
        ),
    )
    react = _write(
        tmp_path / "react.json",
        _eval_payload(
            label="react",
            completion=0.50,
            dangerous=0.0,
            calls=2.0,
            tokens=500.0,
            route_completion=0.40,
        ),
    )
    reflex = _write(
        tmp_path / "reflex.json",
        _eval_payload(
            label="reflex",
            completion=0.80,
            dangerous=1.0,
            calls=1.0,
            tokens=60.0,
            route_completion=0.85,
        ),
    )
    unified = _write(
        tmp_path / "unified.json",
        _eval_payload(
            label="unified",
            completion=0.96,
            dangerous=1.0,
            calls=0.9,
            tokens=250.0,
            route_completion=0.86,
        ),
    )
    audit = _write(tmp_path / "generalization.json", {"passed": True})
    overfit = _audit(tmp_path / "overfit.json")

    report = check_phase2b_gates(
        unified_eval_paths=[unified],
        prompt_only_eval_path=prompt,
        react_eval_path=react,
        reflex_eval_path=reflex,
        generalization_audit_path=audit,
        overfit_audit_path=overfit,
    )

    assert report["passed"] is False
    checks = report["unified_assessments"][0]["checks"]
    assert checks["route_sensitive_gain_vs_reflex"]["passed"] is False


def test_phase2b_gate_rejects_failed_generalization_audit(tmp_path: Path) -> None:
    prompt = _write(
        tmp_path / "prompt.json",
        _eval_payload(
            label="prompt",
            completion=0.30,
            dangerous=0.0,
            calls=2.0,
            tokens=500.0,
            route_completion=0.20,
        ),
    )
    react = _write(
        tmp_path / "react.json",
        _eval_payload(
            label="react",
            completion=0.55,
            dangerous=0.0,
            calls=2.2,
            tokens=520.0,
            route_completion=0.40,
        ),
    )
    reflex = _write(
        tmp_path / "reflex.json",
        _eval_payload(
            label="reflex",
            completion=0.80,
            dangerous=1.0,
            calls=1.0,
            tokens=60.0,
            route_completion=0.55,
        ),
    )
    unified = _write(
        tmp_path / "unified.json",
        _eval_payload(
            label="unified",
            completion=0.96,
            dangerous=1.0,
            calls=0.9,
            tokens=250.0,
            route_completion=0.72,
        ),
    )
    audit = _write(tmp_path / "generalization.json", {"passed": False})
    overfit = _audit(tmp_path / "overfit.json")

    report = check_phase2b_gates(
        unified_eval_paths=[unified],
        prompt_only_eval_path=prompt,
        react_eval_path=react,
        reflex_eval_path=reflex,
        generalization_audit_path=audit,
        overfit_audit_path=overfit,
    )

    assert report["passed"] is False
    assert report["generalization_audit"]["passed"] is False


def test_phase2b_gate_rejects_failed_overfit_audit(tmp_path: Path) -> None:
    prompt = _write(
        tmp_path / "prompt.json",
        _eval_payload(
            label="prompt",
            completion=0.30,
            dangerous=0.0,
            calls=2.0,
            tokens=500.0,
            route_completion=0.20,
        ),
    )
    react = _write(
        tmp_path / "react.json",
        _eval_payload(
            label="react",
            completion=0.55,
            dangerous=0.0,
            calls=2.2,
            tokens=520.0,
            route_completion=0.40,
        ),
    )
    reflex = _write(
        tmp_path / "reflex.json",
        _eval_payload(
            label="reflex",
            completion=0.80,
            dangerous=1.0,
            calls=1.0,
            tokens=60.0,
            route_completion=0.55,
        ),
    )
    unified = _write(
        tmp_path / "unified.json",
        _eval_payload(
            label="unified",
            completion=0.96,
            dangerous=1.0,
            calls=0.9,
            tokens=250.0,
            route_completion=0.72,
        ),
    )
    audit = _write(tmp_path / "generalization.json", {"passed": True})
    overfit = _audit(tmp_path / "overfit.json", passed=False)

    report = check_phase2b_gates(
        unified_eval_paths=[unified],
        prompt_only_eval_path=prompt,
        react_eval_path=react,
        reflex_eval_path=reflex,
        generalization_audit_path=audit,
        overfit_audit_path=overfit,
    )

    assert report["passed"] is False
    assert report["overfit_audit"]["passed"] is False
