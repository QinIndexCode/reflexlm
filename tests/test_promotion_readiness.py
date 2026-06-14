from pathlib import Path

from reflexlm.cli.check_promotion_readiness import assess_promotion_readiness


def _aggregate(
    *,
    completion: float,
    hard: float,
    latency: float,
    passed: bool,
    completion_gain: float = 0.0,
    hard_gain: float = 0.0,
) -> dict[str, object]:
    return {
        "mean_total_completion": completion,
        "mean_hard_completion": hard,
        "mean_reaction_latency_ms": latency,
        "mean_state_hallucination_rate": 0.0,
        "mean_stale_state_action_rate": 0.0,
        "mean_total_completion_gain_vs_baseline": completion_gain,
        "mean_hard_completion_gain_vs_baseline": hard_gain,
        "mean_latency_delta_ms_vs_baseline": -0.1,
        "passed_mean_gate": passed,
        "mean_per_task_completion": {
            "common_error_recovery_routine": completion,
            "test_failure_reflex": completion,
        },
    }


def _summary(
    *,
    baseline: dict[str, object],
    candidate: dict[str, object],
    passed_labels: list[str],
) -> dict[str, object]:
    return {
        "passed_mean_labels": passed_labels,
        "aggregate_by_label": {
            "flat": baseline,
            "nsi": candidate,
        },
        "comparisons_by_seed": [
            {"seed": 13, "label": "nsi", "passed": "nsi" in passed_labels},
            {"seed": 29, "label": "nsi", "passed": "nsi" in passed_labels},
            {"seed": 47, "label": "nsi", "passed": "nsi" in passed_labels},
        ],
    }


def test_promotion_readiness_accepts_reflex_gate_and_debug_non_regression(
    tmp_path: Path,
) -> None:
    pause_lock = tmp_path / "phase2_7b.paused"
    pause_lock.write_text("paused", encoding="utf-8")
    all_summary = _summary(
        baseline=_aggregate(completion=0.49, hard=0.42, latency=0.48, passed=False),
        candidate=_aggregate(
            completion=0.79,
            hard=0.58,
            latency=0.34,
            completion_gain=0.30,
            hard_gain=0.16,
            passed=True,
        ),
        passed_labels=["nsi"],
    )
    reflex_summary = _summary(
        baseline=_aggregate(completion=0.54, hard=0.35, latency=0.49, passed=False),
        candidate=_aggregate(
            completion=0.90,
            hard=0.83,
            latency=0.33,
            completion_gain=0.36,
            hard_gain=0.48,
            passed=True,
        ),
        passed_labels=["nsi"],
    )
    debug_summary = _summary(
        baseline=_aggregate(completion=0.2546583, hard=0.2546583, latency=0.48, passed=False),
        candidate=_aggregate(completion=0.2546580, hard=0.2546580, latency=0.37, passed=False),
        passed_labels=[],
    )

    readiness = assess_promotion_readiness(
        all_task_summary=all_summary,
        reflex_layer_summary=reflex_summary,
        debug_cortex_summary=debug_summary,
        candidate_label="nsi",
        baseline_label="flat",
        phase2_pause_lock=pause_lock,
        debug_nonregression_tolerance=1.0e-6,
    )

    assert readiness["ready_for_7b_validation"] is True
    assert readiness["debug_cortex"]["gain_claimed"] is False
    assert readiness["checks"]["phase2_7b_pause_lock_present"] is True


def test_promotion_readiness_accepts_strict_floors(tmp_path: Path) -> None:
    pause_lock = tmp_path / "phase2_7b.paused"
    pause_lock.write_text("paused", encoding="utf-8")
    all_summary = _summary(
        baseline=_aggregate(completion=0.56, hard=0.57, latency=0.47, passed=False),
        candidate=_aggregate(
            completion=0.87,
            hard=0.74,
            latency=0.42,
            completion_gain=0.31,
            hard_gain=0.17,
            passed=True,
        ),
        passed_labels=["nsi"],
    )
    reflex_summary = _summary(
        baseline=_aggregate(completion=0.55, hard=0.36, latency=0.47, passed=False),
        candidate=_aggregate(
            completion=1.0,
            hard=1.0,
            latency=0.42,
            completion_gain=0.45,
            hard_gain=0.64,
            passed=True,
        ),
        passed_labels=["nsi"],
    )
    debug_summary = _summary(
        baseline=_aggregate(completion=0.20, hard=0.20, latency=0.47, passed=False),
        candidate=_aggregate(completion=0.22, hard=0.22, latency=0.42, passed=False),
        passed_labels=[],
    )

    readiness = assess_promotion_readiness(
        all_task_summary=all_summary,
        reflex_layer_summary=reflex_summary,
        debug_cortex_summary=debug_summary,
        candidate_label="nsi",
        baseline_label="flat",
        phase2_pause_lock=pause_lock,
        require_all_seed_gates=True,
        min_reflex_layer_completion=0.95,
        min_common_recovery_completion=0.85,
    )

    assert readiness["ready_for_7b_validation"] is True
    assert readiness["checks"]["all_task_all_seed_gates_passed"] is True
    assert readiness["checks"]["reflex_layer_completion_floor"] is True
    assert readiness["checks"]["common_recovery_completion_floor"] is True


def test_promotion_readiness_rejects_debug_regression(tmp_path: Path) -> None:
    pause_lock = tmp_path / "phase2_7b.paused"
    pause_lock.write_text("paused", encoding="utf-8")
    passing_summary = _summary(
        baseline=_aggregate(completion=0.5, hard=0.4, latency=0.5, passed=False),
        candidate=_aggregate(
            completion=0.8,
            hard=0.7,
            latency=0.3,
            completion_gain=0.3,
            hard_gain=0.3,
            passed=True,
        ),
        passed_labels=["nsi"],
    )
    debug_summary = _summary(
        baseline=_aggregate(completion=0.25, hard=0.25, latency=0.5, passed=False),
        candidate=_aggregate(completion=0.20, hard=0.20, latency=0.3, passed=False),
        passed_labels=[],
    )

    readiness = assess_promotion_readiness(
        all_task_summary=passing_summary,
        reflex_layer_summary=passing_summary,
        debug_cortex_summary=debug_summary,
        candidate_label="nsi",
        baseline_label="flat",
        phase2_pause_lock=pause_lock,
    )

    assert readiness["ready_for_7b_validation"] is False
    assert "debug_cortex_completion_non_regression" in readiness["failed_checks"]
