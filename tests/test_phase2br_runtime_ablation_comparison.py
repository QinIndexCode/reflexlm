from reflexlm.cli.run_phase2br_runtime_ablation_comparison import (
    _aggregate_subreports,
    _mode_comparison,
)


def _episode(*, completed: bool, recovered: bool, latency: float) -> dict:
    return {
        "requires_failure": True,
        "task_completion_success": completed,
        "recovery_success": recovered,
        "policy_debug_steps": [{"decision_latency_ms": latency}],
        "token_equivalent_cost": 12,
        "model_calls": 1,
    }


def test_phase2br_aggregates_runtime_measurements_and_mechanism_delta() -> None:
    native = _aggregate_subreports(
        [
            {
                "metrics": {"executed_actions": 2, "rejected_actions": 0},
                "episode_reports": [
                    _episode(completed=True, recovered=True, latency=2.0),
                    _episode(completed=True, recovered=True, latency=4.0),
                ],
            }
        ]
    )
    ablation = _aggregate_subreports(
        [
            {
                "metrics": {"executed_actions": 1, "rejected_actions": 1},
                "episode_reports": [
                    _episode(completed=False, recovered=False, latency=1.0),
                    _episode(completed=True, recovered=True, latency=3.0),
                ],
            }
        ]
    )

    comparison = _mode_comparison(native, ablation)

    assert native["task_completion_success_rate"] == 1.0
    assert native["failure_recovery_success_rate"] == 1.0
    assert native["mean_decision_latency_ms"] == 3.0
    assert native["token_equivalent_cost"] == 24
    assert native["model_calls"] == 2
    assert comparison["task_completion_rate_delta"] == 0.5
    assert comparison["failure_recovery_rate_delta"] == 0.5
