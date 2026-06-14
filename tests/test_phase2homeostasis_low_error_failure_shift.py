from reflexlm.cli.audit_phase2homeostasis_low_error_failure_shift import (
    audit_phase2homeostasis_low_error_failure_shift,
)


def test_low_error_failure_shift_audit_proves_bounded_adaptive_utility() -> None:
    report = audit_phase2homeostasis_low_error_failure_shift(
        calibrated_threshold=0.50,
        prediction_errors=(0.10, 0.20, 0.30),
        max_failure_observations=64,
        stable_recovery_observations=4,
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_low_error_failure_shift_utility_claim"] is True
    assert all(
        row["wake_observation"] is not None
        for row in report["metrics"]["adaptive_rows"]
    )
    assert all(
        row["wake_observation"] is None
        for row in report["metrics"]["fixed_rows"]
    )


def test_low_error_failure_shift_audit_rejects_insufficient_observation_horizon() -> None:
    report = audit_phase2homeostasis_low_error_failure_shift(
        calibrated_threshold=0.50,
        prediction_errors=(0.10,),
        max_failure_observations=1,
        stable_recovery_observations=2,
    )

    assert report["passed"] is False
    assert (
        report["checks"][
            "adaptive_control_eventually_wakes_on_all_low_error_failures"
        ]
        is False
    )
