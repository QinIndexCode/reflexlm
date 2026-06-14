from reflexlm.cli.audit_phase2homeostasis_cross_episode_memory import (
    audit_phase2homeostasis_cross_episode_memory,
)


def test_cross_episode_memory_accumulates_weak_failure_evidence() -> None:
    report = audit_phase2homeostasis_cross_episode_memory(
        calibrated_threshold=0.50,
        prediction_error=0.20,
        observations_per_episode=2,
        max_episodes=12,
        stable_recovery_episodes=3,
    )

    assert report["passed"] is True
    assert report["ready_for_bounded_cross_episode_homeostatic_memory_claim"] is True
    assert report["metrics"]["persistent"]["wake_episode"] is not None
    assert report["metrics"]["erased"]["wake_episode"] is None


def test_cross_episode_memory_audit_rejects_too_short_horizon() -> None:
    report = audit_phase2homeostasis_cross_episode_memory(
        calibrated_threshold=0.50,
        prediction_error=0.10,
        observations_per_episode=1,
        max_episodes=1,
        stable_recovery_episodes=1,
    )

    assert report["passed"] is False
    assert report["checks"]["persistent_memory_eventually_wakes"] is False
