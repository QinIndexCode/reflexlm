from reflexlm.cli.audit_phase2bi_temporal_prediction_error import (
    audit_phase2bi_temporal_prediction_error,
)


def test_phase2bi_audit_proves_runtime_observed_prediction_error_and_gate() -> None:
    report = audit_phase2bi_temporal_prediction_error()

    assert report["passed"] is True
    assert report["ready_for_runtime_observed_prediction_error_claim"] is True
    assert report["ready_for_trained_world_model_accuracy_claim"] is False
    assert report["checks"]["changed_transition_uses_observed_temporal_error"] is True
    assert report["checks"]["high_error_blocks_direct_hybrid_reflex"] is True


def test_phase2bi_audit_rejects_an_unreachable_error_margin() -> None:
    report = audit_phase2bi_temporal_prediction_error(min_changed_error_delta=2.0)

    assert report["passed"] is False
    assert report["checks"]["changed_transition_error_exceeds_stable_by_margin"] is False
