import json
from pathlib import Path

from reflexlm.cli.build_phase2at_evidence_sufficiency_report import (
    build_phase2at_evidence_sufficiency_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _data_health(*, passed: bool = True, holdout: int = 32) -> dict:
    return {
        "passed": passed,
        "metrics": {
            "split_counts": {"train": 32, "val": 32, "holdout": holdout},
            "best_non_full_baseline_accuracy": 0.5,
        },
    }


def _readiness(*, passed: bool = True) -> dict:
    return {"passed": passed, "ready_for_training": passed}


def _multiseed(*, passed: bool = True, seeds: int = 3) -> dict:
    return {
        "passed": passed,
        "metrics": {
            "unique_seeds": [13, 17, 23][:seeds],
            "unique_seed_count": seeds,
            "metric_summary": {
                "command_slot_accuracy": {"min": 1.0, "mean": 1.0, "std": 0.0},
                "patch_template_slot_accuracy": {"min": 0.91, "mean": 0.95, "std": 0.03},
            },
        },
        "unsupported_claims": [
            "learned_freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
    }


def _cross_model(*, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "metrics": {
            "model_count": 2,
            "models": ["qwen1.5b", "qwen3b"],
            "min_metrics": {
                "command_slot_accuracy": 1.0,
                "patch_template_slot_accuracy": 0.91,
            },
        },
        "unsupported_claims": [
            "learned_freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
    }


def _package_schema_gate(*, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "metrics": {
            "patch_proposal_strategy": "learned_bounded_candidate",
            "learned_patch_generation_enabled": True,
            "patch_candidate_schema_version": "phase2at.learned_bounded_patch_candidate.v1",
        },
        "unsupported_claims": [
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
    }


def _runtime_smoke_audit(*, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "claim_boundary": "bounded_runtime_symbolic_structural_patch_proposal_only_not_open_ended_repair",
        "metrics": {
            "row_count": 1,
            "success_rate": 1.0,
            "symbolic_structural_mode": True,
        },
        "blocked_actions": [
            "do_not_claim_freeform_model_generated_patch_repair",
            "do_not_claim_open_ended_debugging_generalization",
        ],
    }


def _runtime_delta_gate(*, passed: bool = True) -> dict:
    return {
        "passed": passed,
        "metrics": {
            "full_success_rate": 1.0,
            "control_success_rate": 0.7 if passed else 1.0,
            "full_minus_control_success_rate": 0.3 if passed else 0.0,
        },
        "unsupported_claims": [
            "learned_freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
    }


def test_phase2at_evidence_sufficiency_accepts_bounded_three_seed_smoke(
    tmp_path: Path,
) -> None:
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health()),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed()),
    )

    assert report["passed"] is True
    assert report["supported_claims"] == [
        "phase2at_nonsealed_data_ready_for_bounded_descriptor_training",
        "phase2at_same_model_three_seed_descriptor_smoke_stable",
    ]
    assert "epoch_making_architecture" in report["unsupported_claims"]
    assert "cross_model_descriptor_smoke_after_same_model_seed_gate" in report[
        "next_required_evidence"
    ]
    assert "larger_nonsealed_public_repo_origin_disjoint_descriptor_holdout" in report[
        "next_required_evidence"
    ]


def test_phase2at_evidence_sufficiency_accepts_optional_cross_model_smoke(
    tmp_path: Path,
) -> None:
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health()),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed()),
        cross_model_smoke_json=_write(tmp_path / "cross.json", _cross_model()),
    )

    assert report["passed"] is True
    assert "phase2at_initial_same_split_cross_model_descriptor_smoke_supported" in report[
        "supported_claims"
    ]
    assert "cross_model_descriptor_smoke_after_same_model_seed_gate" not in report[
        "next_required_evidence"
    ]


def test_phase2at_evidence_sufficiency_removes_larger_holdout_after_threshold(
    tmp_path: Path,
) -> None:
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health(holdout=96)),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed()),
        cross_model_smoke_json=_write(tmp_path / "cross.json", _cross_model()),
        larger_holdout_row_threshold=96,
    )

    assert report["passed"] is True
    assert "larger_nonsealed_public_repo_origin_disjoint_descriptor_holdout" not in report[
        "next_required_evidence"
    ]
    assert report["metrics"]["larger_holdout_row_threshold_met"] is True


def test_phase2at_evidence_sufficiency_records_schema_ready_package_without_claim_upgrade(
    tmp_path: Path,
) -> None:
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health(holdout=96)),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed()),
        cross_model_smoke_json=_write(tmp_path / "cross.json", _cross_model()),
        package_schema_gate_json=_write(
            tmp_path / "package_schema_gate.json", _package_schema_gate()
        ),
        larger_holdout_row_threshold=96,
    )

    assert report["passed"] is True
    assert (
        "phase2at_package_schema_ready_for_learned_bounded_candidate_eval"
        in report["supported_claims"]
    )
    assert "learned_freeform_patch_generation" in report["unsupported_claims"]
    assert (
        "do_not_treat_phase2at_schema_ready_package_as_claim_bearing_generation_without_full_nonsealed_delta"
        in report["blocked_actions"]
    )
    assert report["checks"]["package_schema_gate_passed"] is True
    assert report["metrics"]["package_schema_gate_passed"] is True
    assert (
        "claim_bearing_package_release_only_after_full_nonsealed_delta_gate"
        in report["next_required_evidence"]
    )


def test_phase2at_evidence_sufficiency_records_bounded_runtime_smoke_without_freeform_claim(
    tmp_path: Path,
) -> None:
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health(holdout=96)),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed()),
        cross_model_smoke_json=_write(tmp_path / "cross.json", _cross_model()),
        package_schema_gate_json=_write(
            tmp_path / "package_schema_gate.json", _package_schema_gate()
        ),
        package_runtime_smoke_audit_json=_write(
            tmp_path / "runtime_smoke.json", _runtime_smoke_audit()
        ),
        larger_holdout_row_threshold=96,
    )

    assert report["passed"] is True
    assert (
        "phase2at_package_runtime_bounded_symbolic_smoke_passed"
        in report["supported_claims"]
    )
    assert "learned_freeform_patch_generation" in report["unsupported_claims"]
    assert report["checks"]["package_runtime_smoke_audit_passed"] is True
    assert (
        report["checks"]["package_runtime_smoke_blocks_freeform_and_openended"]
        is True
    )


def test_phase2at_evidence_sufficiency_rejects_runtime_smoke_boundary_drift(
    tmp_path: Path,
) -> None:
    audit = _runtime_smoke_audit()
    audit["claim_boundary"] = "open_ended_repair"
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health(holdout=96)),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed()),
        cross_model_smoke_json=_write(tmp_path / "cross.json", _cross_model()),
        package_runtime_smoke_audit_json=_write(tmp_path / "runtime_smoke.json", audit),
        larger_holdout_row_threshold=96,
    )

    assert report["passed"] is False
    assert (
        report["checks"]["package_runtime_smoke_uses_bounded_symbolic_boundary"]
        is False
    )


def test_phase2at_evidence_sufficiency_rejects_failed_runtime_delta_gate(
    tmp_path: Path,
) -> None:
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health(holdout=256)),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed()),
        cross_model_smoke_json=_write(tmp_path / "cross.json", _cross_model()),
        package_schema_gate_json=_write(
            tmp_path / "package_schema_gate.json", _package_schema_gate()
        ),
        package_runtime_smoke_audit_json=_write(
            tmp_path / "runtime_smoke.json", _runtime_smoke_audit()
        ),
        package_runtime_delta_gate_json=_write(
            tmp_path / "runtime_delta.json", _runtime_delta_gate(passed=False)
        ),
        larger_holdout_row_threshold=256,
    )

    assert report["passed"] is False
    assert report["checks"]["package_runtime_delta_gate_passed"] is False
    assert "phase2at_package_runtime_delta_supported" not in report["supported_claims"]
    assert (
        "nonsealed_task_where_loaded_package_beats_no_policy_control"
        in report["next_required_evidence"]
    )


def test_phase2at_evidence_sufficiency_accepts_runtime_delta_gate(
    tmp_path: Path,
) -> None:
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health(holdout=256)),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed()),
        cross_model_smoke_json=_write(tmp_path / "cross.json", _cross_model()),
        package_schema_gate_json=_write(
            tmp_path / "package_schema_gate.json", _package_schema_gate()
        ),
        package_runtime_smoke_audit_json=_write(
            tmp_path / "runtime_smoke.json", _runtime_smoke_audit()
        ),
        package_runtime_delta_gate_json=_write(
            tmp_path / "runtime_delta.json", _runtime_delta_gate()
        ),
        larger_holdout_row_threshold=256,
    )

    assert report["passed"] is True
    assert "phase2at_package_runtime_delta_supported" in report["supported_claims"]
    assert (
        "nonsealed_task_where_loaded_package_beats_no_policy_control"
        not in report["next_required_evidence"]
    )


def test_phase2at_evidence_sufficiency_rejects_schema_gate_boundary_drift(
    tmp_path: Path,
) -> None:
    gate = _package_schema_gate()
    gate["unsupported_claims"] = ["production_autonomy"]
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health(holdout=96)),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed()),
        cross_model_smoke_json=_write(tmp_path / "cross.json", _cross_model()),
        package_schema_gate_json=_write(tmp_path / "package_schema_gate.json", gate),
        larger_holdout_row_threshold=96,
    )

    assert report["passed"] is False
    assert report["checks"]["package_schema_boundary_retained"] is False


def test_phase2at_evidence_sufficiency_rejects_cross_model_boundary_drift(
    tmp_path: Path,
) -> None:
    cross_model = _cross_model()
    cross_model["unsupported_claims"] = ["learned_freeform_patch_generation"]
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health()),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed()),
        cross_model_smoke_json=_write(tmp_path / "cross.json", cross_model),
    )

    assert report["passed"] is False
    assert report["checks"]["cross_model_boundary_retained"] is False


def test_phase2at_evidence_sufficiency_rejects_failed_multiseed(
    tmp_path: Path,
) -> None:
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health()),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed(passed=False)),
    )

    assert report["passed"] is False
    assert report["checks"]["multiseed_smoke_passed"] is False


def test_phase2at_evidence_sufficiency_rejects_missing_boundary_claims(
    tmp_path: Path,
) -> None:
    multiseed = _multiseed()
    multiseed["unsupported_claims"] = ["learned_freeform_patch_generation"]
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health()),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", multiseed),
    )

    assert report["passed"] is False
    assert report["checks"]["unsupported_claims_retained"] is False


def test_phase2at_evidence_sufficiency_rejects_tiny_holdout(
    tmp_path: Path,
) -> None:
    report = build_phase2at_evidence_sufficiency_report(
        data_health_json=_write(tmp_path / "data.json", _data_health(holdout=12)),
        pretrain_readiness_json=_write(tmp_path / "ready.json", _readiness()),
        multiseed_smoke_json=_write(tmp_path / "multi.json", _multiseed()),
    )

    assert report["passed"] is False
    assert report["checks"]["holdout_row_minimum_met"] is False
