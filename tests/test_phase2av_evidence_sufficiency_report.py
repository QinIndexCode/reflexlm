import json
from pathlib import Path

from reflexlm.cli.build_phase2av_evidence_sufficiency_report import (
    build_phase2av_evidence_sufficiency_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _postflight(tmp_path: Path, name: str, *, command: float = 0.95, delta: float = 0.25) -> Path:
    return _write(
        tmp_path / f"{name}.json",
        {
            "passed": True,
            "metrics": {
                "command_slot_accuracy": command,
                "model_minus_source_overlap_accuracy": delta,
                "patch_operation_accuracy": 0.96,
                "patch_template_slot_accuracy": 0.95,
            },
        },
    )


def _full_readiness(tmp_path: Path, *, boundary_ok: bool = True) -> Path:
    unsupported = [
        "phase2av_package_ready",
        "sealed_cross_model_transfer",
        "freeform_patch_generation",
        "open_ended_debugging_generalization",
        "production_autonomy",
        "epoch_making_architecture",
    ]
    if not boundary_ok:
        unsupported = ["sealed_cross_model_transfer"]
    return _write(
        tmp_path / "full_readiness.json",
        {
            "passed": True,
            "ready_for_package": False,
            "ready_for_sealed_eval": False,
            "metrics": {
                "effective_train_examples": 422,
                "split_counts": {"train": 144, "val": 155, "holdout": 156},
            },
            "unsupported_claims": unsupported,
        },
    )


def test_phase2av_evidence_sufficiency_keeps_bounded_boundary(tmp_path: Path) -> None:
    report = build_phase2av_evidence_sufficiency_report(
        pretrain_gate_json=_write(
            tmp_path / "pretrain.json",
            {"passed": True, "ready_for_phase2av_training": True},
        ),
        val_postflight_json=_postflight(tmp_path, "val", command=1.0, delta=0.56),
        holdout_postflight_json=_postflight(tmp_path, "holdout", command=1.0, delta=0.56),
        full_readiness_json=_full_readiness(tmp_path),
        prior_failure_audit_json=_write(
            tmp_path / "failure.json",
            {"failure_modes": ["command_slot_identity_transfer_below_holdout_gate"]},
        ),
    )

    assert report["passed"] is True
    assert report["claim_scope"] == "phase2av_bounded_nonsealed_descriptor_runtime_candidate_selection"
    assert "phase2av_package_ready" in report["unsupported_claims"]
    assert "do_not_run_sealed_eval_from_this_report" in report["blocked_actions"]
    assert "cross_model_nonsealed_descriptor_runtime_reproduction" in report["next_required_evidence"]


def test_phase2av_evidence_sufficiency_rejects_weak_holdout_delta(tmp_path: Path) -> None:
    report = build_phase2av_evidence_sufficiency_report(
        pretrain_gate_json=_write(
            tmp_path / "pretrain.json",
            {"passed": True, "ready_for_phase2av_training": True},
        ),
        val_postflight_json=_postflight(tmp_path, "val", command=1.0, delta=0.56),
        holdout_postflight_json=_postflight(tmp_path, "holdout", command=0.9, delta=0.02),
        full_readiness_json=_full_readiness(tmp_path),
    )

    assert report["passed"] is False
    assert report["checks"]["holdout_model_delta_gate"] is False
    assert report["supported_claims"] == []


def test_phase2av_evidence_sufficiency_rejects_boundary_drift(tmp_path: Path) -> None:
    report = build_phase2av_evidence_sufficiency_report(
        pretrain_gate_json=_write(
            tmp_path / "pretrain.json",
            {"passed": True, "ready_for_phase2av_training": True},
        ),
        val_postflight_json=_postflight(tmp_path, "val", command=1.0, delta=0.56),
        holdout_postflight_json=_postflight(tmp_path, "holdout", command=1.0, delta=0.56),
        full_readiness_json=_full_readiness(tmp_path, boundary_ok=False),
    )

    assert report["passed"] is False
    assert report["checks"]["boundary_claims_blocked"] is False


def test_phase2av_evidence_sufficiency_accepts_legacy_failure_mode_name(
    tmp_path: Path,
) -> None:
    report = build_phase2av_evidence_sufficiency_report(
        pretrain_gate_json=_write(
            tmp_path / "pretrain.json",
            {"passed": True, "ready_for_phase2av_training": True},
        ),
        val_postflight_json=_postflight(tmp_path, "val", command=1.0, delta=0.56),
        holdout_postflight_json=_postflight(tmp_path, "holdout", command=1.0, delta=0.56),
        full_readiness_json=_full_readiness(tmp_path),
        prior_failure_audit_json=_write(
            tmp_path / "failure.json",
            {
                "passed": False,
                "failure_modes": ["command_slot_accuracy_below_gate"],
                "metrics": {"command_slot_accuracy": 0.78},
            },
        ),
    )

    assert report["passed"] is True


def test_phase2av_evidence_sufficiency_can_include_runtime_execution_gate(
    tmp_path: Path,
) -> None:
    report = build_phase2av_evidence_sufficiency_report(
        pretrain_gate_json=_write(
            tmp_path / "pretrain.json",
            {"passed": True, "ready_for_phase2av_training": True},
        ),
        val_postflight_json=_postflight(tmp_path, "val", command=1.0, delta=0.56),
        holdout_postflight_json=_postflight(tmp_path, "holdout", command=1.0, delta=0.56),
        full_readiness_json=_full_readiness(tmp_path),
        runtime_execution_gate_json=_write(
            tmp_path / "runtime_gate.json",
            {
                "passed": True,
                "metrics": {
                    "full_success_rate": 1.0,
                    "control_success_rate": 0.25,
                    "full_minus_control_success_rate": 0.75,
                },
            },
        ),
    )

    assert report["passed"] is True
    assert (
        "phase2av_bounded_descriptor_selected_symbolic_execution_delta_supported"
        in report["supported_claims"]
    )
    assert report["metrics"]["runtime_execution_full_minus_control_success_rate"] == 0.75


def test_phase2av_evidence_sufficiency_removes_completed_reproduction_evidence(
    tmp_path: Path,
) -> None:
    report = build_phase2av_evidence_sufficiency_report(
        pretrain_gate_json=_write(
            tmp_path / "pretrain.json",
            {"passed": True, "ready_for_phase2av_training": True},
        ),
        val_postflight_json=_postflight(tmp_path, "val", command=1.0, delta=0.56),
        holdout_postflight_json=_postflight(tmp_path, "holdout", command=1.0, delta=0.56),
        full_readiness_json=_full_readiness(tmp_path),
        runtime_execution_gate_json=_write(
            tmp_path / "runtime_gate.json",
            {
                "passed": True,
                "metrics": {
                    "full_success_rate": 0.92,
                    "control_success_rate": 0.34,
                    "full_minus_control_success_rate": 0.58,
                },
            },
        ),
        multiseed_report_json=_write(
            tmp_path / "multiseed.json",
            {
                "passed": True,
                "metrics": {
                    "run_count": 3,
                    "holdout_command_slot_accuracy_min": 1.0,
                },
            },
        ),
        cross_model_report_json=_write(
            tmp_path / "cross_model.json",
            {
                "passed": True,
                "metrics": {
                    "model_count": 2,
                    "holdout_command_slot_accuracy_min": 1.0,
                },
            },
        ),
    )

    assert report["passed"] is True
    assert "phase2av_nonsealed_multiseed_descriptor_runtime_reproduced" in report[
        "supported_claims"
    ]
    assert "phase2av_nonsealed_cross_model_descriptor_runtime_reproduced" in report[
        "supported_claims"
    ]
    assert report["next_required_evidence"] == [
        "package_gate_only_after_full_nonsealed_runtime_delta"
    ]


def test_phase2av_evidence_sufficiency_rejects_failed_cross_model_report(
    tmp_path: Path,
) -> None:
    report = build_phase2av_evidence_sufficiency_report(
        pretrain_gate_json=_write(
            tmp_path / "pretrain.json",
            {"passed": True, "ready_for_phase2av_training": True},
        ),
        val_postflight_json=_postflight(tmp_path, "val", command=1.0, delta=0.56),
        holdout_postflight_json=_postflight(tmp_path, "holdout", command=1.0, delta=0.56),
        full_readiness_json=_full_readiness(tmp_path),
        cross_model_report_json=_write(tmp_path / "cross_model.json", {"passed": False}),
    )

    assert report["passed"] is False
    assert report["checks"]["cross_model_report_optional_or_passed"] is False
    assert "cross_model_nonsealed_descriptor_runtime_reproduction" in report[
        "next_required_evidence"
    ]


def test_phase2av_evidence_sufficiency_rejects_failed_runtime_execution_gate(
    tmp_path: Path,
) -> None:
    report = build_phase2av_evidence_sufficiency_report(
        pretrain_gate_json=_write(
            tmp_path / "pretrain.json",
            {"passed": True, "ready_for_phase2av_training": True},
        ),
        val_postflight_json=_postflight(tmp_path, "val", command=1.0, delta=0.56),
        holdout_postflight_json=_postflight(tmp_path, "holdout", command=1.0, delta=0.56),
        full_readiness_json=_full_readiness(tmp_path),
        runtime_execution_gate_json=_write(
            tmp_path / "runtime_gate.json",
            {"passed": False, "metrics": {"full_minus_control_success_rate": 0.0}},
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["runtime_execution_gate_optional_or_passed"] is False


def test_phase2av_evidence_sufficiency_reads_current_pretrain_ready_field(
    tmp_path: Path,
) -> None:
    report = build_phase2av_evidence_sufficiency_report(
        pretrain_gate_json=_write(
            tmp_path / "pretrain.json",
            {"passed": True, "ready_for_phase2av_smoke_training": True},
        ),
        val_postflight_json=_postflight(tmp_path, "val", command=1.0, delta=0.56),
        holdout_postflight_json=_postflight(tmp_path, "holdout", command=1.0, delta=0.56),
        full_readiness_json=_full_readiness(tmp_path),
    )

    assert report["passed"] is True
    assert report["metrics"]["pretrain_ready"] is True
