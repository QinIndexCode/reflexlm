import json
from pathlib import Path

from reflexlm.cli.build_phase2z_evidence_sufficiency_report import (
    build_phase2z_evidence_sufficiency_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2z_sufficiency_report_keeps_recorded_patch_boundary(tmp_path: Path) -> None:
    report = build_phase2z_evidence_sufficiency_report(
        nonliteral_gap_audit_json=_write(
            tmp_path / "nonliteral.json",
            {"passed": True, "metrics": {"structural_nonliteral_rows": 72, "multifile_rows": 34}},
        ),
        data_health_json=_write(tmp_path / "data.json", {"passed": True}),
        training_summary_json=_write(
            tmp_path / "training.json",
            {
                "low_level_qwen_calls_target": 0,
                "use_pairwise_command_reranker": False,
                "command_candidate_encoder": "features_only",
                "history": [
                    {
                        "train_components": {
                            "patch_proposal": 0.02,
                            "test_selection": 0.02,
                            "rollback_safety": 0.03,
                            "bounded_edit_scope": 0.02,
                            "progress_monitor": 0.03,
                            "verification_state": 0.04,
                        },
                        "val_metrics": {"command_slot_accuracy": 1.0},
                    }
                ],
            },
        ),
        postflight_json=_write(
            tmp_path / "postflight.json",
            {
                "passed": True,
                "metrics": {
                    "source_overlap_val_accuracy": 0.5,
                    "model_minus_source_overlap_accuracy": 0.5,
                },
            },
        ),
        execution_audit_json=_write(
            tmp_path / "execution.json",
            {
                "passed": True,
                "claim_bearing_execution_evidence": False,
                "metrics": {"success_rate": 1.0, "success_count": 24, "row_count": 24},
            },
        ),
    )

    assert report["passed"] is True
    assert "model_generated_patch_diff" in report["unsupported_claims"]
    assert "do_not_claim_model_generated_patch_repair" in report["blocked_actions"]
    assert "public_repo_structural_nonliteral_repair_runtime_control_supported" in report["supported_claims"]


def test_phase2z_sufficiency_report_accepts_phase2aq_symbolic_patch_evidence(
    tmp_path: Path,
) -> None:
    common_kwargs = {
        "nonliteral_gap_audit_json": _write(
            tmp_path / "nonliteral.json",
            {"passed": True, "metrics": {"structural_nonliteral_rows": 72, "multifile_rows": 34}},
        ),
        "data_health_json": _write(tmp_path / "data.json", {"passed": True}),
        "training_summary_json": _write(
            tmp_path / "training.json",
            {
                "low_level_qwen_calls_target": 0,
                "use_pairwise_command_reranker": False,
                "command_candidate_encoder": "features_only",
                "history": [
                    {
                        "train_components": {
                            "patch_proposal": 0.02,
                            "test_selection": 0.02,
                            "rollback_safety": 0.03,
                            "bounded_edit_scope": 0.02,
                            "progress_monitor": 0.03,
                            "verification_state": 0.04,
                        },
                        "val_metrics": {"command_slot_accuracy": 1.0},
                    }
                ],
            },
        ),
        "postflight_json": _write(
            tmp_path / "postflight.json",
            {
                "passed": True,
                "metrics": {
                    "source_overlap_val_accuracy": 0.5,
                    "model_minus_source_overlap_accuracy": 0.5,
                },
            },
        ),
        "execution_audit_json": _write(
            tmp_path / "execution.json",
            {
                "passed": True,
                "claim_bearing_execution_evidence": False,
                "metrics": {"success_rate": 1.0, "success_count": 24, "row_count": 24},
            },
        ),
    }
    report = build_phase2z_evidence_sufficiency_report(
        **common_kwargs,
        symbolic_patch_audit_json=_write(
            tmp_path / "phase2aq.json",
            {
                "passed": True,
                "claim_bearing_execution_evidence": True,
                "evidence_level": "holdout24",
                "claim_boundary": "bounded_runtime_symbolic_patch_proposal_only_not_open_ended_repair",
                "metrics": {"success_rate": 1.0, "success_count": 24, "row_count": 24},
            },
        ),
    )

    assert report["passed"] is True
    assert report["phase2aq_checks"]["symbolic_patch_holdout24"] is True
    assert (
        "bounded_runtime_symbolic_text_membership_patch_proposal_holdout24_supported"
        in report["supported_claims"]
    )
    assert (
        "bounded_patch_proposal_motor_or_patch_slot_outputs_not_recorded_diff_replay"
        not in report["next_required_evidence"]
    )


def test_phase2z_sufficiency_report_rejects_missing_open_control_training(
    tmp_path: Path,
) -> None:
    report = build_phase2z_evidence_sufficiency_report(
        nonliteral_gap_audit_json=_write(tmp_path / "nonliteral.json", {"passed": True, "metrics": {}}),
        data_health_json=_write(tmp_path / "data.json", {"passed": True}),
        training_summary_json=_write(
            tmp_path / "training.json",
            {"history": [{"train_components": {}, "val_metrics": {"command_slot_accuracy": 1.0}}]},
        ),
        postflight_json=_write(tmp_path / "postflight.json", {"passed": True, "metrics": {}}),
        execution_audit_json=_write(
            tmp_path / "execution.json",
            {"passed": True, "claim_bearing_execution_evidence": False, "metrics": {"success_rate": 1.0}},
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["training_open_control_losses_present"] is False
