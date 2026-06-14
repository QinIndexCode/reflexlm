import json
from pathlib import Path

from reflexlm.cli.build_phase2ar_evidence_sufficiency_report import (
    build_phase2ar_evidence_sufficiency_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2ar_sufficiency_report_accepts_diverse_execution(tmp_path: Path) -> None:
    report = build_phase2ar_evidence_sufficiency_report(
        data_health_json=_write(
            tmp_path / "data.json",
            {
                "passed": True,
                "checks": {
                    "holdout_required_patch_kinds_present": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {
                    "rows_by_split": {"train": 8, "val": 8, "holdout": 8},
                    "patch_kinds_by_split": {
                        "holdout": {"text_membership": 4, "ast_attribute_restoration": 4}
                    },
                },
            },
        ),
        execution_audit_json=_write(
            tmp_path / "execution.json",
            {
                "passed": True,
                "checks": {
                    "required_patch_kinds_present": True,
                    "no_rows_use_recorded_patch_as_proposal": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {
                    "row_count": 32,
                    "success_rate": 1.0,
                    "patch_kind_counts": {
                        "text_membership": 4,
                        "ast_attribute_restoration": 4,
                    },
                },
            },
        ),
        control_delta_json=_write(
            tmp_path / "control.json",
            {
                "passed": True,
                "checks": {
                    "controls_nonzero": True,
                    "best_control_below_ceiling": True,
                    "full_minus_best_control_met": True,
                },
                "metrics": {
                    "full_success_rate": 1.0,
                    "best_control_success_rate": 0.4375,
                    "full_minus_best_control": 0.5625,
                },
            },
        ),
    )

    assert report["passed"] is True
    assert (
        "bounded_runtime_symbolic_structural_patch_proposal_diverse_holdout_supported"
        in report["supported_claims"]
    )
    assert (
        "phase2ar_full_symbolic_structural_beats_nonzero_restricted_controls"
        in report["supported_claims"]
    )
    assert "candidate_patch_baselines_with_nonzero_controls" not in report["next_required_evidence"]
    assert "larger_nonsealed_public_repo_diverse_patch_holdout" not in report["next_required_evidence"]
    assert "epoch_making_architecture" in report["unsupported_claims"]


def test_phase2ar_sufficiency_report_rejects_nondiverse_execution(tmp_path: Path) -> None:
    report = build_phase2ar_evidence_sufficiency_report(
        data_health_json=_write(
            tmp_path / "data.json",
            {
                "passed": True,
                "checks": {
                    "holdout_required_patch_kinds_present": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {},
            },
        ),
        execution_audit_json=_write(
            tmp_path / "execution.json",
            {
                "passed": True,
                "checks": {
                    "required_patch_kinds_present": False,
                    "no_rows_use_recorded_patch_as_proposal": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {"success_rate": 1.0},
            },
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["diverse_patch_kinds_in_execution"] is False


def test_phase2ar_sufficiency_report_records_failed_reproduction_as_boundary(
    tmp_path: Path,
) -> None:
    report = build_phase2ar_evidence_sufficiency_report(
        data_health_json=_write(
            tmp_path / "data.json",
            {
                "passed": True,
                "checks": {
                    "holdout_required_patch_kinds_present": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {},
            },
        ),
        execution_audit_json=_write(
            tmp_path / "execution.json",
            {
                "passed": True,
                "checks": {
                    "required_patch_kinds_present": True,
                    "no_rows_use_recorded_patch_as_proposal": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {"row_count": 32, "success_rate": 1.0},
            },
        ),
        control_delta_json=_write(
            tmp_path / "control.json",
            {
                "passed": True,
                "checks": {
                    "controls_nonzero": True,
                    "best_control_below_ceiling": True,
                    "full_minus_best_control_met": True,
                },
                "metrics": {"full_minus_best_control": 0.5},
            },
        ),
        reproduction_audit_json=_write(
            tmp_path / "repro.json",
            {
                "passed": False,
                "metrics": {
                    "reproduction_success_rate": 0.21875,
                    "failure_reasons": {"rollback_safety_head_not_authorized": 25},
                },
            },
        ),
    )

    assert report["passed"] is True
    assert "phase2ar_cross_package_reproduction_supported" in report["unsupported_claims"]
    assert "do_not_claim_cross_package_reproduction_from_failed_run" in report["blocked_actions"]
    assert report["metrics"]["reproduction"]["reproduction_success_rate"] == 0.21875


def test_phase2ar_sufficiency_report_includes_two_seed_reproduction_smoke(
    tmp_path: Path,
) -> None:
    report = build_phase2ar_evidence_sufficiency_report(
        data_health_json=_write(
            tmp_path / "data.json",
            {
                "passed": True,
                "checks": {
                    "holdout_required_patch_kinds_present": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {},
            },
        ),
        execution_audit_json=_write(
            tmp_path / "execution.json",
            {
                "passed": True,
                "checks": {
                    "required_patch_kinds_present": True,
                    "no_rows_use_recorded_patch_as_proposal": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {"row_count": 32, "success_rate": 1.0},
            },
        ),
        control_delta_json=_write(
            tmp_path / "control.json",
            {
                "passed": True,
                "checks": {
                    "controls_nonzero": True,
                    "best_control_below_ceiling": True,
                    "full_minus_best_control_met": True,
                },
                "metrics": {"full_minus_best_control": 0.5},
            },
        ),
        reproduction_audit_json=_write(
            tmp_path / "repro.json",
            {
                "passed": True,
                "supported_claims": [
                    "phase2ar_cross_package_reproduction_supported",
                    "phase2ar_two_seed_reproduction_smoke_supported",
                ],
                "unsupported_claims": ["multi_seed_reproduction_3plus"],
                "blocked_actions": ["do_not_claim_robust_multi_seed_reproduction_until_3plus_seeds"],
                "metrics": {
                    "reproduction_success_rate": 1.0,
                    "training_contract": {"seed_changed": True},
                },
            },
        ),
    )

    assert report["passed"] is True
    assert "phase2ar_two_seed_reproduction_smoke_supported" in report["supported_claims"]
    assert "multi_seed_reproduction_3plus" in report["unsupported_claims"]


def test_phase2ar_sufficiency_report_includes_three_seed_aggregate(
    tmp_path: Path,
) -> None:
    report = build_phase2ar_evidence_sufficiency_report(
        data_health_json=_write(
            tmp_path / "data.json",
            {
                "passed": True,
                "checks": {
                    "holdout_required_patch_kinds_present": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {},
            },
        ),
        execution_audit_json=_write(
            tmp_path / "execution.json",
            {
                "passed": True,
                "checks": {
                    "required_patch_kinds_present": True,
                    "no_rows_use_recorded_patch_as_proposal": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {"row_count": 32, "success_rate": 1.0},
            },
        ),
        control_delta_json=_write(
            tmp_path / "control.json",
            {
                "passed": True,
                "checks": {
                    "controls_nonzero": True,
                    "best_control_below_ceiling": True,
                    "full_minus_best_control_met": True,
                },
                "metrics": {"full_minus_best_control": 0.5},
            },
        ),
        multiseed_reproduction_json=_write(
            tmp_path / "multiseed.json",
            {
                "passed": True,
                "supported_claims": [
                    "phase2ar_three_seed_same_model_reproduction_supported"
                ],
                "unsupported_claims": ["cross_model_reproduction"],
                "blocked_actions": [
                    "do_not_claim_cross_model_reproduction_from_same_model_seed_runs"
                ],
                "metrics": {"unique_seeds": [13, 17, 23]},
            },
        ),
    )

    assert report["passed"] is True
    assert (
        "phase2ar_three_seed_same_model_reproduction_supported"
        in report["supported_claims"]
    )
    assert (
        "multi_seed_or_cross_model_reproduction_after_nonsealed_gates"
        not in report["next_required_evidence"]
    )
    assert (
        "cross_model_reproduction_after_same_model_seed_gate"
        in report["next_required_evidence"]
    )
    assert "multi_seed_reproduction_3plus" not in report["unsupported_claims"]


def test_phase2ar_sufficiency_report_includes_cross_model_reproduction(
    tmp_path: Path,
) -> None:
    report = build_phase2ar_evidence_sufficiency_report(
        data_health_json=_write(
            tmp_path / "data.json",
            {
                "passed": True,
                "checks": {
                    "holdout_required_patch_kinds_present": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {},
            },
        ),
        execution_audit_json=_write(
            tmp_path / "execution.json",
            {
                "passed": True,
                "checks": {
                    "required_patch_kinds_present": True,
                    "no_rows_use_recorded_patch_as_proposal": True,
                    "sealed_feedback_absent": True,
                },
                "metrics": {"row_count": 32, "success_rate": 1.0},
            },
        ),
        control_delta_json=_write(
            tmp_path / "control.json",
            {
                "passed": True,
                "checks": {
                    "controls_nonzero": True,
                    "best_control_below_ceiling": True,
                    "full_minus_best_control_met": True,
                },
                "metrics": {"full_minus_best_control": 0.5},
            },
        ),
        multiseed_reproduction_json=_write(
            tmp_path / "multiseed.json",
            {
                "passed": True,
                "supported_claims": [
                    "phase2ar_three_seed_same_model_reproduction_supported"
                ],
                "unsupported_claims": ["cross_model_reproduction"],
                "blocked_actions": [
                    "do_not_claim_cross_model_reproduction_from_same_model_seed_runs"
                ],
                "metrics": {"unique_seeds": [13, 17, 23]},
            },
        ),
        cross_model_reproduction_json=_write(
            tmp_path / "cross_model.json",
            {
                "passed": True,
                "supported_claims": [
                    "phase2ar_qwen7b_to_qwen3b_same_family_reproduction_supported"
                ],
                "unsupported_claims": ["sealed_cross_model_transfer"],
                "blocked_actions": [
                    "do_not_claim_sealed_cross_model_transfer_from_nonsealed_phase2ar"
                ],
                "metrics": {"cross_model": "qwen3b"},
            },
        ),
    )

    assert report["passed"] is True
    assert (
        "phase2ar_qwen7b_to_qwen3b_same_family_reproduction_supported"
        in report["supported_claims"]
    )
    assert "cross_model_reproduction" not in report["unsupported_claims"]
    assert "cross_model_reproduction_after_same_model_seed_gate" not in report["next_required_evidence"]
