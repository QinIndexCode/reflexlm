import json
from pathlib import Path

from reflexlm.cli.build_phase2as_architecture_boundary_report import (
    build_phase2as_architecture_boundary_report,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase2as_boundary_report_accepts_symbolic_evidence_and_rejected_relabel(
    tmp_path: Path,
) -> None:
    report = build_phase2as_architecture_boundary_report(
        symbolic_evidence_report_json=_write(
            tmp_path / "symbolic.json",
            {
                "passed": True,
                "supported_claims": [
                    "bounded_runtime_symbolic_structural_patch_proposal_diverse_holdout_supported",
                    "phase2ar_three_seed_same_model_reproduction_supported",
                ],
                "unsupported_claims": [
                    "sealed_cross_model_transfer",
                    "epoch_making_architecture",
                ],
                "blocked_actions": ["do_not_claim_freeform_patch_generation"],
                "metrics": {"execution_success_rate": 1.0},
            },
        ),
        learned_relabel_audit_json=_write(
            tmp_path / "relabel.json",
            {
                "passed": False,
                "claimed_capability": "learned_patch_generation",
                "unsupported_claims": ["learned_patch_generation"],
                "blocked_actions": [
                    "do_not_relabel_symbolic_phase2ar_as_learned_patch_generation"
                ],
                "metrics": {
                    "evidence_class_counts": {"symbolic_runtime_generator": 32}
                },
            },
        ),
    )

    assert report["passed"] is True
    assert (
        "phase2as_claim_boundary_prevents_learned_repair_overclaim"
        in report["supported_claims"]
    )
    assert "learned_patch_generation" in report["unsupported_claims"]
    assert (
        "phase2at_native_package_learned_bounded_patch_candidate_generation"
        in report["next_required_evidence"]
    )


def test_phase2as_boundary_report_rejects_if_epoch_claim_not_blocked(
    tmp_path: Path,
) -> None:
    report = build_phase2as_architecture_boundary_report(
        symbolic_evidence_report_json=_write(
            tmp_path / "symbolic.json",
            {
                "passed": True,
                "supported_claims": [
                    "bounded_runtime_symbolic_structural_patch_proposal_diverse_holdout_supported"
                ],
                "unsupported_claims": ["sealed_cross_model_transfer"],
                "metrics": {},
            },
        ),
        learned_relabel_audit_json=_write(
            tmp_path / "relabel.json",
            {
                "passed": False,
                "claimed_capability": "learned_patch_generation",
                "unsupported_claims": ["learned_patch_generation"],
                "blocked_actions": [
                    "do_not_relabel_symbolic_phase2ar_as_learned_patch_generation"
                ],
                "metrics": {},
            },
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["epoch_making_architecture_not_claimed"] is False
