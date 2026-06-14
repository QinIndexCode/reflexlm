import json
from pathlib import Path

from reflexlm.cli.decide_phase2i_next_step import build_phase2i_next_step_decision


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _latent_audit(*, passed: bool, architecture_identifiable: bool) -> dict:
    return {
        "audit_family": "phase2i_latent_necessity",
        "passed": passed,
        "checks": {
            "nsi_latent_command_identity_available": architecture_identifiable,
            "head_latent_coverage": passed,
            "latent_challenge_source_overlap_not_sufficient": True,
            "latent_challenge_same_intent_ambiguous": True,
        },
    }


def _root_cause(delta: float) -> dict:
    return {
        "analysis_family": "phase2i_pairwiseopt_gate_root_cause_after_continuation_fix",
        "deltas": {"full_minus_no_nsi_latent": delta},
    }


def test_phase2i_next_step_blocks_current_architecture_when_latent_is_not_identifiable(
    tmp_path: Path,
) -> None:
    latent = _write(
        tmp_path / "latent.json",
        _latent_audit(passed=False, architecture_identifiable=False),
    )
    root = _write(tmp_path / "root.json", _root_cause(0.0))
    gate = _write(tmp_path / "gate.json", {"passed": False})

    report = build_phase2i_next_step_decision(
        latent_necessity_audit_json=latent,
        prepackage_gate_json=gate,
        root_cause_json=root,
    )

    assert report["passed"] is False
    assert report["current_architecture_training_allowed"] is False
    assert report["recommended_direction"] == (
        "freeze_phase2i_bounded_claim_and_do_not_retrain_same_architecture"
    )
    assert "do_not_retrain_current_phase2i_architecture_for_nsi_latent_claim" in report[
        "blocked_actions"
    ]


def test_phase2i_next_step_requires_data_repair_when_architecture_is_identifiable(
    tmp_path: Path,
) -> None:
    latent = _write(
        tmp_path / "latent.json",
        _latent_audit(passed=False, architecture_identifiable=True),
    )

    report = build_phase2i_next_step_decision(latent_necessity_audit_json=latent)

    assert report["passed"] is False
    assert report["recommended_direction"] == "repair_nonsealed_latent_identifiability_before_training"
    assert "do_not_start_full_pairwise_training" in report["blocked_actions"]


def test_phase2i_next_step_allows_only_smoke_when_audits_pass(tmp_path: Path) -> None:
    latent = _write(
        tmp_path / "latent.json",
        _latent_audit(passed=True, architecture_identifiable=True),
    )

    report = build_phase2i_next_step_decision(latent_necessity_audit_json=latent)

    assert report["passed"] is True
    assert report["current_architecture_training_allowed"] is True
    assert report["paper_claim_upgrade_allowed"] is False
    assert report["recommended_direction"] == "run_nonsealed_smoke_only"
    assert report["allowed_actions"] == ["run_small_nonsealed_smoke_before_any_full_training"]


def test_phase2i_next_step_keeps_claim_bounded_on_failed_sealed_delta(tmp_path: Path) -> None:
    latent = _write(
        tmp_path / "latent.json",
        _latent_audit(passed=True, architecture_identifiable=True),
    )
    root = _write(tmp_path / "root.json", _root_cause(0.05))

    report = build_phase2i_next_step_decision(
        latent_necessity_audit_json=latent,
        root_cause_json=root,
    )

    assert report["passed"] is False
    assert report["recommended_direction"] == (
        "do_not_promote_current_package_run_new_smoke_only_if_mechanism_changes"
    )
    assert "do_not_upgrade_paper_claim_from_bounded" in report["blocked_actions"]
