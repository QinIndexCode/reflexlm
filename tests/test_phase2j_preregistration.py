import json
from pathlib import Path

from reflexlm.cli.check_phase2j_preregistration import build_phase2j_preregistration_check


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _phase2i_frozen_decision() -> dict:
    return {
        "decision_family": "phase2i_next_step_decision",
        "current_architecture_training_allowed": False,
        "paper_claim_upgrade_allowed": False,
        "blocked_actions": [
            "do_not_retrain_current_phase2i_architecture_for_nsi_latent_claim",
            "do_not_upgrade_paper_claim_from_bounded",
        ],
    }


def _valid_phase2j_proposal() -> dict:
    return {
        "experiment_id": "phase2j_semantic_command_identity_latent_smoke",
        "phase": "Phase2J",
        "mechanism": {
            "uses_current_phase2i_architecture": False,
            "uses_command_or_slot_identity_latent": True,
            "mechanism_scope_changed": True,
            "command_identity_provenance": "runtime receptor observed evidence and static analysis",
            "derives_identity_from_gold_label": False,
            "runtime_available_without_target_label": True,
            "generalizes_beyond_named_tests": True,
        },
        "data_policy": {
            "training_roots": ["artifacts/datasets/phase2j_semantic_train"],
            "validation_roots": ["artifacts/datasets/phase2j_semantic_val"],
            "eval_only_roots": ["artifacts/datasets/phase2i_external_trace_v3_semantic_required"],
            "uses_sealed_for_training": False,
            "uses_sealed_for_tuning": False,
            "uses_sealed_failures_for_analysis_feedback": False,
        },
        "gate_policy": {
            "latent_necessity_audit_required": True,
            "effective_split_hash_required": True,
            "source_overlap_baseline_required": True,
            "same_intent_ambiguity_required": True,
            "train_val_intent_coverage_required": True,
            "smoke_first_required": True,
            "full_train_requires_smoke_pass": True,
            "package_requires_prepackage_gate": True,
            "sealed_eval_requires_package": True,
        },
        "execution_plan": {
            "next_action": "prepare_nonsealed_data_and_audit",
            "starts_full_training": False,
            "starts_package": False,
            "starts_sealed_eval": False,
        },
        "paper_claim": {
            "upgrades_phase2i_claim": False,
            "bounded_until_gates_pass": True,
            "separate_phase2j_claim_required": True,
        },
    }


def test_phase2j_preregistration_accepts_separate_nonsealed_smoke_plan(tmp_path: Path) -> None:
    decision = _write(tmp_path / "decision.json", _phase2i_frozen_decision())
    proposal = _write(tmp_path / "proposal.json", _valid_phase2j_proposal())

    report = build_phase2j_preregistration_check(
        phase2i_decision_json=decision,
        proposal_json=proposal,
    )

    assert report["passed"] is True
    assert report["next_action"] == "prepare_nonsealed_phase2j_data_and_latent_necessity_audit"
    assert report["checks"]["no_sealed_training_or_tuning"] is True


def test_phase2j_preregistration_rejects_same_architecture_retrain(tmp_path: Path) -> None:
    decision = _write(tmp_path / "decision.json", _phase2i_frozen_decision())
    payload = _valid_phase2j_proposal()
    payload["mechanism"]["uses_current_phase2i_architecture"] = True
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2j_preregistration_check(
        phase2i_decision_json=decision,
        proposal_json=proposal,
    )

    assert report["passed"] is False
    assert "do_not_retrain_current_phase2i_architecture" in report["blocked_actions"]


def test_phase2j_preregistration_rejects_sealed_training_paths(tmp_path: Path) -> None:
    decision = _write(tmp_path / "decision.json", _phase2i_frozen_decision())
    payload = _valid_phase2j_proposal()
    payload["data_policy"]["training_roots"] = [
        "artifacts/datasets/phase2i_external_trace_v3_semantic_required"
    ]
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2j_preregistration_check(
        phase2i_decision_json=decision,
        proposal_json=proposal,
    )

    assert report["passed"] is False
    assert "do_not_use_sealed_data_for_training_tuning_or_feedback" in report["blocked_actions"]


def test_phase2j_preregistration_does_not_reject_nonsealed_name(tmp_path: Path) -> None:
    decision = _write(tmp_path / "decision.json", _phase2i_frozen_decision())
    payload = _valid_phase2j_proposal()
    payload["data_policy"]["training_roots"] = ["artifacts/datasets/phase2j_nonsealed_train"]
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2j_preregistration_check(
        phase2i_decision_json=decision,
        proposal_json=proposal,
    )

    assert report["passed"] is True
    assert report["observations"]["sealed_training_paths"] == []


def test_phase2j_preregistration_rejects_command_identity_under_phase2i_scope(
    tmp_path: Path,
) -> None:
    decision = _write(tmp_path / "decision.json", _phase2i_frozen_decision())
    payload = _valid_phase2j_proposal()
    payload["experiment_id"] = "phase2i_semantic_pairwise_retry"
    payload["phase"] = "Phase2I"
    payload["mechanism"]["mechanism_scope_changed"] = False
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2j_preregistration_check(
        phase2i_decision_json=decision,
        proposal_json=proposal,
    )

    assert report["passed"] is False
    assert "do_not_add_command_identity_latent_under_phase2i_claim" in report["blocked_actions"]


def test_phase2j_preregistration_rejects_gold_label_identity_provenance(tmp_path: Path) -> None:
    decision = _write(tmp_path / "decision.json", _phase2i_frozen_decision())
    payload = _valid_phase2j_proposal()
    payload["mechanism"]["command_identity_provenance"] = "gold target label"
    payload["mechanism"]["derives_identity_from_gold_label"] = True
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2j_preregistration_check(
        phase2i_decision_json=decision,
        proposal_json=proposal,
    )

    assert report["passed"] is False
    assert (
        "do_not_use_gold_label_test_name_or_sealed_failure_as_latent_identity"
        in report["blocked_actions"]
    )


def test_phase2j_preregistration_rejects_package_before_gates(tmp_path: Path) -> None:
    decision = _write(tmp_path / "decision.json", _phase2i_frozen_decision())
    payload = _valid_phase2j_proposal()
    payload["execution_plan"]["starts_package"] = True
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2j_preregistration_check(
        phase2i_decision_json=decision,
        proposal_json=proposal,
    )

    assert report["passed"] is False
    assert "do_not_package_or_run_sealed_eval_before_gates" in report["blocked_actions"]
