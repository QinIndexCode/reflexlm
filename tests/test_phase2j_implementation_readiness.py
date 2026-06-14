import json
from pathlib import Path

from reflexlm.cli.audit_phase2j_implementation_readiness import (
    build_phase2j_implementation_readiness_audit,
)
from reflexlm.llm.receptor_latent import COMMAND_IDENTITY_LATENT_FIELDS


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _proposal() -> dict:
    return {
        "mechanism": {
            "uses_command_or_slot_identity_latent": True,
            "command_identity_provenance": "runtime receptor observed evidence and static analysis",
            "derives_identity_from_gold_label": False,
            "runtime_available_without_target_label": True,
        },
        "data_policy": {
            "train_profile": "phase2j_semantic_train",
            "val_profile": "phase2j_semantic_val",
        }
    }


def test_phase2j_readiness_blocks_missing_fields_and_profiles(tmp_path: Path) -> None:
    prereg = _write(
        tmp_path / "prereg.json",
        {
            "passed": True,
            "next_action": "prepare_nonsealed_phase2j_data_and_latent_necessity_audit",
        },
    )
    proposal = _write(
        tmp_path / "proposal.json",
        {
            **_proposal(),
            "data_policy": {"train_profile": "missing_train", "val_profile": "missing_val"},
        },
    )

    report = build_phase2j_implementation_readiness_audit(
        preregistration_check_json=prereg,
        proposal_json=proposal,
        nsi_latent_fields=["salience", "risk"],
    )

    assert report["passed"] is False
    assert report["ready_for_training"] is False
    assert "do_not_generate_phase2j_head_split_until_latent_fields_exist" in report[
        "blocked_actions"
    ]
    assert "do_not_generate_phase2j_data_until_nonsealed_profiles_exist" in report[
        "blocked_actions"
    ]
    assert "do_not_generate_phase2j_data_until_explicit_nonsealed_profiles_are_selected" in report[
        "blocked_actions"
    ]


def test_phase2j_readiness_accepts_when_fields_and_profiles_exist(tmp_path: Path) -> None:
    prereg = _write(
        tmp_path / "prereg.json",
        {
            "passed": True,
            "next_action": "prepare_nonsealed_phase2j_data_and_latent_necessity_audit",
        },
    )
    proposal = _write(
        tmp_path / "proposal.json",
        _proposal(),
    )

    report = build_phase2j_implementation_readiness_audit(
        preregistration_check_json=prereg,
        proposal_json=proposal,
        nsi_latent_fields=["salience", *COMMAND_IDENTITY_LATENT_FIELDS],
    )

    assert report["passed"] is True
    assert report["ready_for_data_generation"] is True
    assert report["ready_for_training"] is False


def test_phase2j_readiness_accepts_source_overlap_hard_profiles(tmp_path: Path) -> None:
    prereg = _write(
        tmp_path / "prereg.json",
        {
            "passed": True,
            "next_action": "prepare_nonsealed_phase2j_data_and_latent_necessity_audit",
        },
    )
    payload = _proposal()
    payload["data_policy"] = {
        "train_profile": "phase2j_source_overlap_hard_train",
        "val_profile": "phase2j_source_overlap_hard_val",
    }
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2j_implementation_readiness_audit(
        preregistration_check_json=prereg,
        proposal_json=proposal,
        nsi_latent_fields=["salience", *COMMAND_IDENTITY_LATENT_FIELDS],
    )

    assert report["passed"] is True
    assert report["checks"]["phase2j_profiles_are_explicit_nonsealed"] is True


def test_phase2j_readiness_accepts_source_overlap_hard_actiongate_profiles(
    tmp_path: Path,
) -> None:
    prereg = _write(
        tmp_path / "prereg.json",
        {
            "passed": True,
            "next_action": "prepare_nonsealed_phase2j_data_and_latent_necessity_audit",
        },
    )
    payload = _proposal()
    payload["data_policy"] = {
        "train_profile": "phase2j_source_overlap_hard_actiongate_train",
        "val_profile": "phase2j_source_overlap_hard_actiongate_val",
    }
    proposal = _write(tmp_path / "proposal.json", payload)

    report = build_phase2j_implementation_readiness_audit(
        preregistration_check_json=prereg,
        proposal_json=proposal,
        nsi_latent_fields=["salience", *COMMAND_IDENTITY_LATENT_FIELDS],
    )

    assert report["passed"] is True
    assert report["checks"]["phase2j_profiles_are_explicit_nonsealed"] is True


def test_phase2j_readiness_rejects_unknown_profile_fallback(tmp_path: Path) -> None:
    prereg = _write(
        tmp_path / "prereg.json",
        {
            "passed": True,
            "next_action": "prepare_nonsealed_phase2j_data_and_latent_necessity_audit",
        },
    )
    proposal = _write(
        tmp_path / "proposal.json",
        {
            **_proposal(),
            "data_policy": {"train_profile": "missing_train", "val_profile": "missing_val"},
        },
    )

    report = build_phase2j_implementation_readiness_audit(
        preregistration_check_json=prereg,
        proposal_json=proposal,
        nsi_latent_fields=COMMAND_IDENTITY_LATENT_FIELDS,
    )

    assert report["passed"] is False
    assert report["checks"]["phase2j_train_profile_exists"] is False
    assert "do_not_generate_phase2j_data_until_nonsealed_profiles_exist" in report[
        "blocked_actions"
    ]


def test_phase2j_readiness_rejects_training_oriented_preregistration(tmp_path: Path) -> None:
    prereg = _write(
        tmp_path / "prereg.json",
        {"passed": True, "next_action": "run_full_training"},
    )
    proposal = _write(tmp_path / "proposal.json", _proposal())

    report = build_phase2j_implementation_readiness_audit(
        preregistration_check_json=prereg,
        proposal_json=proposal,
        nsi_latent_fields=COMMAND_IDENTITY_LATENT_FIELDS,
    )

    assert report["passed"] is False
    assert "do_not_start_training_from_preregistration" in report["blocked_actions"]


def test_phase2j_readiness_rejects_gold_label_identity_provenance(tmp_path: Path) -> None:
    prereg = _write(
        tmp_path / "prereg.json",
        {
            "passed": True,
            "next_action": "prepare_nonsealed_phase2j_data_and_latent_necessity_audit",
        },
    )
    proposal = _write(
        tmp_path / "proposal.json",
        {
            **_proposal(),
            "mechanism": {
                "uses_command_or_slot_identity_latent": True,
                "command_identity_provenance": "gold label correct target slot",
                "derives_identity_from_gold_label": True,
                "runtime_available_without_target_label": False,
            },
        },
    )

    report = build_phase2j_implementation_readiness_audit(
        preregistration_check_json=prereg,
        proposal_json=proposal,
        nsi_latent_fields=COMMAND_IDENTITY_LATENT_FIELDS,
    )

    assert report["passed"] is False
    assert "do_not_generate_phase2j_data_until_nonlabel_identity_provenance_is_recorded" in report[
        "blocked_actions"
    ]
