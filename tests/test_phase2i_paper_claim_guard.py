import json
from pathlib import Path

from reflexlm.cli.audit_phase2i_paper_claims import build_phase2i_paper_claim_audit


def _write(path: Path, payload: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _frozen_decision() -> dict:
    return {
        "recommended_direction": "freeze_phase2i_bounded_claim_and_do_not_retrain_same_architecture",
        "current_architecture_training_allowed": False,
        "paper_claim_upgrade_allowed": False,
        "blocked_actions": [
            "do_not_retrain_current_phase2i_architecture_for_nsi_latent_claim",
            "do_not_upgrade_paper_claim_from_bounded",
        ],
    }


def test_phase2i_paper_claim_guard_accepts_bounded_wording(tmp_path: Path) -> None:
    decision = _write(
        tmp_path / "phase2i_next_step_decision_after_latent_identifiability.json",
        _frozen_decision(),
    )
    paper = _write(
        tmp_path / "paper.md",
        "\n".join(
            [
                "# Paper",
                "",
                "Phase2I remains a bounded claim: it does not prove NSI latent necessity, "
                "and the current architecture training is not allowed. Evidence is recorded in "
                "phase2i_next_step_decision_after_latent_identifiability.json.",
            ]
        ),
    )

    report = build_phase2i_paper_claim_audit(paper_path=paper, decision_json=decision)

    assert report["passed"] is True
    assert report["checks"]["phase2i_bounded_statement_present"] is True
    assert report["checks"]["phase2i_decision_artifact_referenced"] is True


def test_phase2i_paper_claim_guard_rejects_upgrade_wording(tmp_path: Path) -> None:
    decision = _write(tmp_path / "decision.json", _frozen_decision())
    paper = _write(
        tmp_path / "paper.md",
        "Phase2I proves the semantic-required NSI latent native nervous-interface mechanism.",
    )

    report = build_phase2i_paper_claim_audit(paper_path=paper, decision_json=decision)

    assert report["passed"] is False
    assert report["checks"]["no_forbidden_phase2i_upgrade_claim"] is False
    assert report["forbidden_upgrade_claims"]


def test_phase2i_paper_claim_guard_requires_decision_artifact_reference(tmp_path: Path) -> None:
    decision = _write(
        tmp_path / "phase2i_next_step_decision_after_latent_identifiability.json",
        _frozen_decision(),
    )
    paper = _write(
        tmp_path / "paper.md",
        "Phase2I remains bounded and does not prove NSI latent necessity.",
    )

    report = build_phase2i_paper_claim_audit(paper_path=paper, decision_json=decision)

    assert report["passed"] is False
    assert report["checks"]["phase2i_decision_artifact_referenced"] is False
