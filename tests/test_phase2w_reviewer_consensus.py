import json
from pathlib import Path

from reflexlm.cli.audit_phase2w_reviewer_consensus import (
    audit_phase2w_reviewer_consensus,
)


ROLES = [
    "architecture_mechanism_reviewer",
    "software_repair_reviewer",
    "reproducibility_reviewer",
    "adversarial_overclaim_reviewer",
]


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _review(*, must_fix: bool = False, read_only: bool = True) -> dict:
    return {
        "read_only": read_only,
        "does_not_create_new_claims": True,
        "reviews": [
            {
                "role": role,
                "verdict": "approve_epoch_claim_debate_after_measured_gates",
                "must_fix_before_epoch_claim": ["missing external runner"] if must_fix else [],
            }
            for role in ROLES
        ],
    }


def test_phase2w_reviewer_consensus_accepts_unanimous_readonly_review(tmp_path: Path) -> None:
    report = audit_phase2w_reviewer_consensus(
        review_json=_write(tmp_path / "review.json", _review())
    )
    assert report["passed"] is True
    assert report["read_only"] is True
    assert report["unanimous"] is True


def test_phase2w_reviewer_consensus_rejects_must_fix_items(tmp_path: Path) -> None:
    report = audit_phase2w_reviewer_consensus(
        review_json=_write(tmp_path / "review.json", _review(must_fix=True))
    )
    assert report["passed"] is False
    assert report["must_fix_before_epoch_claim"]


def test_phase2w_reviewer_consensus_rejects_non_readonly_review(tmp_path: Path) -> None:
    report = audit_phase2w_reviewer_consensus(
        review_json=_write(tmp_path / "review.json", _review(read_only=False))
    )
    assert report["passed"] is False
    assert report["checks"]["read_only"] is False
