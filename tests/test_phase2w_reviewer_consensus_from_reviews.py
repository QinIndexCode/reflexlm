import json
from pathlib import Path

from reflexlm.cli.build_phase2w_reviewer_consensus_from_reviews import (
    build_phase2w_reviewer_consensus_from_reviews,
)


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _review(role: str, *, verdict: str = "major_blockers") -> dict:
    return {
        "role": role,
        "read_only": True,
        "verdict": verdict,
        "must_fix_before_epoch_claim": ["missing live-agent baseline"]
        if verdict != "approve_epoch_claim_debate_after_measured_gates"
        else [],
    }


def test_phase2w_reviewer_consensus_from_reviews_preserves_must_fix_items(
    tmp_path: Path,
) -> None:
    report = build_phase2w_reviewer_consensus_from_reviews(
        review_json_paths=[
            _write(tmp_path / "a.json", _review("architecture_mechanism_reviewer")),
            _write(tmp_path / "b.json", _review("software_repair_reviewer")),
        ]
    )
    assert report["read_only"] is True
    assert report["unanimous"] is False
    assert report["must_fix_before_epoch_claim"]


def test_phase2w_reviewer_consensus_from_reviews_detects_unanimity(
    tmp_path: Path,
) -> None:
    verdict = "approve_epoch_claim_debate_after_measured_gates"
    roles = [
        "architecture_mechanism_reviewer",
        "software_repair_reviewer",
        "reproducibility_reviewer",
        "adversarial_overclaim_reviewer",
    ]
    report = build_phase2w_reviewer_consensus_from_reviews(
        review_json_paths=[
            _write(tmp_path / f"{role}.json", _review(role, verdict=verdict))
            for role in roles
        ]
    )
    assert report["role_coverage_complete"] is True
    assert report["unanimous"] is True
    assert report["must_fix_before_epoch_claim"] == []
