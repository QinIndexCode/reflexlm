from pathlib import Path

from reflexlm.cli.review_paper_b_manuscript_boundary import (
    build_paper_b_manuscript_boundary_review,
)


ROOT = Path(__file__).resolve().parents[1]


def test_paper_b_boundary_review_preserves_bounded_submission_frame() -> None:
    report = build_paper_b_manuscript_boundary_review(
        manuscript_tex=ROOT / "docs" / "paper_b" / "main.tex",
        claim_boundary_tex=ROOT / "docs" / "paper_b" / "tables" / "claim_boundary.tex",
        positive_evidence_tex=ROOT / "docs" / "paper_b" / "tables" / "positive_evidence_matrix.tex",
        negative_evidence_tex=ROOT / "docs" / "paper_b" / "tables" / "negative_evidence.tex",
        baseline_zero_summary_tex=ROOT / "docs" / "paper_b" / "tables" / "baseline_zero_summary.tex",
    )

    assert report["passed"] is True
    assert report["bounded_submission_candidate"] is True
    assert report["top_tier_strong_claim_ready"] is False
    assert (
        report["supported_submission_frame"]
        == "bounded_mechanism_manuscript_with_explicit_negative_evidence_and_scope_controls"
    )
    assert "top-tier strong architecture claim paper" in report["unsupported_submission_frame"]
    assert report["checks"]["claim_boundary_marks_autonomy_unsupported"] is True
    assert report["checks"]["claim_boundary_marks_epoch_making_unsupported"] is True
    assert report["checks"]["negative_evidence_retains_failed_phases"] is True
    assert report["metrics"]["figure_reference_count"] >= 6
    assert report["metrics"]["table_reference_count"] >= 4
