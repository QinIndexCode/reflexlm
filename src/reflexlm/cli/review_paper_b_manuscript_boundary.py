from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _contains_all(text: str, phrases: list[str]) -> bool:
    normalized = re.sub(r"\s+", " ", text).lower()
    return all(phrase.lower() in normalized for phrase in phrases)


def build_paper_b_manuscript_boundary_review(
    *,
    manuscript_tex: str | Path,
    claim_boundary_tex: str | Path,
    positive_evidence_tex: str | Path,
    negative_evidence_tex: str | Path,
    baseline_zero_summary_tex: str | Path,
) -> dict[str, Any]:
    manuscript = _read(manuscript_tex)
    claim_boundary = _read(claim_boundary_tex)
    positive_evidence = _read(positive_evidence_tex)
    negative_evidence = _read(negative_evidence_tex)
    baseline_zero = _read(baseline_zero_summary_tex)

    figure_refs = re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", manuscript)
    table_refs = re.findall(r"\\input\{(tables/[^}]+\.tex)\}", manuscript)

    checks = {
        "has_claim_boundary_section": r"\section{Claim Boundary}" in manuscript,
        "has_threats_section": r"\section{Threats to Validity}" in manuscript,
        "has_remaining_boundary_section": r"\section{Remaining Evidence Boundary}" in manuscript,
        "uses_generated_figures": len(figure_refs) >= 6,
        "uses_generated_tables": len(table_refs) >= 4,
        "claim_boundary_marks_autonomy_unsupported": _contains_all(
            claim_boundary,
            ["unsupported", "production autonomy", "open-ended repair"],
        ),
        "claim_boundary_marks_epoch_making_unsupported": _contains_all(
            claim_boundary,
            ["unsupported", "epoch-making architecture status"],
        ),
        "positive_evidence_declares_same_family_boundary": _contains_all(
            positive_evidence,
            ["Phase2S", "same-family", "not production autonomy"],
        ),
        "negative_evidence_retains_failed_phases": _contains_all(
            negative_evidence,
            ["Phase2I", "Phase2J initial smoke", "Phase2K sealed gate", "Phase2L sealed gate", "Phase2M synthetic-safe smoke"],
        ),
        "baseline_zero_summary_present": _contains_all(
            baseline_zero,
            ["Interpretability audit for zero-valued controls"],
        ),
        "manuscript_explicitly_rejects_strong_claims": _contains_all(
            manuscript,
            [
                "does not support production autonomy",
                "epoch-making architecture claim",
                "independent reproduction",
                "open-ended repair",
            ],
        ),
    }

    bounded_submission_candidate = all(checks.values())
    top_tier_strong_claim_ready = False

    return {
        "artifact_family": "paper_b_manuscript_boundary_review",
        "passed": bounded_submission_candidate,
        "bounded_submission_candidate": bounded_submission_candidate,
        "top_tier_strong_claim_ready": top_tier_strong_claim_ready,
        "checks": checks,
        "metrics": {
            "figure_reference_count": len(figure_refs),
            "table_reference_count": len(table_refs),
        },
        "supported_submission_frame": (
            "bounded_mechanism_manuscript_with_explicit_negative_evidence_and_scope_controls"
            if bounded_submission_candidate
            else "manuscript_boundary_not_yet_hard_locked"
        ),
        "unsupported_submission_frame": [
            "top-tier strong architecture claim paper",
            "production-autonomy paper",
            "epoch-making architecture paper",
        ],
        "required_for_stronger_claims": [
            "independent external reproduction beyond the local machine",
            "cross-family replication rather than only local same-family reproduction",
            "open-ended repair tasks with patch generation, rollback, and stop-condition metrics",
            "sealed transfer that does not collapse on the retained negative phases",
            "modern agent-loop baselines under the same bounded task family",
        ],
        "manuscript_paths": {
            "manuscript_tex": str(manuscript_tex),
            "claim_boundary_tex": str(claim_boundary_tex),
            "positive_evidence_tex": str(positive_evidence_tex),
            "negative_evidence_tex": str(negative_evidence_tex),
            "baseline_zero_summary_tex": str(baseline_zero_summary_tex),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper B Manuscript Boundary Review",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Bounded submission candidate: `{report['bounded_submission_candidate']}`",
        f"- Top-tier strong-claim ready: `{report['top_tier_strong_claim_ready']}`",
        f"- Supported submission frame: `{report['supported_submission_frame']}`",
        "",
        "## Checks",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Required For Stronger Claims"])
    lines.extend(f"- {item}" for item in report["required_for_stronger_claims"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Review Paper B manuscript claim boundary.")
    parser.add_argument("--manuscript-tex", required=True)
    parser.add_argument("--claim-boundary-tex", required=True)
    parser.add_argument("--positive-evidence-tex", required=True)
    parser.add_argument("--negative-evidence-tex", required=True)
    parser.add_argument("--baseline-zero-summary-tex", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = build_paper_b_manuscript_boundary_review(
        manuscript_tex=args.manuscript_tex,
        claim_boundary_tex=args.claim_boundary_tex,
        positive_evidence_tex=args.positive_evidence_tex,
        negative_evidence_tex=args.negative_evidence_tex,
        baseline_zero_summary_tex=args.baseline_zero_summary_tex,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
