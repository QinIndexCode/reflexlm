from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_phase2w_reviewer_consensus_from_reviews(
    *,
    review_json_paths: list[str | Path],
) -> dict[str, Any]:
    reviews = []
    for raw_path in review_json_paths:
        review = _read_json(raw_path)
        review["source_path"] = str(Path(raw_path))
        reviews.append(review)
    must_fix = [
        finding
        for review in reviews
        for finding in (review.get("must_fix_before_epoch_claim") or [])
    ]
    required_roles = {
        "architecture_mechanism_reviewer",
        "software_repair_reviewer",
        "reproducibility_reviewer",
        "adversarial_overclaim_reviewer",
    }
    roles = {str(review.get("role")) for review in reviews}
    unanimous = bool(reviews) and all(
        review.get("verdict") == "approve_epoch_claim_debate_after_measured_gates"
        for review in reviews
    )
    return {
        "artifact_family": "phase2w_reviewer_consensus_source",
        "read_only": all(review.get("read_only") is True for review in reviews),
        "does_not_create_new_claims": True,
        "role_coverage_complete": required_roles.issubset(roles),
        "unanimous": unanimous,
        "reviews": reviews,
        "must_fix_before_epoch_claim": must_fix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2W reviewer consensus source JSON.")
    parser.add_argument("--review-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2w_reviewer_consensus_from_reviews(
        review_json_paths=args.review_json
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
