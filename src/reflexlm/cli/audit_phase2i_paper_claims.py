from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PHASE2I_RE = re.compile(r"\bphase\s*2i\b|\bphase2i\b", re.IGNORECASE)
UPGRADE_RE = re.compile(
    r"\b(proves?|proven|supports?|supported|validates?|validated|demonstrates?|"
    r"demonstrated|confirms?|confirmed|establishes?|established)\b",
    re.IGNORECASE,
)
SEMANTIC_NSI_RE = re.compile(
    r"(semantic[- ]required|nsi[- ]latent|native nervous[- ]interface mechanism|"
    r"semantic command[- ]slot)",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(not|does not|do not|cannot|failed|fails|insufficient|not yet|no measurable|"
    r"bounded rather than upgraded|remains bounded)\b",
    re.IGNORECASE,
)
BOUNDED_RE = re.compile(
    r"(Phase\s*2I|Phase2I).{0,220}(bounded|not prove|does not prove|"
    r"not supported|do not retrain|full[- ]minus[- ]no[- ]NSI|no[- ]NSI|"
    r"not upgrade|current architecture training)",
    re.IGNORECASE | re.DOTALL,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _forbidden_upgrade_claims(text: str) -> list[str]:
    hits: list[str] = []
    for paragraph in _paragraphs(text):
        if not PHASE2I_RE.search(paragraph):
            continue
        if not UPGRADE_RE.search(paragraph) or not SEMANTIC_NSI_RE.search(paragraph):
            continue
        if NEGATION_RE.search(paragraph):
            continue
        hits.append(" ".join(paragraph.split())[:500])
    return hits


def build_phase2i_paper_claim_audit(
    *,
    paper_path: str | Path,
    decision_json: str | Path,
) -> dict[str, Any]:
    paper = Path(paper_path)
    decision = _load_json(decision_json)
    text = paper.read_text(encoding="utf-8")

    training_allowed = decision.get("current_architecture_training_allowed") is True
    claim_upgrade_allowed = decision.get("paper_claim_upgrade_allowed") is True
    recommended_direction = str(decision.get("recommended_direction") or "")
    blocked_actions = set(decision.get("blocked_actions") or [])
    frozen_by_decision = (
        not training_allowed
        or not claim_upgrade_allowed
        or "do_not_upgrade_paper_claim_from_bounded" in blocked_actions
        or recommended_direction.startswith("freeze_phase2i")
    )
    forbidden_claims = _forbidden_upgrade_claims(text)
    phase2i_mentioned = bool(PHASE2I_RE.search(text))
    bounded_statement_present = bool(BOUNDED_RE.search(text))
    decision_artifact_referenced = Path(decision_json).name in text

    checks = {
        "paper_exists": paper.exists(),
        "decision_freezes_phase2i_claim": frozen_by_decision,
        "phase2i_bounded_statement_present": (
            True if not frozen_by_decision else phase2i_mentioned and bounded_statement_present
        ),
        "phase2i_decision_artifact_referenced": (
            True if not frozen_by_decision else decision_artifact_referenced
        ),
        "no_forbidden_phase2i_upgrade_claim": not forbidden_claims,
    }
    return {
        "audit_family": "phase2i_paper_claim_guard",
        "passed": all(checks.values()),
        "checks": checks,
        "recommended_direction": recommended_direction,
        "paper_claim_upgrade_allowed": claim_upgrade_allowed,
        "current_architecture_training_allowed": training_allowed,
        "forbidden_upgrade_claims": forbidden_claims,
        "inputs": {
            "paper_path": str(paper),
            "decision_json": str(Path(decision_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard Phase2I paper claims against overstatement.")
    parser.add_argument("--paper-path", default="paper_draft.md")
    parser.add_argument("--decision-json", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = build_phase2i_paper_claim_audit(
        paper_path=args.paper_path,
        decision_json=args.decision_json,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
