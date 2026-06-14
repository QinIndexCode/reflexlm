from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_ROLES = {
    "architecture_mechanism_reviewer",
    "software_repair_reviewer",
    "reproducibility_reviewer",
    "adversarial_overclaim_reviewer",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def audit_phase2w_reviewer_consensus(*, review_json: str | Path) -> dict[str, Any]:
    review = _read_json(review_json)
    reviews = [item for item in _list(review.get("reviews")) if isinstance(item, dict)]
    roles = {str(item.get("role")) for item in reviews}
    must_fix = [
        finding
        for item in reviews
        for finding in _list(item.get("must_fix_before_epoch_claim"))
    ]
    approvals = [
        item.get("verdict") == "approve_epoch_claim_debate_after_measured_gates"
        for item in reviews
    ]
    checks = {
        "read_only": review.get("read_only") is True,
        "required_roles_present": REQUIRED_ROLES.issubset(roles),
        "unanimous": bool(approvals) and all(approvals),
        "no_must_fix_before_epoch_claim": not must_fix,
        "does_not_create_new_claims": review.get("does_not_create_new_claims") is True,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2w_reviewer_consensus",
        "passed": passed,
        "read_only": review.get("read_only") is True,
        "unanimous": checks["unanimous"],
        "checks": checks,
        "roles": sorted(roles),
        "must_fix_before_epoch_claim": must_fix,
        "blocked_actions": []
        if passed
        else ["do_not_use_reviewer_simulation_to_upgrade_claim"],
        "inputs": {"review_json": str(Path(review_json))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2W reviewer consensus.")
    parser.add_argument("--review-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2w_reviewer_consensus(review_json=args.review_json)
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
