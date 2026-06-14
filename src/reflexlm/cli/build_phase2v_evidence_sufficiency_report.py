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


def build_phase2v_evidence_sufficiency_report(
    *,
    phase2u_evidence_json: str | Path,
    phase2v_postflight_json: str | Path,
    phase2v_sealed_block_json: str | Path,
    phase2v_independence_json: str | Path | None = None,
) -> dict[str, Any]:
    phase2u = _read_json(phase2u_evidence_json)
    postflight = _read_json(phase2v_postflight_json)
    sealed_block = _read_json(phase2v_sealed_block_json)
    independence = _read_json(phase2v_independence_json) if phase2v_independence_json else {}
    checks = {
        "phase2u_two_layer_boundary_passed": phase2u.get("passed") is True,
        "phase2u_sealed_curve_not_claimed": phase2u.get("sealed_mechanism_curve_supported")
        is False,
        "phase2v_postflight_passed": postflight.get("passed") is True,
        "phase2v_controls_nonzero": bool(postflight.get("nonzero_controls")),
        "phase2v_independence_passed": independence.get("passed") is True
        if independence
        else True,
        "phase2v_sealed_blocked": sealed_block.get("ready_for_sealed_eval") is False,
        "phase2v_no_architecture_upgrade": "do_not_upgrade_to_production_autonomy_or_epoch_making_claim"
        in list(sealed_block.get("blocked_actions") or []),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2v_evidence_sufficiency_report",
        "passed": passed,
        "claim_scope": "phase2v_graded_transfer_bounded_mechanism_evidence"
        if passed
        else "phase2v_evidence_incomplete",
        "checks": checks,
        "metrics": {
            "phase2u_claim_scope": phase2u.get("claim_scope"),
            "phase2v_full_task_success": postflight.get("metrics", {}).get(
                "full_task_success"
            ),
            "phase2v_best_nonfull_task_success": postflight.get("metrics", {}).get(
                "best_nonfull_task_success"
            ),
            "phase2v_full_minus_best_nonfull": postflight.get("metrics", {}).get(
                "full_minus_best_nonfull_task_success"
            ),
            "phase2v_nonzero_controls": postflight.get("nonzero_controls"),
            "phase2v_independence_identity_hashes": independence.get("identity_hashes"),
        },
        "supported_claims": [
            "Phase2V supports held-out graded transfer evidence because controls are nonzero and full still beats the best non-full baseline."
        ]
        if passed
        else [],
        "unsupported_claims": [
            "Phase2V does not prove production autonomy.",
            "Phase2V does not prove open-ended debugging generalization.",
            "Phase2V does not prove an epoch-making architecture.",
            "Phase2V does not authorize sealed-v3 tuning or additional sealed evaluation.",
        ],
        "inputs": {
            "phase2u_evidence_json": str(Path(phase2u_evidence_json)),
            "phase2v_postflight_json": str(Path(phase2v_postflight_json)),
            "phase2v_sealed_block_json": str(Path(phase2v_sealed_block_json)),
            "phase2v_independence_json": str(Path(phase2v_independence_json))
            if phase2v_independence_json
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2V evidence sufficiency report.")
    parser.add_argument("--phase2u-evidence-json", required=True)
    parser.add_argument("--phase2v-postflight-json", required=True)
    parser.add_argument("--phase2v-sealed-block-json", required=True)
    parser.add_argument("--phase2v-independence-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2v_evidence_sufficiency_report(
        phase2u_evidence_json=args.phase2u_evidence_json,
        phase2v_postflight_json=args.phase2v_postflight_json,
        phase2v_sealed_block_json=args.phase2v_sealed_block_json,
        phase2v_independence_json=args.phase2v_independence_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
