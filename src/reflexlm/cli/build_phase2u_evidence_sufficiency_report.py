from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_phase2u_evidence_sufficiency_report(
    *,
    nonsealed_full_postflight_json: str | Path,
    sealed_transfer_report_json: str | Path,
    min_nonzero_controls: int = 3,
    min_nonsealed_full_minus_best: float = 0.10,
) -> dict[str, Any]:
    nonsealed = _read_json(nonsealed_full_postflight_json)
    sealed = _read_json(sealed_transfer_report_json)
    nonzero_controls = list(nonsealed.get("nonzero_controls") or [])
    nonsealed_metrics = nonsealed.get("metrics", {})
    full_minus_best = nonsealed_metrics.get("full_minus_best_non_full_task_success")
    checks = {
        "nonsealed_full_postflight_passed": nonsealed.get("passed") is True,
        "nonsealed_controls_nonzero": len(nonzero_controls) >= min_nonzero_controls,
        "nonsealed_full_beats_best_nonfull": isinstance(full_minus_best, (int, float))
        and float(full_minus_best) >= min_nonsealed_full_minus_best,
        "sealed_extreme_stress_passed": sealed.get("passed") is True,
        "sealed_mechanism_sufficiency_not_claimed": sealed.get(
            "mechanism_sufficiency_passed"
        )
        is False,
        "sealed_all_zero_caveat_recorded": "sealed-v3 all-zero controls do not prove a graded mechanism curve"
        in list(sealed.get("unsupported_claims") or []),
    }
    nonsealed_mechanism_curve_supported = all(
        checks[key]
        for key in [
            "nonsealed_full_postflight_passed",
            "nonsealed_controls_nonzero",
            "nonsealed_full_beats_best_nonfull",
        ]
    )
    sealed_stress_observation_supported = all(
        checks[key]
        for key in [
            "sealed_extreme_stress_passed",
            "sealed_mechanism_sufficiency_not_claimed",
            "sealed_all_zero_caveat_recorded",
        ]
    )
    passed = nonsealed_mechanism_curve_supported and sealed_stress_observation_supported
    return {
        "artifact_family": "phase2u_evidence_sufficiency_report",
        "passed": passed,
        "claim_scope": "phase2u_two_layer_bounded_evidence"
        if passed
        else "phase2u_evidence_chain_incomplete",
        "checks": checks,
        "nonsealed_mechanism_curve_supported": nonsealed_mechanism_curve_supported,
        "sealed_stress_observation_supported": sealed_stress_observation_supported,
        "sealed_mechanism_curve_supported": False,
        "metrics": {
            "nonsealed_full_task_success": nonsealed_metrics.get("full_task_success"),
            "nonsealed_best_non_full_task_success": nonsealed_metrics.get(
                "best_non_full_task_success"
            ),
            "nonsealed_full_minus_best_non_full_task_success": full_minus_best,
            "nonsealed_nonzero_controls": nonzero_controls,
            "sealed_claim_scope": sealed.get("claim_scope"),
            "sealed_full_completion": sealed.get("metrics", {}).get("full_completion"),
            "sealed_best_mechanism_completion": sealed.get("metrics", {}).get(
                "best_mechanism_completion"
            ),
        },
        "supported_claims": [
            "Non-sealed graded Phase2U sanity supports a bounded mechanism curve because controls are nonzero and full beats the best non-full baseline.",
            "Sealed-v3 Phase2U supports only an extreme stress observation, not a graded mechanism curve.",
        ]
        if passed
        else [],
        "unsupported_claims": [
            "The sealed-v3 all-zero control field does not independently prove mechanism sufficiency.",
            "The current evidence does not prove production autonomy, open-ended debugging generalization, or an epoch-making architecture.",
            "A stricter claim requires a held-out transfer benchmark where controls have nonzero measured feasibility and full still wins.",
        ],
        "blocked_actions": [
            "do_not_describe_sealed_v3_all_zero_deltas_as_sufficient_proof",
            "do_not_upgrade_architecture_claim_without_nonzero_transfer_controls",
        ],
        "inputs": {
            "nonsealed_full_postflight_json": str(Path(nonsealed_full_postflight_json)),
            "sealed_transfer_report_json": str(Path(sealed_transfer_report_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2U evidence sufficiency report.")
    parser.add_argument("--nonsealed-full-postflight-json", required=True)
    parser.add_argument("--sealed-transfer-report-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-nonzero-controls", type=int, default=3)
    parser.add_argument("--min-nonsealed-full-minus-best", type=float, default=0.10)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2u_evidence_sufficiency_report(
        nonsealed_full_postflight_json=args.nonsealed_full_postflight_json,
        sealed_transfer_report_json=args.sealed_transfer_report_json,
        min_nonzero_controls=args.min_nonzero_controls,
        min_nonsealed_full_minus_best=args.min_nonsealed_full_minus_best,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
