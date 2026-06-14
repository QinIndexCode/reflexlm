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


def build_phase2ad_residual_budget_pressure_comparison(
    *,
    data_health_json: str | Path,
    full_summary_json: str | Path,
    policyless_summary_json: str | Path,
    output_json: str | Path,
    min_full_success_rate: float = 0.85,
    min_full_minus_best_non_full: float = 0.10,
) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    full_summary = _read_json(full_summary_json)
    policyless_summary = _read_json(policyless_summary_json)
    full_success = float(full_summary.get("success_rate") or 0.0)
    policyless_success = float(policyless_summary.get("success_rate") or 0.0)
    best_non_full = float(
        data_health.get("metrics", {}).get("best_non_full_baseline_accuracy", 0.0) or 0.0
    )
    report = {
        "artifact_family": "phase2ad_residual_budget_pressure_comparison",
        "passed": full_success >= min_full_success_rate
        and (full_success - best_non_full) >= min_full_minus_best_non_full,
        "metrics": {
            "full_success_rate": full_success,
            "policyless_slot0_budget2_success_rate": policyless_success,
            "best_non_full_baseline_accuracy": best_non_full,
            "full_minus_policyless_slot0_budget2": full_success - policyless_success,
            "full_minus_best_non_full_baseline": full_success - best_non_full,
            "identity_heuristic_holdout_accuracy": float(
                data_health.get("metrics", {})
                .get("identity_heuristic_accuracy", {})
                .get("holdout", 0.0)
                or 0.0
            ),
        },
        "interpretation": {
            "residual_selector_supported": full_success >= min_full_success_rate
            and (full_success - best_non_full) >= min_full_minus_best_non_full,
            "selector_insufficiency_observed": full_success < best_non_full,
            "claim_boundary": (
                "current package fails residual budget-pressure cases; positional sanity baseline is stronger"
                if full_success < best_non_full
                else "current package does not clear the residual budget-pressure success gate"
                if full_success < min_full_success_rate
                else "residual budget-pressure gate passed"
            ),
        },
        "thresholds": {
            "min_full_success_rate": min_full_success_rate,
            "min_full_minus_best_non_full": min_full_minus_best_non_full,
        },
        "unsupported_claims": [
            "learned_residual_selector_sufficiency",
            "freeform_patch_generation",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "full_summary_json": str(Path(full_summary_json)),
            "policyless_summary_json": str(Path(policyless_summary_json)),
        },
    }
    _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Phase2AD residual budget-pressure execution against controls."
    )
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--full-summary-json", required=True)
    parser.add_argument("--policyless-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2ad_residual_budget_pressure_comparison(
        data_health_json=args.data_health_json,
        full_summary_json=args.full_summary_json,
        policyless_summary_json=args.policyless_summary_json,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
