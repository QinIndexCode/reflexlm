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


def _success_rate(summary: dict[str, Any]) -> float:
    if "success_rate" in summary:
        return float(summary.get("success_rate") or 0.0)
    rows = float(summary.get("rows") or 0.0)
    successes = float(summary.get("successes") or 0.0)
    return successes / rows if rows > 0 else 0.0


def audit_phase2ar_symbolic_control_delta(
    *,
    full_summary_json: str | Path,
    text_control_summary_json: str | Path,
    attribute_control_summary_json: str | Path,
    min_full_success_rate: float = 1.0,
    min_best_control_success_rate: float = 0.20,
    max_best_control_success_rate: float = 0.75,
    min_full_minus_best_control: float = 0.15,
) -> dict[str, Any]:
    full = _read_json(full_summary_json)
    text_control = _read_json(text_control_summary_json)
    attribute_control = _read_json(attribute_control_summary_json)
    control_rates = {
        "text_only": _success_rate(text_control),
        "attribute_only": _success_rate(attribute_control),
    }
    full_rate = _success_rate(full)
    best_control_name, best_control_rate = max(control_rates.items(), key=lambda item: item[1])
    checks = {
        "full_success_rate_met": full_rate >= min_full_success_rate,
        "controls_nonzero": any(rate > 0.0 for rate in control_rates.values()),
        "best_control_above_floor": best_control_rate >= min_best_control_success_rate,
        "best_control_below_ceiling": best_control_rate <= max_best_control_success_rate,
        "full_minus_best_control_met": (full_rate - best_control_rate)
        >= min_full_minus_best_control,
        "control_summaries_are_non_claim": text_control.get("claim_boundary")
        == "phase2ar_restricted_symbolic_control_not_claim_bearing"
        and attribute_control.get("claim_boundary")
        == "phase2ar_restricted_symbolic_control_not_claim_bearing",
        "row_counts_match": full.get("rows")
        == text_control.get("rows")
        == attribute_control.get("rows"),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2ar_symbolic_control_delta_audit",
        "passed": passed,
        "claim_boundary": (
            "Phase2AR full bounded symbolic structural operator must beat measured "
            "restricted symbolic controls; this is not freeform repair evidence."
        ),
        "checks": checks,
        "metrics": {
            "full_success_rate": full_rate,
            "control_success_rates": control_rates,
            "best_control_name": best_control_name,
            "best_control_success_rate": best_control_rate,
            "full_minus_best_control": full_rate - best_control_rate,
            "row_count": full.get("rows"),
        },
        "thresholds": {
            "min_full_success_rate": min_full_success_rate,
            "min_best_control_success_rate": min_best_control_success_rate,
            "max_best_control_success_rate": max_best_control_success_rate,
            "min_full_minus_best_control": min_full_minus_best_control,
        },
        "supported_claims": [
            "phase2ar_full_symbolic_structural_beats_nonzero_restricted_controls"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "inputs": {
            "full_summary_json": str(Path(full_summary_json)),
            "text_control_summary_json": str(Path(text_control_summary_json)),
            "attribute_control_summary_json": str(Path(attribute_control_summary_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AR symbolic control deltas.")
    parser.add_argument("--full-summary-json", required=True)
    parser.add_argument("--text-control-summary-json", required=True)
    parser.add_argument("--attribute-control-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = audit_phase2ar_symbolic_control_delta(
        full_summary_json=args.full_summary_json,
        text_control_summary_json=args.text_control_summary_json,
        attribute_control_summary_json=args.attribute_control_summary_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
