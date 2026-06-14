from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _initial_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = sum(
        1
        for row in rows
        if row.get("initial_selected_patch_candidate_slot") == row.get("expected_patch_candidate_slot")
    )
    return correct / len(rows)


def _retry_recovery_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("initial_selected_patch_candidate_slot") != row.get("expected_patch_candidate_slot")
        and row.get("selected_patch_candidate_slot") == row.get("expected_patch_candidate_slot")
    )


def build_phase2ac_budget_pressure_comparison(
    *,
    data_health_json: str | Path,
    full_summary_json: str | Path,
    full_results_jsonl: str | Path,
    policyless_summary_json: str | Path,
    policyless_results_jsonl: str | Path,
    output_json: str | Path,
    min_full_success_rate: float = 0.85,
    min_full_minus_policyless: float = 0.15,
) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    full_summary = _read_json(full_summary_json)
    policyless_summary = _read_json(policyless_summary_json)
    full_rows = _read_jsonl(full_results_jsonl)
    policyless_rows = _read_jsonl(policyless_results_jsonl)
    full_success = float(full_summary.get("success_rate") or 0.0)
    policyless_success = float(policyless_summary.get("success_rate") or 0.0)
    identity_holdout = float(
        data_health.get("metrics", {})
        .get("identity_heuristic_accuracy", {})
        .get("holdout", 0.0)
        or 0.0
    )
    report = {
        "artifact_family": "phase2ac_budget_pressure_comparison",
        "passed": full_success >= min_full_success_rate
        and (full_success - policyless_success) >= min_full_minus_policyless,
        "metrics": {
            "full_success_rate": full_success,
            "policyless_slot0_budget2_success_rate": policyless_success,
            "full_minus_policyless_slot0_budget2": full_success - policyless_success,
            "deterministic_identity_heuristic_holdout_accuracy": identity_holdout,
            "full_minus_deterministic_identity_heuristic": full_success - identity_holdout,
            "full_initial_selection_accuracy": _initial_accuracy(full_rows),
            "policyless_initial_selection_accuracy": _initial_accuracy(policyless_rows),
            "full_retry_recovery_count": _retry_recovery_count(full_rows),
            "policyless_retry_recovery_count": _retry_recovery_count(policyless_rows),
            "row_count": min(len(full_rows), len(policyless_rows)),
        },
        "interpretation": {
            "budget_constrained_advantage_over_policyless_supported": (
                full_success - policyless_success
            )
            >= min_full_minus_policyless,
            "phase2ac_passes_success_gate": full_success >= min_full_success_rate,
            "learned_native_head_advantage_over_identity_heuristic_supported": (
                full_success - identity_holdout
            )
            > 0.0,
            "claim_boundary": (
                "full package improves over policyless slot0 under budget pressure, "
                "but current run fails the absolute success gate"
                if full_success < min_full_success_rate
                and (full_success - policyless_success) >= min_full_minus_policyless
                else "Phase2AC budget-pressure gate passed"
                if full_success >= min_full_success_rate
                and (full_success - policyless_success) >= min_full_minus_policyless
                else "Phase2AC does not support a budget-pressure mechanism claim"
            ),
        },
        "thresholds": {
            "min_full_success_rate": min_full_success_rate,
            "min_full_minus_policyless": min_full_minus_policyless,
        },
        "unsupported_claims": [
            "learned_native_head_advantage_over_deterministic_identity_heuristic",
            "freeform_patch_generation",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "full_summary_json": str(Path(full_summary_json)),
            "full_results_jsonl": str(Path(full_results_jsonl)),
            "policyless_summary_json": str(Path(policyless_summary_json)),
            "policyless_results_jsonl": str(Path(policyless_results_jsonl)),
        },
    }
    _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Phase2AC full package against policyless budget-pressure retry."
    )
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--full-summary-json", required=True)
    parser.add_argument("--full-results-jsonl", required=True)
    parser.add_argument("--policyless-summary-json", required=True)
    parser.add_argument("--policyless-results-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2ac_budget_pressure_comparison(
        data_health_json=args.data_health_json,
        full_summary_json=args.full_summary_json,
        full_results_jsonl=args.full_results_jsonl,
        policyless_summary_json=args.policyless_summary_json,
        policyless_results_jsonl=args.policyless_results_jsonl,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
