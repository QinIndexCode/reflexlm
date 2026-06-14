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


def build_phase2ab_retry_baseline_comparison(
    *,
    full_summary_json: str | Path,
    full_results_jsonl: str | Path,
    policyless_summary_json: str | Path,
    policyless_results_jsonl: str | Path,
    output_json: str | Path,
) -> dict[str, Any]:
    full_summary = _read_json(full_summary_json)
    policyless_summary = _read_json(policyless_summary_json)
    full_rows = _read_jsonl(full_results_jsonl)
    policyless_rows = _read_jsonl(policyless_results_jsonl)
    full_success = float(full_summary.get("success_rate") or 0.0)
    policyless_success = float(policyless_summary.get("success_rate") or 0.0)
    report = {
        "artifact_family": "phase2ab_retry_baseline_comparison",
        "passed": True,
        "metrics": {
            "full_success_rate": full_success,
            "policyless_slot0_retry_success_rate": policyless_success,
            "full_minus_policyless_slot0_retry": full_success - policyless_success,
            "full_initial_selection_accuracy": _initial_accuracy(full_rows),
            "policyless_initial_selection_accuracy": _initial_accuracy(policyless_rows),
            "row_count": min(len(full_rows), len(policyless_rows)),
        },
        "interpretation": {
            "bounded_verification_loop_supported": full_success >= 0.85 and policyless_success >= 0.85,
            "full_package_unique_advantage_supported": (full_success - policyless_success) >= 0.10,
            "claim_boundary": (
                "bounded verification retry is sufficient on this Phase2AB benchmark; "
                "full package unique advantage is not supported"
                if (full_success - policyless_success) < 0.10
                else "full package has a measured advantage over policyless retry"
            ),
        },
        "unsupported_claims_if_no_delta": [
            "learned_native_head_necessity_over_exhaustive_bounded_retry",
            "nsi_latent_necessity_over_exhaustive_bounded_retry",
            "production_autonomy",
            "freeform_patch_generation",
            "epoch_making_architecture",
        ],
        "inputs": {
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
        description="Compare Phase2AB full package retry against policyless slot0 retry baseline."
    )
    parser.add_argument("--full-summary-json", required=True)
    parser.add_argument("--full-results-jsonl", required=True)
    parser.add_argument("--policyless-summary-json", required=True)
    parser.add_argument("--policyless-results-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2ab_retry_baseline_comparison(
        full_summary_json=args.full_summary_json,
        full_results_jsonl=args.full_results_jsonl,
        policyless_summary_json=args.policyless_summary_json,
        policyless_results_jsonl=args.policyless_results_jsonl,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
