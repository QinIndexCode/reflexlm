from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def audit_phase2ae_learning_gap(
    *,
    results_jsonl: str | Path,
    output_json: str | Path | None = None,
    min_initial_selection_accuracy_for_learned_head_claim: float = 0.85,
) -> dict[str, Any]:
    rows = _read_jsonl(results_jsonl)
    total = len(rows)
    initial_correct = 0
    final_correct = 0
    successes = 0
    rescued_by_identity_retry = 0
    retry_attempted = 0
    for row in rows:
        expected = row.get("expected_patch_candidate_slot")
        initial = row.get("initial_selected_patch_candidate_slot")
        final = row.get("selected_patch_candidate_slot")
        identity_retry = row.get("identity_retry_slot")
        initial_ok = initial == expected
        final_ok = final == expected
        if initial_ok:
            initial_correct += 1
        if final_ok:
            final_correct += 1
        if row.get("success") is True:
            successes += 1
        if identity_retry is not None:
            retry_attempted += 1
        if not initial_ok and final_ok and identity_retry == expected and row.get("success") is True:
            rescued_by_identity_retry += 1

    initial_accuracy = initial_correct / total if total else 0.0
    final_selection_accuracy = final_correct / total if total else 0.0
    success_rate = successes / total if total else 0.0
    retry_rescue_rate = rescued_by_identity_retry / total if total else 0.0
    checks = {
        "results_present": total > 0,
        "initial_policy_selection_supports_learned_head_claim": initial_accuracy
        >= min_initial_selection_accuracy_for_learned_head_claim,
        "identity_retry_is_explicitly_accounted_for": retry_attempted == total,
    }
    report = {
        "artifact_family": "phase2ae_learning_gap_audit",
        "passed": checks["results_present"]
        and checks["identity_retry_is_explicitly_accounted_for"],
        "checks": checks,
        "metrics": {
            "rows": total,
            "initial_policy_selection_accuracy": initial_accuracy,
            "final_patch_candidate_selection_accuracy": final_selection_accuracy,
            "success_rate": success_rate,
            "identity_retry_rescue_rate": retry_rescue_rate,
            "identity_retry_attempted_rate": retry_attempted / total if total else 0.0,
        },
        "interpretation": {
            "supported": [
                "structural sidecar retry can rescue bounded candidate execution"
            ]
            if checks["results_present"] and retry_rescue_rate > 0.0
            else [],
            "unsupported": [
                "learned native-head initial candidate selection"
            ]
            if not checks["initial_policy_selection_supports_learned_head_claim"]
            else [],
            "claim_boundary": (
                "Do not describe Phase2AE directcmd success as learned-head candidate selection unless initial policy selection accuracy clears the preregistered threshold."
            ),
        },
        "thresholds": {
            "min_initial_selection_accuracy_for_learned_head_claim": min_initial_selection_accuracy_for_learned_head_claim,
        },
        "inputs": {"results_jsonl": str(Path(results_jsonl))},
    }
    if output_json is not None:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether Phase2AE success comes from initial learned policy selection or structural sidecar retry."
    )
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--min-initial-selection-accuracy-for-learned-head-claim",
        type=float,
        default=0.85,
    )
    args = parser.parse_args()
    report = audit_phase2ae_learning_gap(
        results_jsonl=args.results_jsonl,
        output_json=args.output_json,
        min_initial_selection_accuracy_for_learned_head_claim=args.min_initial_selection_accuracy_for_learned_head_claim,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
