from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _metric(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else None


def audit_phase2af_hardened_structural_sidecar(
    *,
    baseline_report_json: str | Path,
    min_full_accuracy: float = 0.85,
    min_full_minus_no_nsi_textablated: float = 0.15,
    max_raw_source_overlap_accuracy: float = 0.75,
    max_runtime_identity_heuristic_accuracy: float = 0.90,
    min_identity_text_ablated_source_overlap_accuracy: float = 0.05,
    max_identity_text_ablated_source_overlap_accuracy: float = 0.75,
) -> dict[str, Any]:
    baseline = _load(baseline_report_json)
    full = _metric(baseline, "full_summary", "patch_candidate_selection_accuracy")
    no_nsi = _metric(baseline, "no_nsi_summary", "patch_candidate_selection_accuracy")
    raw_source = _metric(baseline, "baseline_metrics", "source_overlap", "accuracy")
    identity_text_ablated_source = _metric(
        baseline,
        "baseline_metrics",
        "source_overlap_identity_text_ablated",
        "accuracy",
    )
    identity_heuristic = _metric(
        baseline,
        "baseline_metrics",
        "runtime_identity_heuristic",
        "accuracy",
    )
    full_minus_no_nsi = (
        full - no_nsi
        if isinstance(full, float) and isinstance(no_nsi, float)
        else None
    )
    checks = {
        "full_accuracy_gate": isinstance(full, float) and full >= min_full_accuracy,
        "full_beats_no_nsi_textablated": isinstance(full_minus_no_nsi, float)
        and full_minus_no_nsi >= min_full_minus_no_nsi_textablated,
        "raw_source_overlap_not_ceiling": isinstance(raw_source, float)
        and raw_source <= max_raw_source_overlap_accuracy,
        "runtime_identity_heuristic_not_sufficient_alone": isinstance(identity_heuristic, float)
        and identity_heuristic <= max_runtime_identity_heuristic_accuracy,
        "identity_text_ablated_source_overlap_nonzero_feasible": isinstance(
            identity_text_ablated_source, float
        )
        and identity_text_ablated_source >= min_identity_text_ablated_source_overlap_accuracy,
        "identity_text_ablated_source_overlap_not_ceiling": isinstance(
            identity_text_ablated_source, float
        )
        and identity_text_ablated_source <= max_identity_text_ablated_source_overlap_accuracy,
    }
    passed = all(checks.values())
    blocked_actions = []
    if not passed:
        blocked_actions.extend(
            [
                "do_not_train_phase2af_full",
                "do_not_package_phase2af",
                "do_not_claim_hardened_structural_sidecar_mechanism",
            ]
        )
    return {
        "artifact_family": "phase2af_hardened_structural_sidecar_gate",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "full_accuracy": full,
            "no_nsi_textablated_accuracy": no_nsi,
            "full_minus_no_nsi_textablated": full_minus_no_nsi,
            "raw_source_overlap_accuracy": raw_source,
            "identity_text_ablated_source_overlap_accuracy": identity_text_ablated_source,
            "runtime_identity_heuristic_accuracy": identity_heuristic,
        },
        "thresholds": {
            "min_full_accuracy": min_full_accuracy,
            "min_full_minus_no_nsi_textablated": min_full_minus_no_nsi_textablated,
            "max_raw_source_overlap_accuracy": max_raw_source_overlap_accuracy,
            "max_runtime_identity_heuristic_accuracy": max_runtime_identity_heuristic_accuracy,
            "min_identity_text_ablated_source_overlap_accuracy": min_identity_text_ablated_source_overlap_accuracy,
            "max_identity_text_ablated_source_overlap_accuracy": max_identity_text_ablated_source_overlap_accuracy,
        },
        "blocked_actions": blocked_actions,
        "claim_boundary": (
            "Phase2AF requires a graded benchmark where raw source-overlap and raw runtime "
            "identity heuristics do not solve the split, while full still beats text-ablated controls."
        ),
        "next_step": (
            "build_hardened_nonsealed_split"
            if not passed
            else "train_or_evaluate_phase2af_hardened_structural_sidecar"
        ),
        "inputs": {"baseline_report_json": str(Path(baseline_report_json))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AF hardened structural-sidecar gate.")
    parser.add_argument("--baseline-report-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-full-accuracy", type=float, default=0.85)
    parser.add_argument("--min-full-minus-no-nsi-textablated", type=float, default=0.15)
    parser.add_argument("--max-raw-source-overlap-accuracy", type=float, default=0.75)
    parser.add_argument("--max-runtime-identity-heuristic-accuracy", type=float, default=0.90)
    parser.add_argument("--min-identity-text-ablated-source-overlap-accuracy", type=float, default=0.05)
    parser.add_argument("--max-identity-text-ablated-source-overlap-accuracy", type=float, default=0.75)
    args = parser.parse_args()
    report = audit_phase2af_hardened_structural_sidecar(
        baseline_report_json=args.baseline_report_json,
        min_full_accuracy=args.min_full_accuracy,
        min_full_minus_no_nsi_textablated=args.min_full_minus_no_nsi_textablated,
        max_raw_source_overlap_accuracy=args.max_raw_source_overlap_accuracy,
        max_runtime_identity_heuristic_accuracy=args.max_runtime_identity_heuristic_accuracy,
        min_identity_text_ablated_source_overlap_accuracy=args.min_identity_text_ablated_source_overlap_accuracy,
        max_identity_text_ablated_source_overlap_accuracy=args.max_identity_text_ablated_source_overlap_accuracy,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
