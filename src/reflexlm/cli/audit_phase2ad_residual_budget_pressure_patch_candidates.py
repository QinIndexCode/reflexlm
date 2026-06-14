from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import (
    _expected_slot,
    _identity_heuristic_prediction,
)
from reflexlm.cli.build_phase2ad_residual_budget_pressure_patch_candidates import CLAIM_BOUNDARY


MARKER_TERMS = ("gold", "candidate_0", "candidate_1", "slot id", "sealed")


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


def _row_has_marker_leak(row: dict[str, Any]) -> bool:
    visible = str(row.get("current_visible_text") or "").lower()
    return any(term in visible for term in MARKER_TERMS)


def _baseline_accuracy(rows: list[dict[str, Any]], name: str) -> float | None:
    total = 0
    correct = 0
    for row in rows:
        baselines = row.get("baselines") if isinstance(row.get("baselines"), dict) else {}
        prediction = baselines.get(name)
        expected = row.get("expected_repair_action")
        if prediction is None:
            continue
        total += 1
        correct += int(prediction == expected)
    if total == 0:
        return None
    return correct / total


def _repo_sets(splits: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    return {
        split: {str(row.get("repo_id") or "") for row in rows if row.get("repo_id")}
        for split, rows in splits.items()
    }


def _identity_heuristic_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(
        1
        for row in rows
        if _identity_heuristic_prediction(row) == _expected_slot(row)
    ) / len(rows)


def audit_phase2ad_residual_budget_pressure_patch_candidates(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    min_train_rows: int = 24,
    min_val_rows: int = 12,
    min_holdout_rows: int = 24,
    min_best_non_full: float = 0.05,
    max_best_non_full: float = 0.75,
) -> dict[str, Any]:
    splits = {
        "train": _read_jsonl(train_jsonl),
        "val": _read_jsonl(val_jsonl),
        "holdout": _read_jsonl(holdout_jsonl),
    }
    all_rows = [row for rows in splits.values() for row in rows]
    repo_sets = _repo_sets(splits)
    baseline_names = sorted(
        {
            name
            for row in [*splits["val"], *splits["holdout"]]
            if isinstance(row.get("baselines"), dict)
            for name in row["baselines"]
        }
    )
    baseline_metrics = {
        split: {name: _baseline_accuracy(rows, name) for name in baseline_names}
        for split, rows in {"val": splits["val"], "holdout": splits["holdout"]}.items()
    }
    best_non_full = max(
        [
            value
            for metrics in baseline_metrics.values()
            for value in metrics.values()
            if value is not None
        ]
        or [0.0]
    )
    nonzero_controls = {
        split: sum(1 for value in metrics.values() if value is not None and value > 0.0)
        for split, metrics in baseline_metrics.items()
    }
    identity_metrics = {
        split: _identity_heuristic_accuracy(rows)
        for split, rows in {"val": splits["val"], "holdout": splits["holdout"]}.items()
    }
    checks = {
        "train_row_minimum_met": len(splits["train"]) >= min_train_rows,
        "val_row_minimum_met": len(splits["val"]) >= min_val_rows,
        "holdout_row_minimum_met": len(splits["holdout"]) >= min_holdout_rows,
        "all_rows_boundary_correct": all(row.get("claim_boundary") == CLAIM_BOUNDARY for row in all_rows),
        "all_rows_public_repo": all(row.get("source_kind") == "public_repo" for row in all_rows),
        "repo_origin_disjoint": repo_sets["train"].isdisjoint(repo_sets["val"])
        and repo_sets["train"].isdisjoint(repo_sets["holdout"])
        and repo_sets["val"].isdisjoint(repo_sets["holdout"]),
        "candidate_count_exceeds_budget": all(
            len(row.get("repair_candidates", [])) > int(row.get("attempt_budget") or 0)
            for row in all_rows
        ),
        "expected_slot_outside_policyless_budget": all(
            row.get("expected_slot_outside_policyless_budget") is True
            and _expected_slot(row) >= int(row.get("attempt_budget") or 0)
            for row in all_rows
        ),
        "identity_heuristic_not_solving_selected_rows": all(
            row.get("identity_heuristic_correct") is False
            and _identity_heuristic_prediction(row) != _expected_slot(row)
            for row in all_rows
        ),
        "policyless_slot0_budget_expected_to_fail": all(
            row.get("policyless_slot0_budget_expected_success") is False for row in all_rows
        ),
        "no_marker_leak_in_visible_text": not any(_row_has_marker_leak(row) for row in all_rows),
        "non_full_controls_nonzero": all(value >= 1 for value in nonzero_controls.values()),
        "best_non_full_not_all_zero": best_non_full >= min_best_non_full,
        "best_non_full_not_ceiling": best_non_full <= max_best_non_full,
        "no_freeform_patch_generation_claim": all(
            row.get("freeform_patch_generation") is not True
            and all(
                candidate.get("freeform_patch_generation") is False
                for candidate in row.get("patch_candidates", [])
                if isinstance(candidate, dict)
            )
            for row in all_rows
        ),
        "sealed_feedback_absent": all(
            row.get("normalization", {}).get("sealed_feedback_absent") is True
            or row.get("sealed_feedback_used") is False
            or "sealed_feedback_used" not in row
            for row in all_rows
        ),
    }
    return {
        "artifact_family": "phase2ad_residual_budget_pressure_data_health",
        "passed": all(checks.values()),
        "claim_boundary": CLAIM_BOUNDARY,
        "checks": checks,
        "metrics": {
            "split_counts": {split: len(rows) for split, rows in splits.items()},
            "repo_sets": {split: sorted(values) for split, values in repo_sets.items()},
            "expected_slot_distribution": {
                split: dict(sorted(Counter(row.get("expected_patch_candidate_slot") for row in rows).items()))
                for split, rows in splits.items()
            },
            "candidate_count_distribution": {
                split: dict(sorted(Counter(len(row.get("repair_candidates", [])) for row in rows).items()))
                for split, rows in splits.items()
            },
            "baseline_accuracy": baseline_metrics,
            "nonzero_control_count": nonzero_controls,
            "best_non_full_baseline_accuracy": best_non_full,
            "identity_heuristic_accuracy": identity_metrics,
        },
        "thresholds": {
            "min_train_rows": min_train_rows,
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "min_best_non_full": min_best_non_full,
            "max_best_non_full": max_best_non_full,
        },
        "supported_claim_if_passed": [
            "residual_budget_pressure_benchmark_ready_for_selector_failure_or_learning_test"
        ],
        "unsupported_claims": [
            "claim_bearing_success_before_execution",
            "freeform_patch_generation",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AD residual budget-pressure bounded patch candidate splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = audit_phase2ad_residual_budget_pressure_patch_candidates(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
