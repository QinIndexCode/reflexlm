from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import _expected_slot
from reflexlm.cli.build_phase2ae_structural_sidecar_budget_pressure_patch_candidates import (
    CLAIM_BOUNDARY,
    structural_sidecar_prediction,
)


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


def _repo_sets(splits: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    return {
        split: {str(row.get("repo_id") or "") for row in rows if row.get("repo_id")}
        for split, rows in splits.items()
    }


def _structural_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if structural_sidecar_prediction(row) == _expected_slot(row)) / len(rows)


def _stripped_identity_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get("stripped_identity_heuristic_correct") is True) / len(rows)


def audit_phase2ae_structural_sidecar_budget_pressure_patch_candidates(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    min_train_rows: int = 24,
    min_val_rows: int = 12,
    min_holdout_rows: int = 24,
) -> dict[str, Any]:
    splits = {
        "train": _read_jsonl(train_jsonl),
        "val": _read_jsonl(val_jsonl),
        "holdout": _read_jsonl(holdout_jsonl),
    }
    all_rows = [row for rows in splits.values() for row in rows]
    repo_sets = _repo_sets(splits)
    checks = {
        "train_row_minimum_met": len(splits["train"]) >= min_train_rows,
        "val_row_minimum_met": len(splits["val"]) >= min_val_rows,
        "holdout_row_minimum_met": len(splits["holdout"]) >= min_holdout_rows,
        "all_rows_boundary_correct": all(row.get("claim_boundary") == CLAIM_BOUNDARY for row in all_rows),
        "all_rows_public_repo": all(row.get("source_kind") == "public_repo" for row in all_rows),
        "repo_origin_disjoint": repo_sets["train"].isdisjoint(repo_sets["val"])
        and repo_sets["train"].isdisjoint(repo_sets["holdout"])
        and repo_sets["val"].isdisjoint(repo_sets["holdout"]),
        "expected_slot_outside_policyless_budget": all(
            row.get("expected_slot_outside_policyless_budget") is True
            and _expected_slot(row) >= int(row.get("attempt_budget") or 0)
            for row in all_rows
        ),
        "stripped_identity_not_solving": all(
            row.get("stripped_identity_heuristic_correct") is False for row in all_rows
        ),
        "legacy_identity_neutralized_for_structural_sidecar": all(
            row.get("legacy_identity_neutralized_for_structural_sidecar") is True
            and all(
                str(candidate.get("edit_scope") or "") == "bounded_public_source_patch"
                and str(candidate.get("target_symbol") or "") == ""
                for candidate in row.get("repair_candidates", [])
                if isinstance(candidate, dict)
            )
            for row in all_rows
        ),
        "structural_sidecar_solves_selected_rows": all(
            row.get("structural_sidecar_correct") is True
            and structural_sidecar_prediction(row) == _expected_slot(row)
            for row in all_rows
        ),
        "policyless_slot0_budget_expected_to_fail": all(
            row.get("policyless_slot0_budget_expected_success") is False for row in all_rows
        ),
        "no_marker_leak_in_visible_text": not any(_row_has_marker_leak(row) for row in all_rows),
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
        "artifact_family": "phase2ae_structural_sidecar_data_health",
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
            "stripped_identity_accuracy": {
                split: _stripped_identity_accuracy(rows) for split, rows in splits.items()
            },
            "structural_sidecar_accuracy": {
                split: _structural_accuracy(rows) for split, rows in splits.items()
            },
        },
        "thresholds": {
            "min_train_rows": min_train_rows,
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
        },
        "supported_claim_if_passed": [
            "structural_sidecar_budget_pressure_benchmark_ready"
        ],
        "unsupported_claims": [
            "learned_head_advantage_over_structural_sidecar",
            "freeform_patch_generation",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AE structural-sidecar residual budget-pressure splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = audit_phase2ae_structural_sidecar_budget_pressure_patch_candidates(
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
