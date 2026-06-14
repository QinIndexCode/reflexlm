from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import (
    CLAIM_BOUNDARY,
    _identity_heuristic_prediction,
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


def _expected_slot(row: dict[str, Any]) -> int | None:
    actions = [
        str(candidate.get("repair_action") or "")
        for candidate in row.get("repair_candidates", [])
        if isinstance(candidate, dict)
    ]
    expected = str(row.get("expected_repair_action") or "")
    if expected not in actions:
        return None
    return actions.index(expected)


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


def _identity_heuristic_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        correct += int(_identity_heuristic_prediction(row) == _expected_slot(row))
    return correct / len(rows)


def _repo_sets(splits: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    return {
        split: {str(row.get("repo_id") or "") for row in rows if row.get("repo_id")}
        for split, rows in splits.items()
    }


def _has_identity_shortcut(row: dict[str, Any]) -> bool:
    evidence = row.get("runtime_visible_evidence")
    if isinstance(evidence, dict):
        if evidence.get("structural_probe_hashes") or evidence.get("expected_literal_hash"):
            return True
    for candidate in row.get("repair_candidates", []):
        if not isinstance(candidate, dict):
            continue
        if (
            candidate.get("structural_probe_hash")
            or candidate.get("target_literal_hash")
            or candidate.get("target_line")
            or candidate.get("target_col")
        ):
            return True
    return False


def audit_phase2ab_identity_ambiguous_patch_candidates(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    min_train_rows: int = 48,
    min_val_rows: int = 32,
    min_holdout_rows: int = 64,
    min_best_non_full: float = 0.20,
    max_best_non_full: float = 0.75,
    max_identity_heuristic_accuracy: float = 0.10,
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
        "all_rows_have_patch_candidates": all(
            isinstance(row.get("patch_candidates"), list) and len(row["patch_candidates"]) >= 2
            for row in all_rows
        ),
        "expected_slots_covered": all(
            len(
                {
                    row.get("expected_patch_candidate_slot")
                    for row in rows
                    if row.get("expected_patch_candidate_slot") is not None
                }
            )
            >= 2
            for rows in splits.values()
        ),
        "no_marker_leak_in_visible_text": not any(_row_has_marker_leak(row) for row in all_rows),
        "identity_shortcuts_absent": not any(_has_identity_shortcut(row) for row in all_rows),
        "selected_rows_are_identity_ambiguous": all(
            row.get("identity_heuristic_correct") is False
            and _identity_heuristic_prediction(row) != _expected_slot(row)
            for row in all_rows
        ),
        "identity_heuristic_not_solving_val_or_holdout": all(
            value <= max_identity_heuristic_accuracy for value in identity_metrics.values()
        ),
        "non_full_controls_nonzero": all(value >= 3 for value in nonzero_controls.values()),
        "best_non_full_not_all_zero": best_non_full >= min_best_non_full,
        "best_non_full_not_ceiling": best_non_full <= max_best_non_full,
        "retry_declared_as_required": all(
            row.get("requires_bounded_verification_retry") is True for row in all_rows
        ),
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
        "artifact_family": "phase2ab_identity_ambiguous_data_health",
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
            "max_identity_heuristic_accuracy": max_identity_heuristic_accuracy,
        },
        "supported_claim_if_passed": [
            "identity_ambiguous_bounded_candidate_benchmark_ready_for_retry_evaluation"
        ],
        "unsupported_claims": [
            "single_shot_learned_head_necessity",
            "freeform_patch_generation",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AB identity-ambiguous bounded patch candidate splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = audit_phase2ab_identity_ambiguous_patch_candidates(
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
