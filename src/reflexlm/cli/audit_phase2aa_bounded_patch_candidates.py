from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2aa_bounded_patch_candidates import CLAIM_BOUNDARY


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


def _required_artifacts_exist(row: dict[str, Any], artifact_root: Path) -> bool:
    artifact_paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    patch_rel = artifact_paths.get("patch_diff")
    if not patch_rel or not (artifact_root / str(patch_rel)).is_file():
        return False
    generated_rel = artifact_paths.get("generated_test")
    if generated_rel and not (artifact_root / str(generated_rel)).is_file():
        return False
    return True


def audit_phase2aa_bounded_patch_candidates(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    artifact_root: str | Path | None = None,
    min_val_rows: int = 24,
    min_holdout_rows: int = 24,
    min_best_non_full: float = 0.20,
    max_best_non_full: float = 0.85,
) -> dict[str, Any]:
    splits = {
        "train": _read_jsonl(train_jsonl),
        "val": _read_jsonl(val_jsonl),
        "holdout": _read_jsonl(holdout_jsonl),
    }
    all_rows = [row for rows in splits.values() for row in rows]
    slot_distribution = {
        split: dict(sorted(Counter(row.get("expected_patch_candidate_slot") for row in rows).items()))
        for split, rows in splits.items()
    }
    baseline_names = sorted(
        {
            name
            for row in [*splits["val"], *splits["holdout"]]
            if isinstance(row.get("baselines"), dict)
            for name in row["baselines"]
        }
    )
    baseline_metrics = {
        split: {
            name: _baseline_accuracy(rows, name)
            for name in baseline_names
        }
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
    artifact_base = Path(artifact_root) if artifact_root is not None else None
    artifact_paths_available = (
        True
        if artifact_base is None
        else all(_required_artifacts_exist(row, artifact_base) for row in all_rows)
    )
    checks = {
        "val_row_minimum_met": len(splits["val"]) >= min_val_rows,
        "holdout_row_minimum_met": len(splits["holdout"]) >= min_holdout_rows,
        "all_rows_boundary_correct": all(row.get("claim_boundary") == CLAIM_BOUNDARY for row in all_rows),
        "all_rows_public_repo": all(row.get("source_kind") == "public_repo" for row in all_rows),
        "all_rows_have_patch_candidates": all(
            isinstance(row.get("patch_candidates"), list) and len(row["patch_candidates"]) >= 2
            for row in all_rows
        ),
        "expected_slots_covered": all(
            len({slot for slot in distribution if slot is not None}) >= 2
            for distribution in slot_distribution.values()
            if distribution
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
        "best_non_full_not_all_zero": best_non_full >= min_best_non_full,
        "best_non_full_not_ceiling": best_non_full <= max_best_non_full,
        "required_runtime_artifacts_available": artifact_paths_available,
    }
    return {
        "artifact_family": "phase2aa_bounded_patch_candidate_data_health",
        "passed": all(checks.values()),
        "claim_boundary": CLAIM_BOUNDARY,
        "checks": checks,
        "metrics": {
            "split_counts": {split: len(rows) for split, rows in splits.items()},
            "expected_slot_distribution": slot_distribution,
            "baseline_accuracy": baseline_metrics,
            "best_non_full_baseline_accuracy": best_non_full,
        },
        "thresholds": {
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "min_best_non_full": min_best_non_full,
            "max_best_non_full": max_best_non_full,
            "artifact_root": str(artifact_base) if artifact_base is not None else None,
        },
        "blocked_actions": [
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_open_ended_debugging_generalization",
            "do_not_use_sealed_feedback_for_sampling_or_tuning",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AA bounded patch candidate split readiness."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--artifact-root",
        help=(
            "Optional root used to verify patch_diff/generated_test artifacts referenced "
            "by rows. When omitted, artifact portability is not checked."
        ),
    )
    parser.add_argument("--min-val-rows", type=int, default=24)
    parser.add_argument("--min-holdout-rows", type=int, default=24)
    args = parser.parse_args()
    report = audit_phase2aa_bounded_patch_candidates(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        artifact_root=args.artifact_root,
        min_val_rows=args.min_val_rows,
        min_holdout_rows=args.min_holdout_rows,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
