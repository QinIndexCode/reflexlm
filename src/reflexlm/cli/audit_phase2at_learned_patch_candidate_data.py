from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


CLAIM_BOUNDARY = "phase2at_learned_bounded_patch_candidate_generation_pretrain_only"
SCHEMA_VERSION = "phase2at.learned_bounded_patch_candidate.v1"
REQUIRED_TARGET_FIELDS = (
    "target_path",
    "operation",
    "anchor",
    "before_fragment_hash",
    "after_fragment_template_id",
    "literal_or_symbol_payload",
    "safety_constraints",
    "verification_command_slot",
)
FORBIDDEN_VISIBLE_MARKERS = (
    "candidate_0",
    "candidate_1",
    "slot id",
    "gold",
    "sealed",
)
FORBIDDEN_TARGET_KEYS = (
    "patch_text",
    "patch_diff",
    "diff_text",
    "unified_diff",
    "recorded_patch",
)
FORBIDDEN_TARGET_SOURCES = (
    "recorded_correct_patch_artifact",
    "recorded_public_structural_patch_diff_operator",
    "package_runtime_symbolic_structural_patch_proposal",
    "package_runtime_symbolic_text_membership_patch_proposal",
    "control_runtime_symbolic_ast_attribute_patch",
    "control_runtime_symbolic_text_patch",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _row_has_visible_marker_leak(row: dict[str, Any]) -> bool:
    visible_parts = [
        str(row.get("current_visible_text") or ""),
        _canonical_json(row.get("runtime_visible_evidence") or {}),
    ]
    visible = "\n".join(visible_parts).lower()
    return any(marker in visible for marker in FORBIDDEN_VISIBLE_MARKERS)


def _baseline_accuracy(rows: list[dict[str, Any]], name: str) -> float | None:
    total = 0
    correct = 0
    for row in rows:
        baselines = row.get("baselines") if isinstance(row.get("baselines"), dict) else {}
        expected = row.get("expected_repair_action")
        prediction = baselines.get(name)
        if prediction is None:
            continue
        total += 1
        correct += int(prediction == expected)
    if total == 0:
        return None
    return correct / total


def _target_failure_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    target = row.get("learned_patch_candidate_target")
    if not isinstance(target, dict):
        return ["missing_learned_patch_candidate_target"]
    schema = str(target.get("schema_version") or row.get("patch_candidate_schema_version") or "")
    if schema != SCHEMA_VERSION:
        reasons.append("schema_version_mismatch")
    for field in REQUIRED_TARGET_FIELDS:
        if field not in target:
            reasons.append(f"missing_target_field:{field}")
    source = str(target.get("target_source") or target.get("patch_source") or "")
    if source in FORBIDDEN_TARGET_SOURCES:
        reasons.append("forbidden_target_source")
    if any(key in target for key in FORBIDDEN_TARGET_KEYS):
        reasons.append("freeform_or_recorded_diff_target_present")
    payload = _canonical_json(target).lower()
    if any(marker in payload for marker in ("sealed_v2", "sealed_v3", "gold_patch")):
        reasons.append("sealed_or_gold_target_marker")
    if str(target.get("operation") or "") not in {
        "replace_symbol",
        "replace_attribute",
        "insert_import",
        "replace_literal",
        "insert_guard",
    }:
        reasons.append("operation_not_allowlisted")
    return reasons


def audit_phase2at_learned_patch_candidate_data(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
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
    target_failures: Counter[str] = Counter()
    for row in all_rows:
        target_failures.update(_target_failure_reasons(row))
        candidate_lists = [
            row.get("patch_candidates", []) or [],
            row.get("repair_candidates", []) or [],
        ]
        for candidates in candidate_lists:
            if not isinstance(candidates, list):
                continue
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                if str(candidate.get("patch_source") or "") in FORBIDDEN_TARGET_SOURCES:
                    target_failures["candidate_uses_forbidden_patch_source"] += 1
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
    repo_sets = {
        split: {str(row.get("repo_url_or_origin") or row.get("repo_id") or "") for row in rows}
        for split, rows in splits.items()
    }
    split_hashes = {
        split: _sha256_text(_canonical_json(rows)) for split, rows in splits.items()
    }
    checks = {
        "train_rows_present": len(splits["train"]) > 0,
        "val_row_minimum_met": len(splits["val"]) >= min_val_rows,
        "holdout_row_minimum_met": len(splits["holdout"]) >= min_holdout_rows,
        "all_rows_public_repo": all(row.get("source_kind") == "public_repo" for row in all_rows),
        "claim_boundary_correct": all(
            row.get("claim_boundary") == CLAIM_BOUNDARY for row in all_rows
        ),
        "schema_version_correct": all(
            (
                isinstance(row.get("learned_patch_candidate_target"), dict)
                and row["learned_patch_candidate_target"].get("schema_version") == SCHEMA_VERSION
            )
            for row in all_rows
        ),
        "all_targets_have_required_fields": all(
            not any(reason.startswith("missing_target_field:") for reason in _target_failure_reasons(row))
            for row in all_rows
        ),
        "no_recorded_or_symbolic_generation_targets": not any(
            reason
            in {
                "forbidden_target_source",
                "freeform_or_recorded_diff_target_present",
                "candidate_uses_forbidden_patch_source",
            }
            for reason in target_failures
        ),
        "no_patch_diff_artifact_available_as_training_target": all(
            "patch_diff"
            not in (
                row.get("artifact_paths")
                if isinstance(row.get("artifact_paths"), dict)
                else {}
            )
            for row in all_rows
        ),
        "operations_allowlisted": target_failures.get("operation_not_allowlisted", 0) == 0,
        "no_visible_marker_leak": not any(_row_has_visible_marker_leak(row) for row in all_rows),
        "sealed_feedback_absent": all(
            row.get("sealed_feedback_used") is False
            or row.get("normalization", {}).get("sealed_feedback_absent") is True
            for row in all_rows
        ),
        "repo_origin_disjoint_val_holdout": repo_sets["val"].isdisjoint(repo_sets["holdout"]),
        "best_non_full_not_all_zero": best_non_full >= min_best_non_full,
        "best_non_full_not_ceiling": best_non_full <= max_best_non_full,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2at_learned_patch_candidate_data_health",
        "passed": passed,
        "claim_boundary": CLAIM_BOUNDARY,
        "schema_version": SCHEMA_VERSION,
        "checks": checks,
        "metrics": {
            "split_counts": {split: len(rows) for split, rows in splits.items()},
            "split_hashes": split_hashes,
            "repo_counts": {split: len(repos) for split, repos in repo_sets.items()},
            "baseline_accuracy": baseline_metrics,
            "best_non_full_baseline_accuracy": best_non_full,
            "target_failure_reasons": dict(sorted(target_failures.items())),
        },
        "thresholds": {
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "min_best_non_full": min_best_non_full,
            "max_best_non_full": max_best_non_full,
        },
        "supported_claims": [
            "phase2at_data_ready_for_learned_bounded_patch_candidate_training"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "learned_patch_generation",
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ]
        if not passed
        else [
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_train_phase2at_until_data_health_passes",
            "do_not_use_recorded_patch_artifacts_or_symbolic_generators_as_generation_targets",
            "do_not_use_sealed_feedback_for_sampling_or_tuning",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AT learned bounded patch candidate data health."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-val-rows", type=int, default=24)
    parser.add_argument("--min-holdout-rows", type=int, default=24)
    args = parser.parse_args()
    report = audit_phase2at_learned_patch_candidate_data(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        min_val_rows=args.min_val_rows,
        min_holdout_rows=args.min_holdout_rows,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
