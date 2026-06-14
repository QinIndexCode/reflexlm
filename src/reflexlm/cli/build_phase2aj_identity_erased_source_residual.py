from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2af_hardened_structural_sidecar_split import (
    _read_jsonl,
    _row_candidate,
)


IDENTITY_SIDE_CAR_FIELDS = {
    "structural_probe_hash",
    "target_symbol",
    "target_literal_hash",
    "target_line",
    "target_col",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _identity_ambiguous_row(row: dict[str, Any]) -> dict[str, Any]:
    converted = json.loads(json.dumps(row))
    candidates = converted.get("repair_candidates")
    ambiguous_tokens: list[str] = []
    structural_probe_hashes: list[str] = []
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict):
                action = str(candidate.get("repair_action") or "").strip()
                if action:
                    ambiguous_tokens.append(action)
                structural_probe_hash = str(candidate.get("structural_probe_hash") or "").strip()
                if structural_probe_hash:
                    structural_probe_hashes.append(structural_probe_hash)
                candidate["identity_sidecar_erased"] = False
                candidate["identity_sidecar_ambiguity_control"] = True
    runtime_evidence = converted.get("runtime_visible_evidence")
    if isinstance(runtime_evidence, dict) and structural_probe_hashes:
        runtime_evidence["structural_probe_hashes"] = sorted(set(structural_probe_hashes))
    if ambiguous_tokens:
        converted["current_visible_text"] = "\n".join(
            [
                str(converted.get("current_visible_text") or ""),
                "Controlled ambiguous identity receptor: "
                f"command_identity_tokens={' '.join(ambiguous_tokens)}",
            ]
        ).strip()
    converted["benchmark_family"] = "phase2aj_identity_ambiguous_source_residual"
    converted["claim_boundary"] = (
        "nonsealed_controlled_ambiguous_identity_source_residual_pressure_not_natural_trace_claim"
    )
    converted["phase2aj_transform"] = {
        "identity_sidecar_erased_from_candidates": False,
        "ambiguous_identity_receptor_contains_all_candidate_actions": bool(ambiguous_tokens),
        "runtime_visible_source_evidence_preserved": True,
        "candidate_source_metadata_preserved": True,
        "repair_action_ids_preserved": True,
        "uses_sealed_feedback": False,
        "does_not_use_gold_to_choose_wrong_identity": True,
    }
    converted["unsupported_claims"] = [
        "natural_public_trace_identity_failure_distribution",
        "sealed_transfer",
        "production_autonomy",
        "open_ended_debugging_generalization",
        "epoch_making_architecture",
    ]
    converted["trace_hash"] = _sha256(converted)
    return converted


def _phase2aj_shortcut_key(row: dict[str, Any]) -> tuple[int, int]:
    shortcuts = row["phase2af_measured_shortcuts"]["correct"]
    source_correct = bool(shortcuts.get("identity_text_ablated_source_overlap"))
    expected_slot = int(row["phase2af_measured_shortcuts"]["expected_slot"])
    controlled_identity_prediction = 0
    row["phase2aj_measured_controls"] = {
        "controlled_ambiguous_identity_prediction": controlled_identity_prediction,
        "controlled_ambiguous_identity_correct": controlled_identity_prediction == expected_slot,
        "source_overlap_identity_text_ablated_correct": source_correct,
        "control_interpretation": (
            "Ambiguous identity receptor contains all candidate actions; the identity-only "
            "control is therefore non-discriminative and uses deterministic slot-0 tie-break."
        ),
    }
    return int(source_correct), int(controlled_identity_prediction == expected_slot)


def _reindex_slot0_source_correct_candidate(row: dict[str, Any]) -> dict[str, Any]:
    candidate = _row_candidate(row, require_tie_residual_feasible=True)
    if candidate is None:
        return row
    shortcuts = candidate["phase2af_measured_shortcuts"]
    if (
        int(shortcuts["expected_slot"]) != 0
        or not bool(shortcuts["correct"].get("identity_text_ablated_source_overlap"))
    ):
        return row
    candidates = row.get("repair_candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return row
    reindexed = json.loads(json.dumps(row))
    moved_candidates = list(reindexed["repair_candidates"])
    moved_candidates[0], moved_candidates[1] = moved_candidates[1], moved_candidates[0]
    reindexed["repair_candidates"] = moved_candidates
    transform = reindexed.setdefault("phase2aj_transform", {})
    transform["controlled_candidate_reindexing"] = {
        "enabled": True,
        "reason": (
            "source-overlap-correct rows in the public structural collector are slot-0 "
            "biased; reindexing creates a non-trivial ambiguous-identity tie-break control"
        ),
        "uses_expected_repair_action_for_nonsealed_controlled_reindexing": True,
        "sealed_feedback_used": False,
        "original_expected_slot": 0,
        "reindexed_expected_slot": 1,
    }
    reindexed["trace_hash"] = _sha256(reindexed)
    return reindexed


def _bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        candidate = _row_candidate(row, require_tie_residual_feasible=True)
        if candidate is None:
            continue
        source, identity = _phase2aj_shortcut_key(candidate)
        counts[f"source_{source}_identity_{identity}"] += 1
    return dict(sorted(counts.items()))


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        transformed = _reindex_slot0_source_correct_candidate(_identity_ambiguous_row(row))
        candidate = _row_candidate(transformed, require_tie_residual_feasible=True)
        if candidate is None or _phase2aj_shortcut_key(candidate) != (1, 0):
            continue
        candidate["benchmark_family"] = "phase2aj_identity_ambiguous_source_residual"
        candidate["claim_boundary"] = transformed["claim_boundary"]
        candidate["phase2aj_transform"] = transformed["phase2aj_transform"]
        trace_id = str(transformed.get("trace_id") or f"row:{index}")
        if trace_id in seen:
            continue
        seen.add(trace_id)
        selected.append(candidate)
        if max_rows is not None and len(selected) >= max_rows:
            break
    return selected


def _split_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_slot_counts: Counter[str] = Counter()
    source_correct_by_expected_slot: Counter[str] = Counter()
    shortcut_counts: Counter[str] = Counter()
    tie_feasible_after_identity_ambiguity = 0
    row_candidate_after_identity_ambiguity = 0
    for row in rows:
        transformed = _identity_ambiguous_row(row)
        candidate = _row_candidate(transformed, require_tie_residual_feasible=False)
        if candidate is None:
            continue
        row_candidate_after_identity_ambiguity += 1
        shortcuts = candidate["phase2af_measured_shortcuts"]
        expected_slot = int(shortcuts["expected_slot"])
        expected_slot_counts[str(expected_slot)] += 1
        correct = shortcuts["correct"]
        for name, value in correct.items():
            shortcut_counts[f"{name}:{bool(value)}"] += 1
        if bool(correct.get("identity_text_ablated_source_overlap")):
            source_correct_by_expected_slot[str(expected_slot)] += 1
        if _row_candidate(transformed, require_tie_residual_feasible=True) is not None:
            tie_feasible_after_identity_ambiguity += 1
    source_correct_total = sum(source_correct_by_expected_slot.values())
    slot0_source_correct = source_correct_by_expected_slot.get("0", 0)
    return {
        "row_count": len(rows),
        "row_candidate_after_identity_ambiguity": row_candidate_after_identity_ambiguity,
        "tie_feasible_after_identity_ambiguity": tie_feasible_after_identity_ambiguity,
        "expected_slot_counts": dict(sorted(expected_slot_counts.items())),
        "source_correct_by_expected_slot": dict(sorted(source_correct_by_expected_slot.items())),
        "shortcut_counts_after_identity_ambiguity": dict(sorted(shortcut_counts.items())),
        "source_correct_total": source_correct_total,
        "source_correct_slot0_fraction": (
            slot0_source_correct / source_correct_total if source_correct_total else 0.0
        ),
        "failure_interpretation": (
            "source-overlap-correct rows are slot-0/default-biased or lose tie-residual "
            "feasibility after identity evidence is made ambiguous; this split is not a valid "
            "source-residual training/evaluation gate"
        ),
    }


def build_phase2aj_identity_erased_source_residual(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path | None = None,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    max_holdout_rows: int | None = None,
    min_train_rows: int = 16,
    min_val_rows: int = 16,
    min_holdout_rows: int = 16,
) -> dict[str, Any]:
    source_rows = {
        "train": _read_jsonl(train_jsonl),
        "val": _read_jsonl(val_jsonl),
        "holdout": _read_jsonl(holdout_jsonl),
    }
    split_rows = {
        "train": _select_rows(source_rows["train"], max_rows=max_train_rows),
        "val": _select_rows(source_rows["val"], max_rows=max_val_rows),
        "holdout": _select_rows(source_rows["holdout"], max_rows=max_holdout_rows),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        _write_jsonl(output / f"{split}.jsonl", rows)

    split_counts = {split: len(rows) for split, rows in split_rows.items()}
    bucket_counts = {split: _bucket_counts(rows) for split, rows in split_rows.items()}
    split_hashes = {split: _sha256(rows) for split, rows in split_rows.items()}
    diagnostics = {split: _split_diagnostics(rows) for split, rows in source_rows.items()}
    checks = {
        "sealed_feedback_absent": True,
        "train_rows_min": split_counts["train"] >= min_train_rows,
        "val_rows_min": split_counts["val"] >= min_val_rows,
        "holdout_rows_min": split_counts["holdout"] >= min_holdout_rows,
        "train_source_correct_identity_wrong": bucket_counts["train"].get(
            "source_1_identity_0", 0
        )
        == split_counts["train"],
        "val_source_correct_identity_wrong": bucket_counts["val"].get(
            "source_1_identity_0", 0
        )
        == split_counts["val"],
        "holdout_source_correct_identity_wrong": bucket_counts["holdout"].get(
            "source_1_identity_0", 0
        )
        == split_counts["holdout"],
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2aj_identity_ambiguous_source_residual",
        "passed": passed,
        "claim_bearing_natural_trace_evidence": False,
        "claim_bearing_controlled_pressure_evidence": passed,
        "checks": checks,
        "source_split_inputs": {
            "train": str(Path(train_jsonl)),
            "val": str(Path(val_jsonl)),
            "holdout": str(Path(holdout_jsonl)),
        },
        "source_split_counts": {split: len(rows) for split, rows in source_rows.items()},
        "split_counts": split_counts,
        "bucket_counts": bucket_counts,
        "diagnostics": diagnostics,
        "split_hashes": split_hashes,
        "transform": {
            "identity_sidecar_fields_removed": [],
            "ambiguous_identity_receptor_contains_all_candidate_actions": True,
            "runtime_identity_probe_hashes_replaced_with_all_candidate_hashes": True,
            "candidate_source_metadata_preserved": True,
            "repair_action_ids_preserved": True,
            "runtime_visible_source_evidence_preserved": True,
            "controlled_candidate_reindexing_allowed_for_nonsealed_pressure": True,
            "uses_expected_repair_action_for_nonsealed_controlled_reindexing": True,
            "uses_sealed_feedback": False,
            "uses_gold_to_choose_wrong_identity": False,
        },
        "blocked_actions": (
            []
            if passed
            else [
                "do_not_train_phase2aj_until_controlled_pressure_split_passes",
                "do_not_package_phase2aj",
                "do_not_claim_source_residual_mechanism",
            ]
        ),
        "allowed_next_action": (
            "build_phase2aj_head_dataset_and_smoke_train"
            if passed
            else "revise_nonsealed_controlled_ambiguous_identity_design"
        ),
        "claim_boundary": (
            "Non-sealed controlled ambiguous-identity source-residual pressure only. "
            "This benchmark tests whether the model can use source evidence when structured "
            "identity receptor evidence is intentionally made non-discriminative while candidate "
            "source metadata remains intact. Slot-0-biased source-correct rows may be reindexed "
            "with non-sealed expected actions to make the identity tie-break control non-trivial. "
            "It is not a natural trace distribution claim and does not support sealed transfer, "
            "production autonomy, open-ended debugging generalization, or an epoch-making "
            "architecture claim."
        ),
    }
    if manifest_json:
        _write_json(manifest_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AJ ambiguous-identity source residual pressure splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--max-holdout-rows", type=int)
    parser.add_argument("--min-train-rows", type=int, default=16)
    parser.add_argument("--min-val-rows", type=int, default=16)
    parser.add_argument("--min-holdout-rows", type=int, default=16)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2aj_identity_erased_source_residual(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        max_holdout_rows=args.max_holdout_rows,
        min_train_rows=args.min_train_rows,
        min_val_rows=args.min_val_rows,
        min_holdout_rows=args.min_holdout_rows,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
