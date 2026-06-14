from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import _expected_slot
from reflexlm.cli.build_phase2af_hardened_structural_sidecar_split import (
    _read_jsonl,
    _row_candidate,
    _shortcut_key,
    _split_metrics,
)
from reflexlm.cli.build_phase2ak_source_cue_balanced_identity_ambiguous import (
    _assert_literal,
)


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


def _phase2al_row(row: dict[str, Any]) -> dict[str, Any] | None:
    literal = _assert_literal(row)
    if not literal:
        return None
    converted = json.loads(json.dumps(row))
    candidates = converted.get("repair_candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return None
    try:
        _expected_slot(converted)
    except ValueError:
        return None

    for candidate in candidates:
        if not isinstance(candidate, dict):
            return None
        candidate["target_symbol"] = literal
        candidate["phase2al_shared_source_cue"] = literal
        candidate["phase2al_source_cue_kind"] = "runtime_visible_assert_literal_shared_by_all_candidates"
        candidate["phase2al_structural_identity_control"] = (
            "runtime structural_probe_hashes remain unchanged; source cue is shared"
        )

    converted["current_visible_text"] = "\n".join(
        [
            str(converted.get("current_visible_text") or ""),
            "Phase2AL controlled pressure: assertion literal is shared by every "
            "candidate, so lexical source-overlap should not uniquely solve the slot; "
            "runtime structural identity evidence is preserved.",
        ]
    ).strip()
    converted["benchmark_family"] = "phase2al_shared_source_cue_structural_identity"
    converted["claim_boundary"] = (
        "nonsealed_controlled_shared_source_cue_structural_identity_pressure_not_natural_trace_claim"
    )
    converted["phase2al_transform"] = {
        "source_cue_from_runtime_visible_assert_literal": True,
        "candidate_target_symbol_replaced_with_same_source_cue_for_all_candidates": True,
        "runtime_structural_probe_hashes_preserved": True,
        "candidate_order_preserved": True,
        "uses_expected_repair_action_for_nonsealed_controlled_source_cue_assignment": False,
        "uses_sealed_feedback": False,
    }
    converted["unsupported_claims"] = [
        "natural_public_trace_distribution",
        "sealed_transfer",
        "production_autonomy",
        "open_ended_debugging_generalization",
        "epoch_making_architecture",
    ]
    candidate = _row_candidate(converted, require_tie_residual_feasible=True)
    if candidate is None:
        return None
    candidate["benchmark_family"] = converted["benchmark_family"]
    candidate["claim_boundary"] = converted["claim_boundary"]
    candidate["phase2al_transform"] = converted["phase2al_transform"]
    candidate["trace_hash"] = _sha256(candidate)
    return candidate


def _select_rows(rows: list[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        converted = _phase2al_row(row)
        if converted is None:
            continue
        trace_id = str(converted.get("trace_id") or f"row:{index}")
        if trace_id in seen:
            continue
        seen.add(trace_id)
        selected.append(converted)
        if max_rows is not None and len(selected) >= max_rows:
            break
    return selected


def _bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        source, identity = _shortcut_key(row)
        counts[f"source_{source}_identity_{identity}"] += 1
    return dict(sorted(counts.items()))


def _metric_accuracy(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, dict):
        return 0.0
    accuracy = value.get("accuracy")
    return float(accuracy) if isinstance(accuracy, (int, float)) else 0.0


def _split_checks(
    rows: list[dict[str, Any]],
    *,
    min_rows: int,
    min_source_overlap_accuracy: float,
    max_source_overlap_accuracy: float,
    min_identity_accuracy: float,
) -> dict[str, bool]:
    metrics = _split_metrics(rows)
    source = _metric_accuracy(metrics, "identity_text_ablated_source_overlap")
    identity = _metric_accuracy(metrics, "runtime_identity_heuristic")
    return {
        "rows_min": len(rows) >= min_rows,
        "source_overlap_nonzero": source >= min_source_overlap_accuracy,
        "source_overlap_not_ceiling": source <= max_source_overlap_accuracy,
        "runtime_identity_high": identity >= min_identity_accuracy,
    }


def build_phase2al_shared_source_cue_structural_identity(
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
    min_source_overlap_accuracy: float = 0.20,
    max_source_overlap_accuracy: float = 0.75,
    min_identity_accuracy: float = 0.95,
) -> dict[str, Any]:
    source_rows = {
        "train": _read_jsonl(train_jsonl),
        "val": _read_jsonl(val_jsonl),
        "holdout": _read_jsonl(holdout_jsonl),
    }
    split_rows = {
        "train": _select_rows(source_rows["train"], max_train_rows),
        "val": _select_rows(source_rows["val"], max_val_rows),
        "holdout": _select_rows(source_rows["holdout"], max_holdout_rows),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        _write_jsonl(output / f"{split}.jsonl", rows)

    split_counts = {split: len(rows) for split, rows in split_rows.items()}
    split_metrics = {split: _split_metrics(rows) for split, rows in split_rows.items()}
    split_checks = {
        "train": _split_checks(
            split_rows["train"],
            min_rows=min_train_rows,
            min_source_overlap_accuracy=min_source_overlap_accuracy,
            max_source_overlap_accuracy=max_source_overlap_accuracy,
            min_identity_accuracy=min_identity_accuracy,
        ),
        "val": _split_checks(
            split_rows["val"],
            min_rows=min_val_rows,
            min_source_overlap_accuracy=min_source_overlap_accuracy,
            max_source_overlap_accuracy=max_source_overlap_accuracy,
            min_identity_accuracy=min_identity_accuracy,
        ),
        "holdout": _split_checks(
            split_rows["holdout"],
            min_rows=min_holdout_rows,
            min_source_overlap_accuracy=min_source_overlap_accuracy,
            max_source_overlap_accuracy=max_source_overlap_accuracy,
            min_identity_accuracy=min_identity_accuracy,
        ),
    }
    checks = {
        "sealed_feedback_absent": True,
        "all_split_gates_pass": all(
            all(values.values()) for values in split_checks.values()
        ),
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2al_shared_source_cue_structural_identity",
        "passed": passed,
        "claim_bearing_natural_trace_evidence": False,
        "claim_bearing_controlled_pressure_evidence": passed,
        "checks": checks,
        "split_checks": split_checks,
        "source_split_inputs": {
            "train": str(Path(train_jsonl)),
            "val": str(Path(val_jsonl)),
            "holdout": str(Path(holdout_jsonl)),
        },
        "source_split_counts": {split: len(rows) for split, rows in source_rows.items()},
        "split_counts": split_counts,
        "bucket_counts": {split: _bucket_counts(rows) for split, rows in split_rows.items()},
        "split_metrics": split_metrics,
        "thresholds": {
            "min_source_overlap_accuracy": min_source_overlap_accuracy,
            "max_source_overlap_accuracy": max_source_overlap_accuracy,
            "min_identity_accuracy": min_identity_accuracy,
        },
        "split_hashes": {split: _sha256(rows) for split, rows in split_rows.items()},
        "transform": {
            "source_cue_from_runtime_visible_assert_literal": True,
            "shared_source_cue_all_candidates": True,
            "runtime_structural_probe_hashes_preserved": True,
            "uses_sealed_feedback": False,
        },
        "blocked_actions": (
            []
            if passed
            else [
                "do_not_train_phase2al_until_shared_source_cue_gate_passes",
                "do_not_package_phase2al",
                "do_not_claim_source_overlap_delta",
            ]
        ),
        "allowed_next_action": (
            "build_phase2al_head_dataset_and_smoke_train"
            if passed
            else "revise_nonsealed_shared_source_cue_design"
        ),
        "claim_boundary": (
            "Phase2AL is a non-sealed controlled pressure benchmark. It makes the "
            "assertion-literal source cue shared across candidates, preserving runtime "
            "structural identity evidence so source-overlap is feasible but not ceiling. "
            "It cannot support natural trace distribution, sealed transfer, production "
            "autonomy, open-ended debugging generalization, or epoch-making architecture claims."
        ),
    }
    if manifest_json:
        _write_json(manifest_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AL shared-source-cue structural-identity controlled pressure splits."
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
    parser.add_argument("--min-source-overlap-accuracy", type=float, default=0.20)
    parser.add_argument("--max-source-overlap-accuracy", type=float, default=0.75)
    parser.add_argument("--min-identity-accuracy", type=float, default=0.95)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = build_phase2al_shared_source_cue_structural_identity(
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
        min_source_overlap_accuracy=args.min_source_overlap_accuracy,
        max_source_overlap_accuracy=args.max_source_overlap_accuracy,
        min_identity_accuracy=args.min_identity_accuracy,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
