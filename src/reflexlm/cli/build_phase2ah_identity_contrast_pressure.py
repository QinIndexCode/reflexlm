from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import _expected_slot
from reflexlm.cli.build_phase2af_hardened_structural_sidecar_split import (
    _row_candidate,
    _shortcut_key,
)


CLAIM_BOUNDARY = (
    "nonsealed_adversarial_identity_contrast_pressure_not_natural_trace_claim"
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


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _stable_candidate_literal(row: dict[str, Any], candidate: dict[str, Any], slot: int) -> str:
    existing = candidate.get("target_literal_hash")
    if existing:
        return str(existing)
    basis = _canonical_json(
        {
            "trace_id": row.get("trace_id"),
            "repair_action": candidate.get("repair_action"),
            "edit_scope": candidate.get("edit_scope"),
            "target_symbol": candidate.get("target_symbol"),
            "slot": slot,
        }
    )
    return "phase2ah_" + _sha256_text(basis)[:16]


def _source_identity_bucket(row: dict[str, Any]) -> tuple[int, int] | None:
    candidate = _row_candidate(row, require_tie_residual_feasible=True)
    if candidate is None:
        return None
    return _shortcut_key(candidate)


def phase2ah_identity_contrast_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return a non-sealed row whose structured identity sidecar points to a wrong candidate.

    The transformed row is accepted only if measured source evidence still selects the
    expected candidate while runtime identity selects a different candidate. This creates a
    pressure benchmark for sidecar robustness without using sealed failures or candidate
    slot markers.
    """

    candidates = row.get("repair_candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return None
    try:
        expected_slot = _expected_slot(row)
    except (KeyError, TypeError, ValueError):
        return None
    wrong_slots = [slot for slot in range(len(candidates)) if slot != expected_slot]
    for wrong_slot in wrong_slots:
        converted = json.loads(json.dumps(row))
        converted_candidates = converted.get("repair_candidates")
        if not isinstance(converted_candidates, list):
            continue
        for slot, candidate in enumerate(converted_candidates):
            if not isinstance(candidate, dict):
                continue
            candidate["target_literal_hash"] = _stable_candidate_literal(converted, candidate, slot)
            candidate.pop("structural_probe_hash", None)
            candidate.pop("target_line", None)
            candidate.pop("target_col", None)
        wrong_candidate = converted_candidates[wrong_slot]
        if not isinstance(wrong_candidate, dict):
            continue
        evidence = converted.setdefault("runtime_visible_evidence", {})
        if not isinstance(evidence, dict):
            continue
        evidence["expected_literal_hash"] = wrong_candidate["target_literal_hash"]
        evidence.pop("structural_probe_hashes", None)
        evidence.pop("target_location", None)
        converted["benchmark_family"] = "phase2ah_identity_contrast_pressure"
        converted["claim_boundary"] = CLAIM_BOUNDARY
        converted["sealed_feedback_used"] = False
        converted["current_visible_text"] = (
            str(converted.get("current_visible_text") or "")
            + " Phase2AH identity-contrast pressure: structured identity sidecar is stale or misleading; "
            "the bounded selector must use public source/runtime evidence instead of blindly following identity."
        ).strip()
        converted["phase2ah_identity_contrast"] = {
            "source": "nonsealed_public_repair_row",
            "uses_expected_repair_action_for_counterfactual_construction": True,
            "wrong_identity_slot": wrong_slot,
            "expected_slot": expected_slot,
            "sealed_feedback_used": False,
            "candidate_slot_markers_visible": False,
        }
        converted["trace_hash"] = _sha256_text(_canonical_json(converted))
        if _source_identity_bucket(converted) == (1, 0):
            return converted
    return None


def _convert_rows(rows: list[dict[str, Any]], *, max_rows: int | None) -> list[dict[str, Any]]:
    converted = [
        item
        for item in (phase2ah_identity_contrast_row(row) for row in rows)
        if item is not None
    ]
    converted.sort(key=lambda row: (str(row.get("repo_id") or ""), str(row.get("trace_id") or "")))
    if max_rows is not None:
        converted = converted[:max_rows]
    return converted


def _bucket_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    buckets = Counter(_source_identity_bucket(row) for row in rows)
    return {
        f"source_{source}_identity_{identity}": int(count)
        for (source, identity), count in sorted(
            (key, value) for key, value in buckets.items() if key is not None
        )
    }


def build_phase2ah_identity_contrast_pressure(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    max_holdout_rows: int | None = None,
    min_val_rows: int = 16,
    min_holdout_rows: int = 16,
) -> dict[str, Any]:
    output = Path(output_dir)
    split_inputs = {
        "train": Path(train_jsonl),
        "val": Path(val_jsonl),
        "holdout": Path(holdout_jsonl),
    }
    max_by_split = {
        "train": max_train_rows,
        "val": max_val_rows,
        "holdout": max_holdout_rows,
    }
    split_rows: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, int] = {}
    for split, path in split_inputs.items():
        source = _read_jsonl(path)
        rows = _convert_rows(source, max_rows=max_by_split[split])
        source_counts[split] = len(source)
        split_rows[split] = rows
        _write_jsonl(output / f"{split}.jsonl", rows)
    checks = {
        "sealed_feedback_absent": True,
        "train_rows_present": len(split_rows["train"]) > 0,
        "val_rows_min": len(split_rows["val"]) >= min_val_rows,
        "holdout_rows_min": len(split_rows["holdout"]) >= min_holdout_rows,
        "val_identity_wrong_source_correct": _bucket_counts(split_rows["val"]).get(
            "source_1_identity_0", 0
        )
        == len(split_rows["val"]),
        "holdout_identity_wrong_source_correct": _bucket_counts(split_rows["holdout"]).get(
            "source_1_identity_0", 0
        )
        == len(split_rows["holdout"]),
    }
    passed = all(checks.values())
    manifest = {
        "artifact_family": "phase2ah_identity_contrast_pressure",
        "passed": passed,
        "claim_boundary": CLAIM_BOUNDARY,
        "claim_bearing_natural_trace_evidence": False,
        "claim_bearing_adversarial_pressure_evidence": passed,
        "checks": checks,
        "source_split_inputs": {split: str(path) for split, path in split_inputs.items()},
        "source_split_counts": source_counts,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_hashes": {
            split: _sha256_text(_canonical_json(rows)) for split, rows in split_rows.items()
        },
        "bucket_counts": {split: _bucket_counts(rows) for split, rows in split_rows.items()},
        "selection_rule": (
            "construct non-sealed identity-contrast pressure rows only when measured source evidence "
            "remains correct and structured runtime identity becomes wrong; this tests shortcut "
            "robustness but is not a natural public-trace proof"
        ),
        "unsupported_claims": [
            "sealed_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
            "natural_public_trace_identity_failure_distribution",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_train_phase2ah",
            "do_not_package_phase2ah",
            "do_not_claim_model_beats_runtime_identity",
        ],
        "allowed_next_action": "build_phase2ah_head_dataset_and_smoke_train"
        if passed
        else "collect_more_nonsealed_source_correct_rows",
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AH non-sealed identity-contrast pressure splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--max-holdout-rows", type=int)
    parser.add_argument("--min-val-rows", type=int, default=16)
    parser.add_argument("--min-holdout-rows", type=int, default=16)
    args = parser.parse_args()
    manifest = build_phase2ah_identity_contrast_pressure(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        max_holdout_rows=args.max_holdout_rows,
        min_val_rows=args.min_val_rows,
        min_holdout_rows=args.min_holdout_rows,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
