from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


CLAIM_BOUNDARY = "bounded_patch_candidate_selection_not_freeform_patch_generation"


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


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _patch_candidate_from_repair_candidate(
    candidate: dict[str, Any],
    *,
    slot: int,
    expected_slot: int,
) -> dict[str, Any]:
    action = str(candidate.get("repair_action") or f"patch_candidate_{slot}")
    is_expected = slot == expected_slot
    return {
        "slot": slot,
        "candidate_id": _sha256_text(f"{slot}:{action}:{candidate.get('structural_probe_hash')}")[:16],
        "repair_action": action,
        "intent": str(candidate.get("intent") or "apply_patch_and_rerun_tests"),
        "edit_scope": "bounded_public_source_patch",
        "structural_probe_hash": str(candidate.get("structural_probe_hash") or ""),
        "target_symbol": str(candidate.get("target_symbol") or ""),
        "patch_source": "recorded_correct_patch_artifact"
        if is_expected
        else "runtime_generated_bounded_distractor_patch",
        "claim_role": "correct_bounded_candidate" if is_expected else "distractor_bounded_candidate",
        "freeform_patch_generation": False,
    }


def phase2z_row_to_phase2aa(row: dict[str, Any]) -> dict[str, Any]:
    candidates = row.get("repair_candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError(f"row lacks at least two repair candidates: {row.get('trace_id')}")
    actions = [str(candidate.get("repair_action") or "") for candidate in candidates]
    expected = str(row.get("expected_repair_action") or "")
    if expected not in actions:
        raise ValueError(f"expected repair action missing from candidates: {row.get('trace_id')}")
    expected_slot = actions.index(expected)
    converted = dict(row)
    converted["benchmark_family"] = "phase2aa_bounded_patch_candidate_selection"
    converted["claim_boundary"] = CLAIM_BOUNDARY
    converted["expected_patch_candidate_slot"] = expected_slot
    converted["patch_candidates"] = [
        _patch_candidate_from_repair_candidate(candidate, slot=slot, expected_slot=expected_slot)
        for slot, candidate in enumerate(candidates)
        if isinstance(candidate, dict)
    ]
    converted["current_visible_text"] = (
        str(row.get("current_visible_text") or "")
        + " Select the bounded patch candidate slot from runtime-visible evidence; "
        "candidate slots do not expose oracle labels and patch content is not generated freely."
    ).strip()
    converted["claim_boundary_notes"] = [
        "patch candidates are bounded artifacts or runtime-generated distractor controls",
        "model chooses a candidate slot through native command-slot control",
        "this is stronger than recorded patch replay but still not free-form patch generation",
    ]
    converted["trace_hash"] = _sha256_text(_canonical_json(converted))
    return converted


def build_phase2aa_bounded_patch_candidates(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    split_inputs = {
        "train": Path(train_jsonl),
        "val": Path(val_jsonl),
        "holdout": Path(holdout_jsonl),
    }
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split, path in split_inputs.items():
        rows = [phase2z_row_to_phase2aa(row) for row in _read_jsonl(path)]
        split_rows[split] = rows
        _write_jsonl(output / f"{split}.jsonl", rows)

    slot_distribution = {
        split: dict(sorted(Counter(row["expected_patch_candidate_slot"] for row in rows).items()))
        for split, rows in split_rows.items()
    }
    split_hashes = {
        split: _sha256_text(_canonical_json(rows)) for split, rows in split_rows.items()
    }
    manifest = {
        "artifact_family": "phase2aa_bounded_patch_candidate_split",
        "claim_boundary": CLAIM_BOUNDARY,
        "output_dir": str(output),
        "source_split_inputs": {split: str(path) for split, path in split_inputs.items()},
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_hashes": split_hashes,
        "expected_slot_distribution": slot_distribution,
        "freeform_patch_generation": False,
        "sealed_feedback_used": False,
        "next_gate": "phase2aa_bounded_patch_candidate_data_health",
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AA bounded patch candidate splits from Phase2Z public structural rows."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    args = parser.parse_args()
    report = build_phase2aa_bounded_patch_candidates(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
