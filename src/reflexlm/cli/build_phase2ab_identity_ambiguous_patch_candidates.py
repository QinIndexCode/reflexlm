from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2aa_bounded_patch_candidates import (
    CLAIM_BOUNDARY as PHASE2AA_CLAIM_BOUNDARY,
    phase2z_row_to_phase2aa,
)
from reflexlm.cli.build_phase2s_head_dataset import (
    _candidate_commands,
    _command_identity_signal,
)


CLAIM_BOUNDARY = (
    "identity_ambiguous_bounded_patch_candidate_selection_with_verification_retry"
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


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _expected_slot(row: dict[str, Any]) -> int:
    actions = [
        str(candidate.get("repair_action") or "")
        for candidate in row.get("repair_candidates", [])
        if isinstance(candidate, dict)
    ]
    expected = str(row.get("expected_repair_action") or "")
    if expected not in actions:
        raise ValueError(f"expected repair action missing from candidates: {row.get('trace_id')}")
    return actions.index(expected)


def _identity_heuristic_prediction(row: dict[str, Any]) -> int | None:
    commands = _candidate_commands(row)
    if not commands:
        return None
    signal = _command_identity_signal(row, commands)
    scores = [
        float(signal.get(f"command_identity_slot:{index}") or 0.0)
        for index in range(len(commands))
    ]
    if not scores:
        return None
    best = max(scores)
    if best <= 0.0 or scores.count(best) != 1:
        return None
    return scores.index(best)


def _strip_identity_shortcuts(row: dict[str, Any]) -> dict[str, Any]:
    converted = json.loads(json.dumps(row))
    evidence = converted.get("runtime_visible_evidence")
    if isinstance(evidence, dict):
        evidence.pop("structural_probe_hashes", None)
        evidence.pop("expected_literal_hash", None)
        target_location = evidence.get("target_location")
        if isinstance(target_location, dict):
            target_location.pop("literal_hash", None)
    for candidate in converted.get("repair_candidates", []):
        if not isinstance(candidate, dict):
            continue
        candidate.pop("structural_probe_hash", None)
        candidate.pop("target_literal_hash", None)
        candidate.pop("target_line", None)
        candidate.pop("target_col", None)
    return converted


def phase2s_row_to_phase2ab(row: dict[str, Any]) -> dict[str, Any]:
    source = _strip_identity_shortcuts(row)
    expected_slot = _expected_slot(source)
    predicted_slot = _identity_heuristic_prediction(source)
    converted = phase2z_row_to_phase2aa(source)
    converted["benchmark_family"] = "phase2ab_identity_ambiguous_bounded_patch_candidates"
    converted["claim_boundary"] = CLAIM_BOUNDARY
    converted["phase2aa_source_boundary"] = PHASE2AA_CLAIM_BOUNDARY
    converted["identity_heuristic_prediction"] = predicted_slot
    converted["identity_heuristic_correct"] = predicted_slot == expected_slot
    converted["requires_bounded_verification_retry"] = True
    converted["current_visible_text"] = (
        str(source.get("current_visible_text") or "")
        + " Identity-ambiguous residual case: direct hash/literal identity shortcuts are absent; "
        "multiple bounded same-intent candidates may share file or symbol evidence, so any strong "
        "claim must include measured verification, rollback, and stop-condition behavior."
    ).strip()
    converted["claim_boundary_notes"] = [
        "rows are selected from public non-sealed repair traces where deterministic command identity is tied or wrong",
        "single-shot candidate choice is not enough for a strong claim on these rows",
        "bounded verification retry is the claim-bearing mechanism; this is still not free-form patch generation",
    ]
    converted["phase2ab_selection_rule"] = {
        "source": "phase2s_public_repair",
        "deterministic_identity_heuristic_must_not_solve_row": True,
        "uses_expected_repair_action_for_offline_filter_only": True,
        "uses_sealed_feedback": False,
    }
    converted["trace_hash"] = _sha256_text(_canonical_json(converted))
    return converted


def _select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        stripped = _strip_identity_shortcuts(row)
        if _identity_heuristic_prediction(stripped) != _expected_slot(stripped):
            selected.append(phase2s_row_to_phase2ab(stripped))
    return selected


def build_phase2ab_identity_ambiguous_patch_candidates(
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
    source_counts: dict[str, int] = {}
    for split, path in split_inputs.items():
        raw_rows = _read_jsonl(path)
        rows = _select_rows(raw_rows)
        source_counts[split] = len(raw_rows)
        split_rows[split] = rows
        _write_jsonl(output / f"{split}.jsonl", rows)

    split_hashes = {
        split: _sha256_text(_canonical_json(rows)) for split, rows in split_rows.items()
    }
    manifest = {
        "artifact_family": "phase2ab_identity_ambiguous_patch_candidate_split",
        "claim_boundary": CLAIM_BOUNDARY,
        "output_dir": str(output),
        "source_split_inputs": {split: str(path) for split, path in split_inputs.items()},
        "source_split_counts": source_counts,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_hashes": split_hashes,
        "expected_slot_distribution": {
            split: dict(sorted(Counter(row["expected_patch_candidate_slot"] for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "candidate_count_distribution": {
            split: dict(sorted(Counter(len(row.get("repair_candidates", [])) for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "freeform_patch_generation": False,
        "sealed_feedback_used": False,
        "selection_rule": "keep rows where deterministic command identity heuristic is tied or wrong",
        "next_gate": "phase2ab_identity_ambiguous_data_health",
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AB identity-ambiguous bounded patch candidate splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    args = parser.parse_args()
    report = build_phase2ab_identity_ambiguous_patch_candidates(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
