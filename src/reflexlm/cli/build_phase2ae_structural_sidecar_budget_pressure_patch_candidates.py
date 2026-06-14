from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2aa_bounded_patch_candidates import phase2z_row_to_phase2aa
from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import (
    _expected_slot,
    _identity_heuristic_prediction,
    _strip_identity_shortcuts,
)
from reflexlm.cli.build_phase2ac_budget_pressure_patch_candidates import _round_robin_by_repo


CLAIM_BOUNDARY = "structural_sidecar_closes_residual_budget_pressure_gap"


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


def structural_sidecar_prediction(row: dict[str, Any]) -> int | None:
    evidence = row.get("runtime_visible_evidence") if isinstance(row.get("runtime_visible_evidence"), dict) else {}
    target_location = evidence.get("target_location") if isinstance(evidence.get("target_location"), dict) else {}
    target_path = str(target_location.get("path") or "").replace("\\", "/").lower()
    try:
        target_line = int(target_location.get("line"))
        target_col = int(target_location.get("col"))
    except (TypeError, ValueError):
        return None
    literal_hash = str(evidence.get("expected_literal_hash") or "").lower()
    scores: list[float] = []
    for candidate in row.get("repair_candidates", []):
        if not isinstance(candidate, dict):
            scores.append(0.0)
            continue
        score = 0.0
        edit_scope = str(candidate.get("edit_scope") or "").replace("\\", "/").lower()
        if target_path and edit_scope and (target_path == edit_scope or target_path.endswith(edit_scope)):
            score += 2.0
        if candidate.get("target_line") == target_line and candidate.get("target_col") == target_col:
            score += 3.0
        if literal_hash and str(candidate.get("target_literal_hash") or "").lower() == literal_hash:
            score += 3.0
        scores.append(score)
    if not scores:
        return None
    best = max(scores)
    if best <= 0.0 or scores.count(best) != 1:
        return None
    return scores.index(best)


def _neutralize_legacy_identity_for_structural_sidecar(row: dict[str, Any]) -> dict[str, Any]:
    converted = json.loads(json.dumps(row))
    for candidate in converted.get("repair_candidates", []):
        if not isinstance(candidate, dict):
            continue
        candidate["edit_scope"] = "bounded_public_source_patch"
        candidate["target_symbol"] = ""
    converted["legacy_identity_neutralized_for_structural_sidecar"] = True
    return converted


def phase2s_row_to_phase2ae(row: dict[str, Any], *, attempt_budget: int = 2) -> dict[str, Any]:
    source = _neutralize_legacy_identity_for_structural_sidecar(row)
    expected_slot = _expected_slot(source)
    stripped_identity_pred = _identity_heuristic_prediction(_strip_identity_shortcuts(source))
    structural_pred = structural_sidecar_prediction(source)
    converted = phase2z_row_to_phase2aa(source)
    converted["benchmark_family"] = "phase2ae_structural_sidecar_budget_pressure_patch_candidates"
    converted["claim_boundary"] = CLAIM_BOUNDARY
    converted["attempt_budget"] = attempt_budget
    converted["expected_slot_outside_policyless_budget"] = expected_slot >= attempt_budget
    converted["stripped_identity_heuristic_prediction"] = stripped_identity_pred
    converted["stripped_identity_heuristic_correct"] = stripped_identity_pred == expected_slot
    converted["structural_sidecar_prediction"] = structural_pred
    converted["structural_sidecar_correct"] = structural_pred == expected_slot
    converted["policyless_slot0_budget_expected_success"] = expected_slot < attempt_budget
    converted["current_visible_text"] = (
        str(row.get("current_visible_text") or "")
        + " Structural sidecar budget-pressure case: target path, literal location, and "
        "literal identity are treated as runtime-visible receptor evidence from the generated "
        "repair test, not as a hidden label."
    ).strip()
    converted["claim_boundary_notes"] = [
        "stripped residual identity heuristic does not solve the row",
        "structural sidecar can uniquely rank the correct bounded candidate",
        "this supports structural receptor sidecar sufficiency, not learned-head advantage over the sidecar",
    ]
    converted["phase2ae_selection_rule"] = {
        "stripped_identity_heuristic_must_not_solve_row": True,
        "structural_sidecar_must_solve_row": True,
        "correct_candidate_outside_attempt_budget": True,
        "legacy_file_symbol_identity_neutralized": True,
        "uses_expected_repair_action_for_offline_filter_only": True,
        "uses_sealed_feedback": False,
    }
    converted["trace_hash"] = _sha256_text(_canonical_json(converted))
    return converted


def _select_rows(rows: list[dict[str, Any]], *, attempt_budget: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        source = _neutralize_legacy_identity_for_structural_sidecar(row)
        candidates = row.get("repair_candidates")
        if not isinstance(candidates, list) or len(candidates) <= attempt_budget:
            continue
        expected_slot = _expected_slot(source)
        if expected_slot < attempt_budget:
            continue
        if _identity_heuristic_prediction(_strip_identity_shortcuts(source)) == expected_slot:
            continue
        if structural_sidecar_prediction(source) != expected_slot:
            continue
        selected.append(phase2s_row_to_phase2ae(source, attempt_budget=attempt_budget))
    return _round_robin_by_repo(selected)


def build_phase2ae_structural_sidecar_budget_pressure_patch_candidates(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
    attempt_budget: int = 2,
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
        rows = _select_rows(raw_rows, attempt_budget=attempt_budget)
        source_counts[split] = len(raw_rows)
        split_rows[split] = rows
        _write_jsonl(output / f"{split}.jsonl", rows)
    manifest = {
        "artifact_family": "phase2ae_structural_sidecar_budget_pressure_patch_candidate_split",
        "claim_boundary": CLAIM_BOUNDARY,
        "attempt_budget": attempt_budget,
        "output_dir": str(output),
        "source_split_inputs": {split: str(path) for split, path in split_inputs.items()},
        "source_split_counts": source_counts,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_hashes": {
            split: _sha256_text(_canonical_json(rows)) for split, rows in split_rows.items()
        },
        "expected_slot_distribution": {
            split: dict(sorted(Counter(row["expected_patch_candidate_slot"] for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "candidate_count_distribution": {
            split: dict(sorted(Counter(len(row.get("repair_candidates", [])) for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "selection_rule": (
            "keep public residual rows where structural target sidecar uniquely identifies "
            "the correct candidate outside the verification budget"
        ),
        "output_ordering": "repo_round_robin_to_reduce_prefix_eval_bias",
        "freeform_patch_generation": False,
        "sealed_feedback_used": False,
        "next_gate": "phase2ae_structural_sidecar_data_health",
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AE structural-sidecar residual budget-pressure splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--attempt-budget", type=int, default=2)
    args = parser.parse_args()
    report = build_phase2ae_structural_sidecar_budget_pressure_patch_candidates(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        attempt_budget=args.attempt_budget,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
