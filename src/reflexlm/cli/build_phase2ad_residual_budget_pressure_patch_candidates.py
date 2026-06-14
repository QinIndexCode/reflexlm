from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import (
    _expected_slot,
    _identity_heuristic_prediction,
    _strip_identity_shortcuts,
    phase2s_row_to_phase2ab,
)


CLAIM_BOUNDARY = "residual_budget_pressure_requires_learned_or_structural_selector"


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


def _round_robin_by_repo(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("repo_id") or ""), []).append(row)
    ordered: list[dict[str, Any]] = []
    repo_ids = sorted(grouped)
    index = 0
    while True:
        appended = False
        for repo_id in repo_ids:
            bucket = grouped[repo_id]
            if index < len(bucket):
                ordered.append(bucket[index])
                appended = True
        if not appended:
            return ordered
        index += 1


def phase2s_row_to_phase2ad(row: dict[str, Any], *, attempt_budget: int = 2) -> dict[str, Any]:
    stripped = _strip_identity_shortcuts(row)
    expected_slot = _expected_slot(stripped)
    identity_pred = _identity_heuristic_prediction(stripped)
    converted = phase2s_row_to_phase2ab(stripped)
    converted["benchmark_family"] = "phase2ad_residual_budget_pressure_patch_candidates"
    converted["claim_boundary"] = CLAIM_BOUNDARY
    converted["attempt_budget"] = attempt_budget
    converted["expected_slot_outside_policyless_budget"] = expected_slot >= attempt_budget
    converted["identity_heuristic_prediction"] = identity_pred
    converted["identity_heuristic_correct"] = identity_pred == expected_slot
    converted["policyless_slot0_budget_expected_success"] = expected_slot < attempt_budget
    candidates = [
        candidate
        for candidate in converted.get("repair_candidates", [])
        if isinstance(candidate, dict)
    ]
    first_outside_budget_action = (
        str(candidates[attempt_budget].get("repair_action"))
        if attempt_budget < len(candidates)
        else None
    )
    if first_outside_budget_action is not None:
        baselines = dict(converted.get("baselines") if isinstance(converted.get("baselines"), dict) else {})
        baselines["first_outside_budget_slot"] = first_outside_budget_action
        converted["baselines"] = baselines
        baseline_metadata = dict(
            converted.get("baseline_metadata")
            if isinstance(converted.get("baseline_metadata"), dict)
            else {}
        )
        baseline_metadata["first_outside_budget_slot"] = {
            "measured": True,
            "method": "fixed_first_candidate_slot_outside_attempt_budget",
            "uses_expected_repair_action": False,
            "uses_sealed_feedback": False,
            "purpose": "positional sanity control for residual budget-pressure rows",
        }
        converted["baseline_metadata"] = baseline_metadata
    converted["current_visible_text"] = (
        str(stripped.get("current_visible_text") or "")
        + " Residual budget-pressure case: deterministic identity evidence is absent, tied, "
        "or wrong, and the correct bounded candidate is outside the policyless verification budget."
    ).strip()
    converted["claim_boundary_notes"] = [
        "deterministic command identity does not solve the selected row",
        "correct candidate is outside the policyless slot0 verification budget",
        "passing this benchmark would require learned residual selection or additional non-oracle structural evidence",
    ]
    converted["phase2ad_selection_rule"] = {
        "deterministic_identity_heuristic_must_not_solve_row": True,
        "correct_candidate_outside_attempt_budget": True,
        "uses_expected_repair_action_for_offline_filter_only": True,
        "uses_sealed_feedback": False,
    }
    converted["trace_hash"] = _sha256_text(_canonical_json(converted))
    return converted


def _select_rows(rows: list[dict[str, Any]], *, attempt_budget: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        stripped = _strip_identity_shortcuts(row)
        candidates = stripped.get("repair_candidates")
        if not isinstance(candidates, list) or len(candidates) <= attempt_budget:
            continue
        expected_slot = _expected_slot(stripped)
        if expected_slot < attempt_budget:
            continue
        if _identity_heuristic_prediction(stripped) == expected_slot:
            continue
        selected.append(phase2s_row_to_phase2ad(stripped, attempt_budget=attempt_budget))
    return _round_robin_by_repo(selected)


def build_phase2ad_residual_budget_pressure_patch_candidates(
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
        "artifact_family": "phase2ad_residual_budget_pressure_patch_candidate_split",
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
            "keep public rows where correct candidate is outside slot0 budget "
            "and deterministic identity heuristic is tied or wrong"
        ),
        "output_ordering": "repo_round_robin_to_reduce_prefix_eval_bias",
        "freeform_patch_generation": False,
        "sealed_feedback_used": False,
        "next_gate": "phase2ad_residual_budget_pressure_data_health",
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AD residual budget-pressure bounded patch candidate splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--attempt-budget", type=int, default=2)
    args = parser.parse_args()
    report = build_phase2ad_residual_budget_pressure_patch_candidates(
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
