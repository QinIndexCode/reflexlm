from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2aa_bounded_patch_candidates import phase2z_row_to_phase2aa
from reflexlm.cli.build_phase2s_head_dataset import _candidate_commands, _command_identity_signal


CLAIM_BOUNDARY = "budget_constrained_identity_guided_bounded_patch_candidate_selection"


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


def identity_heuristic_prediction(row: dict[str, Any]) -> int | None:
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


def phase2s_row_to_phase2ac(row: dict[str, Any], *, attempt_budget: int = 2) -> dict[str, Any]:
    expected_slot = _expected_slot(row)
    identity_pred = identity_heuristic_prediction(row)
    converted = phase2z_row_to_phase2aa(row)
    converted["benchmark_family"] = "phase2ac_budget_pressure_patch_candidates"
    converted["claim_boundary"] = CLAIM_BOUNDARY
    converted["attempt_budget"] = attempt_budget
    converted["expected_slot_outside_policyless_budget"] = expected_slot >= attempt_budget
    converted["identity_heuristic_prediction"] = identity_pred
    converted["identity_heuristic_correct"] = identity_pred == expected_slot
    converted["policyless_slot0_budget_expected_success"] = expected_slot < attempt_budget
    converted["current_visible_text"] = (
        str(row.get("current_visible_text") or "")
        + " Budget-constrained candidate selection: only a small number of bounded candidates "
        "may be verified, so the selector must prioritize candidates using runtime-visible "
        "identity evidence rather than exhaustive retry."
    ).strip()
    converted["claim_boundary_notes"] = [
        "correct candidate is outside the policyless slot0 verification budget",
        "runtime-visible identity evidence is allowed and measured as a deterministic baseline",
        "this can support evidence-guided bounded candidate prioritization, not free-form patch generation",
    ]
    converted["trace_hash"] = _sha256_text(_canonical_json(converted))
    return converted


def _select_rows(rows: list[dict[str, Any]], *, attempt_budget: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        candidates = row.get("repair_candidates")
        if not isinstance(candidates, list) or len(candidates) <= attempt_budget:
            continue
        expected_slot = _expected_slot(row)
        if expected_slot < attempt_budget:
            continue
        if identity_heuristic_prediction(row) != expected_slot:
            continue
        selected.append(phase2s_row_to_phase2ac(row, attempt_budget=attempt_budget))
    return _round_robin_by_repo(selected)


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


def build_phase2ac_budget_pressure_patch_candidates(
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
        "artifact_family": "phase2ac_budget_pressure_patch_candidate_split",
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
            "and runtime identity heuristic uniquely identifies it"
        ),
        "output_ordering": "repo_round_robin_to_reduce_prefix_eval_bias",
        "freeform_patch_generation": False,
        "sealed_feedback_used": False,
        "next_gate": "phase2ac_budget_pressure_data_health",
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AC budget-constrained bounded patch candidate splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--attempt-budget", type=int, default=2)
    args = parser.parse_args()
    report = build_phase2ac_budget_pressure_patch_candidates(
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
