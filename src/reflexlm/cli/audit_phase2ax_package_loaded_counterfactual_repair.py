from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2ax_package_loaded_counterfactual_repair import BENCHMARK_FAMILY


FORBIDDEN_VISIBLE_MARKERS = (
    "candidate_0",
    "candidate_1",
    "expected_slot",
    "gold_hint=",
    "\"gold\":",
    "external_trace_v3",
    "correct_repair_action",
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _candidate_actions(row: dict[str, Any]) -> list[str]:
    candidates = row.get("repair_candidates")
    if not isinstance(candidates, list):
        return []
    return [
        str(candidate.get("repair_action") or "")
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("repair_action")
    ]


def _visible_text(row: dict[str, Any]) -> str:
    visible = {
        "current_visible_text": row.get("current_visible_text"),
        "phase2ax_current_repair_surface": row.get("phase2ax_current_repair_surface"),
        "runtime_visible_evidence": row.get("runtime_visible_evidence"),
        "runtime_visible_contract": row.get("runtime_visible_contract"),
        "repair_candidates": row.get("repair_candidates"),
    }
    return json.dumps(visible, ensure_ascii=False, sort_keys=True).lower()


def _source_overlap_slot0_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    correct = 0
    for row in rows:
        actions = _candidate_actions(row)
        expected = str(row.get("expected_repair_action") or "")
        if not actions or expected not in actions:
            continue
        total += 1
        correct += int(actions.index(expected) == 0)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "baseline": "slot0_current_surface_baseline",
    }


def audit_phase2ax_package_loaded_counterfactual_repair(
    *,
    tasks_jsonl: str | Path,
    metadata_json: str | Path,
    min_pairs: int = 16,
    max_current_only_baseline: float = 0.75,
) -> dict[str, Any]:
    rows = _read_jsonl(tasks_jsonl)
    metadata = _read_json(metadata_json)
    metadata_rows = metadata if isinstance(metadata, list) else []
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    meta_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row.get("phase2ax_pair_id"))].append(row)
    for row in metadata_rows:
        meta_by_pair[str(row.get("pair_id"))].append(row)

    pair_details: dict[str, Any] = {}
    for pair_id, pair_rows in sorted(by_pair.items()):
        metas = meta_by_pair.get(pair_id, [])
        current_hashes = {str(meta.get("current_surface_hash")) for meta in metas}
        prior_hashes = {str(meta.get("prior_context_hash")) for meta in metas}
        candidate_hashes = {str(meta.get("candidate_actions_hash")) for meta in metas}
        expected_actions = {str(row.get("expected_repair_action")) for row in pair_rows}
        members = {str(row.get("phase2ax_pair_member")) for row in pair_rows}
        expected_slots = {
            int(meta.get("expected_slot"))
            for meta in metas
            if isinstance(meta.get("expected_slot"), int)
        }
        pair_details[pair_id] = {
            "row_count": len(pair_rows),
            "metadata_count": len(metas),
            "members": sorted(members),
            "current_surface_hash_count": len(current_hashes),
            "prior_context_hash_count": len(prior_hashes),
            "candidate_actions_hash_count": len(candidate_hashes),
            "expected_action_count": len(expected_actions),
            "expected_slots": sorted(expected_slots),
            "passed": (
                len(pair_rows) == 2
                and len(metas) == 2
                and members == {"a", "b"}
                and len(current_hashes) == 1
                and len(prior_hashes) == 2
                and len(candidate_hashes) == 1
                and len(expected_actions) == 2
                and len(expected_slots) == 2
            ),
        }

    serialized_visible = "\n".join(_visible_text(row) for row in rows)
    serialized_all = json.dumps(rows, ensure_ascii=False, sort_keys=True).lower()
    baseline = _source_overlap_slot0_accuracy(rows)
    pair_count = len(pair_details)
    checks = {
        "rows_present": len(rows) > 0,
        "metadata_present": len(metadata_rows) > 0,
        "benchmark_family_correct": all(row.get("benchmark_family") == BENCHMARK_FAMILY for row in rows),
        "min_pair_count_met": pair_count >= min_pairs,
        "all_pairs_complete": all(detail["passed"] for detail in pair_details.values()) if pair_details else False,
        "current_surface_identical_within_pairs": all(
            detail["current_surface_hash_count"] == 1 for detail in pair_details.values()
        ),
        "prior_context_differs_within_pairs": all(
            detail["prior_context_hash_count"] == 2 for detail in pair_details.values()
        ),
        "expected_action_differs_within_pairs": all(
            detail["expected_action_count"] == 2 for detail in pair_details.values()
        ),
        "current_only_baseline_measured": baseline["total"] == len(rows),
        "current_only_baseline_below_threshold": isinstance(baseline["accuracy"], float)
        and baseline["accuracy"] <= max_current_only_baseline,
        "no_forbidden_visible_markers": not any(
            marker in serialized_visible for marker in FORBIDDEN_VISIBLE_MARKERS
        ),
        "no_sealed_feedback_in_any_row": "external_trace_v3" not in serialized_all
        and '"sealed_feedback_used": true' not in serialized_all,
        "prior_evidence_present": all(
            isinstance(row.get("phase2ax_prior_runtime_evidence"), dict) for row in rows
        ),
        "freeform_patch_generation_not_claimed": all(
            row.get("runtime_visible_contract", {}).get("no_freeform_patch_generation") is True
            for row in rows
            if isinstance(row.get("runtime_visible_contract"), dict)
        ),
    }
    blocked_actions: list[str] = []
    if not all(checks.values()):
        blocked_actions.append("do_not_train_phase2ax_until_data_health_passes")
    if not checks["current_only_baseline_below_threshold"]:
        blocked_actions.append("do_not_train_when_current_surface_baseline_solves_phase2ax")
    if not checks["no_forbidden_visible_markers"]:
        blocked_actions.append("remove_visible_marker_leakage_before_training")
    return {
        "audit_family": "phase2ax_package_loaded_counterfactual_repair_data_health",
        "passed": all(checks.values()),
        "allowed_next_action": (
            "build_phase2ax_runtime_runner_and_smoke_only"
            if all(checks.values())
            else "revise_phase2ax_data_before_training"
        ),
        "blocked_actions": blocked_actions,
        "checks": checks,
        "thresholds": {
            "min_pairs": min_pairs,
            "max_current_only_baseline": max_current_only_baseline,
        },
        "metrics": {
            "row_count": len(rows),
            "pair_count": pair_count,
            "passed_pair_count": sum(1 for detail in pair_details.values() if detail["passed"]),
            "current_only_baseline": baseline,
        },
        "pair_details": pair_details,
        "claim_boundary": "phase2ax_data_health_only_not_training_or_claim_evidence",
        "inputs": {
            "tasks_jsonl": str(Path(tasks_jsonl)),
            "metadata_json": str(Path(metadata_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AX package-loaded counterfactual repair data."
    )
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-pairs", type=int, default=16)
    parser.add_argument("--max-current-only-baseline", type=float, default=0.75)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ax_package_loaded_counterfactual_repair(
        tasks_jsonl=args.tasks_jsonl,
        metadata_json=args.metadata_json,
        min_pairs=args.min_pairs,
        max_current_only_baseline=args.max_current_only_baseline,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
