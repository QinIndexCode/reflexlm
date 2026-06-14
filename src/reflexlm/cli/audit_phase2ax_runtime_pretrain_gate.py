from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2ax_package_loaded_counterfactual_repair import (
    audit_phase2ax_package_loaded_counterfactual_repair,
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


def _expected_slot(row: dict[str, Any]) -> int | None:
    expected = str(row.get("expected_repair_action") or "")
    actions = _candidate_actions(row)
    if expected in actions:
        return actions.index(expected)
    return None


def _prior_tokens(prior: dict[str, Any]) -> list[str]:
    raw_tokens: list[str] = []
    for key in (
        "structural_probe_hashes",
        "runtime_structural_probe_hashes",
        "repair_modes",
        "descriptor_operation",
        "descriptor_template",
        "target_symbol_hash",
    ):
        value = prior.get(key)
        if isinstance(value, list):
            raw_tokens.extend(str(item) for item in value if item is not None)
        elif value is not None:
            raw_tokens.append(str(value))

    tokens: list[str] = []
    for raw in raw_tokens:
        token = raw.strip().lower()
        if len(token) < 4:
            continue
        tokens.append(token)
        if len(token) >= 12:
            tokens.append(token[:12])
        if len(token) >= 16:
            tokens.append(token[:16])
            tokens.append(f"structural_repair_{token[:12]}")
    return sorted(set(tokens), key=lambda item: (-len(item), item))


def _select_by_prior(row: dict[str, Any], prior: dict[str, Any]) -> int | None:
    actions = [action.lower() for action in _candidate_actions(row)]
    tokens = _prior_tokens(prior)
    scores: list[int] = []
    for action in actions:
        score = 0
        for token in tokens:
            if token and token in action:
                score += 1
        scores.append(score)
    if not scores or max(scores) <= 0:
        return None
    if scores.count(max(scores)) != 1:
        return None
    return scores.index(max(scores))


def _accuracy(rows: list[dict[str, Any]], predictions: dict[str, int | None]) -> dict[str, Any]:
    total = 0
    correct = 0
    missing = 0
    examples: list[dict[str, Any]] = []
    for row in rows:
        expected = _expected_slot(row)
        if expected is None:
            continue
        total += 1
        predicted = predictions.get(str(row.get("task_id")))
        if predicted is None:
            missing += 1
        correct += int(predicted == expected)
        if len(examples) < 8:
            examples.append(
                {
                    "task_id": row.get("task_id"),
                    "pair_id": row.get("phase2ax_pair_id"),
                    "member": row.get("phase2ax_pair_member"),
                    "expected_slot": expected,
                    "predicted_slot": predicted,
                    "correct": predicted == expected,
                }
            )
    return {
        "total": total,
        "correct": correct,
        "missing_predictions": missing,
        "accuracy": correct / total if total else None,
        "examples": examples,
    }


def _paired_prior_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[str(row.get("phase2ax_pair_id"))].append(row)
    paired: dict[str, dict[str, Any]] = {}
    for pair_rows in by_pair.values():
        if len(pair_rows) != 2:
            continue
        left, right = pair_rows
        left_prior = right.get("phase2ax_prior_runtime_evidence")
        right_prior = left.get("phase2ax_prior_runtime_evidence")
        if isinstance(left_prior, dict):
            paired[str(left.get("task_id"))] = left_prior
        if isinstance(right_prior, dict):
            paired[str(right.get("task_id"))] = right_prior
    return paired


def build_phase2ax_runtime_pretrain_gate(
    *,
    tasks_jsonl: str | Path,
    metadata_json: str | Path,
    data_health_json: str | Path | None = None,
    min_prior_resolver_accuracy: float = 0.85,
    min_prior_minus_current_only: float = 0.25,
    min_prior_minus_wrong_cache: float = 0.25,
    max_current_only_accuracy: float = 0.75,
) -> dict[str, Any]:
    rows = _read_jsonl(tasks_jsonl)
    if data_health_json:
        data_health = _read_json(data_health_json)
    else:
        data_health = audit_phase2ax_package_loaded_counterfactual_repair(
            tasks_jsonl=tasks_jsonl,
            metadata_json=metadata_json,
        )

    current_predictions = {
        str(row.get("task_id")): 0 if _candidate_actions(row) else None for row in rows
    }
    prior_predictions = {
        str(row.get("task_id")): _select_by_prior(
            row,
            row.get("phase2ax_prior_runtime_evidence")
            if isinstance(row.get("phase2ax_prior_runtime_evidence"), dict)
            else {},
        )
        for row in rows
    }
    paired_prior = _paired_prior_by_task(rows)
    wrong_cache_predictions = {
        str(row.get("task_id")): _select_by_prior(row, paired_prior.get(str(row.get("task_id")), {}))
        for row in rows
    }
    current = _accuracy(rows, current_predictions)
    prior = _accuracy(rows, prior_predictions)
    wrong_cache = _accuracy(rows, wrong_cache_predictions)

    def _delta(left: dict[str, Any], right: dict[str, Any]) -> float | None:
        if isinstance(left.get("accuracy"), float) and isinstance(right.get("accuracy"), float):
            return float(left["accuracy"]) - float(right["accuracy"])
        return None

    prior_minus_current = _delta(prior, current)
    prior_minus_wrong_cache = _delta(prior, wrong_cache)
    serialized_inputs = json.dumps(
        {"tasks_jsonl": str(tasks_jsonl), "metadata_json": str(metadata_json)},
        sort_keys=True,
    )
    serialized_rows = json.dumps(rows, ensure_ascii=False, sort_keys=True).lower()
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "rows_present": len(rows) > 0,
        "current_only_baseline_measured": current["total"] == len(rows),
        "current_only_baseline_nonzero": isinstance(current["accuracy"], float)
        and current["accuracy"] > 0.0,
        "current_only_baseline_below_threshold": isinstance(current["accuracy"], float)
        and current["accuracy"] <= max_current_only_accuracy,
        "prior_runtime_resolver_measured": prior["total"] == len(rows),
        "prior_runtime_resolver_passes_accuracy": isinstance(prior["accuracy"], float)
        and prior["accuracy"] >= min_prior_resolver_accuracy,
        "prior_resolver_beats_current_only": isinstance(prior_minus_current, float)
        and prior_minus_current >= min_prior_minus_current_only,
        "wrong_cache_control_measured": wrong_cache["total"] == len(rows),
        "prior_resolver_beats_wrong_cache": isinstance(prior_minus_wrong_cache, float)
        and prior_minus_wrong_cache >= min_prior_minus_wrong_cache,
        "candidate_prior_link_present": prior["missing_predictions"] == 0,
        "phase2aw_pretest_leakage_absent": "runtime-visible repair evidence:" not in serialized_rows
        and "structured command identity sidecar:" not in serialized_rows,
        "sealed_v3_not_used_for_pretrain_gate": "external_trace_v3_semantic_required"
        not in serialized_inputs
        and "external_trace_v3_semantic_required" not in serialized_rows,
        "no_freeform_patch_generation_claim": '"freeform_patch_generation": true'
        not in serialized_rows
        and '"no_freeform_patch_generation": false' not in serialized_rows,
    }
    blocked_actions: list[str] = []
    if not all(checks.values()):
        blocked_actions.append("do_not_train_phase2ax_until_runtime_pretrain_gate_passes")
    if not checks["prior_runtime_resolver_passes_accuracy"]:
        blocked_actions.append("revise_phase2ax_candidate_prior_link_before_training")
    if not checks["prior_resolver_beats_wrong_cache"]:
        blocked_actions.append("do_not_train_without_wrong_cache_counterfactual_delta")
    if not checks["phase2aw_pretest_leakage_absent"]:
        blocked_actions.append("do_not_reuse_phase2aw_runner_for_phase2ax_current_surface")
    passed = all(checks.values())
    return {
        "audit_family": "phase2ax_package_loaded_counterfactual_repair_runtime_pretrain_gate",
        "passed": passed,
        "allowed_next_action": (
            "build_phase2ax_head_split_and_run_smoke_only"
            if passed
            else "revise_phase2ax_runtime_or_data_before_training"
        ),
        "blocked_actions": blocked_actions,
        "checks": checks,
        "metrics": {
            "current_only": current,
            "prior_runtime_resolver": prior,
            "wrong_cache": wrong_cache,
            "prior_minus_current_only": prior_minus_current,
            "prior_minus_wrong_cache": prior_minus_wrong_cache,
        },
        "thresholds": {
            "min_prior_resolver_accuracy": min_prior_resolver_accuracy,
            "min_prior_minus_current_only": min_prior_minus_current_only,
            "min_prior_minus_wrong_cache": min_prior_minus_wrong_cache,
            "max_current_only_accuracy": max_current_only_accuracy,
        },
        "claim_boundary": "phase2ax_pretrain_gate_only_not_model_or_claim_evidence",
        "inputs": {
            "tasks_jsonl": str(Path(tasks_jsonl)),
            "metadata_json": str(Path(metadata_json)),
            "data_health_json": str(Path(data_health_json)) if data_health_json else None,
        },
        "notes": [
            "This gate checks runtime-visible prior/candidate resolvability before any training.",
            "It is not model evidence and does not support package or sealed evaluation by itself.",
            "Phase2AW public runner pre-test identity leakage is explicitly blocked for Phase2AX.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AX runtime pretrain gate.")
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--data-health-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-prior-resolver-accuracy", type=float, default=0.85)
    parser.add_argument("--min-prior-minus-current-only", type=float, default=0.25)
    parser.add_argument("--min-prior-minus-wrong-cache", type=float, default=0.25)
    parser.add_argument("--max-current-only-accuracy", type=float, default=0.75)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2ax_runtime_pretrain_gate(
        tasks_jsonl=args.tasks_jsonl,
        metadata_json=args.metadata_json,
        data_health_json=args.data_health_json,
        min_prior_resolver_accuracy=args.min_prior_resolver_accuracy,
        min_prior_minus_current_only=args.min_prior_minus_current_only,
        min_prior_minus_wrong_cache=args.min_prior_minus_wrong_cache,
        max_current_only_accuracy=args.max_current_only_accuracy,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
