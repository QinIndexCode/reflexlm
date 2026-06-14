from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reflexlm.llm.candidate_features import source_overlap_command_slot_prediction


def _read_json(path: str | Path) -> dict[str, Any]:
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


def _candidate_count(row: dict[str, Any]) -> int:
    candidates = row.get("candidate_commands")
    return len(candidates) if isinstance(candidates, list) else 0


def _slot(row: dict[str, Any]) -> int | None:
    slot = row.get("command_slot")
    count = _candidate_count(row)
    return slot if isinstance(slot, int) and 0 <= slot < count else None


def _accuracy(rows: list[dict[str, Any]], predict) -> dict[str, Any]:
    total = 0
    correct = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        gold = _slot(row)
        if gold is None:
            continue
        pred = predict(row)
        if not isinstance(pred, int):
            continue
        total += 1
        correct += int(pred == gold)
        confusion[str(gold)][str(pred)] += 1
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "confusion": {gold: dict(preds) for gold, preds in sorted(confusion.items())},
    }


def _slot_prior(rows: list[dict[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in rows:
        slot = _slot(row)
        if slot is not None:
            counts[slot] += 1
    return counts


def _count_conditioned_prior(rows: list[dict[str, Any]]) -> dict[int, Counter[int]]:
    priors: dict[int, Counter[int]] = defaultdict(Counter)
    for row in rows:
        slot = _slot(row)
        count = _candidate_count(row)
        if slot is not None and count > 0:
            priors[count][slot] += 1
    return priors


def _best_valid_slot(prior: Counter[int], count: int) -> int:
    valid = [(prior.get(slot, 0), -slot, slot) for slot in range(count)]
    if not valid:
        return 0
    return max(valid)[2]


def _eval_accuracy(eval_report: dict[str, Any]) -> float | None:
    metrics = eval_report.get("eval_metrics")
    value = metrics.get("command_slot_accuracy") if isinstance(metrics, dict) else None
    return float(value) if isinstance(value, int | float) else None


def audit_phase2ao_residual_baselines(
    *,
    train_jsonl: str | Path,
    eval_jsonl: str | Path,
    erased_eval_json: str | Path,
    training_summary_json: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    train_rows = _read_jsonl(train_jsonl)
    eval_rows = _read_jsonl(eval_jsonl)
    erased_eval = _read_json(erased_eval_json)
    model_accuracy = _eval_accuracy(erased_eval)
    train_prior = _slot_prior(train_rows)
    eval_prior = _slot_prior(eval_rows)
    count_prior = _count_conditioned_prior(train_rows)
    training_summary = _read_json(training_summary_json) if training_summary_json else {}
    summary_train_slots = (
        training_summary.get("slot_intent_distribution", {})
        .get("train", {})
        .get("command_slots", {})
        if isinstance(training_summary.get("slot_intent_distribution"), dict)
        else {}
    )
    if isinstance(summary_train_slots, dict) and summary_train_slots:
        train_prior = Counter({int(slot): int(count) for slot, count in summary_train_slots.items()})

    baselines = {
        "first_slot": _accuracy(eval_rows, lambda row: 0),
        "last_valid_slot": _accuracy(eval_rows, lambda row: max(0, _candidate_count(row) - 1)),
        "train_slot_prior": _accuracy(
            eval_rows, lambda row: _best_valid_slot(train_prior, _candidate_count(row))
        ),
        "train_candidate_count_conditioned_slot_prior": _accuracy(
            eval_rows,
            lambda row: _best_valid_slot(
                count_prior.get(_candidate_count(row), Counter()),
                _candidate_count(row),
            ),
        ),
        "source_overlap": _accuracy(
            eval_rows,
            lambda row: source_overlap_command_slot_prediction(
                str(row.get("state_prompt") or ""),
                list(row.get("candidate_commands") or []),
            ),
        ),
        "eval_majority_upper_bound_leaky": _accuracy(
            eval_rows, lambda row: _best_valid_slot(eval_prior, _candidate_count(row))
        ),
    }
    nonleaky_names = [
        "first_slot",
        "last_valid_slot",
        "train_slot_prior",
        "train_candidate_count_conditioned_slot_prior",
        "source_overlap",
    ]
    max_nonleaky = max(
        float(baselines[name]["accuracy"])
        for name in nonleaky_names
        if isinstance(baselines[name].get("accuracy"), int | float)
    )
    checks = {
        "model_accuracy_present": isinstance(model_accuracy, float),
        "model_exceeds_max_nonleaky_baseline": isinstance(model_accuracy, float)
        and model_accuracy > max_nonleaky,
        "model_exceeds_eval_majority_leaky_upper_bound": isinstance(model_accuracy, float)
        and isinstance(baselines["eval_majority_upper_bound_leaky"].get("accuracy"), float)
        and model_accuracy > float(baselines["eval_majority_upper_bound_leaky"]["accuracy"]),
    }
    report = {
        "artifact_family": "phase2ao_residual_baselines",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "erased_model_accuracy": model_accuracy,
            "max_nonleaky_baseline_accuracy": max_nonleaky,
            "model_minus_max_nonleaky_baseline": (
                model_accuracy - max_nonleaky if isinstance(model_accuracy, float) else None
            ),
            "train_slot_prior": dict(sorted(train_prior.items())),
            "eval_slot_distribution": dict(sorted(eval_prior.items())),
        },
        "baselines": baselines,
        "interpretation": (
            "This report tests whether erased-sidecar residual performance is explainable by simple "
            "slot priors, candidate count, or source-overlap baselines. It is diagnostic and does not "
            "use sealed data."
        ),
        "claim_boundary": (
            "Residual above these baselines can motivate stronger order/feature controls, but it does "
            "not by itself prove open-ended debugging or production autonomy."
        ),
        "inputs": {
            "train_jsonl": str(Path(train_jsonl)),
            "eval_jsonl": str(Path(eval_jsonl)),
            "erased_eval_json": str(Path(erased_eval_json)),
            "training_summary_json": str(Path(training_summary_json)) if training_summary_json else None,
        },
    }
    if output_json:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AO erased residual baselines.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--erased-eval-json", required=True)
    parser.add_argument("--training-summary-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ao_residual_baselines(
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        erased_eval_json=args.erased_eval_json,
        training_summary_json=args.training_summary_json,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
