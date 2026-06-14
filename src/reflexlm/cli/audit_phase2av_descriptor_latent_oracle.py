from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CLAIM_BOUNDARY = (
    "Descriptor-latent oracle is a non-training separability audit. It may diagnose "
    "whether runtime-visible descriptor_failure_family can explain descriptor labels, "
    "but it does not authorize package, sealed evaluation, production autonomy, or "
    "epoch-making architecture claims."
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _family(row: dict[str, Any]) -> str:
    reference = row.get("nsi_reference")
    if isinstance(reference, dict) and str(reference.get("descriptor_failure_family") or ""):
        return str(reference["descriptor_failure_family"])
    evidence = row.get("runtime_visible_evidence")
    if isinstance(evidence, dict) and str(evidence.get("descriptor_failure_family") or ""):
        return str(evidence["descriptor_failure_family"])
    return "other"


def _label(row: dict[str, Any], key: str) -> int | None:
    try:
        value = int(row.get(key, -100))
    except (TypeError, ValueError):
        return None
    return value if value != -100 else None


def _majority_mapping(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        label = _label(row, key)
        if label is None:
            continue
        counts[_family(row)][label] += 1
    mapping = {}
    for family, counter in counts.items():
        mapping[family] = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return mapping


def _accuracy(rows: list[dict[str, Any]], key: str, mapping: dict[str, int]) -> dict[str, Any]:
    total = 0
    correct = 0
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    missing_families: Counter[str] = Counter()
    for row in rows:
        label = _label(row, key)
        if label is None:
            continue
        family = _family(row)
        prediction = mapping.get(family)
        if prediction is None:
            missing_families[family] += 1
            continue
        total += 1
        hit = int(prediction == label)
        correct += hit
        by_family[family]["total"] += 1
        by_family[family]["correct"] += hit
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_family": {
            family: {
                "total": bucket["total"],
                "correct": bucket["correct"],
                "accuracy": bucket["correct"] / bucket["total"] if bucket["total"] else 0.0,
            }
            for family, bucket in sorted(by_family.items())
        },
        "missing_families": dict(sorted(missing_families.items())),
    }


def _family_label_counts(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        label = _label(row, key)
        if label is None:
            continue
        counts[_family(row)][str(label)] += 1
    return {
        family: dict(sorted(counter.items(), key=lambda item: int(item[0])))
        for family, counter in sorted(counts.items())
    }


def audit_phase2av_descriptor_latent_oracle(
    *,
    train_jsonl: str | Path,
    eval_jsonl: str | Path,
    min_oracle_accuracy: float = 0.85,
) -> dict[str, Any]:
    train_rows = _read_jsonl(train_jsonl)
    eval_rows = _read_jsonl(eval_jsonl)
    operation_mapping = _majority_mapping(train_rows, "patch_operation_label")
    template_mapping = _majority_mapping(train_rows, "patch_template_slot")
    operation_accuracy = _accuracy(eval_rows, "patch_operation_label", operation_mapping)
    template_accuracy = _accuracy(eval_rows, "patch_template_slot", template_mapping)
    passed = (
        operation_accuracy["accuracy"] >= min_oracle_accuracy
        and template_accuracy["accuracy"] >= min_oracle_accuracy
        and not operation_accuracy["missing_families"]
        and not template_accuracy["missing_families"]
    )
    return {
        "artifact_family": "phase2av_descriptor_latent_oracle",
        "passed": passed,
        "claim_boundary": CLAIM_BOUNDARY,
        "sealed_feedback_used": False,
        "uses_model_training": False,
        "uses_gold_for_prediction": False,
        "oracle_source": "train_majority_descriptor_failure_family_mapping",
        "thresholds": {"min_oracle_accuracy": min_oracle_accuracy},
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_family_label_counts": {
            "patch_operation_label": _family_label_counts(train_rows, "patch_operation_label"),
            "patch_template_slot": _family_label_counts(train_rows, "patch_template_slot"),
        },
        "mappings": {
            "patch_operation_label": dict(sorted(operation_mapping.items())),
            "patch_template_slot": dict(sorted(template_mapping.items())),
        },
        "metrics": {
            "patch_operation_oracle": operation_accuracy,
            "patch_template_oracle": template_accuracy,
        },
        "blocked_actions": [
            "do_not_package_from_oracle",
            "do_not_run_sealed_eval_from_oracle",
        ],
        "unsupported_claims": [
            "learned_descriptor_runtime_delta_sufficient_for_full_training",
            "freeform_patch_generation",
            "sealed_cross_model_transfer",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AV descriptor label separability from runtime-visible descriptor latent family."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--min-oracle-accuracy", type=float, default=0.85)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    report = audit_phase2av_descriptor_latent_oracle(
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        min_oracle_accuracy=args.min_oracle_accuracy,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
