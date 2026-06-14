from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _int(value: Any) -> int | None:
    number = _num(value)
    return int(number) if number is not None else None


def _metric(postflight: dict[str, Any], name: str) -> float | None:
    return _num(_dict(postflight.get("metrics")).get(name))


def _eval_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = _dict(summary.get("eval_metrics"))
    if metrics:
        return metrics
    history = summary.get("history")
    if isinstance(history, list) and history:
        return _dict(_dict(history[-1]).get("val_metrics"))
    return {}


def _slot_confusion_errors(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = _eval_metrics(summary)
    confusion = _dict(_dict(metrics.get("slot_confusion")).get("command_slot"))
    error_edges: list[dict[str, Any]] = []
    total = 0
    correct = 0
    by_target: dict[str, dict[str, int]] = {}
    for target, predictions in sorted(confusion.items(), key=lambda item: int(item[0])):
        target_total = 0
        target_correct = 0
        for predicted, count_value in sorted(_dict(predictions).items(), key=lambda item: int(item[0])):
            count = int(count_value)
            target_total += count
            total += count
            if predicted == target:
                target_correct += count
                correct += count
            else:
                error_edges.append(
                    {
                        "target_slot": target,
                        "predicted_slot": predicted,
                        "count": count,
                    }
                )
        by_target[target] = {
            "total": target_total,
            "correct": target_correct,
            "errors": target_total - target_correct,
        }
    return {
        "total": total,
        "correct": correct,
        "errors": total - correct,
        "by_target": by_target,
        "error_edges": sorted(
            error_edges,
            key=lambda row: (-int(row["count"]), int(row["target_slot"]), int(row["predicted_slot"])),
        ),
    }


def _classification(postflight: dict[str, Any]) -> str:
    checks = _dict(postflight.get("checks"))
    if postflight.get("passed") is True:
        return "no_failure_postflight_passed"
    if (
        checks.get("command_slot_accuracy_min") is True
        and checks.get("model_beats_source_overlap") is True
        and checks.get("model_beats_runtime_identity") is False
    ):
        return "runtime_identity_residual_shortcut_not_broken"
    if checks.get("command_slot_accuracy_min") is False:
        return "model_accuracy_below_minimum"
    if checks.get("model_beats_source_overlap") is False:
        return "source_overlap_shortcut_not_broken"
    return "mixed_holdout_gate_failure"


def build_phase2af_holdout_failure_audit(
    *,
    postflight_json: str | Path,
    eval_summary_json: str | Path,
    manifest_json: str | Path | None = None,
    output_json: str | Path,
) -> dict[str, Any]:
    postflight = _read_json(postflight_json)
    summary = _read_json(eval_summary_json)
    manifest = _read_json(manifest_json) if manifest_json else {}
    metrics = _dict(postflight.get("metrics"))
    thresholds = _dict(postflight.get("thresholds"))
    count = _int(metrics.get("command_slot_count")) or 0
    accuracy = _metric(postflight, "command_slot_accuracy")
    runtime_identity = _metric(postflight, "runtime_identity_heuristic_accuracy")
    min_delta = _num(thresholds.get("min_model_minus_runtime_identity")) or 0.0
    required_accuracy = (
        runtime_identity + min_delta if runtime_identity is not None else None
    )
    actual_correct = int(round((accuracy or 0.0) * count)) if count else 0
    required_correct = (
        int(math.ceil(required_accuracy * count - 1e-12))
        if required_accuracy is not None and count
        else None
    )
    short_by = (
        max(0, required_correct - actual_correct)
        if required_correct is not None
        else None
    )
    slot_confusion = _slot_confusion_errors(summary)
    train_sampling = _dict(manifest.get("train_sampling"))
    command_intents = (
        _dict(_dict(_dict(summary.get("slot_intent_distribution")).get(str(metrics.get("split") or "holdout"))).get("command_intents"))
    )
    single_command_intent = len(command_intents) == 1 if command_intents else None
    classification = _classification(postflight)
    report = {
        "artifact_family": "phase2af_hardened_structural_sidecar_holdout_failure_audit",
        "passed": False,
        "failure_class": classification,
        "claim_upgrade_allowed": False,
        "training_full_allowed": False,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "metrics": metrics,
        "thresholds": thresholds,
        "runtime_identity_gap": {
            "actual_correct": actual_correct,
            "required_correct_for_gate": required_correct,
            "additional_correct_rows_needed": short_by,
            "required_accuracy_for_gate": required_accuracy,
        },
        "slot_confusion": slot_confusion,
        "diagnosis": {
            "primary_issue": (
                "The adapter learned a high-accuracy command-slot mapping, but holdout performance "
                "does not clear the measured runtime-identity shortcut by the preregistered margin."
            )
            if classification == "runtime_identity_residual_shortcut_not_broken"
            else "The holdout gate failed; inspect failed checks before any further training.",
            "single_command_intent_observed": single_command_intent,
            "train_shortcut_bucket_balanced": train_sampling.get("shortcut_bucket_balanced_train"),
            "train_sampling_duplicates_used": train_sampling.get("duplicates_are_training_sampling_only"),
        },
        "required_next_dataset_or_model_properties": [
            "holdout must keep runtime_identity_heuristic below ceiling and full model must beat it by >=10pp",
            "add non-sealed rows where runtime identity points to the wrong slot but structural evidence points to the correct slot",
            "avoid repeating rows as the only pressure mechanism; prefer new repo-origin-disjoint evidence families",
            "add ablation/control evaluation only after a holdout postflight passes",
            "do not package, do not run sealed, and do not claim epoch-making architecture from this result",
        ],
        "blocked_actions": [
            "do_not_train_phase2af_full_from_this_adapter",
            "do_not_package_phase2af",
            "do_not_run_sealed_phase2af",
            "do_not_claim_hardened_structural_sidecar_mechanism",
        ],
        "inputs": {
            "postflight_json": str(Path(postflight_json)),
            "eval_summary_json": str(Path(eval_summary_json)),
            "manifest_json": str(Path(manifest_json)) if manifest_json else None,
        },
    }
    _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AF holdout failure audit.")
    parser.add_argument("--postflight-json", required=True)
    parser.add_argument("--eval-summary-json", required=True)
    parser.add_argument("--manifest-json")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2af_holdout_failure_audit(
        postflight_json=args.postflight_json,
        eval_summary_json=args.eval_summary_json,
        manifest_json=args.manifest_json,
        output_json=args.output_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
