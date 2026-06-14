from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _baseline_accuracy(summary: dict[str, Any], split: str) -> float | None:
    baseline = _dict(summary.get("source_overlap_command_slot_baseline")).get(split)
    if not isinstance(baseline, dict):
        return None
    value = baseline.get("accuracy")
    return float(value) if isinstance(value, (int, float)) else None


def audit_phase2av_eval_postflight(
    *,
    eval_summary_json: str | Path,
    eval_split: str = "holdout",
    min_command_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    min_descriptor_accuracy: float = 0.85,
) -> dict[str, Any]:
    summary = _read_json(eval_summary_json)
    metrics = _dict(summary.get("eval_metrics"))
    source_overlap = _baseline_accuracy(summary, eval_split)
    command_slot_accuracy = _metric(metrics, "command_slot_accuracy")
    model_minus_source = (
        command_slot_accuracy - source_overlap
        if isinstance(command_slot_accuracy, float) and isinstance(source_overlap, float)
        else None
    )
    patch_operation_accuracy = _metric(metrics, "patch_operation_accuracy")
    patch_template_accuracy = _metric(metrics, "patch_template_slot_accuracy")
    patch_target_file_accuracy = _metric(metrics, "patch_target_file_slot_accuracy")
    patch_operation_count = _metric(metrics, "patch_operation_count")
    patch_template_count = _metric(metrics, "patch_template_slot_count")
    pairwise = _dict(_dict(summary.get("pairwise_candidate_encoding")).get(eval_split))
    max_valid_candidates = _metric(pairwise, "max_valid_candidates_per_row")

    checks = {
        "eval_summary_present": bool(summary),
        "eval_split_expected": summary.get("eval_split") == eval_split,
        "open_repair_heads_enabled": summary.get("open_repair_heads_enabled") is True,
        "pairwise_disabled": summary.get("use_pairwise_command_reranker") is False,
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
        "source_overlap_baseline_present": isinstance(source_overlap, float),
        "source_overlap_nonzero": isinstance(source_overlap, float) and source_overlap > 0.0,
        "command_slot_eval_gate": isinstance(command_slot_accuracy, float)
        and command_slot_accuracy >= min_command_slot_accuracy,
        "model_beats_source_overlap": isinstance(model_minus_source, float)
        and model_minus_source >= min_model_minus_source_overlap,
        "descriptor_operation_gate": isinstance(patch_operation_accuracy, float)
        and patch_operation_accuracy >= min_descriptor_accuracy,
        "descriptor_template_gate": isinstance(patch_template_accuracy, float)
        and patch_template_accuracy >= min_descriptor_accuracy,
        "descriptor_target_file_gate": isinstance(patch_target_file_accuracy, float)
        and patch_target_file_accuracy >= min_descriptor_accuracy,
        "descriptor_counts_present": isinstance(patch_operation_count, float)
        and patch_operation_count > 0.0
        and isinstance(patch_template_count, float)
        and patch_template_count > 0.0,
        "nontrivial_candidate_set": isinstance(max_valid_candidates, float)
        and max_valid_candidates >= 2.0,
    }
    passed = all(checks.values())
    failure_modes: list[str] = []
    if checks["command_slot_eval_gate"] is False:
        failure_modes.append("command_slot_accuracy_below_gate")
    if checks["model_beats_source_overlap"] is False:
        failure_modes.append("model_does_not_beat_source_overlap")
    if checks["descriptor_operation_gate"] is False:
        failure_modes.append("patch_operation_accuracy_below_gate")
    if checks["descriptor_template_gate"] is False:
        failure_modes.append("patch_template_accuracy_below_gate")
    slot_confusion = _dict(_dict(metrics.get("slot_confusion")).get("command_slot"))
    if slot_confusion and all(
        set(preds.keys()) <= {"0"} for preds in slot_confusion.values() if isinstance(preds, dict)
    ):
        failure_modes.append("command_slot_majority_slot0_collapse")
    patch_operation_confusion = _dict(_dict(metrics.get("slot_confusion")).get("patch_operation"))
    if patch_operation_confusion and all(
        set(preds.keys()) <= {"2"}
        for preds in patch_operation_confusion.values()
        if isinstance(preds, dict)
    ):
        failure_modes.append("patch_operation_majority_insert_import_collapse")

    return {
        "artifact_family": "phase2av_eval_postflight",
        "passed": passed,
        "ready_for_phase2av_full_training": False,
        "claim_boundary": (
            "Phase2AV eval postflight validates one non-sealed held-out split. "
            "It does not authorize packaging, sealed evaluation, freeform patch "
            "generation, production autonomy, or epoch-making architecture claims."
        ),
        "checks": checks,
        "metrics": {
            "eval_split": eval_split,
            "eval_examples": summary.get("eval_examples"),
            "eval_rows_hash": summary.get("eval_rows_hash"),
            "command_slot_accuracy": command_slot_accuracy,
            "source_overlap_accuracy": source_overlap,
            "model_minus_source_overlap_accuracy": model_minus_source,
            "patch_operation_accuracy": patch_operation_accuracy,
            "patch_template_slot_accuracy": patch_template_accuracy,
            "patch_target_file_slot_accuracy": patch_target_file_accuracy,
            "patch_operation_count": patch_operation_count,
            "patch_template_slot_count": patch_template_count,
            "max_valid_candidates_per_row": max_valid_candidates,
            "slot_confusion": metrics.get("slot_confusion"),
            "slot_intent_distribution": summary.get("slot_intent_distribution"),
            "effective_split_hashes": summary.get("effective_split_hashes"),
        },
        "failure_modes": sorted(set(failure_modes)),
        "thresholds": {
            "min_command_slot_accuracy": min_command_slot_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_descriptor_accuracy": min_descriptor_accuracy,
        },
        "supported_claims": []
        if not passed
        else ["phase2av_nonsealed_holdout_supports_bounded_descriptor_runtime_learning"],
        "unsupported_claims": [
            "phase2av_full_training_ready",
            "freeform_patch_generation",
            "sealed_cross_model_transfer",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_start_phase2av_full_training",
            "do_not_package_phase2av",
            "do_not_run_sealed_eval_for_phase2av",
            "redesign_nonsealed_phase2av_data_or_training_before_retry",
        ],
        "inputs": {
            "eval_summary_json": str(Path(eval_summary_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AV non-sealed eval postflight.")
    parser.add_argument("--eval-summary-json", required=True)
    parser.add_argument("--eval-split", default="holdout")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-descriptor-accuracy", type=float, default=0.85)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2av_eval_postflight(
        eval_summary_json=args.eval_summary_json,
        eval_split=args.eval_split,
        min_command_slot_accuracy=args.min_command_slot_accuracy,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        min_descriptor_accuracy=args.min_descriptor_accuracy,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
