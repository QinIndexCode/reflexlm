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


def _metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _source_overlap_delta_rows(records: Any) -> int:
    if not isinstance(records, list):
        return 0
    return sum(
        1
        for record in records
        if isinstance(record, dict)
        and record.get("source_overlap_correct") is False
        and record.get("command_slot_correct") is True
    )


def audit_phase2au_eval_postflight(
    *,
    eval_summary_json: str | Path,
    pretrain_gate_json: str | Path,
    min_eval_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    max_source_overlap_baseline: float = 0.95,
    min_descriptor_count: int = 1,
    min_valid_candidates_per_row: int = 2,
    min_source_overlap_delta_rows: int = 1,
) -> dict[str, Any]:
    summary = _read_json(eval_summary_json)
    pretrain = _read_json(pretrain_gate_json)
    metrics = summary.get("eval_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    baseline = summary.get("source_overlap_command_slot_baseline")
    eval_split = str(summary.get("eval_split") or "")
    baseline_split = baseline.get(eval_split) if isinstance(baseline, dict) else {}
    pairwise = summary.get("pairwise_candidate_encoding")
    pairwise_split = pairwise.get(eval_split) if isinstance(pairwise, dict) else {}

    command_slot_accuracy = _metric(metrics, "command_slot_accuracy")
    source_overlap_accuracy = (
        _metric(baseline_split, "accuracy") if isinstance(baseline_split, dict) else None
    )
    model_minus_source = (
        command_slot_accuracy - source_overlap_accuracy
        if isinstance(command_slot_accuracy, float)
        and isinstance(source_overlap_accuracy, float)
        else None
    )
    patch_operation_count = _metric(metrics, "patch_operation_count")
    patch_template_count = _metric(metrics, "patch_template_slot_count")
    max_valid_candidates = (
        _metric(pairwise_split, "max_valid_candidates_per_row")
        if isinstance(pairwise_split, dict)
        else None
    )
    delta_rows = _source_overlap_delta_rows(summary.get("prediction_records"))
    low_level_qwen_calls = _metric(summary, "low_level_qwen_calls_target")

    checks = {
        "pretrain_gate_passed": pretrain.get("passed") is True,
        "eval_summary_present": bool(summary),
        "eval_accuracy_gate": isinstance(command_slot_accuracy, float)
        and command_slot_accuracy >= min_eval_accuracy,
        "source_overlap_not_ceiling": isinstance(source_overlap_accuracy, float)
        and source_overlap_accuracy <= max_source_overlap_baseline,
        "model_minus_source_overlap_gate": isinstance(model_minus_source, float)
        and model_minus_source >= min_model_minus_source_overlap,
        "source_overlap_delta_rows_present": delta_rows >= min_source_overlap_delta_rows,
        "descriptor_operation_evaluable": isinstance(patch_operation_count, float)
        and patch_operation_count >= min_descriptor_count,
        "descriptor_template_evaluable": isinstance(patch_template_count, float)
        and patch_template_count >= min_descriptor_count,
        "nontrivial_command_candidate_set": isinstance(max_valid_candidates, float)
        and max_valid_candidates >= min_valid_candidates_per_row,
        "low_level_qwen_calls_zero": low_level_qwen_calls == 0.0,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2au_policy_required_eval_postflight",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "eval_split": eval_split,
            "command_slot_accuracy": command_slot_accuracy,
            "source_overlap_command_slot_accuracy": source_overlap_accuracy,
            "model_minus_source_overlap": model_minus_source,
            "source_overlap_delta_rows": delta_rows,
            "patch_operation_count": patch_operation_count,
            "patch_template_slot_count": patch_template_count,
            "max_valid_candidates_per_row": max_valid_candidates,
            "eval_examples": summary.get("eval_examples"),
            "eval_rows_hash": summary.get("eval_rows_hash"),
            "config_hash": summary.get("training_summary_config_hash"),
        },
        "claim_boundary": (
            "phase2au_holdout_delta_supported_for_capacity_smoke_not_claim_upgrade"
            if passed
            else "phase2au_holdout_eval_non_evidence_requires_redesign_or_rerun"
        ),
        "blocked_actions": []
        if passed
        else [
            "do_not_package_phase2au",
            "do_not_run_sealed_phase2au",
            "do_not_claim_phase2au_runtime_delta",
        ],
        "unsupported_claims": [
            "freeform_patch_generation",
            "sealed_cross_model_transfer",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "inputs": {
            "eval_summary_json": str(Path(eval_summary_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
        },
        "thresholds": {
            "min_eval_accuracy": min_eval_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "max_source_overlap_baseline": max_source_overlap_baseline,
            "min_descriptor_count": min_descriptor_count,
            "min_valid_candidates_per_row": min_valid_candidates_per_row,
            "min_source_overlap_delta_rows": min_source_overlap_delta_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AU held-out eval postflight.")
    parser.add_argument("--eval-summary-json", required=True)
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-eval-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--max-source-overlap-baseline", type=float, default=0.95)
    parser.add_argument("--min-descriptor-count", type=int, default=1)
    parser.add_argument("--min-valid-candidates-per-row", type=int, default=2)
    parser.add_argument("--min-source-overlap-delta-rows", type=int, default=1)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2au_eval_postflight(
        eval_summary_json=args.eval_summary_json,
        pretrain_gate_json=args.pretrain_gate_json,
        min_eval_accuracy=args.min_eval_accuracy,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        max_source_overlap_baseline=args.max_source_overlap_baseline,
        min_descriptor_count=args.min_descriptor_count,
        min_valid_candidates_per_row=args.min_valid_candidates_per_row,
        min_source_overlap_delta_rows=args.min_source_overlap_delta_rows,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
