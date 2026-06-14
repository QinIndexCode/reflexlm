from __future__ import annotations

import argparse
import json
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


def _metric(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    return float(value) if isinstance(value, (int, float)) else None


def _baseline_accuracy(report: dict[str, Any], split: str) -> float | None:
    baseline = _dict(_dict(report.get("source_overlap_command_slot_baseline")).get(split))
    value = baseline.get("accuracy")
    return float(value) if isinstance(value, (int, float)) else None


def build_phase2at_smoke_postflight(
    *,
    data_health_json: str | Path,
    training_summary_json: str | Path,
    eval_json: str | Path,
    min_command_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    min_descriptor_accuracy: float = 0.85,
) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    summary = _read_json(training_summary_json)
    eval_report = _read_json(eval_json)
    metrics = _dict(eval_report.get("eval_metrics"))
    source_overlap = _baseline_accuracy(eval_report, str(eval_report.get("eval_split") or "val"))
    command_slot = _metric(metrics, "command_slot_accuracy")
    model_minus_source = (
        command_slot - source_overlap
        if isinstance(command_slot, float) and isinstance(source_overlap, float)
        else None
    )
    descriptor_metrics = {
        "patch_operation_accuracy": _metric(metrics, "patch_operation_accuracy"),
        "patch_target_file_slot_accuracy": _metric(metrics, "patch_target_file_slot_accuracy"),
        "patch_template_slot_accuracy": _metric(metrics, "patch_template_slot_accuracy"),
    }
    descriptor_counts = {
        name.replace("_accuracy", "_count"): _metric(metrics, name.replace("_accuracy", "_count"))
        for name in descriptor_metrics
    }
    head_config = _dict(eval_report.get("head_config"))
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "training_summary_has_effective_split_hashes": bool(
            _dict(summary.get("effective_split_hashes"))
        ),
        "eval_rows_hash_present": isinstance(eval_report.get("eval_rows_hash"), str),
        "open_repair_heads_enabled": eval_report.get("open_repair_heads_enabled") is True
        and head_config.get("open_repair_heads_enabled") is True,
        "learned_descriptor_heads_recorded": bool(
            _dict(summary.get("learned_patch_descriptor_heads")).get("enabled")
        ),
        "pairwise_disabled_for_descriptor_smoke": eval_report.get("use_pairwise_command_reranker")
        is False,
        "no_json_motor_target": eval_report.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": eval_report.get("low_level_qwen_calls_target") == 0,
        "source_overlap_baseline_present": isinstance(source_overlap, float),
        "source_overlap_nonzero": isinstance(source_overlap, float) and source_overlap > 0.0,
        "source_overlap_not_ceiling": isinstance(source_overlap, float) and source_overlap < 0.85,
        "command_slot_val_gate": isinstance(command_slot, float)
        and command_slot >= min_command_slot_accuracy,
        "model_beats_source_overlap": isinstance(model_minus_source, float)
        and model_minus_source >= min_model_minus_source_overlap,
        "descriptor_metrics_present": all(
            isinstance(value, float) for value in descriptor_metrics.values()
        ),
        "descriptor_counts_present": all(
            isinstance(value, float) and value > 0.0 for value in descriptor_counts.values()
        ),
        "patch_operation_gate": isinstance(
            descriptor_metrics["patch_operation_accuracy"], float
        )
        and descriptor_metrics["patch_operation_accuracy"] >= min_descriptor_accuracy,
        "patch_target_file_slot_gate": isinstance(
            descriptor_metrics["patch_target_file_slot_accuracy"], float
        )
        and descriptor_metrics["patch_target_file_slot_accuracy"] >= min_descriptor_accuracy,
        "patch_template_slot_gate": isinstance(
            descriptor_metrics["patch_template_slot_accuracy"], float
        )
        and descriptor_metrics["patch_template_slot_accuracy"] >= min_descriptor_accuracy,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2at_learned_patch_candidate_smoke_postflight",
        "passed": passed,
        "claim_boundary": (
            "Non-sealed smoke postflight for bounded descriptor heads only; this does not "
            "prove freeform patch generation, production autonomy, sealed transfer, or an "
            "epoch-making architecture."
        ),
        "checks": checks,
        "metrics": {
            "command_slot_accuracy": command_slot,
            "source_overlap_accuracy": source_overlap,
            "model_minus_source_overlap_accuracy": model_minus_source,
            **descriptor_metrics,
            **descriptor_counts,
        },
        "thresholds": {
            "min_command_slot_accuracy": min_command_slot_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_descriptor_accuracy": min_descriptor_accuracy,
        },
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "training_summary_json": str(Path(training_summary_json)),
            "eval_json": str(Path(eval_json)),
        },
        "supported_claims": [
            "phase2at_nonsealed_smoke_supports_bounded_descriptor_learning"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "learned_freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_start_phase2at_full_training_from_this_smoke",
            "do_not_package_as_learned_patch_generation",
            "do_not_use_this_as_epoch_making_architecture_evidence",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AT descriptor smoke postflight.")
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-descriptor-accuracy", type=float, default=0.85)
    args = parser.parse_args()
    report = build_phase2at_smoke_postflight(
        data_health_json=args.data_health_json,
        training_summary_json=args.training_summary_json,
        eval_json=args.eval_json,
        min_command_slot_accuracy=args.min_command_slot_accuracy,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        min_descriptor_accuracy=args.min_descriptor_accuracy,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
