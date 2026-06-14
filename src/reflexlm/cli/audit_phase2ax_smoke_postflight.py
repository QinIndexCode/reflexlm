from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARTIFACT_FAMILY = "phase2ax_package_loaded_counterfactual_repair_smoke_postflight"


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _last_val_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history")
    if not isinstance(history, list) or not history:
        return {}
    latest = history[-1]
    if not isinstance(latest, dict):
        return {}
    metrics = latest.get("val_metrics")
    return metrics if isinstance(metrics, dict) else {}


def _latest_history(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history")
    if isinstance(history, list) and history and isinstance(history[-1], dict):
        return history[-1]
    return {}


def _val_source_overlap(summary: dict[str, Any]) -> float | None:
    baseline = summary.get("source_overlap_command_slot_baseline")
    if not isinstance(baseline, dict):
        return None
    val = baseline.get("val")
    if not isinstance(val, dict):
        return None
    return _float(val.get("accuracy"))


def audit_phase2ax_smoke_postflight(
    *,
    training_summary_json: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    head_manifest_json: str | Path,
    min_val_command_slot_accuracy: float = 0.85,
    min_val_command_slot_count: int = 32,
    min_model_minus_current_only: float = 0.25,
    min_model_minus_source_overlap: float = 0.25,
) -> dict[str, Any]:
    summary = _read_json(training_summary_json)
    data_health = _read_json(data_health_json)
    pretrain = _read_json(pretrain_gate_json)
    manifest = _read_json(head_manifest_json)
    val_metrics = _last_val_metrics(summary)
    latest = _latest_history(summary)
    val_accuracy = _float(val_metrics.get("command_slot_accuracy"))
    val_count = _float(val_metrics.get("command_slot_count"))
    source_overlap_accuracy = _val_source_overlap(summary)
    current_only_accuracy = _float(
        ((pretrain.get("metrics") or {}).get("current_only") or {}).get("accuracy")
    )
    wrong_cache_accuracy = _float(
        ((pretrain.get("metrics") or {}).get("wrong_cache") or {}).get("accuracy")
    )
    patch_operation_accuracy = _float(val_metrics.get("patch_operation_accuracy"))
    patch_template_accuracy = _float(val_metrics.get("patch_template_slot_accuracy"))
    patch_operation_count = _float(val_metrics.get("patch_operation_count"))
    patch_template_count = _float(val_metrics.get("patch_template_slot_count"))
    model_minus_current = (
        val_accuracy - current_only_accuracy
        if isinstance(val_accuracy, float) and isinstance(current_only_accuracy, float)
        else None
    )
    model_minus_source = (
        val_accuracy - source_overlap_accuracy
        if isinstance(val_accuracy, float) and isinstance(source_overlap_accuracy, float)
        else None
    )
    model_minus_wrong_cache = (
        val_accuracy - wrong_cache_accuracy
        if isinstance(val_accuracy, float) and isinstance(wrong_cache_accuracy, float)
        else None
    )
    adapter_name = str(summary.get("adapter_name") or "")
    pairwise = summary.get("pairwise_candidate_encoding")
    pairwise_val = pairwise.get("val") if isinstance(pairwise, dict) else {}
    pairwise_scored = (
        _float(pairwise_val.get("pairwise_scored_candidates"))
        if isinstance(pairwise_val, dict)
        else None
    )
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "pretrain_gate_passed": pretrain.get("passed") is True,
        "head_manifest_passed": manifest.get("passed") is True,
        "training_summary_present": bool(summary),
        "adapter_is_phase2ax_smoke": "phase2ax_package_loaded_counterfactual_repair" in adapter_name
        and "smoke" in adapter_name,
        "val_command_slot_accuracy_gate": isinstance(val_accuracy, float)
        and val_accuracy >= min_val_command_slot_accuracy,
        "val_command_slot_count_gate": isinstance(val_count, float)
        and val_count >= min_val_command_slot_count,
        "model_beats_current_only": isinstance(model_minus_current, float)
        and model_minus_current >= min_model_minus_current_only,
        "model_beats_source_overlap": isinstance(model_minus_source, float)
        and model_minus_source >= min_model_minus_source_overlap,
        "wrong_cache_control_recorded": isinstance(wrong_cache_accuracy, float),
        "pairwise_disabled_for_unmixed_mechanism": summary.get("use_pairwise_command_reranker")
        is False
        and pairwise_scored == 0.0,
        "open_repair_heads_enabled": summary.get("open_repair_heads_enabled") is True,
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
        "sealed_not_allowed_by_manifest": manifest.get("sealed_eval_allowed") is False,
        "package_not_allowed_by_manifest": manifest.get("package_allowed") is False,
    }
    passed = all(checks.values())
    patch_descriptor_evaluable = (
        isinstance(patch_operation_count, float)
        and patch_operation_count > 0
        and isinstance(patch_template_count, float)
        and patch_template_count > 0
    )
    package_ready = False
    return {
        "artifact_family": ARTIFACT_FAMILY,
        "passed": passed,
        "ready_for_phase2ax_full_nonsealed_training": passed,
        "ready_for_package_or_execution_claim": package_ready,
        "ready_for_sealed_eval": False,
        "checks": checks,
        "metrics": {
            "train_examples": summary.get("train_examples"),
            "val_examples": summary.get("val_examples"),
            "val_command_slot_accuracy": val_accuracy,
            "val_command_slot_count": val_count,
            "current_only_accuracy": current_only_accuracy,
            "source_overlap_val_accuracy": source_overlap_accuracy,
            "wrong_cache_accuracy": wrong_cache_accuracy,
            "model_minus_current_only": model_minus_current,
            "model_minus_source_overlap": model_minus_source,
            "model_minus_wrong_cache": model_minus_wrong_cache,
            "patch_operation_accuracy": patch_operation_accuracy,
            "patch_template_slot_accuracy": patch_template_accuracy,
            "patch_descriptor_evaluable": patch_descriptor_evaluable,
            "train_loss": latest.get("train_loss"),
            "val_loss": val_metrics.get("loss"),
            "train_elapsed_seconds": latest.get("train_elapsed_seconds"),
            "train_steps_per_second": latest.get("train_steps_per_second"),
            "config_hash": summary.get("config_hash"),
            "effective_split_hashes": summary.get("effective_split_hashes"),
            "manifest_effective_split_hashes": manifest.get("effective_split_hashes"),
        },
        "claim_boundary": (
            "phase2ax_smoke_supports_full_nonsealed_training_only_not_package_or_epoch_claim"
            if passed
            else "phase2ax_smoke_failed_or_incomplete_not_claim_evidence"
        ),
        "blocked_actions": [
            "do_not_package_phase2ax_from_smoke",
            "do_not_run_sealed_v3_from_smoke",
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ]
        + ([] if passed else ["do_not_start_phase2ax_full_training"]),
        "unsupported_claims": [
            "full_package_beats_native_head_only_on_phase2ax",
            "sealed_cross_model_transfer",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "thresholds": {
            "min_val_command_slot_accuracy": min_val_command_slot_accuracy,
            "min_val_command_slot_count": min_val_command_slot_count,
            "min_model_minus_current_only": min_model_minus_current_only,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
        },
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "data_health_json": str(Path(data_health_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "head_manifest_json": str(Path(head_manifest_json)),
        },
        "notes": [
            "This is smoke evidence for counterfactual command-slot selection only.",
            "Patch descriptor metrics are recorded but do not make this a repair-execution result.",
            "Package, sealed v3, production autonomy, and epoch-making architecture claims remain blocked.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AX smoke training postflight.")
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--head-manifest-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-val-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-val-command-slot-count", type=int, default=32)
    parser.add_argument("--min-model-minus-current-only", type=float, default=0.25)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.25)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ax_smoke_postflight(
        training_summary_json=args.training_summary_json,
        data_health_json=args.data_health_json,
        pretrain_gate_json=args.pretrain_gate_json,
        head_manifest_json=args.head_manifest_json,
        min_val_command_slot_accuracy=args.min_val_command_slot_accuracy,
        min_val_command_slot_count=args.min_val_command_slot_count,
        min_model_minus_current_only=args.min_model_minus_current_only,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
