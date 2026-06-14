from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ARTIFACT_FAMILY = "phase2ax_package_loaded_counterfactual_repair_full_postflight"


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


def _metric_accuracy(report: dict[str, Any], metric_name: str) -> float | None:
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return None
    metric = metrics.get(metric_name)
    if not isinstance(metric, dict):
        return None
    return _float(metric.get("accuracy"))


def _manifest_check(manifest: dict[str, Any], check_name: str) -> bool:
    checks = manifest.get("checks")
    return isinstance(checks, dict) and checks.get(check_name) is True


def audit_phase2ax_full_postflight(
    *,
    training_summary_json: str | Path,
    full_manifest_json: str | Path,
    train_data_health_json: str | Path,
    val_data_health_json: str | Path,
    train_pretrain_gate_json: str | Path,
    val_pretrain_gate_json: str | Path,
    smoke_postflight_json: str | Path,
    min_val_command_slot_accuracy: float = 0.85,
    min_val_command_slot_count: int = 128,
    min_model_minus_current_only: float = 0.25,
    min_model_minus_source_overlap: float = 0.25,
    min_model_minus_wrong_cache: float = 0.25,
    min_patch_descriptor_accuracy_for_package: float = 0.75,
) -> dict[str, Any]:
    summary = _read_json(training_summary_json)
    manifest = _read_json(full_manifest_json)
    train_data_health = _read_json(train_data_health_json)
    val_data_health = _read_json(val_data_health_json)
    train_pretrain = _read_json(train_pretrain_gate_json)
    val_pretrain = _read_json(val_pretrain_gate_json)
    smoke_postflight = _read_json(smoke_postflight_json)

    val_metrics = _last_val_metrics(summary)
    latest = _latest_history(summary)
    val_accuracy = _float(val_metrics.get("command_slot_accuracy"))
    val_count = _float(val_metrics.get("command_slot_count"))
    source_overlap_accuracy = _val_source_overlap(summary)
    current_only_accuracy = _metric_accuracy(val_pretrain, "current_only")
    wrong_cache_accuracy = _metric_accuracy(val_pretrain, "wrong_cache")
    prior_runtime_accuracy = _metric_accuracy(val_pretrain, "prior_runtime_resolver")
    patch_operation_accuracy = _float(val_metrics.get("patch_operation_accuracy"))
    patch_template_accuracy = _float(val_metrics.get("patch_template_slot_accuracy"))
    patch_target_file_accuracy = _float(val_metrics.get("patch_target_file_slot_accuracy"))
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
    open_repair_contract = summary.get("open_repair_training_contract")
    if not isinstance(open_repair_contract, dict):
        open_repair_contract = {}

    checks = {
        "smoke_postflight_passed": smoke_postflight.get("passed") is True,
        "smoke_allows_full_nonsealed_training": smoke_postflight.get(
            "ready_for_phase2ax_full_nonsealed_training"
        )
        is True,
        "full_manifest_passed": manifest.get("passed") is True,
        "full_manifest_allows_full_training": manifest.get("full_training_allowed") is True,
        "full_manifest_repo_origin_disjoint": manifest.get("repo_origins", {}).get("overlap")
        == []
        and _manifest_check(manifest, "repo_origin_disjoint"),
        "train_data_health_passed": train_data_health.get("passed") is True,
        "val_data_health_passed": val_data_health.get("passed") is True,
        "train_pretrain_gate_passed": train_pretrain.get("passed") is True,
        "val_pretrain_gate_passed": val_pretrain.get("passed") is True,
        "training_summary_present": bool(summary),
        "adapter_is_phase2ax_full": "phase2ax_package_loaded_counterfactual_repair" in adapter_name
        and "full" in adapter_name,
        "train_rows_match_manifest": summary.get("train_examples") == manifest.get("train_rows"),
        "val_rows_match_manifest": summary.get("val_examples") == manifest.get("val_rows"),
        "val_command_slot_accuracy_gate": isinstance(val_accuracy, float)
        and val_accuracy >= min_val_command_slot_accuracy,
        "val_command_slot_count_gate": isinstance(val_count, float)
        and val_count >= min_val_command_slot_count,
        "source_overlap_recorded": isinstance(source_overlap_accuracy, float),
        "current_only_recorded": isinstance(current_only_accuracy, float),
        "wrong_cache_control_recorded": isinstance(wrong_cache_accuracy, float),
        "prior_runtime_resolver_recorded": isinstance(prior_runtime_accuracy, float),
        "model_beats_current_only": isinstance(model_minus_current, float)
        and model_minus_current >= min_model_minus_current_only,
        "model_beats_source_overlap": isinstance(model_minus_source, float)
        and model_minus_source >= min_model_minus_source_overlap,
        "model_beats_wrong_cache": isinstance(model_minus_wrong_cache, float)
        and model_minus_wrong_cache >= min_model_minus_wrong_cache,
        "pairwise_disabled_for_unmixed_mechanism": summary.get("use_pairwise_command_reranker")
        is False
        and pairwise_scored == 0.0,
        "open_repair_heads_enabled": summary.get("open_repair_heads_enabled") is True,
        "no_json_motor_target": summary.get("no_json_motor_target") is True
        and open_repair_contract.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0
        and open_repair_contract.get("low_level_qwen_calls_target") == 0,
        "freeform_patch_text_not_targeted": open_repair_contract.get("freeform_patch_text_target")
        is False,
        "sealed_feedback_not_used": open_repair_contract.get("sealed_feedback_used") is False,
        "manifest_blocks_package": manifest.get("package_allowed") is False,
        "manifest_blocks_sealed_eval": manifest.get("sealed_eval_allowed") is False,
    }
    passed = all(checks.values())
    patch_descriptor_evaluable = (
        isinstance(patch_operation_count, float)
        and patch_operation_count > 0
        and isinstance(patch_template_count, float)
        and patch_template_count > 0
    )
    patch_descriptor_meets_package_bar = (
        isinstance(patch_operation_accuracy, float)
        and isinstance(patch_template_accuracy, float)
        and patch_operation_accuracy >= min_patch_descriptor_accuracy_for_package
        and patch_template_accuracy >= min_patch_descriptor_accuracy_for_package
    )
    patch_descriptor_failure = patch_descriptor_evaluable and not patch_descriptor_meets_package_bar

    return {
        "artifact_family": ARTIFACT_FAMILY,
        "passed": passed,
        "ready_for_phase2ay_runtime_execution_eval": passed,
        "ready_for_phase2ax_package": False,
        "ready_for_package_or_execution_claim": False,
        "ready_for_sealed_eval": False,
        "ready_for_epoch_making_architecture_claim": False,
        "checks": checks,
        "metrics": {
            "train_examples": summary.get("train_examples"),
            "val_examples": summary.get("val_examples"),
            "manifest_train_rows": manifest.get("train_rows"),
            "manifest_val_rows": manifest.get("val_rows"),
            "val_command_slot_accuracy": val_accuracy,
            "val_command_slot_count": val_count,
            "current_only_accuracy": current_only_accuracy,
            "source_overlap_val_accuracy": source_overlap_accuracy,
            "wrong_cache_accuracy": wrong_cache_accuracy,
            "prior_runtime_resolver_accuracy": prior_runtime_accuracy,
            "model_minus_current_only": model_minus_current,
            "model_minus_source_overlap": model_minus_source,
            "model_minus_wrong_cache": model_minus_wrong_cache,
            "patch_operation_accuracy": patch_operation_accuracy,
            "patch_template_slot_accuracy": patch_template_accuracy,
            "patch_target_file_slot_accuracy": patch_target_file_accuracy,
            "patch_descriptor_evaluable": patch_descriptor_evaluable,
            "patch_descriptor_meets_package_bar": patch_descriptor_meets_package_bar,
            "patch_descriptor_failure": patch_descriptor_failure,
            "train_loss": latest.get("train_loss"),
            "val_loss": val_metrics.get("loss"),
            "train_elapsed_seconds": latest.get("train_elapsed_seconds"),
            "train_steps_per_second": latest.get("train_steps_per_second"),
            "config_hash": summary.get("config_hash"),
            "training_effective_split_hashes": summary.get("effective_split_hashes"),
            "manifest_effective_split_hashes": manifest.get("effective_split_hashes"),
            "repo_origin_overlap": manifest.get("repo_origins", {}).get("overlap"),
        },
        "claim_boundary": (
            "phase2ax_full_nonsealed_supports_counterfactual_command_slot_only_not_package_or_epoch_claim"
            if passed
            else "phase2ax_full_postflight_failed_or_incomplete_not_claim_evidence"
        ),
        "blocked_actions": [
            "do_not_package_phase2ax_before_runtime_execution_eval",
            "do_not_run_sealed_v3_from_phase2ax_full_postflight",
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ],
        "unsupported_claims": [
            "full_package_beats_native_head_only_on_phase2ax",
            "runtime_patch_execution_success",
            "sealed_cross_model_transfer",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "next_required_experiment": (
            "phase2ay_runtime_execution_eval"
            if passed
            else "repair_phase2ax_full_postflight_inputs_before_next_experiment"
        ),
        "patch_descriptor_status": (
            "weak_not_repair_execution_evidence"
            if patch_descriptor_failure
            else "recorded_but_still_requires_runtime_execution_eval"
        ),
        "thresholds": {
            "min_val_command_slot_accuracy": min_val_command_slot_accuracy,
            "min_val_command_slot_count": min_val_command_slot_count,
            "min_model_minus_current_only": min_model_minus_current_only,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_model_minus_wrong_cache": min_model_minus_wrong_cache,
            "min_patch_descriptor_accuracy_for_package": min_patch_descriptor_accuracy_for_package,
        },
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "full_manifest_json": str(Path(full_manifest_json)),
            "train_data_health_json": str(Path(train_data_health_json)),
            "val_data_health_json": str(Path(val_data_health_json)),
            "train_pretrain_gate_json": str(Path(train_pretrain_gate_json)),
            "val_pretrain_gate_json": str(Path(val_pretrain_gate_json)),
            "smoke_postflight_json": str(Path(smoke_postflight_json)),
        },
        "notes": [
            "This is full nonsealed evidence for counterfactual command-slot selection only.",
            "Patch descriptor metrics are recorded but do not prove patch execution or package readiness.",
            "Runtime execution, sealed transfer, production autonomy, and epoch-making architecture claims remain blocked.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AX full training postflight.")
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--full-manifest-json", required=True)
    parser.add_argument("--train-data-health-json", required=True)
    parser.add_argument("--val-data-health-json", required=True)
    parser.add_argument("--train-pretrain-gate-json", required=True)
    parser.add_argument("--val-pretrain-gate-json", required=True)
    parser.add_argument("--smoke-postflight-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-val-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-val-command-slot-count", type=int, default=128)
    parser.add_argument("--min-model-minus-current-only", type=float, default=0.25)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.25)
    parser.add_argument("--min-model-minus-wrong-cache", type=float, default=0.25)
    parser.add_argument("--min-patch-descriptor-accuracy-for-package", type=float, default=0.75)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2ax_full_postflight(
        training_summary_json=args.training_summary_json,
        full_manifest_json=args.full_manifest_json,
        train_data_health_json=args.train_data_health_json,
        val_data_health_json=args.val_data_health_json,
        train_pretrain_gate_json=args.train_pretrain_gate_json,
        val_pretrain_gate_json=args.val_pretrain_gate_json,
        smoke_postflight_json=args.smoke_postflight_json,
        min_val_command_slot_accuracy=args.min_val_command_slot_accuracy,
        min_val_command_slot_count=args.min_val_command_slot_count,
        min_model_minus_current_only=args.min_model_minus_current_only,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        min_model_minus_wrong_cache=args.min_model_minus_wrong_cache,
        min_patch_descriptor_accuracy_for_package=args.min_patch_descriptor_accuracy_for_package,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
