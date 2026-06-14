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


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _split_count(manifest: dict[str, Any], split: str) -> int:
    counts = _dict(manifest.get("split_counts"))
    value = counts.get(split)
    return int(value) if isinstance(value, int) else 0


def _operation_counts(manifest: dict[str, Any], split: str) -> dict[str, int]:
    operation_counts = _dict(_dict(manifest.get("operation_counts")).get(split))
    if operation_counts:
        return {
            str(key): int(value)
            for key, value in operation_counts.items()
            if isinstance(value, int)
        }
    report = _dict(_dict(manifest.get("split_reports")).get(split))
    counts = _dict(report.get("selected_operation_counts"))
    return {str(key): int(value) for key, value in counts.items() if isinstance(value, int)}


def _all_split_operations(manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {split: _operation_counts(manifest, split) for split in ("train", "val", "holdout")}


def _operation_diversity_ok(
    split_operations: dict[str, dict[str, int]],
    *,
    min_operations_per_split: int,
    min_examples_per_operation: int,
) -> bool:
    for counts in split_operations.values():
        if len(counts) < min_operations_per_split:
            return False
        if any(value < min_examples_per_operation for value in counts.values()):
            return False
    return True


def _postflight_metric(postflight: dict[str, Any], name: str) -> float | None:
    return _number(_dict(postflight.get("metrics")).get(name))


def _effective_train_examples(manifest: dict[str, Any], summary: dict[str, Any]) -> int:
    for key in ("train_examples", "effective_train_examples", "max_train_records"):
        value = summary.get(key)
        if isinstance(value, int):
            return value
    return _split_count(manifest, "train")


def _split_construction_ok(manifest: dict[str, Any]) -> bool:
    if manifest.get("operation_balanced") is True:
        return True
    return manifest.get("split_construction_family") == "phase2av_repo_operation_disjoint_split"


def _unsupported_claims_blocked(*reports: dict[str, Any]) -> bool:
    required = {
        "freeform_patch_generation",
        "sealed_cross_model_transfer",
        "open_ended_debugging_generalization",
        "production_autonomy",
        "epoch_making_architecture",
    }
    for report in reports:
        claims = report.get("unsupported_claims")
        if not isinstance(claims, list):
            return False
        if not required.issubset({str(claim) for claim in claims}):
            return False
    return True


def audit_phase2av_full_readiness(
    *,
    subset_manifest_json: str | Path,
    training_summary_json: str | Path,
    smoke_postflight_json: str | Path,
    holdout_postflight_json: str | Path,
    data_health_jsons: list[str | Path] | None = None,
    min_train_rows: int = 256,
    min_val_rows: int = 64,
    min_holdout_rows: int = 64,
    min_operations_per_split: int = 3,
    min_examples_per_operation: int = 5,
    min_command_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    min_descriptor_accuracy: float = 0.85,
    min_command_identity_logit_bias: float = 0.1,
) -> dict[str, Any]:
    manifest = _read_json(subset_manifest_json)
    summary = _read_json(training_summary_json)
    smoke_postflight = _read_json(smoke_postflight_json)
    holdout_postflight = _read_json(holdout_postflight_json)
    data_health_reports = [_read_json(path) for path in data_health_jsons or []]

    split_counts = {
        "train": _split_count(manifest, "train"),
        "val": _split_count(manifest, "val"),
        "holdout": _split_count(manifest, "holdout"),
    }
    effective_train_examples = _effective_train_examples(manifest, summary)
    split_operations = _all_split_operations(manifest)
    smoke_metrics = _dict(smoke_postflight.get("metrics"))
    holdout_metrics = _dict(holdout_postflight.get("metrics"))
    identity_bias = _number(summary.get("command_identity_logit_bias"))

    smoke_command = _postflight_metric(smoke_postflight, "command_slot_accuracy")
    holdout_command = _postflight_metric(holdout_postflight, "command_slot_accuracy")
    smoke_delta = _postflight_metric(smoke_postflight, "model_minus_source_overlap_accuracy")
    holdout_delta = _postflight_metric(holdout_postflight, "model_minus_source_overlap_accuracy")
    smoke_operation = _postflight_metric(smoke_postflight, "patch_operation_accuracy")
    holdout_operation = _postflight_metric(holdout_postflight, "patch_operation_accuracy")
    smoke_template = _postflight_metric(smoke_postflight, "patch_template_slot_accuracy")
    holdout_template = _postflight_metric(holdout_postflight, "patch_template_slot_accuracy")

    checks = {
        "subset_manifest_present": bool(manifest),
        "subset_manifest_passed": manifest.get("passed") is True,
        "subset_is_not_operation_balanced_shortcut": _split_construction_ok(manifest),
        "split_counts_sufficient": effective_train_examples >= min_train_rows
        and split_counts["val"] >= min_val_rows
        and split_counts["holdout"] >= min_holdout_rows,
        "operation_diversity_sufficient": _operation_diversity_ok(
            split_operations,
            min_operations_per_split=min_operations_per_split,
            min_examples_per_operation=min_examples_per_operation,
        ),
        "data_health_reports_present": bool(data_health_reports),
        "data_health_reports_passed": bool(data_health_reports)
        and all(report.get("passed") is True for report in data_health_reports),
        "training_summary_present": bool(summary),
        "open_repair_heads_enabled": summary.get("open_repair_heads_enabled") is True,
        "pairwise_disabled": summary.get("use_pairwise_command_reranker") is False,
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
        "command_identity_prior_recorded": isinstance(identity_bias, float)
        and identity_bias >= min_command_identity_logit_bias,
        "smoke_postflight_passed": smoke_postflight.get("passed") is True,
        "holdout_postflight_passed": holdout_postflight.get("passed") is True,
        "smoke_command_slot_gate": isinstance(smoke_command, float)
        and smoke_command >= min_command_slot_accuracy,
        "holdout_command_slot_gate": isinstance(holdout_command, float)
        and holdout_command >= min_command_slot_accuracy,
        "smoke_model_delta_gate": isinstance(smoke_delta, float)
        and smoke_delta >= min_model_minus_source_overlap,
        "holdout_model_delta_gate": isinstance(holdout_delta, float)
        and holdout_delta >= min_model_minus_source_overlap,
        "smoke_descriptor_gate": isinstance(smoke_operation, float)
        and smoke_operation >= min_descriptor_accuracy
        and isinstance(smoke_template, float)
        and smoke_template >= min_descriptor_accuracy,
        "holdout_descriptor_gate": isinstance(holdout_operation, float)
        and holdout_operation >= min_descriptor_accuracy
        and isinstance(holdout_template, float)
        and holdout_template >= min_descriptor_accuracy,
        "claim_boundary_preserved": _unsupported_claims_blocked(
            smoke_postflight, holdout_postflight
        ),
    }
    passed = all(checks.values())

    blocking_reasons = [name for name, ok in checks.items() if ok is False]
    return {
        "artifact_family": "phase2av_full_readiness",
        "passed": passed,
        "ready_to_start_phase2av_full_training": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "claim_boundary": (
            "This report can only authorize larger non-sealed Phase2AV training. "
            "It never authorizes packaging, sealed evaluation, freeform patch "
            "generation, production autonomy, open-ended debugging generalization, "
            "or epoch-making architecture claims."
        ),
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "metrics": {
            "split_counts": split_counts,
            "effective_train_examples": effective_train_examples,
            "split_construction_family": manifest.get("split_construction_family"),
            "split_operations": split_operations,
            "command_identity_logit_bias": identity_bias,
            "smoke": {
                "command_slot_accuracy": smoke_command,
                "model_minus_source_overlap_accuracy": smoke_delta,
                "patch_operation_accuracy": smoke_operation,
                "patch_template_slot_accuracy": smoke_template,
                "val_examples": smoke_metrics.get("val_examples"),
                "config_hash": smoke_metrics.get("config_hash"),
            },
            "holdout": {
                "command_slot_accuracy": holdout_command,
                "model_minus_source_overlap_accuracy": holdout_delta,
                "patch_operation_accuracy": holdout_operation,
                "patch_template_slot_accuracy": holdout_template,
                "eval_examples": holdout_metrics.get("eval_examples"),
                "eval_rows_hash": holdout_metrics.get("eval_rows_hash"),
            },
            "thresholds": {
                "min_train_rows": min_train_rows,
                "min_val_rows": min_val_rows,
                "min_holdout_rows": min_holdout_rows,
                "min_operations_per_split": min_operations_per_split,
                "min_examples_per_operation": min_examples_per_operation,
                "min_command_slot_accuracy": min_command_slot_accuracy,
                "min_model_minus_source_overlap": min_model_minus_source_overlap,
                "min_descriptor_accuracy": min_descriptor_accuracy,
                "min_command_identity_logit_bias": min_command_identity_logit_bias,
            },
        },
        "supported_claims": []
        if not passed
        else [
            "phase2av_larger_nonsealed_full_training_is_allowed_for_preregistered_descriptor_runtime_benchmark"
        ],
        "unsupported_claims": [
            "phase2av_package_ready",
            "sealed_cross_model_transfer",
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": [
            "do_not_package_phase2av",
            "do_not_run_sealed_eval_for_phase2av",
            "do_not_claim_freeform_patch_generation",
            "do_not_claim_production_autonomy",
            "do_not_claim_epoch_making_architecture",
        ]
        + ([] if passed else ["do_not_start_phase2av_full_training"]),
        "inputs": {
            "subset_manifest_json": str(Path(subset_manifest_json)),
            "training_summary_json": str(Path(training_summary_json)),
            "smoke_postflight_json": str(Path(smoke_postflight_json)),
            "holdout_postflight_json": str(Path(holdout_postflight_json)),
            "data_health_jsons": [str(Path(path)) for path in data_health_jsons or []],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AV full-training readiness.")
    parser.add_argument("--subset-manifest-json", required=True)
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--smoke-postflight-json", required=True)
    parser.add_argument("--holdout-postflight-json", required=True)
    parser.add_argument("--data-health-json", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-train-rows", type=int, default=256)
    parser.add_argument("--min-val-rows", type=int, default=64)
    parser.add_argument("--min-holdout-rows", type=int, default=64)
    parser.add_argument("--min-operations-per-split", type=int, default=3)
    parser.add_argument("--min-examples-per-operation", type=int, default=5)
    parser.add_argument("--min-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-descriptor-accuracy", type=float, default=0.85)
    parser.add_argument("--min-command-identity-logit-bias", type=float, default=0.1)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2av_full_readiness(
        subset_manifest_json=args.subset_manifest_json,
        training_summary_json=args.training_summary_json,
        smoke_postflight_json=args.smoke_postflight_json,
        holdout_postflight_json=args.holdout_postflight_json,
        data_health_jsons=args.data_health_json,
        min_train_rows=args.min_train_rows,
        min_val_rows=args.min_val_rows,
        min_holdout_rows=args.min_holdout_rows,
        min_operations_per_split=args.min_operations_per_split,
        min_examples_per_operation=args.min_examples_per_operation,
        min_command_slot_accuracy=args.min_command_slot_accuracy,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
        min_descriptor_accuracy=args.min_descriptor_accuracy,
        min_command_identity_logit_bias=args.min_command_identity_logit_bias,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
