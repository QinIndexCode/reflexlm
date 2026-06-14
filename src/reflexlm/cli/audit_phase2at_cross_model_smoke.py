from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2at_multiseed_smoke import METRIC_KEYS


VOLATILE_CONFIG_KEYS = {
    "adapter_name",
    "base_model_name",
    "checkpoint_dir",
    "config_hash",
    "device",
    "progress_log_interval_steps",
    "resume_from_checkpoint",
}


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _summary_path_from_postflight(path: str | Path, postflight: dict[str, Any]) -> Path:
    summary = _dict(postflight.get("inputs")).get("training_summary_json")
    if not isinstance(summary, str) or not summary:
        return Path("")
    summary_path = Path(summary)
    if summary_path.exists():
        return summary_path
    candidate = Path(path).parent / summary_path
    return candidate if candidate.exists() else summary_path


def _canonical_config(summary: dict[str, Any]) -> dict[str, Any]:
    config = _dict(summary.get("config"))
    return {key: value for key, value in config.items() if key not in VOLATILE_CONFIG_KEYS}


def build_phase2at_cross_model_smoke_report(
    *,
    postflight_jsons: list[str | Path],
    min_model_count: int = 2,
    min_command_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.15,
    min_descriptor_accuracy: float = 0.85,
) -> dict[str, Any]:
    postflights = [_read_json(path) for path in postflight_jsons]
    summary_paths = [
        _summary_path_from_postflight(path, postflight)
        for path, postflight in zip(postflight_jsons, postflights)
    ]
    summaries = [
        _read_json(path) if str(path) and Path(path).exists() else {}
        for path in summary_paths
    ]
    missing_summaries = [
        str(path)
        for path in summary_paths
        if not str(path) or not Path(path).exists()
    ]
    failed_postflights = [
        str(path)
        for path, postflight in zip(postflight_jsons, postflights)
        if postflight.get("passed") is not True
    ]
    models = sorted(
        {
            str(summary.get("base_model_name"))
            for summary in summaries
            if summary.get("base_model_name")
        }
    )
    split_hashes = [
        _dict(summary.get("effective_split_hashes"))
        for summary in summaries
        if summary
    ]
    canonical_configs = [_canonical_config(summary) for summary in summaries if summary]
    descriptor_heads = [
        _dict(summary.get("learned_patch_descriptor_heads"))
        for summary in summaries
        if summary
    ]
    train_counts = {summary.get("train_examples") for summary in summaries if summary}
    val_counts = {summary.get("val_examples") for summary in summaries if summary}

    min_metrics: dict[str, float | None] = {}
    per_model_metrics: list[dict[str, Any]] = []
    for summary, postflight in zip(summaries, postflights):
        metrics = _dict(postflight.get("metrics"))
        values = {key: _float(metrics.get(key)) for key in METRIC_KEYS}
        per_model_metrics.append({"base_model_name": summary.get("base_model_name"), **values})
    for key in METRIC_KEYS:
        values = [
            _float(_dict(postflight.get("metrics")).get(key))
            for postflight in postflights
        ]
        numeric_values = [value for value in values if value is not None]
        min_metrics[key] = min(numeric_values) if numeric_values else None

    threshold_checks = {
        "command_slot_min_met": min_metrics["command_slot_accuracy"] is not None
        and min_metrics["command_slot_accuracy"] >= min_command_slot_accuracy,
        "model_minus_source_overlap_min_met": min_metrics[
            "model_minus_source_overlap_accuracy"
        ]
        is not None
        and min_metrics["model_minus_source_overlap_accuracy"]
        >= min_model_minus_source_overlap,
        "patch_operation_min_met": min_metrics["patch_operation_accuracy"] is not None
        and min_metrics["patch_operation_accuracy"] >= min_descriptor_accuracy,
        "patch_target_file_slot_min_met": min_metrics["patch_target_file_slot_accuracy"]
        is not None
        and min_metrics["patch_target_file_slot_accuracy"] >= min_descriptor_accuracy,
        "patch_template_slot_min_met": min_metrics["patch_template_slot_accuracy"] is not None
        and min_metrics["patch_template_slot_accuracy"] >= min_descriptor_accuracy,
    }
    checks = {
        "postflight_count_nonzero": bool(postflights),
        "all_postflights_passed": not failed_postflights,
        "summaries_present": not missing_summaries and len(summaries) == len(postflights),
        "model_count_minimum_met": len(models) >= min_model_count,
        "one_summary_per_model": len(models) == len(summaries),
        "split_hashes_consistent": bool(split_hashes)
        and all(split == split_hashes[0] for split in split_hashes),
        "training_contract_consistent_except_model_and_names": bool(canonical_configs)
        and all(config == canonical_configs[0] for config in canonical_configs),
        "descriptor_heads_consistent": bool(descriptor_heads)
        and all(head == descriptor_heads[0] for head in descriptor_heads)
        and _dict(descriptor_heads[0]).get("enabled") is True,
        "train_count_consistent": len(train_counts) == 1 and None not in train_counts,
        "val_count_consistent": len(val_counts) == 1 and None not in val_counts,
        **threshold_checks,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2at_cross_model_descriptor_smoke",
        "passed": passed,
        "claim_boundary": (
            "Initial cross-model, same-split, non-sealed smoke for bounded descriptor "
            "heads only. It does not prove sealed cross-model transfer, freeform patch "
            "generation, production autonomy, open-ended debugging, or an epoch-making "
            "architecture."
        ),
        "checks": checks,
        "metrics": {
            "models": models,
            "model_count": len(models),
            "min_metrics": min_metrics,
            "per_model_metrics": per_model_metrics,
            "failed_postflights": failed_postflights,
            "missing_summaries": missing_summaries,
            "train_examples": sorted(value for value in train_counts if value is not None),
            "val_examples": sorted(value for value in val_counts if value is not None),
        },
        "thresholds": {
            "min_model_count": min_model_count,
            "min_command_slot_accuracy": min_command_slot_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_descriptor_accuracy": min_descriptor_accuracy,
        },
        "supported_claims": [
            "phase2at_initial_same_split_cross_model_descriptor_smoke_supported"
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
        "blocked_actions": [
            "do_not_claim_sealed_cross_model_transfer_from_phase2at_smoke",
            "do_not_package_phase2at_as_learned_patch_generation",
            "do_not_claim_epoch_making_architecture_from_phase2at_cross_model_smoke",
        ],
        "inputs": {
            "postflight_jsons": [str(Path(path)) for path in postflight_jsons],
            "training_summary_jsons": [str(path) for path in summary_paths],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AT cross-model descriptor smoke.")
    parser.add_argument("--postflight-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-model-count", type=int, default=2)
    parser.add_argument("--min-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-descriptor-accuracy", type=float, default=0.85)
    args = parser.parse_args()
    report = build_phase2at_cross_model_smoke_report(
        postflight_jsons=args.postflight_json,
        min_model_count=args.min_model_count,
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
