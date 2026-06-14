from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


VOLATILE_CONFIG_KEYS = {
    "adapter_name",
    "checkpoint_dir",
    "config_hash",
    "device",
    "progress_log_interval_steps",
    "resume_from_checkpoint",
    "seed",
}

METRIC_KEYS = (
    "command_slot_accuracy",
    "source_overlap_accuracy",
    "model_minus_source_overlap_accuracy",
    "patch_operation_accuracy",
    "patch_target_file_slot_accuracy",
    "patch_template_slot_accuracy",
)


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


def _canonical_config(summary: dict[str, Any]) -> dict[str, Any]:
    config = _dict(summary.get("config"))
    return {key: value for key, value in config.items() if key not in VOLATILE_CONFIG_KEYS}


def _summary_path_from_postflight(path: str | Path, postflight: dict[str, Any]) -> Path:
    summary = _dict(postflight.get("inputs")).get("training_summary_json")
    if not isinstance(summary, str) or not summary:
        return Path("")
    summary_path = Path(summary)
    if summary_path.exists():
        return summary_path
    candidate = Path(path).parent / summary_path
    return candidate if candidate.exists() else summary_path


def _seed(summary: dict[str, Any]) -> int | None:
    seed = _dict(summary.get("config")).get("seed")
    return int(seed) if isinstance(seed, int) else None


def _aggregate(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "std": None}
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {"min": min(values), "mean": statistics.fmean(values), "std": std}


def build_phase2at_multiseed_smoke_report(
    *,
    postflight_jsons: list[str | Path],
    min_unique_seeds: int = 3,
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
    seeds = [_seed(summary) for summary in summaries]
    unique_seeds = sorted({seed for seed in seeds if seed is not None})
    failed_postflights = [
        str(path)
        for path, postflight in zip(postflight_jsons, postflights)
        if postflight.get("passed") is not True
    ]
    missing_summaries = [
        str(path)
        for path in summary_paths
        if not str(path) or not Path(path).exists()
    ]
    base_models = {summary.get("base_model_name") for summary in summaries if summary}
    train_counts = {summary.get("train_examples") for summary in summaries if summary}
    val_counts = {summary.get("val_examples") for summary in summaries if summary}
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

    metric_values: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    for postflight in postflights:
        metrics = _dict(postflight.get("metrics"))
        for key in METRIC_KEYS:
            value = _float(metrics.get(key))
            if value is not None:
                metric_values[key].append(value)

    metric_summary = {key: _aggregate(values) for key, values in metric_values.items()}
    threshold_checks = {
        "command_slot_min_met": (
            metric_summary["command_slot_accuracy"]["min"] is not None
            and metric_summary["command_slot_accuracy"]["min"] >= min_command_slot_accuracy
        ),
        "model_minus_source_overlap_min_met": (
            metric_summary["model_minus_source_overlap_accuracy"]["min"] is not None
            and metric_summary["model_minus_source_overlap_accuracy"]["min"]
            >= min_model_minus_source_overlap
        ),
        "patch_operation_min_met": (
            metric_summary["patch_operation_accuracy"]["min"] is not None
            and metric_summary["patch_operation_accuracy"]["min"] >= min_descriptor_accuracy
        ),
        "patch_target_file_slot_min_met": (
            metric_summary["patch_target_file_slot_accuracy"]["min"] is not None
            and metric_summary["patch_target_file_slot_accuracy"]["min"] >= min_descriptor_accuracy
        ),
        "patch_template_slot_min_met": (
            metric_summary["patch_template_slot_accuracy"]["min"] is not None
            and metric_summary["patch_template_slot_accuracy"]["min"] >= min_descriptor_accuracy
        ),
    }
    checks = {
        "postflight_count_nonzero": bool(postflights),
        "all_postflights_passed": not failed_postflights,
        "summaries_present": not missing_summaries and len(summaries) == len(postflights),
        "unique_seed_minimum_met": len(unique_seeds) >= min_unique_seeds,
        "one_summary_per_unique_seed": len(unique_seeds) == len(summaries),
        "base_model_consistent": len(base_models) == 1 and None not in base_models,
        "train_count_consistent": len(train_counts) == 1 and None not in train_counts,
        "val_count_consistent": len(val_counts) == 1 and None not in val_counts,
        "split_hashes_consistent": bool(split_hashes)
        and all(split == split_hashes[0] for split in split_hashes),
        "training_contract_consistent_except_seed_and_names": bool(canonical_configs)
        and all(config == canonical_configs[0] for config in canonical_configs),
        "descriptor_heads_consistent": bool(descriptor_heads)
        and all(head == descriptor_heads[0] for head in descriptor_heads)
        and _dict(descriptor_heads[0]).get("enabled") is True,
        **threshold_checks,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2at_multiseed_learned_descriptor_smoke",
        "passed": passed,
        "claim_boundary": (
            "Same-model, same-split, non-sealed three-seed smoke for bounded patch "
            "descriptor heads only. This does not prove freeform patch generation, "
            "sealed transfer, production autonomy, open-ended debugging, or an "
            "epoch-making architecture."
        ),
        "checks": checks,
        "metrics": {
            "unique_seeds": unique_seeds,
            "unique_seed_count": len(unique_seeds),
            "base_models": sorted(str(value) for value in base_models),
            "train_examples": sorted(value for value in train_counts if value is not None),
            "val_examples": sorted(value for value in val_counts if value is not None),
            "metric_summary": metric_summary,
            "failed_postflights": failed_postflights,
            "missing_summaries": missing_summaries,
        },
        "thresholds": {
            "min_unique_seeds": min_unique_seeds,
            "min_command_slot_accuracy": min_command_slot_accuracy,
            "min_model_minus_source_overlap": min_model_minus_source_overlap,
            "min_descriptor_accuracy": min_descriptor_accuracy,
        },
        "supported_claims": [
            "phase2at_same_model_three_seed_nonsealed_descriptor_smoke_supported"
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
            "do_not_package_as_learned_patch_generation",
            "do_not_claim_freeform_patch_generation_from_descriptor_smoke",
            "do_not_claim_epoch_making_architecture_from_phase2at_smoke",
        ],
        "inputs": {
            "postflight_jsons": [str(Path(path)) for path in postflight_jsons],
            "training_summary_jsons": [str(path) for path in summary_paths],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AT multi-seed descriptor smoke.")
    parser.add_argument("--postflight-json", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-unique-seeds", type=int, default=3)
    parser.add_argument("--min-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.15)
    parser.add_argument("--min-descriptor-accuracy", type=float, default=0.85)
    args = parser.parse_args()
    report = build_phase2at_multiseed_smoke_report(
        postflight_jsons=args.postflight_json,
        min_unique_seeds=args.min_unique_seeds,
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
