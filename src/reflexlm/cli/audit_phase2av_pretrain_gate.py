from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SPLITS = ("train", "val", "holdout")
DESCRIPTOR_ARTIFACT_FAMILY = "phase2at_learned_patch_candidate_split"
DESCRIPTOR_SCHEMA_VERSION = "phase2at.learned_bounded_patch_candidate.v1"
RUNTIME_BUILD_FAMILY = "phase2av_graded_descriptor_runtime_task_builder"
DATA_HEALTH_FAMILY = "phase2av_graded_descriptor_runtime_data_health"
READY_CLAIM_BOUNDARY = "phase2av_graded_descriptor_runtime_ready_for_pretrain_gate"


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _split_payload(paths: dict[str, str | Path], split: str) -> dict[str, Any]:
    path = paths.get(split)
    if path is None:
        return {}
    return _read_json(path)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def build_phase2av_pretrain_gate(
    *,
    descriptor_manifest_json: str | Path,
    runtime_build_jsons: dict[str, str | Path],
    data_health_jsons: dict[str, str | Path],
    min_train_rows: int = 1,
    min_val_rows: int = 1,
    min_holdout_rows: int = 1,
    min_operation_template_pairs: int = 2,
) -> dict[str, Any]:
    descriptor_manifest = _read_json(descriptor_manifest_json)
    split_counts = descriptor_manifest.get("split_counts")
    split_hashes = descriptor_manifest.get("split_hashes")
    if not isinstance(split_counts, dict):
        split_counts = {}
    if not isinstance(split_hashes, dict):
        split_hashes = {}

    runtime_builds = {
        split: _split_payload(runtime_build_jsons, split) for split in REQUIRED_SPLITS
    }
    data_health = {
        split: _split_payload(data_health_jsons, split) for split in REQUIRED_SPLITS
    }
    min_rows = {
        "train": min_train_rows,
        "val": min_val_rows,
        "holdout": min_holdout_rows,
    }

    checks: dict[str, bool] = {
        "descriptor_manifest_family_expected": descriptor_manifest.get("artifact_family")
        == DESCRIPTOR_ARTIFACT_FAMILY,
        "descriptor_schema_expected": descriptor_manifest.get("schema_version")
        == DESCRIPTOR_SCHEMA_VERSION,
        "descriptor_split_counts_present": all(
            isinstance(split_counts.get(split), int) for split in REQUIRED_SPLITS
        ),
        "descriptor_split_hashes_present": all(
            _is_sha256(split_hashes.get(split)) for split in REQUIRED_SPLITS
        ),
        "descriptor_no_freeform_or_recorded_targets": (
            descriptor_manifest.get("freeform_patch_generation") is False
            and descriptor_manifest.get("recorded_patch_artifact_as_generation_target")
            is False
            and descriptor_manifest.get("symbolic_generator_as_generation_target") is False
            and descriptor_manifest.get("sealed_feedback_used") is False
        ),
    }

    for split in REQUIRED_SPLITS:
        build = runtime_builds[split]
        health = data_health[split]
        metrics = health.get("metrics") if isinstance(health.get("metrics"), dict) else {}
        health_checks = (
            health.get("checks") if isinstance(health.get("checks"), dict) else {}
        )
        checks[f"{split}_runtime_build_passed"] = (
            build.get("artifact_family") == RUNTIME_BUILD_FAMILY
            and build.get("passed") is True
            and _is_sha256(build.get("task_split_sha256"))
        )
        checks[f"{split}_runtime_build_count_matches_descriptor"] = (
            isinstance(split_counts.get(split), int)
            and build.get("source_row_count") == split_counts.get(split)
            and build.get("converted_row_count") == split_counts.get(split)
        )
        checks[f"{split}_runtime_build_multi_template"] = (
            isinstance(build.get("operation_template_pair_count"), int)
            and build.get("operation_template_pair_count", 0)
            >= min_operation_template_pairs
        )
        checks[f"{split}_data_health_passed"] = (
            health.get("artifact_family") == DATA_HEALTH_FAMILY
            and health.get("passed") is True
            and health.get("claim_boundary") == READY_CLAIM_BOUNDARY
        )
        checks[f"{split}_data_health_min_rows"] = (
            isinstance(metrics.get("row_count"), int)
            and metrics.get("row_count", 0) >= min_rows[split]
        )
        checks[f"{split}_data_health_multi_template"] = (
            health_checks.get("operation_template_diversity_met") is True
            and isinstance(metrics.get("operation_template_pair_count"), int)
            and metrics.get("operation_template_pair_count", 0)
            >= min_operation_template_pairs
        )
        checks[f"{split}_data_health_non_parser_oracle"] = (
            health_checks.get("generated_tests_not_parser_oracle_solvable") is True
            and not metrics.get("parser_oracle_rows")
        )
        checks[f"{split}_data_health_no_leakage_or_markers"] = (
            health_checks.get("all_rows_public_repo") is True
            and health_checks.get("no_sealed_feedback") is True
            and health_checks.get("no_candidate_or_gold_markers") is True
            and health_checks.get("contract_blocks_forbidden_targets") is True
        )
        checks[f"{split}_data_health_no_blocked_actions"] = not health.get(
            "blocked_actions"
        )

    passed = all(checks.values())
    blocked_actions = []
    if not passed:
        blocked_actions.extend(
            [
                "do_not_train_phase2av",
                "do_not_package_phase2av",
                "do_not_run_sealed_eval_for_phase2av",
                "fix_phase2av_pretrain_gate_failures_first",
            ]
        )

    return {
        "artifact_family": "phase2av_graded_descriptor_runtime_pretrain_gate",
        "passed": passed,
        "ready_for_phase2av_smoke_training": passed,
        "ready_for_phase2av_full_training": False,
        "claim_boundary": (
            "Phase2AV may start smoke training only after descriptor split hashes, "
            "runtime task hashes, and train/validation/holdout data-health gates "
            "all pass. Passing this gate is not package, sealed-transfer, "
            "freeform-patch, open-ended-repair, production-autonomy, or "
            "epoch-making architecture evidence."
        ),
        "checks": checks,
        "supported_claims": [
            "phase2av_ready_for_small_scale_nonsealed_descriptor_runtime_smoke_training"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "learned_freeform_patch_generation",
            "learned_descriptor_runtime_delta",
            "sealed_transfer_for_phase2av",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": blocked_actions,
        "metrics": {
            "descriptor_split_counts": split_counts,
            "descriptor_split_hashes": split_hashes,
            "runtime_task_hashes": {
                split: runtime_builds[split].get("task_split_sha256")
                for split in REQUIRED_SPLITS
            },
            "data_health": {
                split: data_health[split].get("metrics") for split in REQUIRED_SPLITS
            },
            "thresholds": {
                "min_train_rows": min_train_rows,
                "min_val_rows": min_val_rows,
                "min_holdout_rows": min_holdout_rows,
                "min_operation_template_pairs": min_operation_template_pairs,
            },
        },
        "inputs": {
            "descriptor_manifest_json": str(Path(descriptor_manifest_json)),
            "runtime_build_jsons": {
                split: str(Path(runtime_build_jsons[split]))
                for split in REQUIRED_SPLITS
                if split in runtime_build_jsons
            },
            "data_health_jsons": {
                split: str(Path(data_health_jsons[split]))
                for split in REQUIRED_SPLITS
                if split in data_health_jsons
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AV descriptor-runtime pretrain readiness."
    )
    parser.add_argument("--descriptor-manifest-json", required=True)
    parser.add_argument("--train-runtime-build-json", required=True)
    parser.add_argument("--val-runtime-build-json", required=True)
    parser.add_argument("--holdout-runtime-build-json", required=True)
    parser.add_argument("--train-data-health-json", required=True)
    parser.add_argument("--val-data-health-json", required=True)
    parser.add_argument("--holdout-data-health-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-train-rows", type=int, default=1)
    parser.add_argument("--min-val-rows", type=int, default=1)
    parser.add_argument("--min-holdout-rows", type=int, default=1)
    parser.add_argument("--min-operation-template-pairs", type=int, default=2)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2av_pretrain_gate(
        descriptor_manifest_json=args.descriptor_manifest_json,
        runtime_build_jsons={
            "train": args.train_runtime_build_json,
            "val": args.val_runtime_build_json,
            "holdout": args.holdout_runtime_build_json,
        },
        data_health_jsons={
            "train": args.train_data_health_json,
            "val": args.val_data_health_json,
            "holdout": args.holdout_data_health_json,
        },
        min_train_rows=args.min_train_rows,
        min_val_rows=args.min_val_rows,
        min_holdout_rows=args.min_holdout_rows,
        min_operation_template_pairs=args.min_operation_template_pairs,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
