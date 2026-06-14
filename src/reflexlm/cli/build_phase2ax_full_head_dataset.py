from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2ax_package_loaded_counterfactual_repair import (
    audit_phase2ax_package_loaded_counterfactual_repair,
)
from reflexlm.cli.audit_phase2ax_runtime_pretrain_gate import (
    build_phase2ax_runtime_pretrain_gate,
)
from reflexlm.cli.build_phase2ax_head_dataset import (
    CLAIM_BOUNDARY,
    DATASET_FAMILY,
    _head_row,
    _read_json,
    _read_jsonl,
    _sha256_rows,
    _slot_counts,
    _write_json,
    _write_jsonl,
)
from reflexlm.cli.build_phase2ax_package_loaded_counterfactual_repair import (
    build_phase2ax_package_loaded_counterfactual_repair,
)


FULL_DATASET_FAMILY = "phase2ax_package_loaded_counterfactual_repair_full_head_dataset"
FULL_CLAIM_BOUNDARY = "phase2ax_full_nonsealed_training_rows_not_package_or_claim_evidence"


def _repo_origins(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("repo_origin") or "") for row in rows if str(row.get("repo_origin") or "")}


def _build_task_split(
    *,
    source_jsonl: str | Path,
    tasks_jsonl: str | Path,
    metadata_json: str | Path,
    report_json: str | Path,
    max_pairs: int,
    min_pairs: int,
) -> dict[str, Any]:
    report = build_phase2ax_package_loaded_counterfactual_repair(
        source_tasks_jsonl=source_jsonl,
        output_jsonl=tasks_jsonl,
        metadata_json=metadata_json,
        max_pairs=max_pairs,
        min_pairs=min_pairs,
    )
    _write_json(report_json, report)
    return report


def _audit_task_split(
    *,
    tasks_jsonl: str | Path,
    metadata_json: str | Path,
    data_health_json: str | Path,
    pretrain_gate_json: str | Path,
    min_pairs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_health = audit_phase2ax_package_loaded_counterfactual_repair(
        tasks_jsonl=tasks_jsonl,
        metadata_json=metadata_json,
        min_pairs=min_pairs,
    )
    _write_json(data_health_json, data_health)
    pretrain = build_phase2ax_runtime_pretrain_gate(
        tasks_jsonl=tasks_jsonl,
        metadata_json=metadata_json,
        data_health_json=data_health_json,
    )
    _write_json(pretrain_gate_json, pretrain)
    return data_health, pretrain


def build_phase2ax_full_head_dataset(
    *,
    train_source_jsonl: str | Path,
    val_source_jsonl: str | Path,
    output_dir: str | Path,
    report_dir: str | Path,
    manifest_json: str | Path,
    smoke_postflight_json: str | Path,
    max_train_pairs: int = 64,
    max_val_pairs: int = 64,
    min_train_pairs: int = 16,
    min_val_pairs: int = 16,
) -> dict[str, Any]:
    output = Path(output_dir)
    reports = Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    train_tasks_jsonl = output / "train_tasks.jsonl"
    train_metadata_json = output / "train_metadata.json"
    val_tasks_jsonl = output / "val_tasks.jsonl"
    val_metadata_json = output / "val_metadata.json"
    train_build = _build_task_split(
        source_jsonl=train_source_jsonl,
        tasks_jsonl=train_tasks_jsonl,
        metadata_json=train_metadata_json,
        report_json=reports / "phase2ax_full_train_builder_report.json",
        max_pairs=max_train_pairs,
        min_pairs=min_train_pairs,
    )
    val_build = _build_task_split(
        source_jsonl=val_source_jsonl,
        tasks_jsonl=val_tasks_jsonl,
        metadata_json=val_metadata_json,
        report_json=reports / "phase2ax_full_val_builder_report.json",
        max_pairs=max_val_pairs,
        min_pairs=min_val_pairs,
    )
    train_data_health, train_pretrain = _audit_task_split(
        tasks_jsonl=train_tasks_jsonl,
        metadata_json=train_metadata_json,
        data_health_json=reports / "phase2ax_full_train_data_health.json",
        pretrain_gate_json=reports / "phase2ax_full_train_pretrain_gate.json",
        min_pairs=min_train_pairs,
    )
    val_data_health, val_pretrain = _audit_task_split(
        tasks_jsonl=val_tasks_jsonl,
        metadata_json=val_metadata_json,
        data_health_json=reports / "phase2ax_full_val_data_health.json",
        pretrain_gate_json=reports / "phase2ax_full_val_pretrain_gate.json",
        min_pairs=min_val_pairs,
    )
    smoke_postflight = _read_json(smoke_postflight_json)
    train_tasks = _read_jsonl(train_tasks_jsonl)
    val_tasks = _read_jsonl(val_tasks_jsonl)
    train_origins = _repo_origins(train_tasks)
    val_origins = _repo_origins(val_tasks)
    train_rows = [_head_row(row, split="train", index=index) for index, row in enumerate(train_tasks)]
    val_rows = [_head_row(row, split="val", index=index) for index, row in enumerate(val_tasks)]
    _write_jsonl(output / "train.jsonl", train_rows)
    _write_jsonl(output / "val.jsonl", val_rows)
    repo_origin_disjoint = not (train_origins & val_origins)
    checks = {
        "smoke_postflight_passed": smoke_postflight.get("passed") is True,
        "smoke_allows_full_nonsealed_training": smoke_postflight.get(
            "ready_for_phase2ax_full_nonsealed_training"
        )
        is True,
        "train_builder_passed": train_build.get("passed") is True,
        "val_builder_passed": val_build.get("passed") is True,
        "train_data_health_passed": train_data_health.get("passed") is True,
        "val_data_health_passed": val_data_health.get("passed") is True,
        "train_pretrain_gate_passed": train_pretrain.get("passed") is True,
        "val_pretrain_gate_passed": val_pretrain.get("passed") is True,
        "repo_origin_disjoint": repo_origin_disjoint,
        "train_rows_present": bool(train_rows),
        "val_rows_present": bool(val_rows),
    }
    passed = all(checks.values())
    manifest = {
        "dataset_family": FULL_DATASET_FAMILY,
        "source_dataset_family": DATASET_FAMILY,
        "passed": passed,
        "claim_boundary": FULL_CLAIM_BOUNDARY,
        "source_claim_boundary": CLAIM_BOUNDARY,
        "checks": checks,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_pairs": train_build.get("pair_count"),
        "val_pairs": val_build.get("pair_count"),
        "repo_origins": {
            "train": sorted(train_origins),
            "val": sorted(val_origins),
            "overlap": sorted(train_origins & val_origins),
        },
        "command_slot_distribution": {
            "train": _slot_counts(train_rows),
            "val": _slot_counts(val_rows),
        },
        "effective_split_hashes": {
            "phase2ax_full_head_train": _sha256_rows(train_rows),
            "phase2ax_full_head_val": _sha256_rows(val_rows),
        },
        "source_task_hashes": {
            "phase2ax_full_train_tasks": _sha256_rows(train_tasks),
            "phase2ax_full_val_tasks": _sha256_rows(val_tasks),
        },
        "full_training_allowed": passed,
        "package_allowed": False,
        "sealed_eval_allowed": False,
        "unsupported_claims": [
            "phase2ax_package_or_execution_claim_before_full_postflight",
            "freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "outputs": {
            "train_jsonl": str(output / "train.jsonl"),
            "val_jsonl": str(output / "val.jsonl"),
            "train_tasks_jsonl": str(train_tasks_jsonl),
            "val_tasks_jsonl": str(val_tasks_jsonl),
        },
        "inputs": {
            "train_source_jsonl": str(Path(train_source_jsonl)),
            "val_source_jsonl": str(Path(val_source_jsonl)),
            "smoke_postflight_json": str(Path(smoke_postflight_json)),
        },
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AX full train/val head rows from repo-disjoint source splits."
    )
    parser.add_argument("--train-source-jsonl", required=True)
    parser.add_argument("--val-source-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--smoke-postflight-json", required=True)
    parser.add_argument("--max-train-pairs", type=int, default=64)
    parser.add_argument("--max-val-pairs", type=int, default=64)
    parser.add_argument("--min-train-pairs", type=int, default=16)
    parser.add_argument("--min-val-pairs", type=int, default=16)
    args = parser.parse_args()
    manifest = build_phase2ax_full_head_dataset(
        train_source_jsonl=args.train_source_jsonl,
        val_source_jsonl=args.val_source_jsonl,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        manifest_json=args.manifest_json,
        smoke_postflight_json=args.smoke_postflight_json,
        max_train_pairs=args.max_train_pairs,
        max_val_pairs=args.max_val_pairs,
        min_train_pairs=args.min_train_pairs,
        min_val_pairs=args.min_val_pairs,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
