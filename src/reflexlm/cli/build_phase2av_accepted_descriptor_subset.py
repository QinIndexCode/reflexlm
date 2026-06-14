from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2at_learned_patch_candidate_data import (
    CLAIM_BOUNDARY,
    SCHEMA_VERSION,
)


REQUIRED_SPLITS = ("train", "val", "holdout")
ARTIFACT_FAMILY = "phase2at_learned_patch_candidate_split"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _source_task_id(task: dict[str, Any]) -> str:
    source = task.get("source") if isinstance(task.get("source"), dict) else {}
    return str(source.get("source_task_id") or "")


def _trace_id(row: dict[str, Any]) -> str:
    return str(row.get("trace_id") or "")


def _operation(row: dict[str, Any]) -> str:
    target = (
        row.get("learned_patch_candidate_target")
        if isinstance(row.get("learned_patch_candidate_target"), dict)
        else {}
    )
    return str(target.get("operation") or "<unknown>")


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("artifact_family") != ARTIFACT_FAMILY:
        raise ValueError("descriptor manifest artifact_family is not Phase2AT")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("descriptor manifest schema_version is not Phase2AT v1")
    if manifest.get("sealed_feedback_used") is not False:
        raise ValueError("descriptor manifest must not use sealed feedback")


def _filter_split(
    *,
    descriptor_rows: list[dict[str, Any]],
    runtime_tasks: list[dict[str, Any]],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted_ids = [_source_task_id(task) for task in runtime_tasks]
    if any(not item for item in accepted_ids):
        raise ValueError(f"{split} runtime task without source.source_task_id")
    if len(set(accepted_ids)) != len(accepted_ids):
        raise ValueError(f"{split} runtime tasks contain duplicate source_task_id")

    rows_by_trace = {_trace_id(row): row for row in descriptor_rows}
    missing_ids = [item for item in accepted_ids if item not in rows_by_trace]
    if missing_ids:
        raise ValueError(
            f"{split} accepted runtime task ids missing from descriptor rows: "
            f"{missing_ids[:5]}"
        )

    accepted_rows = [rows_by_trace[item] for item in accepted_ids]
    rejected_ids = sorted(set(rows_by_trace) - set(accepted_ids))
    return accepted_rows, {
        "source_row_count": len(descriptor_rows),
        "accepted_row_count": len(accepted_rows),
        "rejected_row_count": len(rejected_ids),
        "rejected_trace_ids": rejected_ids,
    }


def build_phase2av_accepted_descriptor_subset(
    *,
    descriptor_manifest_json: str | Path,
    train_descriptor_jsonl: str | Path,
    val_descriptor_jsonl: str | Path,
    holdout_descriptor_jsonl: str | Path,
    train_runtime_tasks_jsonl: str | Path,
    val_runtime_tasks_jsonl: str | Path,
    holdout_runtime_tasks_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
) -> dict[str, Any]:
    source_manifest = _read_json(descriptor_manifest_json)
    _validate_manifest(source_manifest)
    descriptor_paths = {
        "train": Path(train_descriptor_jsonl),
        "val": Path(val_descriptor_jsonl),
        "holdout": Path(holdout_descriptor_jsonl),
    }
    runtime_paths = {
        "train": Path(train_runtime_tasks_jsonl),
        "val": Path(val_runtime_tasks_jsonl),
        "holdout": Path(holdout_runtime_tasks_jsonl),
    }
    output = Path(output_dir)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    accepted_subset: dict[str, Any] = {}
    operation_counts: dict[str, dict[str, int]] = {}
    for split in REQUIRED_SPLITS:
        rows, summary = _filter_split(
            descriptor_rows=_read_jsonl(descriptor_paths[split]),
            runtime_tasks=_read_jsonl(runtime_paths[split]),
            split=split,
        )
        split_rows[split] = rows
        accepted_subset[split] = summary
        operation_counts[split] = dict(
            sorted(Counter(_operation(row) for row in rows).items())
        )
        _write_jsonl(output / f"{split}.jsonl", rows)

    manifest = {
        "artifact_family": ARTIFACT_FAMILY,
        "claim_boundary": CLAIM_BOUNDARY,
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output),
        "source_manifest_json": str(Path(descriptor_manifest_json)),
        "source_split_inputs": {
            split: str(descriptor_paths[split]) for split in REQUIRED_SPLITS
        },
        "runtime_accepted_task_inputs": {
            split: str(runtime_paths[split]) for split in REQUIRED_SPLITS
        },
        "accepted_subset_policy": (
            "filter_descriptor_rows_to_runtime_task_source_task_ids_after_quality_rejects"
        ),
        "accepted_subset": accepted_subset,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_hashes": {
            split: _sha256_text(_canonical_json(rows))
            for split, rows in split_rows.items()
        },
        "operation_counts": operation_counts,
        "freeform_patch_generation": False,
        "recorded_patch_artifact_as_generation_target": False,
        "symbolic_generator_as_generation_target": False,
        "sealed_feedback_used": False,
        "next_gate": "phase2av_graded_descriptor_runtime_pretrain_gate",
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Phase2AT descriptor subset aligned to accepted Phase2AV runtime tasks."
    )
    parser.add_argument("--descriptor-manifest-json", required=True)
    parser.add_argument("--train-descriptor-jsonl", required=True)
    parser.add_argument("--val-descriptor-jsonl", required=True)
    parser.add_argument("--holdout-descriptor-jsonl", required=True)
    parser.add_argument("--train-runtime-tasks-jsonl", required=True)
    parser.add_argument("--val-runtime-tasks-jsonl", required=True)
    parser.add_argument("--holdout-runtime-tasks-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    args = parser.parse_args()
    report = build_phase2av_accepted_descriptor_subset(
        descriptor_manifest_json=args.descriptor_manifest_json,
        train_descriptor_jsonl=args.train_descriptor_jsonl,
        val_descriptor_jsonl=args.val_descriptor_jsonl,
        holdout_descriptor_jsonl=args.holdout_descriptor_jsonl,
        train_runtime_tasks_jsonl=args.train_runtime_tasks_jsonl,
        val_runtime_tasks_jsonl=args.val_runtime_tasks_jsonl,
        holdout_runtime_tasks_jsonl=args.holdout_runtime_tasks_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
