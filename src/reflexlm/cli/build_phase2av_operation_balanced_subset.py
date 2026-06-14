from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2av_descriptor_runtime_head_dataset import (
    _descriptor_failure_family,
    _pytest_text,
    _runtime_visible_evidence,
    _traceback_symbols,
)


KNOWN_DESCRIPTOR_FAILURE_FAMILIES = {
    "attribute_missing_runtime",
    "missing_import_or_symbol_runtime",
    "syntax_load_failure_runtime",
    "assertion_behavior_mismatch_runtime",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256_rows(rows: list[dict[str, Any]]) -> str:
    text = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _generated_test_paths(row: dict[str, Any]) -> list[str]:
    artifacts = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    values: list[Any] = []
    for key in ("generated_test", "generated_tests", "test_artifacts"):
        value = artifacts.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value:
            values.append(value)
    return [str(value).replace("\\", "/") for value in values if str(value).strip()]


def _copy_referenced_tests(rows: list[dict[str, Any]], *, source_root: Path, output_root: Path) -> int:
    copied = 0
    seen: set[str] = set()
    for row in rows:
        for rel in _generated_test_paths(row):
            if rel in seen:
                continue
            seen.add(rel)
            source = source_root / rel
            if not source.exists():
                continue
            destination = output_root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied += 1
    return copied


def _operation(row: dict[str, Any]) -> str:
    policy = row.get("expected_policy") if isinstance(row.get("expected_policy"), dict) else {}
    value = policy.get("patch_operation")
    if not isinstance(value, str) or not value:
        descriptor = row.get("learned_patch_descriptor_target")
        if isinstance(descriptor, dict):
            value = descriptor.get("operation")
    if not isinstance(value, str) or not value:
        raise ValueError("row is missing expected patch operation")
    return value


def _repo_origin(row: dict[str, Any]) -> str:
    return str(row.get("repo_origin") or row.get("repo_url_or_origin") or row.get("repo_id") or "")


def _failure_family(row: dict[str, Any]) -> str:
    evidence = _runtime_visible_evidence(row)
    text = _pytest_text(evidence)
    return _descriptor_failure_family(_traceback_symbols(text), text)


def _family_matches_operation(family: str, operation: str) -> bool:
    if family == "attribute_missing_runtime":
        return operation == "replace_attribute"
    if family in {"missing_import_or_symbol_runtime", "syntax_load_failure_runtime"}:
        return operation == "insert_import"
    return family in KNOWN_DESCRIPTOR_FAILURE_FAMILIES


def _eligible_rows(
    rows: list[dict[str, Any]],
    *,
    require_known_failure_family: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not require_known_failure_family:
        return rows, {"excluded_unknown_failure_family": 0, "excluded_family_operation_mismatch": 0}
    selected: list[dict[str, Any]] = []
    excluded_unknown = 0
    excluded_mismatch = 0
    for row in rows:
        operation = _operation(row)
        family = _failure_family(row)
        if family not in KNOWN_DESCRIPTOR_FAILURE_FAMILIES:
            excluded_unknown += 1
            continue
        if not _family_matches_operation(family, operation):
            excluded_mismatch += 1
            continue
        selected.append(row)
    return selected, {
        "excluded_unknown_failure_family": excluded_unknown,
        "excluded_family_operation_mismatch": excluded_mismatch,
        "eligible_failure_family_counts": dict(Counter(_failure_family(row) for row in selected)),
    }


def _balanced_rows(
    rows: list[dict[str, Any]],
    *,
    max_per_operation: int | None,
    require_known_failure_family: bool,
    keep_all_eligible: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, eligibility_report = _eligible_rows(
        rows,
        require_known_failure_family=require_known_failure_family,
    )
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_operation(row)].append(row)
    if len(buckets) < 2:
        raise ValueError("operation-balanced subset requires at least two operations")
    operation_min = min(len(bucket) for bucket in buckets.values())
    limit = operation_min
    if max_per_operation is not None:
        limit = min(limit, max_per_operation)
    selected: list[dict[str, Any]] = []
    if keep_all_eligible:
        selected = list(rows)
    else:
        for operation in sorted(buckets):
            selected.extend(buckets[operation][:limit])
    selected.sort(key=lambda row: str(row.get("task_id") or row.get("source", {}).get("source_task_id") or ""))
    return selected, {
        "source_operation_counts": dict(Counter(_operation(row) for row in rows)),
        "selected_operation_counts": dict(Counter(_operation(row) for row in selected)),
        "selected_failure_family_counts": dict(Counter(_failure_family(row) for row in selected)),
        "selected_repo_origin_count": len({_repo_origin(row) for row in selected}),
        "operation_limit": None if keep_all_eligible else limit,
        "operation_minimum_count": operation_min,
        "keep_all_eligible": keep_all_eligible,
        "require_known_failure_family": require_known_failure_family,
        **eligibility_report,
    }


def build_phase2av_operation_balanced_subset(
    *,
    train_tasks_jsonl: str | Path,
    val_tasks_jsonl: str | Path,
    holdout_tasks_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
    max_per_operation: int | None = None,
    require_known_failure_family: bool = False,
    keep_all_eligible: bool = False,
) -> dict[str, Any]:
    inputs = {
        "train": Path(train_tasks_jsonl),
        "val": Path(val_tasks_jsonl),
        "holdout": Path(holdout_tasks_jsonl),
    }
    output = Path(output_dir)
    split_reports: dict[str, Any] = {}
    split_hashes: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    for split, path in inputs.items():
        rows = _read_jsonl(path)
        selected, report = _balanced_rows(
            rows,
            max_per_operation=max_per_operation,
            require_known_failure_family=require_known_failure_family,
            keep_all_eligible=keep_all_eligible,
        )
        out_path = output / f"{split}.jsonl"
        _write_jsonl(out_path, selected)
        report["copied_generated_tests"] = _copy_referenced_tests(
            selected,
            source_root=path.parent,
            output_root=output,
        )
        split_reports[split] = report
        split_hashes[split] = _sha256_rows(selected)
        split_counts[split] = len(selected)
    passed = all(len(report.get("selected_operation_counts", {})) >= 2 for report in split_reports.values())
    operation_balanced = all(
        len(set(report.get("selected_operation_counts", {}).values())) == 1
        for report in split_reports.values()
    )
    manifest = {
        "artifact_family": "phase2av_operation_balanced_runtime_subset",
        "passed": passed,
        "claim_boundary": "operation-balanced subset for non-sealed smoke diagnostics only; not full/package/sealed evidence",
        "output_dir": str(output),
        "split_counts": split_counts,
        "split_hashes": split_hashes,
        "split_reports": split_reports,
        "max_per_operation": max_per_operation,
        "require_known_failure_family": require_known_failure_family,
        "keep_all_eligible": keep_all_eligible,
        "operation_balanced": operation_balanced,
        "smoke_training_allowed": passed,
        "full_training_allowed": False,
        "package_allowed": False,
        "sealed_eval_allowed": False,
        "unsupported_claims": [
            "full_training_ready",
            "sealed_transfer",
            "freeform_patch_generation",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "inputs": {split: str(path) for split, path in inputs.items()},
    }
    _write_json(Path(manifest_json), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AV operation-balanced runtime subsets.")
    parser.add_argument("--train-tasks-jsonl", required=True)
    parser.add_argument("--val-tasks-jsonl", required=True)
    parser.add_argument("--holdout-tasks-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--max-per-operation", type=int)
    parser.add_argument("--require-known-failure-family", action="store_true")
    parser.add_argument("--keep-all-eligible", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_phase2av_operation_balanced_subset(
                train_tasks_jsonl=args.train_tasks_jsonl,
                val_tasks_jsonl=args.val_tasks_jsonl,
                holdout_tasks_jsonl=args.holdout_tasks_jsonl,
                output_dir=args.output_dir,
                manifest_json=args.manifest_json,
                max_per_operation=args.max_per_operation,
                require_known_failure_family=args.require_known_failure_family,
                keep_all_eligible=args.keep_all_eligible,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
