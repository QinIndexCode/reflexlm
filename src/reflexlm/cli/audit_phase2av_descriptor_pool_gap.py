from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _target(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("learned_patch_descriptor_target", "learned_patch_candidate_target"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    expected = row.get("expected_policy")
    if isinstance(expected, dict):
        return {
            "operation": expected.get("patch_operation"),
            "after_fragment_template_id": expected.get("patch_template"),
        }
    return {}


def _operation(row: dict[str, Any]) -> str:
    return str(_target(row).get("operation") or "<missing>")


def _template(row: dict[str, Any]) -> str:
    return str(_target(row).get("after_fragment_template_id") or "<missing>")


def _failure_family(row: dict[str, Any]) -> str:
    evidence = _dict(row.get("runtime_visible_evidence"))
    for key in ("descriptor_failure_family", "failure_family", "exception_family"):
        value = evidence.get(key) or row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    text = json.dumps(evidence, ensure_ascii=False).lower()
    if "attributeerror" in text or "has no attribute" in text:
        return "attribute_missing_runtime"
    if any(marker in text for marker in ("nameerror", "importerror", "modulenotfounderror")):
        return "missing_import_or_symbol_runtime"
    if "syntaxerror" in text:
        return "syntax_load_failure_runtime"
    return "<unknown>"


def _repo_origin(row: dict[str, Any]) -> str:
    return str(
        row.get("repo_origin")
        or row.get("repo_url_or_origin")
        or row.get("repo_id")
        or "<missing>"
    )


def _split(row: dict[str, Any], fallback: str) -> str:
    value = row.get("split")
    return str(value) if isinstance(value, str) and value.strip() else fallback


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def audit_phase2av_descriptor_pool_gap(
    *,
    jsonl_paths: list[str | Path],
    min_total_rows: int = 384,
    min_rows_per_split: int = 64,
    min_operations_per_split: int = 3,
    min_examples_per_operation: int = 5,
    min_repo_origins_per_split: int = 3,
    min_repo_origins_per_operation: int = 2,
) -> dict[str, Any]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    source_files: dict[str, int] = {}
    for path in jsonl_paths:
        rows = _read_jsonl(path)
        source_files[str(Path(path))] = len(rows)
        fallback_split = Path(path).stem
        for row in rows:
            rows_by_split.setdefault(_split(row, fallback_split), []).append(row)

    split_reports: dict[str, dict[str, Any]] = {}
    for split, rows in sorted(rows_by_split.items()):
        operations = Counter(_operation(row) for row in rows)
        templates = Counter(_template(row) for row in rows)
        families = Counter(_failure_family(row) for row in rows)
        repos = Counter(_repo_origin(row) for row in rows)
        operation_template_pairs = Counter(
            f"{_operation(row)}::{_template(row)}" for row in rows
        )
        operation_repo_origins: dict[str, Counter[str]] = {}
        for row in rows:
            operation_repo_origins.setdefault(_operation(row), Counter())[_repo_origin(row)] += 1
        operation_repo_origin_counts = {
            operation: len(repos_for_operation)
            for operation, repos_for_operation in sorted(operation_repo_origins.items())
        }
        operations_with_too_few_repo_origins = sorted(
            operation
            for operation, repo_count in operation_repo_origin_counts.items()
            if repo_count < min_repo_origins_per_operation
        )
        split_reports[split] = {
            "rows": len(rows),
            "operation_counts": _counter_dict(operations),
            "template_counts": _counter_dict(templates),
            "failure_family_counts": _counter_dict(families),
            "repo_origin_count": len(repos),
            "repo_origin_counts": _counter_dict(repos),
            "operation_repo_origin_counts": operation_repo_origin_counts,
            "operation_repo_origin_details": {
                operation: _counter_dict(repo_counts)
                for operation, repo_counts in sorted(operation_repo_origins.items())
            },
            "operation_template_pair_counts": _counter_dict(operation_template_pairs),
            "operations_with_too_few_examples": sorted(
                operation
                for operation, count in operations.items()
                if count < min_examples_per_operation
            ),
            "operations_with_too_few_repo_origins": operations_with_too_few_repo_origins,
            "operation_diversity_ready": len(operations) >= min_operations_per_split
            and all(count >= min_examples_per_operation for count in operations.values()),
            "operation_repo_origin_ready": not operations_with_too_few_repo_origins,
            "split_size_ready": len(rows) >= min_rows_per_split,
            "repo_origin_ready": len(repos) >= min_repo_origins_per_split,
        }

    full_data_ready = (
        sum(report["rows"] for report in split_reports.values()) >= min_total_rows
        and {"train", "val", "holdout"}.issubset(set(split_reports))
        and all(
            report["split_size_ready"]
            and report["operation_diversity_ready"]
            and report["repo_origin_ready"]
            and report["operation_repo_origin_ready"]
            for report in split_reports.values()
            if report
        )
    )
    missing_splits = sorted({"train", "val", "holdout"} - set(split_reports))
    blocking_reasons: list[str] = []
    if missing_splits:
        blocking_reasons.append("missing_required_splits")
    if sum(report["rows"] for report in split_reports.values()) < min_total_rows:
        blocking_reasons.append("total_rows_below_full_data_threshold")
    for split, report in split_reports.items():
        if not report["split_size_ready"]:
            blocking_reasons.append(f"{split}_rows_below_threshold")
        if not report["operation_diversity_ready"]:
            blocking_reasons.append(f"{split}_operation_diversity_below_threshold")
        if not report["repo_origin_ready"]:
            blocking_reasons.append(f"{split}_repo_origin_below_threshold")
        if not report["operation_repo_origin_ready"]:
            blocking_reasons.append(f"{split}_operation_repo_origin_below_threshold")

    return {
        "artifact_family": "phase2av_descriptor_pool_gap_audit",
        "passed": full_data_ready,
        "ready_for_phase2av_full_data_construction": full_data_ready,
        "claim_boundary": (
            "This is a data-pool audit only. It can identify whether enough "
            "non-sealed descriptor-runtime rows exist for full-data construction, "
            "but it is not training, package, sealed, or architecture proof."
        ),
        "source_files": source_files,
        "total_rows": sum(report["rows"] for report in split_reports.values()),
        "split_reports": split_reports,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "thresholds": {
            "min_total_rows": min_total_rows,
            "min_rows_per_split": min_rows_per_split,
            "min_operations_per_split": min_operations_per_split,
            "min_examples_per_operation": min_examples_per_operation,
            "min_repo_origins_per_split": min_repo_origins_per_split,
            "min_repo_origins_per_operation": min_repo_origins_per_operation,
        },
        "next_data_actions": []
        if full_data_ready
        else [
            "collect_or_generate_nonsealed_rows_for_underrepresented_operations",
            "increase_train_val_holdout_rows_without_using_sealed_feedback",
            "keep_repo_origin_disjoint_and_marker_free_before_training",
            "rerun_phase2av_data_health_and_full_readiness_before_full_training",
        ],
        "unsupported_claims": [
            "phase2av_full_training_ready",
            "phase2av_package_ready",
            "sealed_cross_model_transfer",
            "freeform_patch_generation",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "inputs": {"jsonl_paths": [str(Path(path)) for path in jsonl_paths]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AV descriptor data-pool gaps.")
    parser.add_argument("--jsonl", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-total-rows", type=int, default=384)
    parser.add_argument("--min-rows-per-split", type=int, default=64)
    parser.add_argument("--min-operations-per-split", type=int, default=3)
    parser.add_argument("--min-examples-per-operation", type=int, default=5)
    parser.add_argument("--min-repo-origins-per-split", type=int, default=3)
    parser.add_argument("--min-repo-origins-per-operation", type=int, default=2)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2av_descriptor_pool_gap(
        jsonl_paths=args.jsonl,
        min_total_rows=args.min_total_rows,
        min_rows_per_split=args.min_rows_per_split,
        min_operations_per_split=args.min_operations_per_split,
        min_examples_per_operation=args.min_examples_per_operation,
        min_repo_origins_per_split=args.min_repo_origins_per_split,
        min_repo_origins_per_operation=args.min_repo_origins_per_operation,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
