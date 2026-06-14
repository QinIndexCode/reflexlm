from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2av_descriptor_pool_gap import audit_phase2av_descriptor_pool_gap


DEFAULT_SPLITS = ("train", "val", "holdout")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _rows_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _target(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("learned_patch_candidate_target", "learned_patch_descriptor_target"):
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


def _repo_origin(row: dict[str, Any]) -> str:
    return str(
        row.get("repo_url_or_origin")
        or row.get("repo_origin")
        or row.get("repo_id")
        or "<missing>"
    )


def _split_deficit(
    split_rows: list[dict[str, Any]],
    *,
    min_rows_per_split: int,
    min_examples_per_operation: int,
    min_repo_origins_per_operation: int,
    known_operations: set[str],
) -> int:
    operation_counts = Counter(_operation(row) for row in split_rows)
    operation_repos: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        operation_repos[_operation(row)].add(_repo_origin(row))
    example_deficit = sum(
        max(0, min_examples_per_operation - operation_counts.get(operation, 0))
        for operation in known_operations
    )
    repo_deficit = sum(
        max(0, min_repo_origins_per_operation - len(operation_repos.get(operation, set())))
        for operation in known_operations
    )
    row_deficit = max(0, min_rows_per_split - len(split_rows))
    # Operation coverage is the scientific gate; row count only breaks ties.
    return (example_deficit * 1000) + (repo_deficit * 1000) + row_deficit


def _deficit_reduction(
    split_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    min_rows_per_split: int,
    min_examples_per_operation: int,
    min_repo_origins_per_operation: int,
    known_operations: set[str],
) -> int:
    before = _split_deficit(
        split_rows,
        min_rows_per_split=min_rows_per_split,
        min_examples_per_operation=min_examples_per_operation,
        min_repo_origins_per_operation=min_repo_origins_per_operation,
        known_operations=known_operations,
    )
    after = _split_deficit(
        [*split_rows, *candidate_rows],
        min_rows_per_split=min_rows_per_split,
        min_examples_per_operation=min_examples_per_operation,
        min_repo_origins_per_operation=min_repo_origins_per_operation,
        known_operations=known_operations,
    )
    return before - after


def _assign_repositories(
    rows: list[dict[str, Any]],
    *,
    min_rows_per_split: int,
    min_examples_per_operation: int,
    min_repo_origins_per_operation: int,
) -> dict[str, list[dict[str, Any]]]:
    repo_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        repo_rows[_repo_origin(row)].append(row)
    known_operations = {
        operation for operation in (_operation(row) for row in rows) if operation != "<missing>"
    }
    split_rows = {split: [] for split in DEFAULT_SPLITS}
    repo_items = sorted(
        repo_rows.items(),
        key=lambda item: (
            -len({_operation(row) for row in item[1]}),
            -sum(1 for row in item[1] if _operation(row) == "replace_attribute"),
            -len(item[1]),
            item[0],
        ),
    )
    seeded_items = repo_items[: len(DEFAULT_SPLITS)]
    remaining_items = repo_items[len(DEFAULT_SPLITS) :]
    for split, (_repo, group) in zip(DEFAULT_SPLITS, seeded_items, strict=False):
        split_rows[split].extend(group)

    target_rows_per_split = max(min_rows_per_split, len(rows) // len(DEFAULT_SPLITS))
    for _repo, group in remaining_items:
        best_split = max(
            DEFAULT_SPLITS,
            key=lambda split: (
                _deficit_reduction(
                    split_rows[split],
                    group,
                    min_rows_per_split=min_rows_per_split,
                    min_examples_per_operation=min_examples_per_operation,
                    min_repo_origins_per_operation=min_repo_origins_per_operation,
                    known_operations=known_operations,
                ),
                -max(0, len(split_rows[split]) + len(group) - target_rows_per_split),
                -len(split_rows[split]),
            ),
        )
        split_rows[best_split].extend(group)
    for split in DEFAULT_SPLITS:
        split_rows[split].sort(
            key=lambda row: str(row.get("trace_id") or row.get("task_id") or "")
        )
        for row in split_rows[split]:
            row["split"] = split
    return split_rows


def _operation_counts_by_split(
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    return {
        split: dict(sorted(Counter(_operation(row) for row in rows).items()))
        for split, rows in split_rows.items()
    }


def _all_rows_have_false(rows: list[dict[str, Any]], key: str) -> bool:
    return all(row.get(key) is False for row in rows)


def build_phase2av_repo_operation_disjoint_split(
    *,
    input_jsonl: list[str | Path],
    output_dir: str | Path,
    manifest_json: str | Path,
    min_rows_per_split: int = 64,
    min_examples_per_operation: int = 5,
    min_repo_origins_per_split: int = 3,
    min_repo_origins_per_operation: int = 2,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_files: dict[str, int] = {}
    for path in input_jsonl:
        loaded = _read_jsonl(path)
        source_files[str(Path(path))] = len(loaded)
        rows.extend(dict(row) for row in loaded)
    split_rows = _assign_repositories(
        rows,
        min_rows_per_split=min_rows_per_split,
        min_examples_per_operation=min_examples_per_operation,
        min_repo_origins_per_operation=min_repo_origins_per_operation,
    )
    output = Path(output_dir)
    split_hashes = {}
    split_counts = {}
    for split, split_payload in split_rows.items():
        _write_jsonl(output / f"{split}.jsonl", split_payload)
        split_hashes[split] = _rows_sha256(split_payload)
        split_counts[split] = len(split_payload)
    audit = audit_phase2av_descriptor_pool_gap(
        jsonl_paths=[output / f"{split}.jsonl" for split in DEFAULT_SPLITS],
        min_total_rows=sum(split_counts.values()),
        min_rows_per_split=min_rows_per_split,
        min_operations_per_split=3,
        min_examples_per_operation=min_examples_per_operation,
        min_repo_origins_per_split=min_repo_origins_per_split,
        min_repo_origins_per_operation=min_repo_origins_per_operation,
    )
    manifest = {
        "artifact_family": "phase2at_learned_patch_candidate_split",
        "split_construction_family": "phase2av_repo_operation_disjoint_split",
        "schema_version": "phase2at.learned_bounded_patch_candidate.v1",
        "passed": bool(audit["passed"]),
        "claim_boundary": (
            "Repo-operation-disjoint split construction is a non-sealed data preparation "
            "step only; it does not authorize training claims, packaging, sealed transfer, "
            "or production autonomy."
        ),
        "output_dir": str(output),
        "source_files": source_files,
        "split_counts": split_counts,
        "split_hashes": split_hashes,
        "operation_counts": _operation_counts_by_split(split_rows),
        "freeform_patch_generation": not _all_rows_have_false(
            rows,
            "freeform_patch_generation",
        ),
        "recorded_patch_artifact_as_generation_target": not _all_rows_have_false(
            rows,
            "recorded_patch_artifact_as_generation_target",
        ),
        "symbolic_generator_as_generation_target": not _all_rows_have_false(
            rows,
            "symbolic_generator_as_generation_target",
        ),
        "sealed_feedback_used": not _all_rows_have_false(rows, "sealed_feedback_used"),
        "data_health": audit,
        "next_gate": "phase2av_graded_descriptor_runtime_pretrain_gate",
        "smoke_training_allowed": bool(audit["passed"]),
        "full_training_allowed": False,
        "package_allowed": False,
        "sealed_eval_allowed": False,
        "unsupported_claims": [
            "phase2av_full_training_ready",
            "phase2av_package_ready",
            "sealed_cross_model_transfer",
            "freeform_patch_generation",
            "production_autonomy",
            "epoch_making_architecture",
        ],
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a repo-origin-disjoint Phase2AV split with per-operation repo coverage."
    )
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--min-rows-per-split", type=int, default=64)
    parser.add_argument("--min-examples-per-operation", type=int, default=5)
    parser.add_argument("--min-repo-origins-per-split", type=int, default=3)
    parser.add_argument("--min-repo-origins-per-operation", type=int, default=2)
    args = parser.parse_args()
    report = build_phase2av_repo_operation_disjoint_split(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        min_rows_per_split=args.min_rows_per_split,
        min_examples_per_operation=args.min_examples_per_operation,
        min_repo_origins_per_split=args.min_repo_origins_per_split,
        min_repo_origins_per_operation=args.min_repo_origins_per_operation,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
