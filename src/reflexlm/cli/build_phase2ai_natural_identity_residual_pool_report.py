from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2af_hardened_structural_sidecar_split import (
    _read_jsonl,
    _row_candidate,
    _shortcut_key,
)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_head_dataset(path: Path) -> bool:
    return any("heads" in part.lower() for part in path.parts)


def _is_adversarial_identity_contrast(row: dict[str, Any], path: Path) -> bool:
    text = " ".join(
        [
            str(path).lower(),
            str(row.get("benchmark_family") or "").lower(),
            str(row.get("claim_boundary") or "").lower(),
            str(row.get("phase2af_selection_rule") or "").lower(),
        ]
    )
    return "phase2ah" in text or "adversarial_identity_contrast" in text


def _sealed_or_hidden_marker_present(row: dict[str, Any]) -> bool:
    text = json.dumps(row, ensure_ascii=False).lower()
    return "sealed_v3_used\": true" in text or "sealed_feedback_used\": true" in text


def _source_kind_ok(row: dict[str, Any]) -> bool:
    source_kind = str(row.get("source_kind") or "").lower()
    origin = str(row.get("repo_url_or_origin") or "").lower()
    return source_kind in {"public_repo", "public_trace", ""} or origin.startswith(
        "https://github.com/"
    )


def _find_jsonl_inputs(roots: list[str | Path], *, include_heads: bool) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        candidate = Path(root)
        if candidate.is_file():
            if candidate.suffix == ".jsonl":
                paths.append(candidate)
            continue
        if candidate.is_dir():
            for path in candidate.rglob("*.jsonl"):
                if not include_heads and _is_head_dataset(path):
                    continue
                paths.append(path)
    return sorted(set(paths), key=lambda path: str(path).lower())


def build_phase2ai_natural_identity_residual_pool_report(
    *,
    roots: list[str | Path],
    output_json: str | Path | None = None,
    include_heads: bool = False,
    include_adversarial: bool = False,
    require_tie_residual_feasible: bool = True,
    min_source_correct_identity_wrong_rows: int = 16,
    min_unique_source_correct_identity_wrong_traces: int = 8,
    min_source_correct_identity_wrong_repos: int = 4,
    max_examples_per_bucket: int = 12,
) -> dict[str, Any]:
    paths = _find_jsonl_inputs(roots, include_heads=include_heads)
    aggregate = Counter()
    files: list[dict[str, Any]] = []
    examples: dict[str, list[dict[str, Any]]] = {}
    source_correct_identity_wrong_trace_ids: set[str] = set()
    source_correct_identity_wrong_repos: set[str] = set()
    rejected = Counter()

    for path in paths:
        try:
            raw_rows = _read_jsonl(path)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            rejected["unreadable_jsonl"] += 1
            continue
        file_buckets: Counter[str] = Counter()
        candidate_rows = 0
        for index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, dict):
                rejected["non_dict_row"] += 1
                continue
            if _sealed_or_hidden_marker_present(raw_row):
                rejected["sealed_or_hidden_marker"] += 1
                continue
            if not include_adversarial and _is_adversarial_identity_contrast(raw_row, path):
                rejected["adversarial_identity_contrast"] += 1
                continue
            if not _source_kind_ok(raw_row):
                rejected["non_public_source_kind"] += 1
                continue
            candidate = _row_candidate(
                raw_row,
                require_tie_residual_feasible=require_tie_residual_feasible,
            )
            if candidate is None:
                rejected["not_a_measurable_candidate_row"] += 1
                continue
            candidate_rows += 1
            bucket = _shortcut_key(candidate)
            bucket_key = f"source_{bucket[0]}_identity_{bucket[1]}"
            file_buckets[bucket_key] += 1
            aggregate[bucket_key] += 1
            if bucket_key == "source_1_identity_0":
                trace_id = str(raw_row.get("trace_id") or "")
                repo_id = str(raw_row.get("repo_id") or raw_row.get("repo_url_or_origin") or "")
                if trace_id:
                    source_correct_identity_wrong_trace_ids.add(trace_id)
                if repo_id:
                    source_correct_identity_wrong_repos.add(repo_id)
            bucket_examples = examples.setdefault(bucket_key, [])
            if len(bucket_examples) < max_examples_per_bucket:
                bucket_examples.append(
                    {
                        "path": str(path),
                        "row_index": index,
                        "trace_id": raw_row.get("trace_id"),
                        "repo_id": raw_row.get("repo_id"),
                        "repo_url_or_origin": raw_row.get("repo_url_or_origin"),
                        "expected_slot": candidate["phase2af_measured_shortcuts"].get(
                            "expected_slot"
                        ),
                    }
                )
        if candidate_rows:
            files.append(
                {
                    "path": str(path),
                    "rows": len(raw_rows),
                    "measurable_candidate_rows": candidate_rows,
                    "bucket_counts": dict(sorted(file_buckets.items())),
                }
            )

    source_correct_identity_wrong = int(aggregate.get("source_1_identity_0", 0))
    unique_source_correct_identity_wrong_traces = len(source_correct_identity_wrong_trace_ids)
    source_correct_identity_wrong_repo_count = len(source_correct_identity_wrong_repos)
    checks = {
        "jsonl_inputs_present": bool(paths),
        "measurable_candidate_rows_present": sum(aggregate.values()) > 0,
        "source_correct_identity_wrong_rows_min": (
            source_correct_identity_wrong >= min_source_correct_identity_wrong_rows
        ),
        "source_correct_identity_wrong_unique_traces_min": (
            unique_source_correct_identity_wrong_traces
            >= min_unique_source_correct_identity_wrong_traces
        ),
        "source_correct_identity_wrong_repos_min": (
            source_correct_identity_wrong_repo_count >= min_source_correct_identity_wrong_repos
        ),
        "adversarial_identity_contrast_excluded": not include_adversarial,
        "head_datasets_excluded": not include_heads,
    }
    passed = all(checks.values())
    report = {
        "artifact_family": "phase2ai_natural_identity_residual_pool_report",
        "passed": passed,
        "claim_bearing_training_ready": passed,
        "checks": checks,
        "metrics": {
            "jsonl_inputs_scanned": len(paths),
            "files_with_measurable_candidates": len(files),
            "measurable_candidate_rows": int(sum(aggregate.values())),
            "bucket_counts": dict(sorted(aggregate.items())),
            "source_correct_identity_wrong_rows": source_correct_identity_wrong,
            "source_correct_identity_wrong_unique_traces": unique_source_correct_identity_wrong_traces,
            "source_correct_identity_wrong_repo_count": source_correct_identity_wrong_repo_count,
            "min_source_correct_identity_wrong_rows": min_source_correct_identity_wrong_rows,
            "min_unique_source_correct_identity_wrong_traces": (
                min_unique_source_correct_identity_wrong_traces
            ),
            "min_source_correct_identity_wrong_repos": min_source_correct_identity_wrong_repos,
            "rejected_counts": dict(sorted(rejected.items())),
        },
        "examples": examples,
        "files": files,
        "blocked_actions": (
            []
            if passed
            else [
                "do_not_train_natural_identity_residual_claim_adapter",
                "do_not_package_from_phase2ai_pool",
                "do_not_claim_source_overlap_residual_mechanism",
            ]
        ),
        "allowed_next_action": (
            "build_repo_origin_disjoint_phase2ai_split"
            if passed
            else "collect_more_nonsealed_public_trace_rows_or_design_new_benchmark"
        ),
        "claim_boundary": (
            "This report only audits whether existing non-sealed public-trace pools contain "
            "natural source-correct identity-wrong rows. It does not use sealed feedback and "
            "does not by itself prove sealed transfer, production autonomy, open-ended debugging "
            "generalization, or an epoch-making architecture."
        ),
        "config": {
            "roots": [str(Path(root)) for root in roots],
            "include_heads": include_heads,
            "include_adversarial": include_adversarial,
            "require_tie_residual_feasible": require_tie_residual_feasible,
            "max_examples_per_bucket": max_examples_per_bucket,
        },
    }
    if output_json:
        _write_json(output_json, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan non-sealed public trace pools for natural source-correct identity-wrong rows."
    )
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--include-heads", action="store_true")
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--no-require-tie-residual-feasible", action="store_true")
    parser.add_argument("--min-source-correct-identity-wrong-rows", type=int, default=16)
    parser.add_argument(
        "--min-unique-source-correct-identity-wrong-traces",
        type=int,
        default=8,
    )
    parser.add_argument("--min-source-correct-identity-wrong-repos", type=int, default=4)
    parser.add_argument("--max-examples-per-bucket", type=int, default=12)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2ai_natural_identity_residual_pool_report(
        roots=args.root,
        output_json=args.output_json,
        include_heads=args.include_heads,
        include_adversarial=args.include_adversarial,
        require_tie_residual_feasible=not args.no_require_tie_residual_feasible,
        min_source_correct_identity_wrong_rows=args.min_source_correct_identity_wrong_rows,
        min_unique_source_correct_identity_wrong_traces=(
            args.min_unique_source_correct_identity_wrong_traces
        ),
        min_source_correct_identity_wrong_repos=args.min_source_correct_identity_wrong_repos,
        max_examples_per_bucket=args.max_examples_per_bucket,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
