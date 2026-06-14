from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from reflexlm.experiment import create_experiment_run


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _row_matches(
    row: dict[str, Any],
    *,
    include_task_types: set[str],
    include_routes: set[str],
    include_action_types: set[str],
    exclude_action_types: set[str],
    min_t: int | None,
    max_t: int | None,
) -> bool:
    task_type = str(row.get("task_type", ""))
    route_name = str(row.get("route_name", ""))
    action_type = str(row.get("action_type", ""))
    t_value = int(row.get("t", 0))
    if include_task_types and task_type not in include_task_types:
        return False
    if include_routes and route_name not in include_routes:
        return False
    if include_action_types and action_type not in include_action_types:
        return False
    if exclude_action_types and action_type in exclude_action_types:
        return False
    if min_t is not None and t_value < min_t:
        return False
    if max_t is not None and t_value > max_t:
        return False
    return True


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    include_task_types: set[str],
    include_routes: set[str],
    include_action_types: set[str],
    exclude_action_types: set[str],
    min_t: int | None,
    max_t: int | None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if _row_matches(
            row,
            include_task_types=include_task_types,
            include_routes=include_routes,
            include_action_types=include_action_types,
            exclude_action_types=exclude_action_types,
            min_t=min_t,
            max_t=max_t,
        )
    ]


def _bucket_key(row: dict[str, Any], balance_keys: list[str]) -> str:
    if not balance_keys:
        return "__all__"
    return "||".join(str(row.get(key, "")) for key in balance_keys)


def _balanced_limit_rows(
    rows: list[dict[str, Any]],
    *,
    max_rows: int | None,
    balance_keys: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    if max_rows is None or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(_bucket_key(row, balance_keys), []).append(row)
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
    selected: list[dict[str, Any]] = []
    bucket_names = sorted(buckets)
    while len(selected) < max_rows and bucket_names:
        next_bucket_names: list[str] = []
        for bucket_name in bucket_names:
            bucket_rows = buckets[bucket_name]
            if bucket_rows and len(selected) < max_rows:
                selected.append(bucket_rows.pop())
            if bucket_rows:
                next_bucket_names.append(bucket_name)
        bucket_names = next_bucket_names
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter an existing Phase 2 SFT corpus into a focused subset.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-task-type", action="append")
    parser.add_argument("--include-route", action="append")
    parser.add_argument("--include-action-type", action="append")
    parser.add_argument("--exclude-action-type", action="append")
    parser.add_argument("--min-t", type=int)
    parser.add_argument("--max-t", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument(
        "--balance-key",
        action="append",
        choices=["task_type", "route_name", "action_type"],
        default=[],
        help="Metadata keys used only for deterministic SFT sampling balance.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--run-root")
    parser.add_argument("--run-name", default="phase2_sft_filtered")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    output_dir = Path(args.output_dir)
    include_task_types = set(args.include_task_type or [])
    include_routes = set(args.include_route or [])
    include_action_types = set(args.include_action_type or [])
    exclude_action_types = set(args.exclude_action_type or [])

    run = create_experiment_run(
        kind="sft_filter",
        name=args.run_name,
        config={
            "train_jsonl": str(train_path.resolve()),
            "val_jsonl": str(val_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "include_task_types": sorted(include_task_types),
            "include_routes": sorted(include_routes),
            "include_action_types": sorted(include_action_types),
            "exclude_action_types": sorted(exclude_action_types),
            "min_t": args.min_t,
            "max_t": args.max_t,
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "balance_keys": list(args.balance_key),
            "seed": args.seed,
        },
        run_root=args.run_root,
    )

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    filtered_train = _filter_rows(
        train_rows,
        include_task_types=include_task_types,
        include_routes=include_routes,
        include_action_types=include_action_types,
        exclude_action_types=exclude_action_types,
        min_t=args.min_t,
        max_t=args.max_t,
    )
    filtered_val = _filter_rows(
        val_rows,
        include_task_types=include_task_types,
        include_routes=include_routes,
        include_action_types=include_action_types,
        exclude_action_types=exclude_action_types,
        min_t=args.min_t,
        max_t=args.max_t,
    )
    filtered_train = _balanced_limit_rows(
        filtered_train,
        max_rows=args.max_train_rows,
        balance_keys=list(args.balance_key),
        seed=args.seed,
    )
    filtered_val = _balanced_limit_rows(
        filtered_val,
        max_rows=args.max_val_rows,
        balance_keys=list(args.balance_key),
        seed=args.seed + 1,
    )

    _write_jsonl(output_dir / "train.jsonl", filtered_train)
    _write_jsonl(output_dir / "val.jsonl", filtered_val)

    payload = {
        "train_jsonl": str(train_path.resolve()),
        "val_jsonl": str(val_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "filters": {
            "include_task_types": sorted(include_task_types),
            "include_routes": sorted(include_routes),
            "include_action_types": sorted(include_action_types),
            "exclude_action_types": sorted(exclude_action_types),
            "min_t": args.min_t,
            "max_t": args.max_t,
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "balance_keys": list(args.balance_key),
            "seed": args.seed,
        },
        "counts": {
            "train_input": len(train_rows),
            "train_output": len(filtered_train),
            "val_input": len(val_rows),
            "val_output": len(filtered_val),
        },
    }
    payload["run_manifest"] = run.finalize(payload)
    run.write_json("filter_summary.json", payload)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
