from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2ag_verifiable_candidate_sidecar import _audit_row


CLAIM_BOUNDARY = "verifiable_candidate_sidecar_requires_unique_runtime_visible_probe"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
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


def _repo_id(row: dict[str, Any]) -> str:
    return str(row.get("repo_id") or row.get("source_trace", {}).get("repo_id") or "")


def _canonical_repo_id(repo_id: str) -> str:
    repo = str(repo_id or "").strip()
    for suffix in ("_train", "_val", "_holdout"):
        if repo.endswith(suffix):
            return repo[: -len(suffix)]
    return repo


def _repo_overlaps(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    repos = {
        split: {_canonical_repo_id(repo) for row in rows if (repo := _repo_id(row))}
        for split, rows in split_rows.items()
    }
    names = list(repos)
    overlaps: dict[str, list[str]] = {}
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            overlap = sorted(repos[name] & repos[other])
            if overlap:
                overlaps[f"{name}__{other}"] = overlap
    return overlaps


def _convert_row(row: dict[str, Any], audit: dict[str, Any], *, source_jsonl: str) -> dict[str, Any]:
    converted = json.loads(json.dumps(row))
    converted["benchmark_family"] = "phase2ag_verifiable_candidate_sidecar"
    converted["claim_boundary"] = CLAIM_BOUNDARY
    converted["phase2ag_probe_audit"] = {
        "probe_prediction": audit["probe_prediction"],
        "probe_scores": audit["probe_scores"],
        "probe_overlaps": audit["probe_overlaps"],
        "expected_slot": audit["expected_slot"],
        "marker_leak": audit["marker_leak"],
        "sealed_reference": audit["sealed_reference"],
        "uses_sealed_feedback": False,
        "uses_expected_repair_action_for_offline_filter_only": True,
    }
    converted["phase2ag_source_jsonl"] = source_jsonl
    converted["unsupported_claims"] = sorted(
        set(converted.get("unsupported_claims") or [])
        | {
            "production_autonomy",
            "open_ended_debugging_generalization",
            "sealed_transfer",
            "epoch_making_architecture",
        }
    )
    converted["trace_hash"] = _sha256_text(_canonical_json(converted))
    return converted


def _filter_rows(paths: list[str | Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for path in paths:
        rows = _read_jsonl(path)
        source_name = str(Path(path))
        source_counts[source_name] = len(rows)
        for index, row in enumerate(rows):
            audit = _audit_row(row, index)
            if (
                audit["probe_correct"]
                and not audit["unresolved_probe"]
                and not audit["marker_leak"]
                and not audit["sealed_reference"]
            ):
                accepted.append(_convert_row(row, audit, source_jsonl=source_name))
            else:
                rejected.append(
                    {
                        "source_jsonl": source_name,
                        "row_index": index,
                        "trace_id": row.get("trace_id") or row.get("example_id"),
                        "repo_id": _repo_id(row),
                        "probe_correct": audit["probe_correct"],
                        "unresolved_probe": audit["unresolved_probe"],
                        "marker_leak": audit["marker_leak"],
                        "sealed_reference": audit["sealed_reference"],
                    }
                )
    accepted.sort(key=lambda row: (_repo_id(row), str(row.get("trace_id") or "")))
    return accepted, {
        "source_counts": source_counts,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "rejected_by_repo": dict(sorted(Counter(row["repo_id"] for row in rejected).items())),
        "rejected_examples": rejected[:20],
    }


def _slot_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    slots = Counter()
    for row in rows:
        audit = row.get("phase2ag_probe_audit") if isinstance(row.get("phase2ag_probe_audit"), dict) else {}
        slots[str(audit.get("expected_slot"))] += 1
    return dict(sorted(slots.items()))


def _slot_set(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("phase2ag_probe_audit", {}).get("expected_slot"))
        for row in rows
    }


def build_phase2ag_verifiable_candidate_sidecar_split(
    *,
    train_jsonl: list[str | Path],
    val_jsonl: list[str | Path],
    holdout_jsonl: list[str | Path],
    output_dir: str | Path,
    manifest_json: str | Path,
    min_train_rows: int = 32,
    min_val_rows: int = 32,
    min_holdout_rows: int = 16,
) -> dict[str, Any]:
    split_inputs = {
        "train": train_jsonl,
        "val": val_jsonl,
        "holdout": holdout_jsonl,
    }
    output = Path(output_dir)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    filter_reports: dict[str, dict[str, Any]] = {}
    for split, paths in split_inputs.items():
        rows, filter_report = _filter_rows(paths)
        split_rows[split] = rows
        filter_reports[split] = filter_report
        _write_jsonl(output / f"{split}.jsonl", rows)

    repo_overlaps = _repo_overlaps(split_rows)
    train_slots = _slot_set(split_rows["train"])
    eval_slots = _slot_set(split_rows["val"]) | _slot_set(split_rows["holdout"])
    checks = {
        "min_train_rows": len(split_rows["train"]) >= min_train_rows,
        "min_val_rows": len(split_rows["val"]) >= min_val_rows,
        "min_holdout_rows": len(split_rows["holdout"]) >= min_holdout_rows,
        "repo_disjoint": not repo_overlaps,
        "train_covers_val_and_holdout_slots": eval_slots.issubset(train_slots),
        "all_rows_have_unique_correct_probe": all(
            row.get("phase2ag_probe_audit", {}).get("probe_prediction")
            == row.get("phase2ag_probe_audit", {}).get("expected_slot")
            for rows in split_rows.values()
            for row in rows
        ),
    }
    passed = all(checks.values())
    manifest = {
        "artifact_family": "phase2ag_verifiable_candidate_sidecar_split",
        "claim_boundary": CLAIM_BOUNDARY,
        "passed": passed,
        "checks": checks,
        "blocked_actions": []
        if passed
        else [
            "do_not_train_claim_bearing_phase2ag_adapter",
            "do_not_package_phase2ag",
            "do_not_run_sealed_phase2ag",
            "do_not_claim_verifiable_candidate_sidecar_mechanism",
        ],
        "output_dir": str(output),
        "split_inputs": {split: [str(Path(path)) for path in paths] for split, paths in split_inputs.items()},
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_hashes": {
            split: _sha256_text(_canonical_json(rows)) for split, rows in split_rows.items()
        },
        "slot_distribution": {split: _slot_counts(rows) for split, rows in split_rows.items()},
        "repo_distribution": {
            split: dict(sorted(Counter(_repo_id(row) for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "canonical_repo_distribution": {
            split: dict(
                sorted(Counter(_canonical_repo_id(_repo_id(row)) for row in rows).items())
            )
            for split, rows in split_rows.items()
        },
        "repo_overlaps": repo_overlaps,
        "filter_reports": filter_reports,
        "selection_rule": (
            "accept only non-sealed rows whose runtime-visible verification probe uniquely selects "
            "the expected candidate; expected action is used only for offline filtering and auditing"
        ),
        "next_gate": "phase2ag_verifiable_candidate_sidecar_audit",
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AG verifiable candidate sidecar splits.")
    parser.add_argument("--train-jsonl", action="append", required=True)
    parser.add_argument("--val-jsonl", action="append", required=True)
    parser.add_argument("--holdout-jsonl", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--min-train-rows", type=int, default=32)
    parser.add_argument("--min-val-rows", type=int, default=32)
    parser.add_argument("--min-holdout-rows", type=int, default=16)
    args = parser.parse_args()
    report = build_phase2ag_verifiable_candidate_sidecar_split(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
        min_train_rows=args.min_train_rows,
        min_val_rows=args.min_val_rows,
        min_holdout_rows=args.min_holdout_rows,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
