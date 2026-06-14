from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


BENCHMARK_FAMILY = "graded_transfer_nonzero_controls"
TRACE_CONSTRUCTION_MODE = "phase2v_graded_transfer_nonzero_control_trace"
TIERS = {"control_feasible", "mixed_mechanism", "mechanism_required"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _split_path(source_root: Path, split: str) -> Path:
    return source_root / f"{split}.jsonl"


def _phase2v_tier(row: dict[str, Any]) -> str:
    subset = str(row.get("phase2u_subset") or _dict(row.get("difficulty")).get("phase2u_subset"))
    if subset == "control_feasible_easy":
        return "control_feasible"
    if subset in {"control_feasible_medium", "safety_required", "false_completion_trap"}:
        return "mixed_mechanism"
    return "mechanism_required"


def convert_phase2u_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("phase") != "Phase2U":
        raise ValueError(f"expected Phase2U row: {row.get('trace_id')}")
    source_trace_id = str(row.get("trace_id") or row.get("phase2u_source_trace_id") or _sha256(row)[:16])
    tier = _phase2v_tier(row)
    difficulty = dict(_dict(row.get("difficulty")))
    difficulty["phase2v_tier"] = tier
    converted = dict(row)
    converted.update(
        {
            "phase": "Phase2V",
            "benchmark_family": BENCHMARK_FAMILY,
            "trace_construction_mode": TRACE_CONSTRUCTION_MODE,
            "trace_id": f"phase2v:{source_trace_id}",
            "phase2v_source_trace_id": source_trace_id,
            "phase2v_tier": tier,
            "difficulty": difficulty,
            "phase2v_claim_boundary": {
                "source": "rematerialized_from_nonsealed_public_phase2u_rows",
                "sealed_feedback_used": False,
                "purpose": "heldout_graded_transfer_with_nonzero_controls",
            },
        }
    )
    return converted


def build_phase2v_from_phase2u(
    *,
    source_root: str | Path,
    output_root: str | Path,
    source_manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "holdout"):
        rows = [convert_phase2u_row(row) for row in _read_jsonl(_split_path(source_root, split))]
        split_rows[split] = rows
        _write_jsonl(output_root / f"{split}.jsonl", rows)
    manifest = {
        "phase": "Phase2V",
        "benchmark_family": BENCHMARK_FAMILY,
        "trace_construction_mode": TRACE_CONSTRUCTION_MODE,
        "source_dataset_root": str(source_root),
        "source_manifest_json": str(source_manifest_json) if source_manifest_json else None,
        "sealed_feedback_used": False,
        "claim_boundary": "heldout_graded_transfer_nonzero_controls_only",
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_hashes": {split: _sha256(rows) for split, rows in split_rows.items()},
        "tier_counts": {
            split: dict(sorted(Counter(row["phase2v_tier"] for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "required_tiers": sorted(TIERS),
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2V graded-transfer nonzero-control rows from non-sealed Phase2U rows."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-manifest-json")
    parser.add_argument("--output-manifest-json")
    args = parser.parse_args()
    manifest = build_phase2v_from_phase2u(
        source_root=args.source_root,
        output_root=args.output_root,
        source_manifest_json=args.source_manifest_json,
    )
    if args.output_manifest_json:
        _write_json(args.output_manifest_json, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
