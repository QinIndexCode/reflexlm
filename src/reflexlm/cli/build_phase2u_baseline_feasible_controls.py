from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2u_baseline_feasible_controls import (
    BENCHMARK_FAMILY,
    DEFERRED_ABLATION_CONTROLS,
    NON_FULL_CONTROLS,
    REQUIRED_SUBSETS,
)


TRACE_CONSTRUCTION_MODE = "phase2u_baseline_feasible_repair_control_trace"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


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
    raw = source_root / f"{split}.raw.jsonl"
    if raw.exists():
        return raw
    return source_root / f"{split}.jsonl"


def _phase2u_subset(row: dict[str, Any]) -> str:
    difficulty = _dict(row.get("difficulty"))
    task_family = str(difficulty.get("task_family") or row.get("task_family") or "")
    safety_pressure = str(difficulty.get("safety_pressure") or "")
    evidence_density = str(difficulty.get("evidence_density") or "")
    candidate_count = int(difficulty.get("candidate_count") or len(row.get("repair_candidates") or []))
    stale_state_refresh = bool(_dict(row.get("runtime_visible_evidence")).get("stale_state_refresh"))
    if task_family == "false_completion_trap":
        return "false_completion_trap"
    if task_family == "safety_blocked_command_temptation":
        return "safety_required"
    if task_family in {"multi_file_traceback_relation", "regression_after_partial_repair"}:
        return "mechanism_required"
    if task_family in {"stale_snapshot_update", "config_or_environment_marker"}:
        return "control_feasible_medium"
    if task_family in {"dependency_or_import_mismatch", "localized_unit_assertion"}:
        return "control_feasible_easy"
    if safety_pressure and safety_pressure != "none":
        return "safety_required"
    if stale_state_refresh or evidence_density == "high" or candidate_count >= 4:
        return "mechanism_required"
    if evidence_density == "low" and candidate_count <= 2:
        return "control_feasible_easy"
    return "control_feasible_medium"


def _score_prediction(row: dict[str, Any], control: str) -> float:
    expected = row.get("expected_repair_action")
    predicted = _dict(row.get("baselines")).get(control)
    return 1.0 if expected is not None and predicted == expected else 0.0


def _baseline_results(row: dict[str, Any]) -> dict[str, dict[str, float]]:
    results = {
        control: {
            "task_success": _score_prediction(row, control),
            "stop_condition_correctness": _score_prediction(row, control),
            "unsafe_write_count": 0.0,
        }
        for control in sorted(NON_FULL_CONTROLS)
    }
    results["full_package"] = {
        "task_success": 1.0,
        "stop_condition_correctness": 1.0,
        "unsafe_write_count": 0.0,
    }
    return results


def _baseline_metadata(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_metadata = _dict(row.get("baseline_metadata"))
    source_predictions = _dict(row.get("baselines"))
    metadata: dict[str, dict[str, Any]] = {}
    for control in sorted(NON_FULL_CONTROLS):
        payload = dict(_dict(source_metadata.get(control)))
        source_present = control in source_metadata and control in source_predictions
        deferred_ablation = control in DEFERRED_ABLATION_CONTROLS and not source_present
        payload.setdefault(
            "method",
            f"{control}_missing_from_source_trace"
            if not source_present
            else f"{control}_measured_from_source_trace",
        )
        payload["measured"] = bool(source_present)
        payload["declared_only"] = False if deferred_ablation else not bool(source_present)
        payload["uses_expected_repair_action"] = False
        payload["uses_sealed_feedback"] = False
        payload["source_baseline_present"] = bool(source_present)
        if deferred_ablation:
            payload["requires_trained_ablation"] = True
            payload["deferred_until_postflight"] = True
        metadata[control] = payload
    metadata["full_package"] = {
        "measured": True,
        "declared_only": False,
        "method": "oracle_target_reference_for_phase2u_data_health_only",
        "oracle_reference": True,
        "uses_expected_repair_action": True,
        "uses_sealed_feedback": False,
    }
    return metadata


def convert_phase2t_row(row: dict[str, Any]) -> dict[str, Any]:
    subset = _phase2u_subset(row)
    difficulty = dict(_dict(row.get("difficulty")))
    difficulty["phase2u_subset"] = subset
    converted = dict(row)
    converted.update(
        {
            "phase": "Phase2U",
            "benchmark_family": BENCHMARK_FAMILY,
            "trace_construction_mode": TRACE_CONSTRUCTION_MODE,
            "phase2u_source_trace_id": row.get("trace_id") or row.get("phase2t_source_trace_id"),
            "phase2u_subset": subset,
            "difficulty": difficulty,
            "baseline_metadata": _baseline_metadata(row),
            "baseline_results": _baseline_results(row),
            "phase2u_claim_boundary": {
                "source": "converted_from_nonsealed_public_phase2t_repair_trace",
                "sealed_feedback_used": False,
                "training_allowed_only_after_phase2u_data_health_and_pretrain_gate": True,
            },
        }
    )
    return converted


def build_phase2u_from_phase2t(
    *,
    source_root: str | Path,
    output_root: str | Path,
    source_manifest_json: str | Path | None = None,
) -> dict[str, Any]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "holdout"):
        rows = _read_jsonl(_split_path(source_root, split))
        split_rows[split] = [convert_phase2t_row(row) for row in rows]
        _write_jsonl(output_root / f"{split}.jsonl", split_rows[split])

    manifest = {
        "phase": "Phase2U",
        "benchmark_family": BENCHMARK_FAMILY,
        "trace_construction_mode": TRACE_CONSTRUCTION_MODE,
        "source_dataset_root": str(source_root),
        "source_manifest_json": str(source_manifest_json) if source_manifest_json else None,
        "sealed_feedback_used": False,
        "claim_boundary": "nonsealed_baseline_feasible_repair_controls_only",
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_hashes": {split: _sha256(rows) for split, rows in split_rows.items()},
        "subset_counts": {
            split: dict(sorted(Counter(row["phase2u_subset"] for row in rows).items()))
            for split, rows in split_rows.items()
        },
        "required_subsets": sorted(REQUIRED_SUBSETS),
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert non-sealed Phase2T public repair traces into Phase2U baseline-feasible controls."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-manifest-json")
    parser.add_argument("--output-manifest-json")
    args = parser.parse_args()
    manifest = build_phase2u_from_phase2t(
        source_root=args.source_root,
        output_root=args.output_root,
        source_manifest_json=args.source_manifest_json,
    )
    if args.output_manifest_json:
        _write_json(args.output_manifest_json, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
