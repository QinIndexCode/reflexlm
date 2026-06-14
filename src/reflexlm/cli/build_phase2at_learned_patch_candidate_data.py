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


def _operation_from_repair_modes(modes: list[Any]) -> str:
    primary = str(modes[0]) if modes else ""
    if primary in {"call_attribute_restoration", "behavioral_string_method_restoration"}:
        return "replace_attribute"
    if primary in {"import_restoration", "behavioral_import_restoration"}:
        return "insert_import"
    if primary in {"literal_restoration", "module_constant_literal_restoration"}:
        return "replace_literal"
    return "replace_symbol"


def _template_from_repair_modes(modes: list[Any]) -> str:
    primary = str(modes[0]) if modes else ""
    if primary in {"call_attribute_restoration", "behavioral_string_method_restoration"}:
        return "call_attribute_restoration"
    if primary in {"import_restoration", "behavioral_import_restoration"}:
        return "import_restoration"
    if primary in {"literal_restoration", "module_constant_literal_restoration"}:
        return "literal_restoration"
    if primary == "guard_restoration":
        return "guard_restoration"
    return "symbol_reference_restoration"


def _expected_candidate(row: dict[str, Any]) -> dict[str, Any]:
    expected = str(row.get("expected_repair_action") or "")
    candidates = row.get("repair_candidates") if isinstance(row.get("repair_candidates"), list) else []
    for candidate in candidates:
        if isinstance(candidate, dict) and str(candidate.get("repair_action") or "") == expected:
            return candidate
    raise ValueError(f"expected_repair_action missing from repair_candidates: {row.get('trace_id')}")


def _phase2at_target(row: dict[str, Any]) -> dict[str, Any]:
    evidence = row.get("runtime_visible_evidence") if isinstance(row.get("runtime_visible_evidence"), dict) else {}
    candidate = _expected_candidate(row)
    changed_files = [
        str(item).replace("\\", "/")
        for item in evidence.get("changed_files", [])
        if str(item).strip()
    ]
    if not changed_files:
        raise ValueError(f"row lacks changed_files: {row.get('trace_id')}")
    structural_hashes = [
        str(item)
        for item in evidence.get("structural_probe_hashes", [])
        if str(item).strip()
    ]
    candidate_probe = str(candidate.get("structural_probe_hash") or "")
    repair_modes = evidence.get("repair_modes") if isinstance(evidence.get("repair_modes"), list) else []
    operation = _operation_from_repair_modes(repair_modes)
    template_id = _template_from_repair_modes(repair_modes)
    target_path = changed_files[0]
    anchor_payload = {
        "changed_files_hash": _sha256_text("|".join(changed_files))[:16],
        "candidate_structural_probe_hash": candidate_probe,
        "runtime_structural_probe_hashes": structural_hashes,
        "target_symbol_hash": str(candidate.get("target_symbol") or ""),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "target_source": "runtime_visible_structural_descriptor_not_recorded_patch",
        "target_path": target_path,
        "operation": operation,
        "anchor": {
            "kind": "runtime_structural_probe",
            "probe_hash": candidate_probe,
            "watched_files_hash": _sha256_text(
                "|".join(str(item) for item in evidence.get("watched_files", []))
            )[:16],
        },
        "before_fragment_hash": _sha256_text(
            str(evidence.get("pytest_before_patch", {}).get("stdout_excerpt", ""))
        )[:16],
        "after_fragment_template_id": template_id,
        "literal_or_symbol_payload": anchor_payload,
        "safety_constraints": {
            "max_changed_files": max(1, len(changed_files)),
            "allowed_paths": changed_files,
            "forbid_unbounded_diff_text": True,
            "require_anchor_match": True,
            "require_rollback_verification": True,
        },
        "verification_command_slot": 0,
    }


def phase2z_row_to_phase2at(row: dict[str, Any]) -> dict[str, Any]:
    converted = json.loads(json.dumps(row))
    artifact_paths = converted.get("artifact_paths") if isinstance(converted.get("artifact_paths"), dict) else {}
    artifact_paths.pop("patch_diff", None)
    converted["artifact_paths"] = artifact_paths
    converted["benchmark_family"] = "phase2at_learned_bounded_patch_candidate_generation"
    converted["claim_boundary"] = CLAIM_BOUNDARY
    converted["patch_candidate_schema_version"] = SCHEMA_VERSION
    converted["learned_patch_candidate_target"] = _phase2at_target(converted)
    converted["freeform_patch_generation"] = False
    converted["recorded_patch_artifact_as_generation_target"] = False
    converted["symbolic_generator_as_generation_target"] = False
    converted["sealed_feedback_used"] = False
    converted["phase2at_transform"] = {
        "source_family": "phase2z_public_structural_repair_trace",
        "patch_diff_artifact_removed_from_training_row": True,
        "target_is_bounded_descriptor": True,
        "uses_recorded_patch_text_as_target": False,
        "uses_symbolic_generator_output_as_target": False,
        "uses_expected_repair_action_for_offline_supervised_target_assignment": True,
    }
    converted["trace_hash"] = _sha256_text(_canonical_json(converted))
    return converted


def build_phase2at_learned_patch_candidate_data(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    output_dir: str | Path,
    manifest_json: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    split_inputs = {
        "train": Path(train_jsonl),
        "val": Path(val_jsonl),
        "holdout": Path(holdout_jsonl),
    }
    split_rows: dict[str, list[dict[str, Any]]] = {}
    operation_counts: dict[str, dict[str, int]] = {}
    for split, path in split_inputs.items():
        rows = [phase2z_row_to_phase2at(row) for row in _read_jsonl(path)]
        split_rows[split] = rows
        operation_counts[split] = dict(
            sorted(
                Counter(
                    row["learned_patch_candidate_target"]["operation"] for row in rows
                ).items()
            )
        )
        _write_jsonl(output / f"{split}.jsonl", rows)
    manifest = {
        "artifact_family": "phase2at_learned_patch_candidate_split",
        "claim_boundary": CLAIM_BOUNDARY,
        "schema_version": SCHEMA_VERSION,
        "output_dir": str(output),
        "source_split_inputs": {split: str(path) for split, path in split_inputs.items()},
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "split_hashes": {
            split: _sha256_text(_canonical_json(rows)) for split, rows in split_rows.items()
        },
        "operation_counts": operation_counts,
        "freeform_patch_generation": False,
        "recorded_patch_artifact_as_generation_target": False,
        "symbolic_generator_as_generation_target": False,
        "sealed_feedback_used": False,
        "next_gate": "phase2at_learned_patch_candidate_data_health",
    }
    _write_json(manifest_json, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2AT learned bounded patch candidate descriptor splits."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-json", required=True)
    args = parser.parse_args()
    report = build_phase2at_learned_patch_candidate_data(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        output_dir=args.output_dir,
        manifest_json=args.manifest_json,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
