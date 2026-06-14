from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from reflexlm.cli.build_phase2ae_structural_sidecar_budget_pressure_patch_candidates import (
    structural_sidecar_prediction,
)
from reflexlm.cli.build_phase2ab_identity_ambiguous_patch_candidates import _expected_slot


ABSOLUTE_PATH_RE = re.compile(
    r"(?i)((?<![A-Za-z])[A-Z]:[\\/]|\\\\[A-Za-z0-9_.-]+\\|/(?:Users|home|root|var/folders)/)"
)


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


def _literal_assignments(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {
            "TARGET_REL_PATH",
            "TARGET_LINE",
            "TARGET_COL",
        }:
            values[target.id] = ast.literal_eval(node.value)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        if len(node.test.comparators) != 1:
            continue
        expected_source = ast.get_source_segment(source, node.test.comparators[0])
        if expected_source:
            values["EXPECTED_LITERAL_SOURCE"] = expected_source.strip()
            break
    return values


def _row_provenance(
    row: dict[str, Any],
    *,
    dataset_root: Path,
) -> dict[str, Any]:
    artifact_paths = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    generated_test_rel = str(artifact_paths.get("generated_test") or "")
    generated_test_path = dataset_root / generated_test_rel if generated_test_rel else Path()
    evidence = (
        row.get("runtime_visible_evidence")
        if isinstance(row.get("runtime_visible_evidence"), dict)
        else {}
    )
    target = evidence.get("target_location") if isinstance(evidence.get("target_location"), dict) else {}
    checks = {
        "generated_test_artifact_present": bool(generated_test_rel) and generated_test_path.exists(),
        "generated_test_has_no_absolute_local_path": False,
        "target_location_matches_generated_test": False,
        "expected_literal_hash_matches_generated_test": False,
        "structural_sidecar_unique_and_correct": structural_sidecar_prediction(row)
        == _expected_slot(row),
    }
    details: dict[str, Any] = {
        "trace_id": row.get("trace_id"),
        "generated_test": generated_test_rel,
    }
    if not checks["generated_test_artifact_present"]:
        return {"passed": False, "checks": checks, "details": details}
    source = generated_test_path.read_text(encoding="utf-8")
    assignments = _literal_assignments(source)
    checks["generated_test_has_no_absolute_local_path"] = ABSOLUTE_PATH_RE.search(source) is None
    checks["target_location_matches_generated_test"] = (
        str(target.get("path") or "") == str(assignments.get("TARGET_REL_PATH") or "")
        and int(target.get("line") or -1) == int(assignments.get("TARGET_LINE") or -2)
        and int(target.get("col") or -1) == int(assignments.get("TARGET_COL") or -2)
    )
    expected_literal_source = str(assignments.get("EXPECTED_LITERAL_SOURCE") or "")
    checks["expected_literal_hash_matches_generated_test"] = (
        bool(expected_literal_source)
        and str(evidence.get("expected_literal_hash") or "")
        == _sha256_text(expected_literal_source)[:16]
    )
    details["derived_from_generated_test"] = {
        "target_rel_path": assignments.get("TARGET_REL_PATH"),
        "target_line": assignments.get("TARGET_LINE"),
        "target_col": assignments.get("TARGET_COL"),
        "expected_literal_hash": _sha256_text(expected_literal_source)[:16]
        if expected_literal_source
        else None,
    }
    return {"passed": all(checks.values()), "checks": checks, "details": details}


def audit_phase2ae_structural_sidecar_provenance(
    *,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    holdout_jsonl: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_root)
    splits = {
        "train": _read_jsonl(train_jsonl),
        "val": _read_jsonl(val_jsonl),
        "holdout": _read_jsonl(holdout_jsonl),
    }
    row_reports = {
        split: [_row_provenance(row, dataset_root=root) for row in rows]
        for split, rows in splits.items()
    }
    all_reports = [report for reports in row_reports.values() for report in reports]
    check_names = [
        "generated_test_artifact_present",
        "generated_test_has_no_absolute_local_path",
        "target_location_matches_generated_test",
        "expected_literal_hash_matches_generated_test",
        "structural_sidecar_unique_and_correct",
    ]
    checks = {
        name: all(report["checks"].get(name) is True for report in all_reports)
        for name in check_names
    }
    return {
        "artifact_family": "phase2ae_structural_sidecar_provenance_audit",
        "passed": bool(all_reports) and all(checks.values()),
        "checks": checks,
        "metrics": {
            "split_counts": {split: len(rows) for split, rows in splits.items()},
            "failing_row_count": sum(1 for report in all_reports if not report["passed"]),
        },
        "row_failures": [
            report["details"]
            | {"checks": report["checks"]}
            for report in all_reports
            if not report["passed"]
        ][:20],
        "claim_boundary": (
            "Phase2AE structural sidecar is claim-bearing only when target location and "
            "literal hash are reproducibly derived from saved runtime-visible generated-test artifacts."
        ),
        "unsupported_if_failed": [
            "structural_sidecar_runtime_visible_provenance",
            "claim_bearing_phase2ae_execution",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AE structural-sidecar provenance from saved generated-test artifacts."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--holdout-jsonl", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = audit_phase2ae_structural_sidecar_provenance(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        holdout_jsonl=args.holdout_jsonl,
        dataset_root=args.dataset_root,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
