from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PATCH_OPERATION_ALLOWLIST = {
    "replace_symbol",
    "replace_attribute",
    "insert_import",
    "replace_literal",
    "insert_guard",
}
PATCH_TEMPLATE_ALLOWLIST = {
    "symbol_reference_restoration",
    "call_attribute_restoration",
    "import_restoration",
    "literal_restoration",
    "guard_restoration",
}


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


def _descriptor(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("policy_learned_patch_descriptor_outputs")
    return value if isinstance(value, dict) else {}


def _expected_from_repair_modes(row: dict[str, Any]) -> tuple[str | None, str | None]:
    evidence = (
        row.get("runtime_visible_evidence")
        if isinstance(row.get("runtime_visible_evidence"), dict)
        else {}
    )
    modes = evidence.get("repair_modes") if isinstance(evidence.get("repair_modes"), list) else []
    joined = " ".join(str(mode).lower() for mode in modes)
    if "import" in joined:
        return "insert_import", "import_restoration"
    if "attribute" in joined or "attr" in joined:
        return "replace_attribute", "call_attribute_restoration"
    if "literal" in joined:
        return "replace_literal", "literal_restoration"
    if "guard" in joined:
        return "insert_guard", "guard_restoration"
    if joined:
        return "replace_symbol", "symbol_reference_restoration"
    return None, None


def _indexed_by_trace(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("trace_id") or ""): row for row in rows}


def audit_phase2au_descriptor_runtime_observability(
    *,
    execution_jsonl: str | Path,
    source_rows_jsonl: str | Path | None = None,
    min_rows: int = 20,
    min_operation_template_pairs: int = 2,
) -> dict[str, Any]:
    rows = _read_jsonl(execution_jsonl)
    source_by_trace = _indexed_by_trace(_read_jsonl(source_rows_jsonl)) if source_rows_jsonl else {}
    operations = Counter()
    templates = Counter()
    expected_total = 0
    expected_matches = 0
    missing_descriptor_rows = 0
    invalid_descriptor_rows = 0
    for row in rows:
        descriptor = _descriptor(row)
        if not descriptor:
            missing_descriptor_rows += 1
            continue
        operation = str(descriptor.get("patch_operation") or "")
        template = str(descriptor.get("patch_template") or "")
        operations[operation] += 1
        templates[template] += 1
        if operation not in PATCH_OPERATION_ALLOWLIST or template not in PATCH_TEMPLATE_ALLOWLIST:
            invalid_descriptor_rows += 1
        source = source_by_trace.get(str(row.get("trace_id") or ""))
        if source:
            expected_operation, expected_template = _expected_from_repair_modes(source)
            if expected_operation is not None and expected_template is not None:
                expected_total += 1
                expected_matches += int(
                    operation == expected_operation and template == expected_template
                )
    operation_template_pairs = {
        (
            str(_descriptor(row).get("patch_operation") or ""),
            str(_descriptor(row).get("patch_template") or ""),
        )
        for row in rows
        if _descriptor(row)
    }
    expected_match_rate = expected_matches / expected_total if expected_total else None
    checks = {
        "row_minimum_met": len(rows) >= min_rows,
        "all_rows_policy_loaded": all(row.get("policy_loaded") is True for row in rows),
        "all_rows_successful": all(row.get("success") is True for row in rows),
        "descriptor_outputs_present": missing_descriptor_rows == 0 and bool(rows),
        "descriptor_outputs_allowlisted": invalid_descriptor_rows == 0,
        "sealed_feedback_absent": all(row.get("sealed_feedback_used") is False for row in rows),
        "no_freeform_patch_claim": all(
            row.get("freeform_patch_generation") is False
            and row.get("claim_bearing_freeform_patch_evidence") is False
            for row in rows
        ),
        "descriptor_matches_runtime_visible_repair_modes": (
            expected_total == 0 or expected_match_rate == 1.0
        ),
        "descriptor_operation_template_diversity_met": len(operation_template_pairs)
        >= min_operation_template_pairs,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2au_descriptor_runtime_observability_audit",
        "passed": passed,
        "claim_boundary": (
            "This audit checks whether bounded learned patch descriptor outputs are "
            "present in runtime execution traces and whether they are diverse enough "
            "to support descriptor-runtime evidence. It does not by itself prove "
            "learned freeform patch generation, production autonomy, open-ended "
            "debugging, sealed transfer, or an epoch-making architecture."
        ),
        "checks": checks,
        "metrics": {
            "rows": len(rows),
            "missing_descriptor_rows": missing_descriptor_rows,
            "invalid_descriptor_rows": invalid_descriptor_rows,
            "operation_counts": dict(sorted(operations.items())),
            "template_counts": dict(sorted(templates.items())),
            "operation_template_pair_count": len(operation_template_pairs),
            "expected_match_total": expected_total,
            "expected_match_rate": expected_match_rate,
        },
        "thresholds": {
            "min_rows": min_rows,
            "min_operation_template_pairs": min_operation_template_pairs,
        },
        "supported_claims": [
            "phase2au_runtime_descriptor_outputs_present_and_diverse"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "learned_freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_claim_descriptor_runtime_generation_from_single_template_trace",
            "build_or_collect_graded_descriptor_runtime_tasks_with_multiple_patch_operations",
        ],
        "inputs": {
            "execution_jsonl": str(Path(execution_jsonl)),
            "source_rows_jsonl": str(Path(source_rows_jsonl)) if source_rows_jsonl else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AU runtime learned descriptor observability."
    )
    parser.add_argument("--execution-jsonl", required=True)
    parser.add_argument("--source-rows-jsonl")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=20)
    parser.add_argument("--min-operation-template-pairs", type=int, default=2)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2au_descriptor_runtime_observability(
        execution_jsonl=args.execution_jsonl,
        source_rows_jsonl=args.source_rows_jsonl,
        min_rows=args.min_rows,
        min_operation_template_pairs=args.min_operation_template_pairs,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
