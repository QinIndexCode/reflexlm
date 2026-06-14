from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


RECORDED_PATCH_GENERATORS = {
    "public_structural_recorded_diff_operator_v1",
    "synthetic_safe_recorded_diff_operator_v1",
}
SYMBOLIC_PATCH_PREFIXES = ("bounded_symbolic_",)
BOUNDED_CANDIDATE_GENERATORS = {
    "bounded_patch_candidate_selector_v1",
}
SUPPORTED_CLAIM_TYPES = {
    "symbolic_structural_patch_proposal",
    "bounded_patch_candidate_selection",
    "learned_patch_generation",
}


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


def _is_recorded_patch_control(row: dict[str, Any]) -> bool:
    patch_source = str(row.get("patch_source") or "")
    patch_generator = str(row.get("patch_generator") or "")
    return (
        row.get("recorded_patch_artifact_used") is True
        or patch_generator in RECORDED_PATCH_GENERATORS
        or patch_source.startswith("recorded_")
    )


def _is_symbolic_runtime_generator(row: dict[str, Any]) -> bool:
    patch_source = str(row.get("patch_source") or "")
    patch_generator = str(row.get("patch_generator") or "")
    return patch_source.startswith("package_runtime_symbolic_") or patch_generator.startswith(
        SYMBOLIC_PATCH_PREFIXES
    )


def _is_restricted_symbolic_control(row: dict[str, Any]) -> bool:
    patch_source = str(row.get("patch_source") or "")
    return patch_source.startswith("control_runtime_symbolic_") or (
        row.get("control_execution_evidence") is True
        and row.get("claim_bearing_execution_evidence") is False
    )


def _is_bounded_candidate_selection(row: dict[str, Any]) -> bool:
    return (
        str(row.get("patch_generator") or "") in BOUNDED_CANDIDATE_GENERATORS
        and row.get("claim_bearing_candidate_selection_evidence") is True
        and row.get("claim_bearing_freeform_patch_evidence") is False
        and row.get("freeform_patch_generation") is False
    )


def classify_repair_evidence_row(row: dict[str, Any]) -> str:
    if _is_restricted_symbolic_control(row):
        return "restricted_symbolic_control"
    if _is_symbolic_runtime_generator(row):
        return "symbolic_runtime_generator"
    if _is_bounded_candidate_selection(row):
        return "bounded_patch_candidate_selection"
    if _is_recorded_patch_control(row):
        return "recorded_patch_control"
    if str(row.get("patch_source") or "") in {"patch_not_authorized", "package_runtime_no_patch_authorized"}:
        return "no_patch_or_unauthorized"
    return "invalid_or_mixed"


def audit_phase2as_learned_repair_boundary(
    *,
    execution_results_jsonl: str | Path,
    claimed_capability: str,
    min_rows: int = 8,
    min_success_rate: float = 0.85,
    min_selection_accuracy: float = 0.85,
) -> dict[str, Any]:
    if claimed_capability not in SUPPORTED_CLAIM_TYPES:
        raise ValueError(
            "claimed_capability must be one of: "
            + ", ".join(sorted(SUPPORTED_CLAIM_TYPES))
        )
    rows = _read_jsonl(execution_results_jsonl)
    classifications = [classify_repair_evidence_row(row) for row in rows]
    class_counts = Counter(classifications)
    successes = [row for row in rows if row.get("success") is True]
    correct_selections = [
        row for row in rows if row.get("patch_candidate_selected_correctly") is True
    ]
    success_rate = len(successes) / len(rows) if rows else 0.0
    selection_accuracy = len(correct_selections) / len(rows) if rows else 0.0

    generic_checks = {
        "row_minimum_met": len(rows) >= min_rows,
        "success_rate_minimum_met": success_rate >= min_success_rate,
        "sealed_feedback_absent": all(row.get("sealed_feedback_used") is False for row in rows),
        "no_freeform_patch_claim": all(
            row.get("claim_bearing_freeform_patch_evidence") is not True
            and row.get("freeform_patch_generation") is not True
            for row in rows
        ),
    }
    capability_checks: dict[str, bool]
    supported_claims: list[str] = []
    blocked_actions = [
        "do_not_claim_open_ended_debugging_generalization",
        "do_not_claim_production_autonomy",
        "do_not_claim_epoch_making_architecture_from_phase2as_boundary_audit",
    ]

    if claimed_capability == "symbolic_structural_patch_proposal":
        capability_checks = {
            "all_rows_symbolic_runtime_generator": all(
                item == "symbolic_runtime_generator" for item in classifications
            ),
            "no_rows_recorded_patch_control": class_counts["recorded_patch_control"] == 0,
            "no_rows_bounded_candidate_selection": class_counts[
                "bounded_patch_candidate_selection"
            ]
            == 0,
        }
        if all(generic_checks.values()) and all(capability_checks.values()):
            supported_claims.append(
                "phase2as_symbolic_runtime_generator_boundary_preserved"
            )
    elif claimed_capability == "bounded_patch_candidate_selection":
        capability_checks = {
            "all_rows_bounded_patch_candidate_selection": all(
                item == "bounded_patch_candidate_selection" for item in classifications
            ),
            "selection_accuracy_minimum_met": selection_accuracy >= min_selection_accuracy,
            "candidate_selection_does_not_claim_patch_authorship": all(
                row.get("claim_bearing_freeform_patch_evidence") is False
                for row in rows
            ),
        }
        if all(generic_checks.values()) and all(capability_checks.values()):
            supported_claims.append(
                "phase2as_bounded_patch_candidate_selection_boundary_preserved"
            )
    else:
        capability_checks = {
            "no_rows_symbolic_runtime_generator": class_counts["symbolic_runtime_generator"] == 0,
            "no_rows_recorded_patch_control": class_counts["recorded_patch_control"] == 0,
            "no_rows_restricted_symbolic_control": class_counts[
                "restricted_symbolic_control"
            ]
            == 0,
            "all_rows_have_explicit_learned_patch_generator": all(
                str(row.get("patch_generator") or "").startswith("learned_bounded_patch_")
                for row in rows
            ),
            "all_rows_claim_learned_patch_generation_evidence": all(
                row.get("claim_bearing_learned_patch_generation_evidence") is True
                for row in rows
            ),
        }
        blocked_actions.extend(
            [
                "do_not_relabel_symbolic_phase2ar_as_learned_patch_generation",
                "do_not_relabel_bounded_candidate_selection_as_patch_generation",
            ]
        )
        if all(generic_checks.values()) and all(capability_checks.values()):
            supported_claims.append("phase2as_learned_bounded_patch_generation_supported")

    passed = all(generic_checks.values()) and all(capability_checks.values())
    unsupported_claims = [
        "freeform_patch_generation",
        "open_ended_debugging_generalization",
        "production_autonomy",
        "epoch_making_architecture",
    ]
    if claimed_capability == "learned_patch_generation" and not passed:
        unsupported_claims.append("learned_patch_generation")
    return {
        "artifact_family": "phase2as_learned_repair_boundary_audit",
        "passed": passed,
        "claimed_capability": claimed_capability,
        "claim_boundary": (
            "Phase2AS classifies repair execution evidence by provenance. "
            "Symbolic generators, recorded patch controls, and bounded candidate "
            "selection are distinct from learned patch generation and cannot be "
            "retroactively relabeled as open-ended repair."
        ),
        "checks": {**generic_checks, **capability_checks},
        "metrics": {
            "row_count": len(rows),
            "success_count": len(successes),
            "success_rate": success_rate,
            "selection_accuracy": selection_accuracy,
            "evidence_class_counts": dict(sorted(class_counts.items())),
        },
        "supported_claims": supported_claims if passed else [],
        "unsupported_claims": sorted(set(unsupported_claims)),
        "blocked_actions": sorted(set(blocked_actions)),
        "inputs": {"execution_results_jsonl": str(Path(execution_results_jsonl))},
        "thresholds": {
            "min_rows": min_rows,
            "min_success_rate": min_success_rate,
            "min_selection_accuracy": min_selection_accuracy,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AS learned-vs-symbolic repair claim boundaries."
    )
    parser.add_argument("--execution-results-jsonl", required=True)
    parser.add_argument("--claimed-capability", choices=sorted(SUPPORTED_CLAIM_TYPES), required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--min-success-rate", type=float, default=0.85)
    parser.add_argument("--min-selection-accuracy", type=float, default=0.85)
    args = parser.parse_args()
    report = audit_phase2as_learned_repair_boundary(
        execution_results_jsonl=args.execution_results_jsonl,
        claimed_capability=args.claimed_capability,
        min_rows=args.min_rows,
        min_success_rate=args.min_success_rate,
        min_selection_accuracy=args.min_selection_accuracy,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
