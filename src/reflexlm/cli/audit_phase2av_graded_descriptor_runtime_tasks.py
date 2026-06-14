from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2au_policy_required_runtime_tasks import _test_has_parser_oracle
from reflexlm.cli.build_phase2av_graded_descriptor_runtime_tasks import (
    BENCHMARK_FAMILY,
    CLAIM_BOUNDARY,
)


REQUIRED_KEYS = {
    "task_id",
    "benchmark_family",
    "claim_boundary",
    "split",
    "source_kind",
    "repo_origin",
    "repo_commit",
    "artifact_paths",
    "evaluation_commands",
    "repair_candidates",
    "candidate_policy_commands",
    "expected_repair_action",
    "expected_policy",
    "learned_patch_descriptor_target",
    "runtime_visible_contract",
    "difficulty_axes",
    "sealed_feedback_used",
    "task_spec_sha256",
}
FORBIDDEN_MARKERS = (
    "candidate_0",
    "candidate_1",
    "candidate slot",
    "slot id",
    "gold",
    "sealed_v3",
    "sealed v3",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


def _visible_text(row: dict[str, Any]) -> str:
    return _json_text(
        {
            "problem_statement": row.get("problem_statement"),
            "current_visible_text": row.get("current_visible_text"),
            "runtime_visible_evidence": row.get("runtime_visible_evidence"),
            "evaluation_commands": row.get("evaluation_commands"),
            "candidate_policy_commands": row.get("candidate_policy_commands"),
            "difficulty_axes": row.get("difficulty_axes"),
        }
    )


def _operation_exception_consistency(row: dict[str, Any]) -> tuple[bool, str | None]:
    operation = str(_expected_policy(row).get("patch_operation") or "")
    template = str(_expected_policy(row).get("patch_template") or "")
    evidence_text = _json_text(row.get("runtime_visible_evidence") or {})
    has_name_error = any(
        marker in evidence_text
        for marker in ("nameerror", "importerror", "modulenotfounderror", "is not defined")
    )
    has_attribute_error = any(
        marker in evidence_text
        for marker in ("attributeerror", "has no attribute")
    )
    if has_name_error and has_attribute_error:
        return False, "mixed_name_and_attribute_exception_evidence"
    if operation == "replace_attribute" or template == "call_attribute_restoration":
        if has_name_error:
            return False, "name_error_labeled_as_attribute_restoration"
    if operation == "insert_import" or template == "import_restoration":
        if has_attribute_error:
            return False, "attribute_error_labeled_as_import_restoration"
    return True, None


def _generated_test_paths(row: dict[str, Any], dataset_root: Path) -> list[Path]:
    artifacts = row.get("artifact_paths") if isinstance(row.get("artifact_paths"), dict) else {}
    raw_values: list[Any] = []
    for key in ("generated_test", "generated_tests", "test_artifacts"):
        value = artifacts.get(key)
        if isinstance(value, list):
            raw_values.extend(value)
        elif value:
            raw_values.append(value)
    paths: list[Path] = []
    for value in raw_values:
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        paths.append(path if path.is_absolute() else dataset_root / path)
    return paths


def _expected_policy(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("expected_policy")
    return value if isinstance(value, dict) else {}


def _target(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("learned_patch_descriptor_target")
    return value if isinstance(value, dict) else {}


def _contract(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("runtime_visible_contract")
    return value if isinstance(value, dict) else {}


def _repair_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    values = row.get("repair_candidates")
    return [value for value in values if isinstance(value, dict)] if isinstance(values, list) else []


def _candidate_policy_commands(row: dict[str, Any]) -> list[str]:
    values = row.get("candidate_policy_commands")
    return [str(value) for value in values] if isinstance(values, list) else []


def audit_phase2av_graded_descriptor_runtime_tasks(
    *,
    tasks_jsonl: str | Path,
    dataset_root: str | Path,
    min_rows: int = 64,
    min_repo_origins: int = 4,
    min_operation_template_pairs: int = 2,
) -> dict[str, Any]:
    rows = _read_jsonl(tasks_jsonl)
    root = Path(dataset_root)
    repo_origins = {str(row.get("repo_origin") or "") for row in rows if row.get("repo_origin")}
    split_counts: dict[str, int] = {}
    operation_counts: dict[str, int] = {}
    template_counts: dict[str, int] = {}
    pair_counts: dict[str, int] = {}
    missing_test_rows: list[str] = []
    parser_oracle_rows: list[str] = []
    exception_inconsistent_rows: list[dict[str, str]] = []

    for row in rows:
        split = str(row.get("split") or "")
        split_counts[split] = split_counts.get(split, 0) + 1
        operation = str(_expected_policy(row).get("patch_operation") or "")
        template = str(_expected_policy(row).get("patch_template") or "")
        operation_counts[operation] = operation_counts.get(operation, 0) + 1
        template_counts[template] = template_counts.get(template, 0) + 1
        pair_key = f"{operation}::{template}"
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        test_paths = _generated_test_paths(row, root)
        if not test_paths or not all(path.exists() and path.is_file() for path in test_paths):
            missing_test_rows.append(str(row.get("task_id")))
        elif any(_test_has_parser_oracle(path) for path in test_paths):
            parser_oracle_rows.append(str(row.get("task_id")))
        consistent, reason = _operation_exception_consistency(row)
        if not consistent:
            exception_inconsistent_rows.append(
                {"task_id": str(row.get("task_id")), "reason": str(reason)}
            )

    checks = {
        "rows_minimum_met": len(rows) >= min_rows,
        "row_schema_complete": bool(rows) and all(REQUIRED_KEYS.issubset(row) for row in rows),
        "benchmark_family_expected": bool(rows)
        and {str(row.get("benchmark_family") or "") for row in rows} == {BENCHMARK_FAMILY},
        "claim_boundary_expected": bool(rows)
        and {str(row.get("claim_boundary") or "") for row in rows} == {CLAIM_BOUNDARY},
        "repo_origin_minimum_met": len(repo_origins) >= min_repo_origins,
        "repo_commits_valid": bool(rows)
        and all(
            isinstance(row.get("repo_commit"), str)
            and COMMIT_RE.fullmatch(row["repo_commit"])
            for row in rows
        ),
        "task_spec_hashes_valid": bool(rows)
        and all(
            isinstance(row.get("task_spec_sha256"), str)
            and SHA256_RE.fullmatch(row["task_spec_sha256"])
            for row in rows
        ),
        "all_rows_public_repo": bool(rows) and all(row.get("source_kind") == "public_repo" for row in rows),
        "no_sealed_feedback": bool(rows) and all(row.get("sealed_feedback_used") is False for row in rows),
        "no_candidate_or_gold_markers": not any(
            marker in _visible_text(row) for row in rows for marker in FORBIDDEN_MARKERS
        ),
        "contract_blocks_forbidden_targets": bool(rows)
        and all(
            _contract(row).get("no_recorded_patch_text_target") is True
            and _contract(row).get("no_symbolic_generator_target") is True
            and _contract(row).get("no_freeform_patch_generation") is True
            and _contract(row).get("learned_descriptor_runtime_delta_required") is True
            for row in rows
        ),
        "descriptor_target_matches_policy": bool(rows)
        and all(
            _target(row).get("operation") == _expected_policy(row).get("patch_operation")
            and _target(row).get("after_fragment_template_id")
            == _expected_policy(row).get("patch_template")
            for row in rows
        ),
        "operation_template_diversity_met": len(pair_counts) >= min_operation_template_pairs,
        "nontrivial_repair_candidates": bool(rows)
        and all(len(_repair_candidates(row)) >= 2 for row in rows),
        "expected_repair_action_in_candidates": bool(rows)
        and all(
            str(row.get("expected_repair_action") or "") in {
                str(candidate.get("repair_action") or "") for candidate in _repair_candidates(row)
            }
            for row in rows
        ),
        "nontrivial_candidate_policy_commands": bool(rows)
        and all(len(_candidate_policy_commands(row)) >= 2 for row in rows),
        "generated_tests_present": not missing_test_rows,
        "generated_tests_not_parser_oracle_solvable": not parser_oracle_rows,
        "operation_exception_evidence_consistent": not exception_inconsistent_rows,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2av_graded_descriptor_runtime_data_health",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "repo_origin_count": len(repo_origins),
            "split_counts": dict(sorted(split_counts.items())),
            "operation_counts": dict(sorted(operation_counts.items())),
            "template_counts": dict(sorted(template_counts.items())),
            "operation_template_pair_counts": dict(sorted(pair_counts.items())),
            "operation_template_pair_count": len(pair_counts),
            "missing_test_rows": missing_test_rows[:20],
            "parser_oracle_rows": parser_oracle_rows[:20],
            "exception_inconsistent_rows": exception_inconsistent_rows[:20],
            "exception_inconsistent_row_count": len(exception_inconsistent_rows),
        },
        "thresholds": {
            "min_rows": min_rows,
            "min_repo_origins": min_repo_origins,
            "min_operation_template_pairs": min_operation_template_pairs,
        },
        "claim_boundary": (
            "phase2av_graded_descriptor_runtime_ready_for_pretrain_gate"
            if passed
            else "phase2av_graded_descriptor_runtime_blocked"
        ),
        "supported_claims": [
            "nonsealed_multi_template_descriptor_runtime_task_family_ready"
        ]
        if passed
        else [],
        "unsupported_claims": [
            "learned_freeform_patch_generation",
            "open_ended_debugging_generalization",
            "sealed_transfer_for_phase2au_package",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_train_phase2av",
            "do_not_claim_learned_descriptor_runtime_delta",
            "do_not_package_or_run_sealed_eval",
            "collect_non_parser_oracle_multi_template_public_runtime_tasks_first",
            "reject_operation_exception_inconsistent_rows_before_training",
        ],
        "inputs": {"tasks_jsonl": str(Path(tasks_jsonl)), "dataset_root": str(Path(dataset_root))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AV graded descriptor runtime task specs."
    )
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--min-repo-origins", type=int, default=4)
    parser.add_argument("--min-operation-template-pairs", type=int, default=2)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2av_graded_descriptor_runtime_tasks(
        tasks_jsonl=args.tasks_jsonl,
        dataset_root=args.dataset_root,
        min_rows=args.min_rows,
        min_repo_origins=args.min_repo_origins,
        min_operation_template_pairs=args.min_operation_template_pairs,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
