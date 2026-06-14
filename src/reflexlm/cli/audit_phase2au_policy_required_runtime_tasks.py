from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


BENCHMARK_FAMILY = "phase2au_policy_required_runtime_delta"
REQUIRED_DIFFICULTY_AXES = {
    "ambiguous_nonliteral_semantic",
    "multi_file_interaction",
    "negative_constraint",
    "stateful_verification",
}
REQUIRED_KEYS = {
    "task_id",
    "benchmark_family",
    "split",
    "repo_origin",
    "repo_commit",
    "problem_statement",
    "evaluation_commands",
    "artifact_paths",
    "allowed_write_scope",
    "difficulty_axes",
    "runtime_visible_contract",
    "expected_policy",
    "candidate_policy_commands",
    "expected_repair_action",
    "sealed_feedback_used",
    "task_spec_sha256",
}
FORBIDDEN_MARKERS = (
    "candidate_0",
    "candidate_1",
    "candidate_2",
    "candidate slot",
    "gold",
    "sealed_v3",
    "sealed v3",
)
PARSER_ORACLE_PATTERNS = (
    re.compile(r"assert\s+(?:'[^']*'|\"[^\"]*\")\s+in\s+text"),
    re.compile(r"node\.attr\s*==\s*(?:'[^']*'|\"[^\"]*\")"),
    re.compile(r"phase2z_missing_[A-Za-z0-9_]+"),
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


def _runtime_visible_text(row: dict[str, Any]) -> str:
    payload = {
        "problem_statement": row.get("problem_statement"),
        "evaluation_commands": row.get("evaluation_commands"),
        "allowed_write_scope": row.get("allowed_write_scope"),
        "difficulty_axes": row.get("difficulty_axes"),
    }
    return _json_text(payload)


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


def _test_has_parser_oracle(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    text = path.read_text(encoding="utf-8-sig")
    return any(pattern.search(text) for pattern in PARSER_ORACLE_PATTERNS)


def _contract(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("runtime_visible_contract")
    return value if isinstance(value, dict) else {}


def _expected_policy(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("expected_policy")
    return value if isinstance(value, dict) else {}


def _candidate_policy_commands(row: dict[str, Any]) -> list[str]:
    values = row.get("candidate_policy_commands")
    return [str(item) for item in values] if isinstance(values, list) else []


def audit_phase2au_policy_required_runtime_tasks(
    *,
    tasks_jsonl: str | Path,
    dataset_root: str | Path,
    min_rows: int = 64,
    min_repo_origins: int = 4,
) -> dict[str, Any]:
    rows = _read_jsonl(tasks_jsonl)
    root = Path(dataset_root)
    repo_origins = {
        str(row.get("repo_origin") or "") for row in rows if row.get("repo_origin")
    }
    split_counts: dict[str, int] = {}
    axis_counts: dict[str, int] = {}
    missing_test_rows: list[str] = []
    parser_oracle_rows: list[str] = []
    for row in rows:
        split = str(row.get("split") or "")
        split_counts[split] = split_counts.get(split, 0) + 1
        for axis in row.get("difficulty_axes", []):
            axis_text = str(axis)
            axis_counts[axis_text] = axis_counts.get(axis_text, 0) + 1
        test_paths = _generated_test_paths(row, root)
        if not test_paths or not all(path.exists() for path in test_paths):
            missing_test_rows.append(str(row.get("task_id")))
        elif any(_test_has_parser_oracle(path) for path in test_paths):
            parser_oracle_rows.append(str(row.get("task_id")))

    checks = {
        "rows_minimum_met": len(rows) >= min_rows,
        "row_schema_complete": bool(rows) and all(REQUIRED_KEYS.issubset(row) for row in rows),
        "benchmark_family_expected": bool(rows)
        and {str(row.get("benchmark_family") or "") for row in rows} == {BENCHMARK_FAMILY},
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
        "no_sealed_feedback": bool(rows)
        and all(row.get("sealed_feedback_used") is False for row in rows),
        "no_candidate_or_gold_markers": not any(
            marker in _runtime_visible_text(row)
            for row in rows
            for marker in FORBIDDEN_MARKERS
        ),
        "required_difficulty_axes_present": REQUIRED_DIFFICULTY_AXES.issubset(axis_counts),
        "all_rows_policy_required_contract": bool(rows)
        and all(
            _contract(row).get("policy_required_runtime_delta") is True
            and _contract(row).get("no_policy_symbolic_control_expected_to_fail") is True
            and _contract(row).get("no_direct_text_membership_or_ast_attr_oracle") is True
            for row in rows
        ),
        "expected_policy_requires_learned_patch_heads": bool(rows)
        and all(
            _expected_policy(row).get("patch_proposal") == 1
            and _expected_policy(row).get("patch_operation") is not None
            and _expected_policy(row).get("patch_template") is not None
            for row in rows
        ),
        "nontrivial_policy_candidate_commands": bool(rows)
        and all(len(_candidate_policy_commands(row)) >= 2 for row in rows),
        "expected_repair_action_in_policy_candidates": bool(rows)
        and all(
            any(
                str(row.get("expected_repair_action") or "") in command
                for command in _candidate_policy_commands(row)
            )
            for row in rows
        ),
        "generated_tests_present": not missing_test_rows,
        "generated_tests_not_parser_oracle_solvable": not parser_oracle_rows,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2au_policy_required_runtime_task_gate",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "repo_origin_count": len(repo_origins),
            "split_counts": dict(sorted(split_counts.items())),
            "difficulty_axis_counts": dict(sorted(axis_counts.items())),
            "missing_test_rows": missing_test_rows[:20],
            "parser_oracle_rows": parser_oracle_rows[:20],
        },
        "claim_boundary": (
            "phase2au_policy_required_runtime_tasks_ready"
            if passed
            else "phase2au_policy_required_runtime_tasks_blocked"
        ),
        "supported_claim_if_passed": [
            "nonsealed_runtime_task_family_where_no_policy_symbolic_control_is_preregistered_to_fail"
        ],
        "unsupported_claims": [
            "claim_bearing_runtime_delta_before_execution",
            "learned_freeform_patch_generation",
            "sealed_cross_model_transfer",
            "production_autonomy",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "blocked_actions": []
        if passed
        else [
            "do_not_train_phase2au_until_task_gate_passes",
            "do_not_run_phase2au_package_delta_until_generated_tests_are_non_parser_oracle",
            "do_not_use_sealed_feedback_to_construct_phase2au",
        ],
        "inputs": {
            "tasks_jsonl": str(Path(tasks_jsonl)),
            "dataset_root": str(Path(dataset_root)),
        },
        "thresholds": {
            "min_rows": min_rows,
            "min_repo_origins": min_repo_origins,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AU policy-required runtime task specs."
    )
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--min-repo-origins", type=int, default=4)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2au_policy_required_runtime_tasks(
        tasks_jsonl=args.tasks_jsonl,
        dataset_root=args.dataset_root,
        min_rows=args.min_rows,
        min_repo_origins=args.min_repo_origins,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
