from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_MODES = {
    "nonliteral_symbolic_patch",
    "multi_test_selection",
    "rollback_required",
    "no_edit_control",
}
REQUIRED_KEYS = {
    "task_id",
    "benchmark_family",
    "split",
    "repo_origin",
    "repo_commit",
    "problem_statement",
    "source",
    "repair_mode",
    "requires_patch",
    "patch_type",
    "evaluation_commands",
    "rollback_command",
    "allowed_write_scope",
    "difficulty_axes",
    "expected_policy",
    "runtime_visible_contract",
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


def _json_text(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True).lower()


def _runtime_visible_text(row: dict[str, Any]) -> str:
    payload = {
        "problem_statement": row.get("problem_statement"),
        "evaluation_commands": row.get("evaluation_commands"),
        "allowed_write_scope": row.get("allowed_write_scope"),
        "difficulty_axes": row.get("difficulty_axes"),
        "repair_mode": row.get("repair_mode"),
        "patch_type": row.get("patch_type"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()


def _mode_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        mode = str(row.get("repair_mode") or "")
        counts[mode] = counts.get(mode, 0) + 1
    return dict(sorted(counts.items()))


def _commands(row: dict[str, Any]) -> list[str]:
    values = row.get("evaluation_commands")
    return values if isinstance(values, list) else []


def _materialization_contract_consistent(row: dict[str, Any]) -> bool:
    contract = row.get("runtime_visible_contract")
    if not isinstance(contract, dict):
        return False
    requires_materialization = (
        contract.get("must_materialize_real_tests_before_execution_evidence") is True
    )
    materialized = contract.get("real_test_target_materialized") is True
    return requires_materialization != materialized


def audit_phase2y_open_repair_pressure_tasks(
    *,
    tasks_jsonl: str | Path,
    min_rows: int = 128,
    min_repo_origins: int = 4,
) -> dict[str, Any]:
    rows = _read_jsonl(tasks_jsonl)
    mode_counts = _mode_counts(rows)
    repo_origins = {str(row.get("repo_origin") or "") for row in rows if row.get("repo_origin")}
    benchmark_families = {str(row.get("benchmark_family") or "") for row in rows}
    row_schema_complete = bool(rows) and all(REQUIRED_KEYS.issubset(row) for row in rows)
    forbidden_rows = [
        str(row.get("task_id"))
        for row in rows
        if any(marker in _runtime_visible_text(row) for marker in FORBIDDEN_MARKERS)
    ]
    no_edit_rows = [row for row in rows if row.get("repair_mode") == "no_edit_control"]
    rollback_rows = [row for row in rows if row.get("repair_mode") == "rollback_required"]
    nonliteral_rows = [row for row in rows if row.get("repair_mode") == "nonliteral_symbolic_patch"]
    multitest_rows = [row for row in rows if row.get("repair_mode") == "multi_test_selection"]

    checks = {
        "rows_minimum_met": len(rows) >= min_rows,
        "row_schema_complete": row_schema_complete,
        "benchmark_family_expected": benchmark_families
        == {"phase2y_open_repair_generalization_pressure"},
        "required_modes_present": REQUIRED_MODES.issubset(set(mode_counts)),
        "repo_origin_minimum_met": len(repo_origins) >= min_repo_origins,
        "repo_commits_valid": row_schema_complete
        and all(isinstance(row.get("repo_commit"), str) and COMMIT_RE.fullmatch(row["repo_commit"]) for row in rows),
        "task_spec_hashes_valid": row_schema_complete
        and all(isinstance(row.get("task_spec_sha256"), str) and SHA256_RE.fullmatch(row["task_spec_sha256"]) for row in rows),
        "no_sealed_feedback": row_schema_complete
        and all(row.get("sealed_feedback_used") is False for row in rows),
        "no_candidate_or_gold_markers": not forbidden_rows,
        "nonliteral_rows_are_not_literal_patch": bool(nonliteral_rows)
        and all(row.get("patch_type") == "nonliteral_symbolic" for row in nonliteral_rows),
        "multi_test_rows_have_multiple_commands": bool(multitest_rows)
        and all(len(_commands(row)) >= 2 for row in multitest_rows),
        "rollback_rows_require_rollback_policy": bool(rollback_rows)
        and all(
            isinstance(row.get("expected_policy"), dict)
            and row["expected_policy"].get("rollback_safety") == 1
            and str(row.get("rollback_command") or "").strip()
            for row in rollback_rows
        ),
        "no_edit_controls_deny_patch": bool(no_edit_rows)
        and all(
            row.get("requires_patch") is False
            and isinstance(row.get("expected_policy"), dict)
            and row["expected_policy"].get("patch_proposal") == 0
            for row in no_edit_rows
        ),
        "real_test_materialization_contract_consistent": row_schema_complete
        and all(_materialization_contract_consistent(row) for row in rows),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2y_open_repair_pressure_task_data_health",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "repo_origin_count": len(repo_origins),
            "mode_counts": mode_counts,
            "forbidden_marker_rows": forbidden_rows[:20],
        },
        "claim_boundary": (
            "phase2y_task_specs_ready_not_execution_evidence"
            if passed
            else "phase2y_task_specs_not_ready"
        ),
        "blocked_actions": []
        if passed
        else [
            "do_not_train_phase2y_until_task_data_health_passes",
            "do_not_use_phase2y_task_specs_as_execution_evidence",
        ],
        "inputs": {"tasks_jsonl": str(Path(tasks_jsonl))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2Y open-repair pressure task specs.")
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=128)
    parser.add_argument("--min-repo-origins", type=int, default=4)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2y_open_repair_pressure_tasks(
        tasks_jsonl=args.tasks_jsonl,
        min_rows=args.min_rows,
        min_repo_origins=args.min_repo_origins,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
