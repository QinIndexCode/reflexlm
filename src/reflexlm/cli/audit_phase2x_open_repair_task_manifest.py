from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_DIFFICULTY_AXES = {
    "multi_file_patch",
    "ambiguous_traceback",
    "dependency_or_environment_issue",
    "stale_state_refresh",
    "hidden_side_effect_guard",
}
REQUIRED_ROW_KEYS = {
    "task_id",
    "split",
    "task_family",
    "repo_origin",
    "repo_commit",
    "task_spec_sha256",
    "problem_statement",
    "difficulty_axes",
    "requires_patch",
    "evaluation_command",
    "rollback_command",
    "allowed_write_scope",
    "baseline_budget",
}
FORBIDDEN_TEXT_MARKERS = {
    "candidate_0",
    "candidate_1",
    "candidate_2",
    "candidate_3",
    "gold_slot",
    "gold_label",
    "sealed_v3",
    "phase2i_external_trace_v3",
}
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


def _text_blob(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False).lower()


def _nonempty(row: dict[str, Any], key: str) -> bool:
    return bool(str(row.get(key) or "").strip())


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _axes(rows: list[dict[str, Any]]) -> set[str]:
    axes: set[str] = set()
    for row in rows:
        value = row.get("difficulty_axes")
        if isinstance(value, list):
            axes.update(str(item) for item in value)
    return axes


def _splits(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split") or "")
        counts[split] = counts.get(split, 0) + 1
    return counts


def audit_phase2x_open_repair_task_manifest(
    *,
    tasks_jsonl: str | Path,
    min_rows: int = 64,
    min_holdout_rows: int = 32,
    min_repo_origins: int = 4,
) -> dict[str, Any]:
    rows = _read_jsonl(tasks_jsonl)
    row_schema_ok = bool(rows) and all(REQUIRED_ROW_KEYS.issubset(row) for row in rows)
    repo_origins = {str(row.get("repo_origin") or "") for row in rows}
    split_counts = _splits(rows)
    markers_found = sorted(
        marker
        for marker in FORBIDDEN_TEXT_MARKERS
        if any(marker in _text_blob(row) for row in rows)
    )
    budget_complete = row_schema_ok and all(
        isinstance(row.get("baseline_budget"), dict)
        and row["baseline_budget"].get("max_commands")
        and row["baseline_budget"].get("max_wall_clock_seconds")
        for row in rows
    )
    hashes_valid = row_schema_ok and all(
        _sha256(row.get("task_spec_sha256"))
        and isinstance(row.get("repo_commit"), str)
        and bool(COMMIT_RE.fullmatch(row["repo_commit"]))
        for row in rows
    )
    checks = {
        "rows_minimum_met": len(rows) >= min_rows,
        "holdout_rows_minimum_met": split_counts.get("holdout", 0) >= min_holdout_rows,
        "row_schema_complete": row_schema_ok,
        "task_hashes_valid": hashes_valid,
        "task_family_open_ended": row_schema_ok
        and {str(row.get("task_family")) for row in rows} == {"open_ended_repair"},
        "repo_origin_count_minimum_met": len(repo_origins) >= min_repo_origins,
        "repo_origin_recorded": row_schema_ok
        and all(origin and origin != "sealed_v3" for origin in repo_origins),
        "difficulty_axes_covered": REQUIRED_DIFFICULTY_AXES.issubset(_axes(rows)),
        "requires_patch_all": row_schema_ok
        and all(row.get("requires_patch") is True for row in rows),
        "commands_present": row_schema_ok
        and all(_nonempty(row, "evaluation_command") and _nonempty(row, "rollback_command") for row in rows),
        "write_scope_present": row_schema_ok
        and all(_nonempty(row, "allowed_write_scope") for row in rows),
        "baseline_budget_complete": budget_complete,
        "forbidden_markers_absent": not markers_found,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2x_open_repair_task_manifest_audit",
        "passed": passed,
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "split_counts": split_counts,
            "repo_origin_count": len(repo_origins),
            "difficulty_axes": sorted(_axes(rows)),
            "forbidden_markers_found": markers_found,
        },
        "blocked_actions": []
        if passed
        else [
            "do_not_run_phase2x_open_repair_training_or_claim_open_ended_repair"
        ],
        "inputs": {"tasks_jsonl": str(Path(tasks_jsonl))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2X open-ended repair task manifest.")
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--min-holdout-rows", type=int, default=32)
    parser.add_argument("--min-repo-origins", type=int, default=4)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2x_open_repair_task_manifest(
        tasks_jsonl=args.tasks_jsonl,
        min_rows=args.min_rows,
        min_holdout_rows=args.min_holdout_rows,
        min_repo_origins=args.min_repo_origins,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
