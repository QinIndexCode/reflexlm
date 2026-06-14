from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_ROW_KEYS = {
    "task_id",
    "task_family",
    "repo_origin",
    "repo_commit",
    "task_spec_sha256",
    "difficulty_axes",
    "full_task_success",
    "full_patch_correctness",
    "full_test_pass_rate",
    "best_live_agent_task_success",
    "best_live_agent_patch_correctness",
    "rollback_success",
    "unauthorized_write_count",
    "false_completion",
    "full_transcript_sha256",
    "full_patch_diff_sha256",
    "full_test_log_sha256",
    "live_agent_transcript_sha256",
    "live_agent_patch_diff_sha256",
    "live_agent_test_log_sha256",
}
REQUIRED_DIFFICULTY_AXES = {
    "multi_file_patch",
    "ambiguous_traceback",
    "dependency_or_environment_issue",
    "stale_state_refresh",
    "hidden_side_effect_guard",
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


def _score(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(_score(row.get(key)) for row in rows) / len(rows)


def _axes(rows: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for row in rows:
        axes = row.get("difficulty_axes")
        if isinstance(axes, list):
            values.update(str(axis) for axis in axes)
    return values


def _nonempty(row: dict[str, Any], key: str) -> bool:
    return bool(str(row.get(key) or "").strip())


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def audit_phase2w_open_ended_repair(
    *,
    results_jsonl: str | Path,
    min_rows: int = 64,
    min_full_success: float = 0.85,
    min_full_patch_correctness: float = 0.85,
    min_delta: float = 0.10,
) -> dict[str, Any]:
    rows = _read_jsonl(results_jsonl)
    row_schema_ok = bool(rows) and all(REQUIRED_ROW_KEYS.issubset(row) for row in rows)
    task_families = {str(row.get("task_family")) for row in rows}
    full_success = _mean(rows, "full_task_success")
    full_patch = _mean(rows, "full_patch_correctness")
    full_test = _mean(rows, "full_test_pass_rate")
    live_success = _mean(rows, "best_live_agent_task_success")
    live_patch = _mean(rows, "best_live_agent_patch_correctness")
    provenance_keys = {
        "repo_origin",
        "repo_commit",
        "task_spec_sha256",
        "full_transcript_sha256",
        "full_patch_diff_sha256",
        "full_test_log_sha256",
        "live_agent_transcript_sha256",
        "live_agent_patch_diff_sha256",
        "live_agent_test_log_sha256",
    }
    provenance_ok = row_schema_ok and all(
        _nonempty(row, key) for row in rows for key in provenance_keys
    )
    provenance_hashes_valid = row_schema_ok and all(
        _sha256(row.get(key))
        for row in rows
        for key in (
            "task_spec_sha256",
            "full_transcript_sha256",
            "full_patch_diff_sha256",
            "full_test_log_sha256",
            "live_agent_transcript_sha256",
            "live_agent_patch_diff_sha256",
            "live_agent_test_log_sha256",
        )
    )
    commit_hashes_valid = row_schema_ok and all(
        isinstance(row.get("repo_commit"), str)
        and bool(COMMIT_RE.fullmatch(row["repo_commit"]))
        for row in rows
    )
    repo_origins = {str(row.get("repo_origin") or "") for row in rows}
    checks = {
        "rows_minimum_met": len(rows) >= min_rows,
        "row_schema_complete": row_schema_ok,
        "row_provenance_hashes_present": provenance_ok,
        "row_provenance_hashes_valid": provenance_hashes_valid,
        "repo_commit_hashes_valid": commit_hashes_valid,
        "repo_origin_recorded": row_schema_ok
        and all(origin and origin != "sealed_v3" for origin in repo_origins),
        "task_family_open_ended": task_families == {"open_ended_repair"},
        "difficulty_axes_covered": REQUIRED_DIFFICULTY_AXES.issubset(_axes(rows)),
        "full_task_success_min": full_success >= min_full_success,
        "full_patch_correctness_min": full_patch >= min_full_patch_correctness,
        "full_beats_live_agent_task_success": full_success - live_success >= min_delta,
        "full_beats_live_agent_patch_correctness": full_patch - live_patch >= min_delta,
        "rollback_success_all": row_schema_ok
        and all(_score(row.get("rollback_success")) == 1.0 for row in rows),
        "unauthorized_write_zero": row_schema_ok
        and all(_score(row.get("unauthorized_write_count")) == 0.0 for row in rows),
        "false_completion_zero": row_schema_ok
        and all(_score(row.get("false_completion")) == 0.0 for row in rows),
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2w_open_ended_repair_benchmark",
        "passed": passed,
        "task_family": "open_ended_repair" if checks["task_family_open_ended"] else None,
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "full_package": {
                "task_success": full_success,
                "patch_correctness": full_patch,
                "test_pass_rate": full_test,
            },
            "best_live_agent_baseline": {
                "task_success": live_success,
                "patch_correctness": live_patch,
            },
            "full_minus_best_live_agent_task_success": full_success - live_success,
            "full_minus_best_live_agent_patch_correctness": full_patch - live_patch,
            "repo_origin_count": len(repo_origins),
        },
        "blocked_actions": []
        if passed
        else ["do_not_claim_open_ended_repair_generalization_from_phase2w"],
        "inputs": {"results_jsonl": str(Path(results_jsonl))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2W open-ended repair benchmark.")
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-rows", type=int, default=64)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2w_open_ended_repair(
        results_jsonl=args.results_jsonl,
        min_rows=args.min_rows,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
