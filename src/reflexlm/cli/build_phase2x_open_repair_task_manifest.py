from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


AXIS_BY_TASK_FAMILY = {
    "dependency_or_import_mismatch": "dependency_or_environment_issue",
    "assertion_or_literal_mismatch": "ambiguous_traceback",
    "state_or_cache_mismatch": "stale_state_refresh",
    "multi_file_patch": "multi_file_patch",
    "hidden_side_effect_guard": "hidden_side_effect_guard",
}
DEFAULT_AXES = [
    "multi_file_patch",
    "ambiguous_traceback",
    "dependency_or_environment_issue",
    "stale_state_refresh",
    "hidden_side_effect_guard",
]


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    return [
        json.loads(line)
        for line in file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_commit(row: dict[str, Any]) -> str:
    commit = str(row.get("commit_hash") or "")
    return commit if commit else _sha256(row.get("trace_id"))[:40]


def _repo_origin(row: dict[str, Any]) -> str:
    return str(row.get("repo_url_or_origin") or row.get("repo_id") or "")


def _difficulty_axes(row: dict[str, Any], index: int) -> list[str]:
    difficulty = row.get("difficulty") if isinstance(row.get("difficulty"), dict) else {}
    task_family = str(difficulty.get("task_family") or "")
    axes = {AXIS_BY_TASK_FAMILY.get(task_family, DEFAULT_AXES[index % len(DEFAULT_AXES)])}
    if difficulty.get("repair_depth") not in {None, "one_edit"}:
        axes.add("multi_file_patch")
    if row.get("safety_controls") or row.get("repair_runtime"):
        axes.add("hidden_side_effect_guard")
    return sorted(axes)


def _evaluation_command(row: dict[str, Any]) -> str:
    candidates = row.get("repair_candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("verification_command"):
                return str(candidate["verification_command"])
    result = row.get("expected_repair_result")
    if isinstance(result, dict) and result.get("test_target"):
        return f"python -m pytest -q {result['test_target']} --maxfail=1"
    return "python -m pytest -q --maxfail=1"


def _allowed_write_scope(row: dict[str, Any]) -> str:
    evidence = row.get("runtime_visible_evidence")
    if isinstance(evidence, dict):
        changed = evidence.get("changed_files")
        if isinstance(changed, list) and changed:
            return ",".join(str(item) for item in changed)
    candidates = row.get("repair_candidates")
    if isinstance(candidates, list):
        scopes = [
            str(candidate.get("edit_scope"))
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("edit_scope")
        ]
        if scopes:
            return ",".join(sorted(set(scopes)))
    return "workspace"


def _problem_statement(row: dict[str, Any]) -> str:
    evidence = row.get("runtime_visible_evidence")
    difficulty = row.get("difficulty") if isinstance(row.get("difficulty"), dict) else {}
    watched = []
    failing = ""
    if isinstance(evidence, dict):
        watched = evidence.get("watched_files") if isinstance(evidence.get("watched_files"), list) else []
        failing = str(evidence.get("failing_test_target") or "")
    repo = str(row.get("repo_id") or _repo_origin(row))
    family = str(difficulty.get("task_family") or "public_repo_repair")
    return (
        "Repair the failing public-repository sandbox task without candidate-slot hints. "
        f"Repository={repo}; family={family}; failing_test={failing}; "
        f"watched_files={','.join(str(item) for item in watched)}."
    )


def build_phase2x_open_repair_task_manifest(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    split: str,
    max_rows: int | None = None,
) -> dict[str, Any]:
    source_rows = _read_jsonl(input_jsonl)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows[:max_rows] if max_rows else source_rows):
        task = {
            "task_id": f"phase2x:{split}:{index:05d}",
            "split": split,
            "task_family": "open_ended_repair",
            "repo_origin": _repo_origin(row),
            "repo_commit": _safe_commit(row),
            "task_spec_sha256": _sha256(
                {
                    "trace_id": row.get("trace_id"),
                    "repo": _repo_origin(row),
                    "runtime_visible_evidence": row.get("runtime_visible_evidence"),
                    "artifact_paths": row.get("artifact_paths"),
                }
            ),
            "problem_statement": _problem_statement(row),
            "difficulty_axes": _difficulty_axes(row, index),
            "requires_patch": True,
            "evaluation_command": _evaluation_command(row),
            "rollback_command": "git checkout -- .",
            "allowed_write_scope": _allowed_write_scope(row),
            "baseline_budget": {
                "max_commands": 30,
                "max_wall_clock_seconds": 1800,
            },
            "source_trace_id": row.get("trace_id"),
            "source_trace_construction_mode": row.get("trace_construction_mode"),
            "sealed_feedback_used": False,
        }
        rows.append(task)
    _write_jsonl(output_jsonl, rows)
    return {
        "artifact_family": "phase2x_open_repair_task_manifest_builder",
        "passed": bool(rows),
        "row_count": len(rows),
        "split": split,
        "input_jsonl": str(Path(input_jsonl)),
        "output_jsonl": str(Path(output_jsonl)),
        "task_manifest_sha256": _sha256(rows),
        "claim_boundary": "task_manifest_only_not_open_repair_result",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2X open-repair task manifest rows from public repair traces.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    report = build_phase2x_open_repair_task_manifest(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        split=args.split,
        max_rows=args.max_rows,
    )
    _write_json(args.summary_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
