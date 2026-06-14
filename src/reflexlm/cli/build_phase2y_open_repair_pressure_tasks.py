from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPAIR_MODES = (
    "nonliteral_symbolic_patch",
    "multi_test_selection",
    "rollback_required",
    "no_edit_control",
)


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


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows),
        encoding="utf-8",
    )


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_tasks(inputs: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_path in inputs:
        rows.extend(_read_jsonl(input_path))
    return rows


def _commands_for_mode(task: dict[str, Any], mode: str) -> list[str]:
    command = str(task.get("evaluation_command") or "python -m pytest -q --maxfail=1")
    if mode == "multi_test_selection":
        # The second command is intentionally broad but still allowlisted; later
        # execution must materialize real tests before this can be result evidence.
        return [
            command,
            "python -m pytest -q --maxfail=2",
        ]
    return [command]


def _expected_policy(mode: str) -> dict[str, int]:
    if mode == "no_edit_control":
        return {
            "patch_proposal": 0,
            "bounded_edit_scope": 1,
            "rollback_safety": 0,
            "stop_condition": 1,
        }
    if mode == "rollback_required":
        return {
            "patch_proposal": 1,
            "bounded_edit_scope": 1,
            "rollback_safety": 1,
            "stop_condition": 0,
        }
    return {
        "patch_proposal": 1,
        "bounded_edit_scope": 1,
        "rollback_safety": 1,
        "stop_condition": 0,
    }


def _difficulty_axes(task: dict[str, Any], mode: str) -> list[str]:
    axes = {str(axis) for axis in task.get("difficulty_axes", []) if str(axis).strip()}
    if mode == "nonliteral_symbolic_patch":
        axes.add("nonliteral_symbolic_patch")
    elif mode == "multi_test_selection":
        axes.add("multi_test_selection")
    elif mode == "rollback_required":
        axes.add("rollback_required")
    elif mode == "no_edit_control":
        axes.add("no_edit_control")
    return sorted(axes)


def _phase2y_task(task: dict[str, Any], *, index: int, split: str) -> dict[str, Any]:
    mode = REPAIR_MODES[index % len(REPAIR_MODES)]
    requires_patch = mode != "no_edit_control"
    source = {
        "source_task_id": task.get("task_id"),
        "source_task_spec_sha256": task.get("task_spec_sha256"),
        "source_trace_id": task.get("source_trace_id"),
        "source_split": task.get("split"),
    }
    payload = {
        "task_id": f"phase2y:{split}:{index:05d}",
        "benchmark_family": "phase2y_open_repair_generalization_pressure",
        "split": split,
        "repo_origin": task.get("repo_origin"),
        "repo_commit": task.get("repo_commit"),
        "repo_origin_disjoint_group": task.get("repo_origin"),
        "problem_statement": (
            "Resolve the public-repository repair task using only runtime-visible "
            f"test output, watched files, and the declared bounded write scope; mode={mode}."
        ),
        "source": source,
        "repair_mode": mode,
        "requires_patch": requires_patch,
        "patch_type": "none" if not requires_patch else (
            "nonliteral_symbolic" if mode == "nonliteral_symbolic_patch" else "bounded_runtime_patch"
        ),
        "evaluation_commands": _commands_for_mode(task, mode),
        "rollback_command": str(task.get("rollback_command") or "git checkout -- ."),
        "allowed_write_scope": task.get("allowed_write_scope"),
        "difficulty_axes": _difficulty_axes(task, mode),
        "expected_policy": _expected_policy(mode),
        "runtime_visible_contract": {
            "no_candidate_slot_marker": True,
            "no_gold_hint": True,
            "no_sealed_feedback": True,
            "must_materialize_real_tests_before_execution_evidence": True,
        },
        "claim_boundary": "phase2y_task_spec_only_not_execution_result",
        "sealed_feedback_used": False,
    }
    payload["task_spec_sha256"] = _sha256(
        {
            "repo_origin": payload["repo_origin"],
            "repo_commit": payload["repo_commit"],
            "source": source,
            "repair_mode": mode,
            "evaluation_commands": payload["evaluation_commands"],
            "allowed_write_scope": payload["allowed_write_scope"],
        }
    )
    return payload


def build_phase2y_open_repair_pressure_tasks(
    *,
    input_tasks_jsonl: list[str | Path],
    output_jsonl: str | Path,
    split: str,
    max_rows: int | None = None,
) -> dict[str, Any]:
    source_rows = _load_tasks(input_tasks_jsonl)
    selected = source_rows[:max_rows] if max_rows else source_rows
    rows = [_phase2y_task(row, index=index, split=split) for index, row in enumerate(selected)]
    _write_jsonl(output_jsonl, rows)
    mode_counts: dict[str, int] = {}
    for row in rows:
        mode = str(row.get("repair_mode"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    repo_origins = {str(row.get("repo_origin") or "") for row in rows if row.get("repo_origin")}
    return {
        "artifact_family": "phase2y_open_repair_pressure_task_builder",
        "passed": bool(rows),
        "row_count": len(rows),
        "split": split,
        "mode_counts": dict(sorted(mode_counts.items())),
        "repo_origin_count": len(repo_origins),
        "output_jsonl": str(Path(output_jsonl)),
        "input_tasks_jsonl": [str(Path(path)) for path in input_tasks_jsonl],
        "task_manifest_sha256": _sha256(rows),
        "claim_boundary": "phase2y_task_spec_only_not_execution_result",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Phase2Y open-repair pressure task specs from non-sealed public task manifests."
    )
    parser.add_argument("--input-tasks-jsonl", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    report = build_phase2y_open_repair_pressure_tasks(
        input_tasks_jsonl=args.input_tasks_jsonl,
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
