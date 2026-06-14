from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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


def _trace_index(paths: list[str | Path]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(path):
            trace_id = str(row.get("trace_id") or "")
            if trace_id:
                index[trace_id] = row
    return index


def _test_target(trace: dict[str, Any]) -> str:
    result = trace.get("expected_repair_result")
    if isinstance(result, dict) and result.get("test_target"):
        return str(result["test_target"])
    evidence = trace.get("runtime_visible_evidence")
    if isinstance(evidence, dict) and evidence.get("failing_test_target"):
        return str(evidence["failing_test_target"])
    return ""


def _materialized_commands(row: dict[str, Any], target: str) -> list[str]:
    mode = str(row.get("repair_mode") or "")
    base = f"python -m pytest -q {target} --maxfail=1"
    if mode == "multi_test_selection":
        root = str(Path(target).parent).replace("\\", "/")
        return [base, f"python -m pytest -q {root} --maxfail=2"]
    return [base]


def materialize_phase2y_open_repair_tests(
    *,
    tasks_jsonl: str | Path,
    source_traces_jsonl: list[str | Path],
    output_jsonl: str | Path,
) -> dict[str, Any]:
    rows = _read_jsonl(tasks_jsonl)
    traces = _trace_index(source_traces_jsonl)
    output_rows: list[dict[str, Any]] = []
    missing_trace_rows: list[str] = []
    missing_test_rows: list[str] = []
    for row in rows:
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        trace_id = str(source.get("source_trace_id") or "")
        trace = traces.get(trace_id)
        if not trace:
            missing_trace_rows.append(str(row.get("task_id")))
            output_rows.append(row)
            continue
        target = _test_target(trace)
        if not target:
            missing_test_rows.append(str(row.get("task_id")))
            output_rows.append(row)
            continue
        materialized = dict(row)
        materialized["evaluation_commands"] = _materialized_commands(row, target)
        contract = dict(materialized.get("runtime_visible_contract") or {})
        contract["must_materialize_real_tests_before_execution_evidence"] = False
        contract["real_test_target_materialized"] = True
        materialized["runtime_visible_contract"] = contract
        materialized["materialized_test_target"] = target
        materialized["materialization_source_trace_id"] = trace_id
        materialized["task_spec_sha256"] = _sha256(
            {
                "base_task_spec_sha256": row.get("task_spec_sha256"),
                "materialized_test_target": target,
                "evaluation_commands": materialized["evaluation_commands"],
            }
        )
        output_rows.append(materialized)
    _write_jsonl(output_jsonl, output_rows)
    materialized_count = len(output_rows) - len(missing_trace_rows) - len(missing_test_rows)
    return {
        "artifact_family": "phase2y_open_repair_test_materialization",
        "passed": materialized_count == len(rows) and bool(rows),
        "row_count": len(rows),
        "materialized_count": materialized_count,
        "missing_trace_rows": missing_trace_rows[:20],
        "missing_test_rows": missing_test_rows[:20],
        "output_jsonl": str(Path(output_jsonl)),
        "source_traces_jsonl": [str(Path(path)) for path in source_traces_jsonl],
        "tasks_jsonl": str(Path(tasks_jsonl)),
        "materialized_manifest_sha256": _sha256(output_rows),
        "claim_boundary": "phase2y_test_commands_materialized_not_execution_result",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Phase2Y test commands from non-sealed source traces.")
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--source-traces-jsonl", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()
    report = materialize_phase2y_open_repair_tests(
        tasks_jsonl=args.tasks_jsonl,
        source_traces_jsonl=args.source_traces_jsonl,
        output_jsonl=args.output_jsonl,
    )
    _write_json(args.summary_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
