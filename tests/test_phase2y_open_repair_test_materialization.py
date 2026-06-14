import json
from pathlib import Path

from reflexlm.cli.audit_phase2y_open_repair_pretrain_gate import (
    audit_phase2y_open_repair_pretrain_gate,
)
from reflexlm.cli.materialize_phase2y_open_repair_tests import (
    materialize_phase2y_open_repair_tests,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _task(index: int, mode: str = "nonliteral_symbolic_patch") -> dict:
    return {
        "task_id": f"phase2y:holdout:{index:05d}",
        "repair_mode": mode,
        "evaluation_commands": ["python -m pytest -q <generated_repair_test> --maxfail=1"],
        "runtime_visible_contract": {
            "must_materialize_real_tests_before_execution_evidence": True
        },
        "task_spec_sha256": "a" * 64,
        "source": {"source_trace_id": f"trace-{index}"},
    }


def _trace(index: int) -> dict:
    return {
        "trace_id": f"trace-{index}",
        "expected_repair_result": {
            "test_target": f"phase2s_repair_tests/test_case_{index}.py"
        },
    }


def test_phase2y_materializes_generated_test_placeholder_commands(tmp_path: Path) -> None:
    tasks = [_task(0), _task(1, mode="multi_test_selection")]
    traces = [_trace(0), _trace(1)]
    output = tmp_path / "materialized.tasks.jsonl"

    report = materialize_phase2y_open_repair_tests(
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", tasks),
        source_traces_jsonl=[_write_jsonl(tmp_path / "traces.jsonl", traces)],
        output_jsonl=output,
    )

    assert report["passed"] is True
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert "<generated_repair_test>" not in json.dumps(rows)
    assert rows[0]["runtime_visible_contract"]["must_materialize_real_tests_before_execution_evidence"] is False
    assert len(rows[1]["evaluation_commands"]) == 2


def test_phase2y_materialized_tasks_pass_pretrain_gate(tmp_path: Path) -> None:
    modes = [
        "nonliteral_symbolic_patch",
        "multi_test_selection",
        "rollback_required",
        "no_edit_control",
    ]
    tasks = [_task(index, modes[index % len(modes)]) for index in range(128)]
    traces = [_trace(index) for index in range(128)]
    output = tmp_path / "materialized.tasks.jsonl"
    materialize_phase2y_open_repair_tests(
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", tasks),
        source_traces_jsonl=[_write_jsonl(tmp_path / "traces.jsonl", traces)],
        output_jsonl=output,
    )

    report = audit_phase2y_open_repair_pretrain_gate(
        data_health_json=_write_json(tmp_path / "data.json", {"passed": True}),
        tasks_jsonl=output,
    )

    assert report["passed"] is True


def test_phase2y_materialization_reports_missing_trace(tmp_path: Path) -> None:
    report = materialize_phase2y_open_repair_tests(
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", [_task(0)]),
        source_traces_jsonl=[_write_jsonl(tmp_path / "traces.jsonl", [])],
        output_jsonl=tmp_path / "out.jsonl",
    )

    assert report["passed"] is False
    assert report["missing_trace_rows"] == ["phase2y:holdout:00000"]
