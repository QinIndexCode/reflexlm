import json
from pathlib import Path

from reflexlm.cli.audit_phase2y_open_repair_pressure_tasks import (
    audit_phase2y_open_repair_pressure_tasks,
)
from reflexlm.cli.build_phase2y_open_repair_pressure_tasks import (
    build_phase2y_open_repair_pressure_tasks,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _source_task(index: int) -> dict:
    return {
        "task_id": f"phase2x:holdout:{index:05d}",
        "split": "holdout",
        "task_family": "open_ended_repair",
        "repo_origin": f"https://github.com/example/repo-{index % 4}.git",
        "repo_commit": "a" * 40,
        "task_spec_sha256": "b" * 64,
        "problem_statement": "Repair the public-repository sandbox task.",
        "difficulty_axes": ["ambiguous_traceback"],
        "requires_patch": True,
        "evaluation_command": f"python -m pytest -q tests/test_{index}.py --maxfail=1",
        "rollback_command": "git checkout -- .",
        "allowed_write_scope": f"src/module_{index}.py",
        "source_trace_id": f"trace-{index}",
        "sealed_feedback_used": False,
    }


def _build(tmp_path: Path, count: int = 128) -> Path:
    source = _write_jsonl(tmp_path / "phase2x.tasks.jsonl", [_source_task(i) for i in range(count)])
    output = tmp_path / "phase2y.tasks.jsonl"
    report = build_phase2y_open_repair_pressure_tasks(
        input_tasks_jsonl=[source],
        output_jsonl=output,
        split="holdout",
    )
    assert report["passed"] is True
    return output


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_phase2y_pressure_builder_outputs_auditable_task_specs(tmp_path: Path) -> None:
    output = _build(tmp_path)

    report = audit_phase2y_open_repair_pressure_tasks(tasks_jsonl=output)

    assert report["passed"] is True
    assert report["claim_boundary"] == "phase2y_task_specs_ready_not_execution_evidence"
    assert report["metrics"]["mode_counts"] == {
        "multi_test_selection": 32,
        "no_edit_control": 32,
        "nonliteral_symbolic_patch": 32,
        "rollback_required": 32,
    }


def test_phase2y_pressure_audit_rejects_candidate_or_gold_markers(tmp_path: Path) -> None:
    output = _build(tmp_path)
    rows = _rows(output)
    rows[0]["problem_statement"] = "Use candidate_0 as the gold answer"
    _write_jsonl(output, rows)

    report = audit_phase2y_open_repair_pressure_tasks(tasks_jsonl=output)

    assert report["passed"] is False
    assert report["checks"]["no_candidate_or_gold_markers"] is False


def test_phase2y_pressure_audit_rejects_bad_no_edit_control(tmp_path: Path) -> None:
    output = _build(tmp_path)
    rows = _rows(output)
    target = next(row for row in rows if row["repair_mode"] == "no_edit_control")
    target["expected_policy"]["patch_proposal"] = 1
    target["requires_patch"] = True
    _write_jsonl(output, rows)

    report = audit_phase2y_open_repair_pressure_tasks(tasks_jsonl=output)

    assert report["passed"] is False
    assert report["checks"]["no_edit_controls_deny_patch"] is False


def test_phase2y_pressure_audit_rejects_missing_multi_test_pressure(tmp_path: Path) -> None:
    output = _build(tmp_path)
    rows = _rows(output)
    target = next(row for row in rows if row["repair_mode"] == "multi_test_selection")
    target["evaluation_commands"] = target["evaluation_commands"][:1]
    _write_jsonl(output, rows)

    report = audit_phase2y_open_repair_pressure_tasks(tasks_jsonl=output)

    assert report["passed"] is False
    assert report["checks"]["multi_test_rows_have_multiple_commands"] is False


def test_phase2y_pressure_audit_rejects_literal_nonliteral_rows(tmp_path: Path) -> None:
    output = _build(tmp_path)
    rows = _rows(output)
    target = next(row for row in rows if row["repair_mode"] == "nonliteral_symbolic_patch")
    target["patch_type"] = "bounded_runtime_patch"
    _write_jsonl(output, rows)

    report = audit_phase2y_open_repair_pressure_tasks(tasks_jsonl=output)

    assert report["passed"] is False
    assert report["checks"]["nonliteral_rows_are_not_literal_patch"] is False
