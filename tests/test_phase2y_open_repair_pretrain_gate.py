import json
from pathlib import Path

from reflexlm.cli.audit_phase2y_open_repair_pretrain_gate import (
    audit_phase2y_open_repair_pretrain_gate,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(index: int, mode: str, *, placeholder: bool = False, materialized: bool = True) -> dict:
    command = (
        "python -m pytest -q <generated_repair_test> --maxfail=1"
        if placeholder
        else f"python -m pytest -q tests/test_repair_{index}.py --maxfail=1"
    )
    return {
        "task_id": f"phase2y:holdout:{index:05d}",
        "repair_mode": mode,
        "evaluation_commands": [command],
        "runtime_visible_contract": {
            "must_materialize_real_tests_before_execution_evidence": not materialized
        },
    }


def _rows(*, placeholder: bool = False, materialized: bool = True) -> list[dict]:
    modes = [
        "nonliteral_symbolic_patch",
        "multi_test_selection",
        "rollback_required",
        "no_edit_control",
    ]
    return [
        _row(index, modes[index % len(modes)], placeholder=placeholder, materialized=materialized)
        for index in range(128)
    ]


def test_phase2y_pretrain_gate_blocks_task_specs_with_generated_test_placeholders(
    tmp_path: Path,
) -> None:
    report = audit_phase2y_open_repair_pretrain_gate(
        data_health_json=_write_json(tmp_path / "data.json", {"passed": True}),
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", _rows(placeholder=True, materialized=False)),
    )

    assert report["passed"] is False
    assert report["checks"]["no_generated_test_placeholders"] is False
    assert report["checks"]["real_test_materialization_contract_closed"] is False
    assert "do_not_start_phase2y_training_until_real_tests_are_materialized" in report["blocked_actions"]


def test_phase2y_pretrain_gate_accepts_materialized_real_test_specs(tmp_path: Path) -> None:
    report = audit_phase2y_open_repair_pretrain_gate(
        data_health_json=_write_json(tmp_path / "data.json", {"passed": True}),
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", _rows()),
    )

    assert report["passed"] is True
    assert report["ready_for_phase2y_training"] is True


def test_phase2y_pretrain_gate_rejects_failed_data_health(tmp_path: Path) -> None:
    report = audit_phase2y_open_repair_pretrain_gate(
        data_health_json=_write_json(tmp_path / "data.json", {"passed": False}),
        tasks_jsonl=_write_jsonl(tmp_path / "tasks.jsonl", _rows()),
    )

    assert report["passed"] is False
    assert report["checks"]["data_health_passed"] is False
