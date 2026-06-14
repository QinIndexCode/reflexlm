from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding="utf-8-sig"))


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


def _commands(row: dict[str, Any]) -> list[str]:
    commands = row.get("evaluation_commands")
    return commands if isinstance(commands, list) else []


def _has_placeholder_command(row: dict[str, Any]) -> bool:
    return any("<generated_repair_test>" in str(command) for command in _commands(row))


def _contract_requires_materialization(row: dict[str, Any]) -> bool:
    contract = row.get("runtime_visible_contract")
    return (
        isinstance(contract, dict)
        and contract.get("must_materialize_real_tests_before_execution_evidence") is True
    )


def audit_phase2y_open_repair_pretrain_gate(
    *,
    data_health_json: str | Path,
    tasks_jsonl: str | Path,
) -> dict[str, Any]:
    data_health = _read_json(data_health_json)
    rows = _read_jsonl(tasks_jsonl)
    placeholder_rows = [
        str(row.get("task_id")) for row in rows if _has_placeholder_command(row)
    ]
    materialization_required_rows = [
        str(row.get("task_id")) for row in rows if _contract_requires_materialization(row)
    ]
    mode_counts: dict[str, int] = {}
    for row in rows:
        mode = str(row.get("repair_mode") or "")
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    checks = {
        "data_health_passed": data_health.get("passed") is True,
        "task_rows_present": bool(rows),
        "no_generated_test_placeholders": not placeholder_rows,
        "real_test_materialization_contract_closed": not materialization_required_rows,
        "all_modes_present": all(
            mode_counts.get(mode, 0) > 0
            for mode in (
                "nonliteral_symbolic_patch",
                "multi_test_selection",
                "rollback_required",
                "no_edit_control",
            )
        ),
    }
    ready = all(checks.values())
    return {
        "artifact_family": "phase2y_open_repair_pretrain_gate",
        "passed": ready,
        "ready_for_phase2y_training": ready,
        "checks": checks,
        "metrics": {
            "row_count": len(rows),
            "mode_counts": dict(sorted(mode_counts.items())),
            "placeholder_command_rows": placeholder_rows[:20],
            "materialization_required_rows": materialization_required_rows[:20],
        },
        "claim_boundary": (
            "phase2y_real_test_materialized_training_ready"
            if ready
            else "phase2y_task_specs_only_training_blocked"
        ),
        "blocked_actions": []
        if ready
        else [
            "do_not_start_phase2y_training_until_real_tests_are_materialized",
            "do_not_use_phase2y_task_specs_as_execution_or_training_evidence",
        ],
        "required_next_artifact": None
        if ready
        else "phase2y_real_test_materialization_manifest.json",
        "inputs": {
            "data_health_json": str(Path(data_health_json)),
            "tasks_jsonl": str(Path(tasks_jsonl)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2Y pretrain readiness.")
    parser.add_argument("--data-health-json", required=True)
    parser.add_argument("--tasks-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2y_open_repair_pretrain_gate(
        data_health_json=args.data_health_json,
        tasks_jsonl=args.tasks_jsonl,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
