from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LOW_LEVEL_TASKS = {
    "blocking_input_detection",
    "process_hang_detection",
    "dangerous_action_interception",
    "external_file_change_reflex",
    "common_error_recovery_routine",
}


def _load_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _iter_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _metric_mean(eval_payload: dict[str, Any], task: str, metric: str) -> float | None:
    task_payload = eval_payload.get("metrics", {}).get("per_task", {}).get(task)
    if not isinstance(task_payload, dict):
        return None
    metric_payload = task_payload.get("metrics", {}).get(metric)
    if isinstance(metric_payload, dict):
        value = metric_payload.get("mean")
        return float(value) if isinstance(value, (int, float)) else None
    if isinstance(metric_payload, (int, float)):
        return float(metric_payload)
    return None


def _aggregate_metric(eval_payload: dict[str, Any], metric: str) -> float | None:
    metric_payload = eval_payload.get("metrics", {}).get("aggregate", {}).get(metric)
    if isinstance(metric_payload, dict):
        value = metric_payload.get("mean")
        return float(value) if isinstance(value, (int, float)) else None
    if isinstance(metric_payload, (int, float)):
        return float(metric_payload)
    return None


def _run_path(eval_payload: dict[str, Any]) -> Path | None:
    raw = eval_payload.get("run_path")
    return Path(raw) if raw else None


def _trace_path(eval_payload: dict[str, Any], trace_jsonl: str | Path | None) -> Path | None:
    if trace_jsonl:
        return Path(trace_jsonl)
    run_path = _run_path(eval_payload)
    if run_path is None:
        return None
    candidate = run_path / "trace_rows.jsonl"
    return candidate if candidate.exists() else None


def _trace_audit(eval_payload: dict[str, Any], trace_jsonl: str | Path | None) -> dict[str, Any]:
    path = _trace_path(eval_payload, trace_jsonl)
    if path is None or not path.exists():
        return {"trace_rows_found": False}

    rows = _iter_jsonl(path)
    source_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    predicted_commands_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    oracle_commands_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    low_level_cortex_calls = 0
    debug_cortex_calls = 0
    debug_steps = 0
    debug_correct = 0

    for row in rows:
        task_type = str(row.get("task_type") or "")
        action_source = str(row.get("policy_debug", {}).get("action_source") or "unknown")
        source_by_task[task_type][action_source] += 1
        if task_type in LOW_LEVEL_TASKS and action_source == "native_head_cortex":
            low_level_cortex_calls += 1
        if task_type == "test_failure_reflex":
            debug_steps += 1
            if action_source == "native_head_cortex":
                debug_cortex_calls += 1
            if row.get("correct") is True:
                debug_correct += 1
        action = row.get("action") or {}
        oracle_action = row.get("oracle_action") or {}
        if action.get("command"):
            predicted_commands_by_task[task_type][str(action["command"])] += 1
        if oracle_action.get("command"):
            oracle_commands_by_task[task_type][str(oracle_action["command"])] += 1

    return {
        "trace_rows_found": True,
        "trace_rows": len(rows),
        "low_level_cortex_calls": low_level_cortex_calls,
        "debug_cortex_calls": debug_cortex_calls,
        "debug_steps": debug_steps,
        "debug_correct": debug_correct,
        "source_by_task": {task: dict(counter) for task, counter in sorted(source_by_task.items())},
        "predicted_commands_by_task": {
            task: dict(counter) for task, counter in sorted(predicted_commands_by_task.items())
        },
        "oracle_commands_by_task": {
            task: dict(counter) for task, counter in sorted(oracle_commands_by_task.items())
        },
    }


def _debug_command_sets(split_paths: dict[str, str | Path | None]) -> dict[str, Any]:
    command_sets: dict[str, set[str]] = {}
    intent_counts: dict[str, Counter[str]] = {}
    for split, path in split_paths.items():
        rows = _iter_jsonl(path)
        commands: set[str] = set()
        intents: Counter[str] = Counter()
        for row in rows:
            if row.get("head_scope") != "debug_cortex" or row.get("action_type") != "RUN_COMMAND":
                continue
            command = row.get("command")
            if command:
                commands.add(str(command))
            intent = row.get("command_intent")
            if intent:
                intents[str(intent)] += 1
        command_sets[split] = commands
        intent_counts[split] = intents

    train_commands = command_sets.get("train", set())
    test_commands = command_sets.get("test", set())
    val_commands = command_sets.get("val", set())
    return {
        "command_counts": {split: len(commands) for split, commands in command_sets.items()},
        "intent_counts": {split: dict(counts) for split, counts in intent_counts.items()},
        "train_test_exact_overlap": sorted(train_commands & test_commands),
        "val_test_exact_overlap": sorted(val_commands & test_commands),
        "test_only_commands": sorted(test_commands - train_commands),
    }


def _completion_by_task(eval_payload: dict[str, Any]) -> dict[str, float | None]:
    tasks = sorted(eval_payload.get("metrics", {}).get("per_task", {}).keys())
    return {task: _metric_mean(eval_payload, task, "task_completion_rate") for task in tasks}


def build_phase2c_evidence_audit(
    *,
    eval_json: str | Path,
    gate_json: str | Path | None = None,
    dataset_manifest_json: str | Path | None = None,
    train_head_jsonl: str | Path | None = None,
    val_head_jsonl: str | Path | None = None,
    test_head_jsonl: str | Path | None = None,
    trace_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    eval_payload = _load_json(eval_json) or {}
    gate_payload = _load_json(gate_json) or {}
    manifest_payload = _load_json(dataset_manifest_json) or {}
    trace = _trace_audit(eval_payload, trace_jsonl)
    debug_commands = _debug_command_sets(
        {"train": train_head_jsonl, "val": val_head_jsonl, "test": test_head_jsonl}
    )
    coverage_audit = manifest_payload.get("coverage_audit") if manifest_payload else None
    completion_by_task = _completion_by_task(eval_payload)
    low_level_completion = {
        task: completion_by_task.get(task)
        for task in sorted(LOW_LEVEL_TASKS)
        if task in completion_by_task
    }
    fixed_split_passed = gate_payload.get("passed") is True
    low_level_preserved = all(value == 1.0 for value in low_level_completion.values())
    debug_completion = completion_by_task.get("test_failure_reflex")
    no_exact_debug_command_overlap = not debug_commands["train_test_exact_overlap"]

    evidence_status = {
        "fixed_split_native_head_gate_passed": fixed_split_passed,
        "debug_cortex_test_failure_passed": debug_completion == 1.0,
        "low_level_reflex_preserved": low_level_preserved,
        "low_level_qwen_not_used": trace.get("low_level_cortex_calls") == 0,
        "debug_train_test_exact_command_overlap_zero": no_exact_debug_command_overlap,
        "coverage_audit_passed": (
            coverage_audit is None or coverage_audit.get("passed") is True
        ),
        "external_validity_complete": False,
    }
    evidence_status["fixed_split_claim_supported"] = all(
        [
            evidence_status["fixed_split_native_head_gate_passed"],
            evidence_status["debug_cortex_test_failure_passed"],
            evidence_status["low_level_reflex_preserved"],
            evidence_status["low_level_qwen_not_used"],
            evidence_status["debug_train_test_exact_command_overlap_zero"],
        ]
    )

    next_required = []
    if not evidence_status["coverage_audit_passed"]:
        next_required.append("build a validation split that includes debug_cortex/RUN_COMMAND")
    if debug_commands["command_counts"].get("test", 0) < 3:
        next_required.append("add broader held-out Debug Cortex command families")
    next_required.append("evaluate quasi-real local project trajectories before architecture-level claim")

    return {
        "inputs": {
            "eval_json": str(Path(eval_json)),
            "gate_json": str(Path(gate_json)) if gate_json else None,
            "dataset_manifest_json": str(Path(dataset_manifest_json))
            if dataset_manifest_json
            else None,
            "train_head_jsonl": str(Path(train_head_jsonl)) if train_head_jsonl else None,
            "val_head_jsonl": str(Path(val_head_jsonl)) if val_head_jsonl else None,
            "test_head_jsonl": str(Path(test_head_jsonl)) if test_head_jsonl else None,
            "trace_jsonl": str(Path(trace_jsonl)) if trace_jsonl else None,
        },
        "policy": eval_payload.get("policy", {}),
        "aggregate_metrics": {
            "completion": _aggregate_metric(eval_payload, "task_completion_rate"),
            "reaction_latency_ms": _aggregate_metric(eval_payload, "reaction_latency_ms"),
            "model_calls": _aggregate_metric(eval_payload, "model_calls"),
            "token_equivalent_cost": _aggregate_metric(eval_payload, "token_equivalent_cost"),
            "dangerous_block": _aggregate_metric(eval_payload, "dangerous_action_block_rate"),
            "stale_state_action": _aggregate_metric(eval_payload, "stale_state_action_rate"),
        },
        "completion_by_task": completion_by_task,
        "low_level_completion": low_level_completion,
        "debug_command_generalization": debug_commands,
        "trace_audit": trace,
        "coverage_audit": coverage_audit,
        "gate": {
            "passed": gate_payload.get("passed"),
            "checks": gate_payload.get("checks", {}),
            "evidence_warnings": gate_payload.get("evidence_warnings", []),
        },
        "evidence_status": evidence_status,
        "next_required": next_required,
    }


def _write_markdown(report: dict[str, Any], output_path: str | Path) -> None:
    status = report["evidence_status"]
    metrics = report["aggregate_metrics"]
    lines = [
        "# Phase2C Evidence Audit",
        "",
        "## Status",
        f"- fixed_split_claim_supported: {status['fixed_split_claim_supported']}",
        f"- fixed_split_native_head_gate_passed: {status['fixed_split_native_head_gate_passed']}",
        f"- external_validity_complete: {status['external_validity_complete']}",
        f"- coverage_audit_passed: {status['coverage_audit_passed']}",
        "",
        "## Aggregate Metrics",
        f"- completion: {metrics['completion']}",
        f"- reaction_latency_ms: {metrics['reaction_latency_ms']}",
        f"- model_calls: {metrics['model_calls']}",
        f"- token_equivalent_cost: {metrics['token_equivalent_cost']}",
        "",
        "## Debug Command Generalization",
        f"- train_test_exact_overlap: {report['debug_command_generalization']['train_test_exact_overlap']}",
        f"- val_test_exact_overlap: {report['debug_command_generalization']['val_test_exact_overlap']}",
        f"- intent_counts: {report['debug_command_generalization']['intent_counts']}",
        "",
        "## Low-Level Preservation",
    ]
    for task, value in report["low_level_completion"].items():
        lines.append(f"- {task}: {value}")
    lines.extend(["", "## Next Required"])
    for item in report["next_required"]:
        lines.append(f"- {item}")
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2C evidence readiness.")
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--gate-json")
    parser.add_argument("--dataset-manifest-json")
    parser.add_argument("--train-head-jsonl")
    parser.add_argument("--val-head-jsonl")
    parser.add_argument("--test-head-jsonl")
    parser.add_argument("--trace-jsonl")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    report = build_phase2c_evidence_audit(
        eval_json=args.eval_json,
        gate_json=args.gate_json,
        dataset_manifest_json=args.dataset_manifest_json,
        train_head_jsonl=args.train_head_jsonl,
        val_head_jsonl=args.val_head_jsonl,
        test_head_jsonl=args.test_head_jsonl,
        trace_jsonl=args.trace_jsonl,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(report, output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
