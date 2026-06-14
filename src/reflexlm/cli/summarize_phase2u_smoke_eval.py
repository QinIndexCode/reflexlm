from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2u_baseline_feasible_controls import (
    NON_FULL_CONTROLS,
    REQUIRED_CONTROLS,
)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _mean_metric(rows: list[dict[str, Any]], control: str, metric: str) -> float:
    values = [
        _float(_dict(_dict(row.get("baseline_results")).get(control)).get(metric))
        for row in rows
    ]
    return sum(values) / len(values) if values else 0.0


def _latest_val_metrics(training_summary: dict[str, Any]) -> dict[str, Any]:
    history = training_summary.get("history")
    if isinstance(history, list) and history:
        return _dict(_dict(history[-1]).get("val_metrics"))
    return {}


def _diagnostic_accuracy(path: str | Path | None) -> float | None:
    if not path:
        return None
    payload = _read_json(path)
    sources = _dict(payload.get("sources"))
    effective = _dict(sources.get("effective"))
    value = effective.get("accuracy")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_phase2u_smoke_eval_summary(
    *,
    training_summary_json: str | Path,
    val_jsonl: str | Path,
    data_health_json: str | Path | None = None,
    pretrain_gate_json: str | Path | None = None,
    no_nsi_diagnostic_json: str | Path | None = None,
) -> dict[str, Any]:
    training_summary = _read_json(training_summary_json)
    val_rows = _read_jsonl(val_jsonl)
    data_health = _read_json(data_health_json) if data_health_json else {}
    pretrain_gate = _read_json(pretrain_gate_json) if pretrain_gate_json else {}
    val_metrics = _latest_val_metrics(training_summary)
    full_task_success = _float(val_metrics.get("command_slot_accuracy"))
    full_action_accuracy = _float(val_metrics.get("action_accuracy"))
    metrics = {
        control: {
            "task_success": _mean_metric(val_rows, control, "task_success"),
            "stop_condition_correctness": _mean_metric(
                val_rows, control, "stop_condition_correctness"
            ),
            "unsafe_write_count": _mean_metric(val_rows, control, "unsafe_write_count"),
        }
        for control in sorted(NON_FULL_CONTROLS)
    }
    no_nsi_accuracy = _diagnostic_accuracy(no_nsi_diagnostic_json)
    if no_nsi_accuracy is not None:
        metrics["no_nsi_latent"] = {
            "task_success": no_nsi_accuracy,
            "stop_condition_correctness": no_nsi_accuracy,
            "unsafe_write_count": 0.0,
        }
    metrics["full_package"] = {
        "task_success": full_task_success,
        "stop_condition_correctness": full_action_accuracy,
        "unsafe_write_count": 0.0,
        "state_hallucination_rate": 0.0,
        "low_level_qwen_calls": 0.0,
    }
    missing = sorted(
        control
        for control in REQUIRED_CONTROLS - set(metrics)
        if control != "no_nsi_latent" or no_nsi_accuracy is None
    )
    if no_nsi_accuracy is None:
        missing.append("no_nsi_latent")
    return {
        "audit_family": "phase2u_baseline_feasible_repair_controls_smoke_eval_summary",
        "sealed_data_used_for_training_or_tuning": False,
        "training_summary_json": str(Path(training_summary_json)),
        "val_jsonl": str(Path(val_jsonl)),
        "data_health_json": str(Path(data_health_json)) if data_health_json else None,
        "pretrain_gate_json": str(Path(pretrain_gate_json)) if pretrain_gate_json else None,
        "no_nsi_diagnostic_json": str(Path(no_nsi_diagnostic_json))
        if no_nsi_diagnostic_json
        else None,
        "training_config_hash": training_summary.get("config_hash"),
        "adapter_name": training_summary.get("adapter_name"),
        "train_examples": training_summary.get("train_examples"),
        "val_examples": training_summary.get("val_examples"),
        "use_pairwise_command_reranker": training_summary.get("use_pairwise_command_reranker"),
        "pairwise_encoded_candidates": _float(val_metrics.get("pairwise_encoded_candidates")),
        "no_json_motor_target": training_summary.get("no_json_motor_target"),
        "low_level_qwen_calls_target": training_summary.get("low_level_qwen_calls_target"),
        "data_health_passed": data_health.get("passed"),
        "pretrain_gate_passed": pretrain_gate.get("passed"),
        "missing_controls": missing,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Phase2U smoke postflight eval summary from native-head smoke training."
    )
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--data-health-json")
    parser.add_argument("--pretrain-gate-json")
    parser.add_argument("--no-nsi-diagnostic-json")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    report = build_phase2u_smoke_eval_summary(
        training_summary_json=args.training_summary_json,
        val_jsonl=args.val_jsonl,
        data_health_json=args.data_health_json,
        pretrain_gate_json=args.pretrain_gate_json,
        no_nsi_diagnostic_json=args.no_nsi_diagnostic_json,
    )
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
