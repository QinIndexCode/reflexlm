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


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _last_history(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history")
    if isinstance(history, list) and history:
        item = history[-1]
        return item if isinstance(item, dict) else {}
    return {}


def audit_phase2y_open_repair_smoke_postflight(
    *,
    pretrain_gate_json: str | Path,
    training_summary_json: str | Path,
    runtime_capability_audit_json: str | Path,
    min_val_command_slot_accuracy: float = 0.85,
    min_model_minus_source_overlap: float = 0.10,
) -> dict[str, Any]:
    pretrain = _read_json(pretrain_gate_json)
    summary = _read_json(training_summary_json)
    runtime = _read_json(runtime_capability_audit_json)
    history = _last_history(summary)
    val_metrics = history.get("val_metrics") if isinstance(history.get("val_metrics"), dict) else {}
    source_overlap = summary.get("source_overlap_command_slot_baseline")
    source_val = (
        source_overlap.get("val")
        if isinstance(source_overlap, dict) and isinstance(source_overlap.get("val"), dict)
        else {}
    )
    val_accuracy = _float(val_metrics.get("command_slot_accuracy"))
    source_accuracy = _float(source_val.get("accuracy"))
    delta = val_accuracy - source_accuracy
    checks = {
        "pretrain_gate_passed": pretrain.get("passed") is True,
        "training_summary_present": bool(summary),
        "open_repair_heads_enabled": summary.get("open_repair_heads_enabled") is True,
        "runtime_capability_audit_passed": runtime.get("passed") is True,
        "val_command_slot_accuracy_min": val_accuracy >= min_val_command_slot_accuracy,
        "model_beats_source_overlap": delta >= min_model_minus_source_overlap,
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2y_open_repair_smoke_postflight",
        "passed": passed,
        "ready_for_phase2y_execution_runner_development": passed,
        "ready_for_open_ended_claim": False,
        "checks": checks,
        "metrics": {
            "train_examples": summary.get("train_examples"),
            "val_examples": summary.get("val_examples"),
            "val_command_slot_accuracy": val_accuracy,
            "source_overlap_val_accuracy": source_accuracy,
            "model_minus_source_overlap": delta,
            "train_loss": history.get("train_loss"),
            "val_loss": val_metrics.get("loss"),
            "train_elapsed_seconds": history.get("train_elapsed_seconds"),
        },
        "claim_boundary": (
            "phase2y_smoke_head_control_ready_not_execution_evidence"
            if passed
            else "phase2y_smoke_not_ready"
        ),
        "blocked_actions": [
            "do_not_claim_open_ended_repair_generalization_without_phase2y_execution_results",
            "do_not_claim_production_autonomy_without_live_agent_and_safety_gates",
        ],
        "inputs": {
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "training_summary_json": str(Path(training_summary_json)),
            "runtime_capability_audit_json": str(Path(runtime_capability_audit_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2Y smoke postflight.")
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--runtime-capability-audit-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-val-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-model-minus-source-overlap", type=float, default=0.10)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2y_open_repair_smoke_postflight(
        pretrain_gate_json=args.pretrain_gate_json,
        training_summary_json=args.training_summary_json,
        runtime_capability_audit_json=args.runtime_capability_audit_json,
        min_val_command_slot_accuracy=args.min_val_command_slot_accuracy,
        min_model_minus_source_overlap=args.min_model_minus_source_overlap,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
