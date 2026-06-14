from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _history(summary: dict[str, Any]) -> list[dict[str, Any]]:
    history = summary.get("history")
    return history if isinstance(history, list) else []


def _epoch_metric(epoch: dict[str, Any], key: str) -> float:
    return _float(_dict(epoch.get("val_metrics")).get(key))


def _best_epoch(summary: dict[str, Any], key: str) -> dict[str, Any]:
    history = _history(summary)
    if not history:
        return {"epoch": None, "value": 0.0}
    best = max(history, key=lambda item: _epoch_metric(_dict(item), key))
    return {"epoch": best.get("epoch"), "value": _epoch_metric(_dict(best), key)}


def build_phase2av_smoke_failure_audit(
    *,
    postflight_json: str | Path,
    training_summary_json: str | Path,
    pretrain_gate_json: str | Path,
    pool_gap_json: str | Path,
) -> dict[str, Any]:
    postflight = _read_json(postflight_json)
    summary = _read_json(training_summary_json)
    pretrain_gate = _read_json(pretrain_gate_json)
    pool_gap = _read_json(pool_gap_json)
    metrics = _dict(postflight.get("metrics"))
    history = _history(summary)

    command_slot = _float(metrics.get("command_slot_accuracy"))
    source_overlap = _float(metrics.get("source_overlap_accuracy"))
    delta = _float(metrics.get("model_minus_source_overlap_accuracy"))
    patch_operation = _float(metrics.get("patch_operation_accuracy"))
    patch_template = _float(metrics.get("patch_template_slot_accuracy"))
    patch_target = _float(metrics.get("patch_target_file_slot_accuracy"))

    first_train_loss = (
        _float(history[0].get("first_train_loss")) if history else 0.0
    )
    final_train_loss = (
        _float(history[-1].get("train_loss")) if history else 0.0
    )
    train_loss_reduction = first_train_loss - final_train_loss
    best_command = _best_epoch(summary, "command_slot_accuracy")
    best_operation = _best_epoch(summary, "patch_operation_accuracy")
    best_template = _best_epoch(summary, "patch_template_slot_accuracy")

    issue_classification: list[str] = []
    if pretrain_gate.get("passed") is True and pool_gap.get("passed") is True:
        issue_classification.append("not_a_data_health_or_pool_gap_failure")
    if delta > 0:
        issue_classification.append("nonzero_descriptor_runtime_signal_above_source_overlap")
    else:
        issue_classification.append("source_overlap_tie_or_worse")
    if train_loss_reduction > 5.0 and command_slot < 0.85:
        issue_classification.append("train_loss_drops_but_val_slot_gate_fails")
    if best_command.get("value", 0.0) > command_slot:
        issue_classification.append("late_epoch_overfit_or_val_regression")
    if patch_operation < 0.85 or patch_template < 0.85:
        issue_classification.append("descriptor_operation_template_heads_underfit_or_confused")
    if patch_target >= 0.85 and (patch_operation < 0.85 or patch_template < 0.85):
        issue_classification.append("target_file_head_solved_but_descriptor_labels_not_solved")

    next_actions = [
        "do_not_start_phase2av_full_training",
        "do_not_package_phase2av",
        "do_not_run_sealed_eval_for_phase2av",
        "freeze_this_smoke_as_nonsealed_failure_evidence",
        "inspect_descriptor_operation_template_features_before_more_training",
        "run_nonsealed_ablation_for_descriptor_heads_not_sealed_feedback",
    ]

    return {
        "artifact_family": "phase2av_smoke_failure_audit",
        "passed": False,
        "claim_boundary": (
            "This audit explains a failed non-sealed Phase2AV smoke. It does not "
            "authorize full training, packaging, sealed evaluation, production "
            "autonomy, or epoch-making architecture claims."
        ),
        "preconditions": {
            "pretrain_gate_passed": pretrain_gate.get("passed") is True,
            "pool_gap_passed": pool_gap.get("passed") is True,
            "postflight_passed": postflight.get("passed") is True,
            "sealed_feedback_used": summary.get("open_repair_training_contract", {}).get(
                "sealed_feedback_used"
            )
            is True,
        },
        "metrics": {
            "command_slot_accuracy": command_slot,
            "source_overlap_accuracy": source_overlap,
            "model_minus_source_overlap_accuracy": delta,
            "patch_operation_accuracy": patch_operation,
            "patch_template_slot_accuracy": patch_template,
            "patch_target_file_slot_accuracy": patch_target,
            "first_train_loss": first_train_loss,
            "final_train_loss": final_train_loss,
            "train_loss_reduction": train_loss_reduction,
            "best_epoch": {
                "command_slot_accuracy": best_command,
                "patch_operation_accuracy": best_operation,
                "patch_template_slot_accuracy": best_template,
            },
        },
        "issue_classification": issue_classification,
        "failure_modes": postflight.get("failure_modes", []),
        "blocked_actions": next_actions[:3],
        "recommended_next_actions": next_actions[3:],
        "unsupported_claims": [
            "learned_descriptor_runtime_delta_sufficient_for_full_training",
            "freeform_patch_generation",
            "sealed_cross_model_transfer",
            "open_ended_debugging_generalization",
            "production_autonomy",
            "epoch_making_architecture",
        ],
        "inputs": {
            "postflight_json": str(Path(postflight_json)),
            "training_summary_json": str(Path(training_summary_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
            "pool_gap_json": str(Path(pool_gap_json)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase2AV smoke failure audit.")
    parser.add_argument("--postflight-json", required=True)
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--pool-gap-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = build_phase2av_smoke_failure_audit(
        postflight_json=args.postflight_json,
        training_summary_json=args.training_summary_json,
        pretrain_gate_json=args.pretrain_gate_json,
        pool_gap_json=args.pool_gap_json,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
