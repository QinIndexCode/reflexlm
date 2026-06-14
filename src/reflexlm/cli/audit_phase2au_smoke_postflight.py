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


def _latest_val_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history")
    if not isinstance(history, list) or not history:
        return {}
    metrics = history[-1].get("val_metrics")
    return metrics if isinstance(metrics, dict) else {}


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def audit_phase2au_smoke_postflight(
    *,
    training_summary_json: str | Path,
    pretrain_gate_json: str | Path,
    min_val_accuracy: float = 0.85,
    max_source_overlap_baseline: float = 0.95,
    min_descriptor_count: int = 1,
    min_valid_candidates_per_row: int = 2,
) -> dict[str, Any]:
    summary = _read_json(training_summary_json)
    pretrain = _read_json(pretrain_gate_json)
    val_metrics = _latest_val_metrics(summary)
    baseline = summary.get("source_overlap_command_slot_baseline")
    baseline_val = baseline.get("val") if isinstance(baseline, dict) else {}
    pairwise = summary.get("pairwise_candidate_encoding")
    pairwise_val = pairwise.get("val") if isinstance(pairwise, dict) else {}
    command_slot_accuracy = _metric(val_metrics, "command_slot_accuracy")
    source_overlap_accuracy = _metric(baseline_val, "accuracy") if isinstance(baseline_val, dict) else None
    patch_operation_count = _metric(val_metrics, "patch_operation_count")
    patch_template_count = _metric(val_metrics, "patch_template_slot_count")
    max_valid_candidates = _metric(pairwise_val, "max_valid_candidates_per_row") if isinstance(pairwise_val, dict) else None
    checks = {
        "pretrain_gate_passed": pretrain.get("passed") is True,
        "training_summary_present": bool(summary),
        "val_accuracy_gate": isinstance(command_slot_accuracy, float)
        and command_slot_accuracy >= min_val_accuracy,
        "source_overlap_not_ceiling": isinstance(source_overlap_accuracy, float)
        and source_overlap_accuracy <= max_source_overlap_baseline,
        "descriptor_operation_evaluable": isinstance(patch_operation_count, float)
        and patch_operation_count >= min_descriptor_count,
        "descriptor_template_evaluable": isinstance(patch_template_count, float)
        and patch_template_count >= min_descriptor_count,
        "nontrivial_command_candidate_set": isinstance(max_valid_candidates, float)
        and max_valid_candidates >= min_valid_candidates_per_row,
        "package_and_sealed_not_allowed": True,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2au_policy_required_smoke_postflight",
        "passed": passed,
        "ready_for_phase2au_full_training": passed,
        "checks": checks,
        "metrics": {
            "command_slot_accuracy": command_slot_accuracy,
            "source_overlap_command_slot_accuracy": source_overlap_accuracy,
            "model_minus_source_overlap": (
                command_slot_accuracy - source_overlap_accuracy
                if isinstance(command_slot_accuracy, float)
                and isinstance(source_overlap_accuracy, float)
                else None
            ),
            "patch_operation_count": patch_operation_count,
            "patch_template_slot_count": patch_template_count,
            "max_valid_candidates_per_row": max_valid_candidates,
            "train_examples": summary.get("train_examples"),
            "val_examples": summary.get("val_examples"),
            "config_hash": summary.get("config_hash"),
        },
        "claim_boundary": (
            "phase2au_smoke_ready_for_full_training_not_runtime_delta_evidence"
            if passed
            else "phase2au_smoke_non_evidence_requires_dataset_or_schema_redesign"
        ),
        "blocked_actions": []
        if passed
        else [
            "do_not_start_phase2au_full_training",
            "do_not_package_phase2au",
            "do_not_run_sealed_phase2au",
            "do_not_claim_learned_runtime_delta_from_this_smoke",
        ],
        "unsupported_claims": [
            "learned_runtime_delta_before_nontrivial_runtime_execution",
            "freeform_patch_generation",
            "sealed_cross_model_transfer",
            "open_ended_debugging_generalization",
            "epoch_making_architecture",
        ],
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "pretrain_gate_json": str(Path(pretrain_gate_json)),
        },
        "thresholds": {
            "min_val_accuracy": min_val_accuracy,
            "max_source_overlap_baseline": max_source_overlap_baseline,
            "min_descriptor_count": min_descriptor_count,
            "min_valid_candidates_per_row": min_valid_candidates_per_row,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AU smoke training postflight.")
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--pretrain-gate-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-val-accuracy", type=float, default=0.85)
    parser.add_argument("--max-source-overlap-baseline", type=float, default=0.95)
    parser.add_argument("--min-descriptor-count", type=int, default=1)
    parser.add_argument("--min-valid-candidates-per-row", type=int, default=2)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = audit_phase2au_smoke_postflight(
        training_summary_json=args.training_summary_json,
        pretrain_gate_json=args.pretrain_gate_json,
        min_val_accuracy=args.min_val_accuracy,
        max_source_overlap_baseline=args.max_source_overlap_baseline,
        min_descriptor_count=args.min_descriptor_count,
        min_valid_candidates_per_row=args.min_valid_candidates_per_row,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
