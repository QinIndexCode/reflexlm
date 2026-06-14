from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SUMMARY = Path(
    "artifacts/reports/phase2j_semantic_command_identity/"
    "phase2j_semantic_command_identity_r16_alpha32_lr1e-4_len256_smoke128_val192.training_summary.json"
)
DEFAULT_DATA_HEALTH = Path(
    "artifacts/reports/phase2j_semantic_command_identity/phase2j_data_health_audit.json"
)
DEFAULT_PRETRAIN_GATE = Path(
    "artifacts/reports/phase2j_semantic_command_identity/phase2j_pretrain_gate.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/phase2j_semantic_command_identity/phase2j_smoke_postflight.json"
)


def _load_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload_path = Path(path)
    if not payload_path.exists():
        return {}
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _last_epoch(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history") or []
    return history[-1] if history else {}


def _val_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return _last_epoch(summary).get("val_metrics") or {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_phase2j_smoke_postflight(
    *,
    training_summary_json: str | Path,
    data_health_json: str | Path | None = None,
    pretrain_gate_json: str | Path | None = None,
    min_val_action_accuracy: float = 0.95,
    min_val_command_intent_accuracy: float = 0.95,
    min_val_command_slot_accuracy: float = 0.85,
    max_duration_seconds: float = 3600.0,
    min_mechanism_delta_vs_source_overlap: float = 0.01,
    max_smoke_val_records: int = 512,
) -> dict[str, Any]:
    summary = _load_json(training_summary_json)
    data_health = _load_json(data_health_json)
    pretrain_gate = _load_json(pretrain_gate_json)
    val_metrics = _val_metrics(summary)
    last_epoch = _last_epoch(summary)
    run_manifest = summary.get("run_manifest") or {}
    source_overlap_val = (
        summary.get("source_overlap_command_slot_baseline", {}).get("val", {})
    )
    val_action_accuracy = _float(val_metrics.get("action_accuracy"))
    val_command_intent_accuracy = _float(val_metrics.get("command_intent_accuracy"))
    val_accuracy = _float(val_metrics.get("command_slot_accuracy"))
    source_overlap_accuracy = _float(source_overlap_val.get("accuracy"))
    mechanism_delta = val_accuracy - source_overlap_accuracy
    summary_hashes = summary.get("effective_split_hashes") or {}
    data_hashes = data_health.get("effective_split_hashes") or {}
    duration_seconds = _float(run_manifest.get("duration_seconds"))
    train_pairwise = int(last_epoch.get("train_pairwise_encoded_candidates") or 0)
    val_pairwise = int(_float(val_metrics.get("pairwise_encoded_candidates")))

    checks = {
        "pretrain_gate_passed": (
            True if not pretrain_gate else pretrain_gate.get("passed") is True
        ),
        "data_health_passed": True if not data_health else data_health.get("passed") is True,
        "summary_exists_and_completed": bool(summary and run_manifest.get("finished_at_utc")),
        "smoke_limits_recorded": (
            int(summary.get("config", {}).get("max_train_records") or 0) <= 128
            and int(summary.get("config", {}).get("max_val_records") or 0)
            <= max_smoke_val_records
            and int(summary.get("train_examples") or 0) <= 128
            and int(summary.get("val_examples") or 0) <= max_smoke_val_records
        ),
        "duration_within_smoke_budget": (
            duration_seconds > 0.0 and duration_seconds <= max_duration_seconds
        ),
        "layered_val_metrics_recorded": (
            _float(val_metrics.get("action_count")) > 0.0
            and _float(val_metrics.get("command_intent_count")) > 0.0
            and _float(val_metrics.get("command_slot_count")) > 0.0
        ),
        "val_action_gate_passed": val_action_accuracy >= min_val_action_accuracy,
        "val_command_intent_gate_passed": (
            val_command_intent_accuracy >= min_val_command_intent_accuracy
        ),
        "val_command_slot_gate_passed": val_accuracy >= min_val_command_slot_accuracy,
        "val_command_slot_count_present": _float(val_metrics.get("command_slot_count")) > 0.0,
        "no_json_motor_target": summary.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target") == 0,
        "additive_latent_fusion": (
            summary.get("config", {}).get("latent_fusion") == "additive"
            and summary.get("head_config", {}).get("latent_fusion") == "additive"
        ),
        "command_identity_latent_dim_recorded": (
            int(summary.get("head_config", {}).get("nsi_latent_dim") or 0) >= 30
        ),
        "pairwise_disabled_for_phase2j_isolation": (
            summary.get("use_pairwise_command_reranker") is False
            and summary.get("head_config", {}).get("use_pairwise_command_reranker") is False
            and train_pairwise == 0
            and val_pairwise == 0
        ),
        "effective_hashes_present": bool(summary_hashes.get("phase2c_head_train"))
        and bool(summary_hashes.get("phase2c_head_val")),
        "val_hash_matches_data_health": (
            True
            if not data_hashes
            else summary_hashes.get("phase2c_head_val") == data_hashes.get("phase2j_head_val")
        ),
        "source_overlap_baseline_recorded": source_overlap_val.get("total", 0) > 0,
        "model_beats_source_overlap_baseline": (
            mechanism_delta >= min_mechanism_delta_vs_source_overlap
        ),
    }
    smoke_viable = all(
        checks[name]
        for name in (
            "pretrain_gate_passed",
            "data_health_passed",
            "summary_exists_and_completed",
            "smoke_limits_recorded",
            "duration_within_smoke_budget",
            "layered_val_metrics_recorded",
            "val_action_gate_passed",
            "val_command_intent_gate_passed",
            "val_command_slot_gate_passed",
            "val_command_slot_count_present",
            "no_json_motor_target",
            "low_level_qwen_calls_target_zero",
            "additive_latent_fusion",
            "command_identity_latent_dim_recorded",
            "pairwise_disabled_for_phase2j_isolation",
            "effective_hashes_present",
            "val_hash_matches_data_health",
            "source_overlap_baseline_recorded",
        )
    )
    mechanism_increment_supported = checks["model_beats_source_overlap_baseline"]
    ready_for_full_training = smoke_viable and mechanism_increment_supported
    blocked_actions: list[str] = []
    if not smoke_viable:
        blocked_actions.append("do_not_start_full_training_until_smoke_viability_passes")
    if not checks["val_action_gate_passed"]:
        blocked_actions.append("do_not_start_full_training_until_action_gate_passes")
    if not checks["val_command_intent_gate_passed"]:
        blocked_actions.append("do_not_start_full_training_until_command_intent_gate_passes")
    if not mechanism_increment_supported:
        blocked_actions.append("do_not_start_full_training_without_beating_source_overlap_baseline")
    return {
        "audit_family": "phase2j_smoke_postflight",
        "passed": ready_for_full_training,
        "smoke_viable": smoke_viable,
        "mechanism_increment_supported": mechanism_increment_supported,
        "ready_for_full_training": ready_for_full_training,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "allowed_next_action": (
            "run_phase2j_full_nonsealed_training"
            if ready_for_full_training
            else "revise_phase2j_mechanism_or_data_before_full_training"
        ),
        "blocked_actions": blocked_actions,
        "checks": checks,
        "metrics": {
            "duration_seconds": duration_seconds,
            "train_elapsed_seconds": _float(last_epoch.get("train_elapsed_seconds")),
            "val_elapsed_seconds": _float(val_metrics.get("elapsed_seconds")),
            "train_steps_per_second": _float(last_epoch.get("train_steps_per_second")),
            "val_action_accuracy": val_action_accuracy,
            "val_action_count": _float(val_metrics.get("action_count")),
            "val_command_intent_accuracy": val_command_intent_accuracy,
            "val_command_intent_count": _float(val_metrics.get("command_intent_count")),
            "val_command_slot_accuracy": val_accuracy,
            "val_command_slot_count": _float(val_metrics.get("command_slot_count")),
            "source_overlap_val_accuracy": source_overlap_accuracy,
            "model_minus_source_overlap_accuracy": mechanism_delta,
            "train_pairwise_encoded_candidates": train_pairwise,
            "val_pairwise_encoded_candidates": val_pairwise,
        },
        "thresholds": {
            "min_val_action_accuracy": min_val_action_accuracy,
            "min_val_command_intent_accuracy": min_val_command_intent_accuracy,
            "min_val_command_slot_accuracy": min_val_command_slot_accuracy,
            "max_duration_seconds": max_duration_seconds,
            "min_mechanism_delta_vs_source_overlap": min_mechanism_delta_vs_source_overlap,
            "max_smoke_val_records": max_smoke_val_records,
        },
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "data_health_json": str(Path(data_health_json)) if data_health_json else None,
            "pretrain_gate_json": str(Path(pretrain_gate_json)) if pretrain_gate_json else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Postflight gate for Phase2J non-sealed smoke training."
    )
    parser.add_argument("--training-summary-json", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--data-health-json", default=str(DEFAULT_DATA_HEALTH))
    parser.add_argument("--pretrain-gate-json", default=str(DEFAULT_PRETRAIN_GATE))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--min-val-action-accuracy", type=float, default=0.95)
    parser.add_argument("--min-val-command-intent-accuracy", type=float, default=0.95)
    parser.add_argument("--min-val-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--max-duration-seconds", type=float, default=3600.0)
    parser.add_argument("--min-mechanism-delta-vs-source-overlap", type=float, default=0.01)
    parser.add_argument("--max-smoke-val-records", type=int, default=512)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2j_smoke_postflight(
        training_summary_json=args.training_summary_json,
        data_health_json=args.data_health_json,
        pretrain_gate_json=args.pretrain_gate_json,
        min_val_action_accuracy=args.min_val_action_accuracy,
        min_val_command_intent_accuracy=args.min_val_command_intent_accuracy,
        min_val_command_slot_accuracy=args.min_val_command_slot_accuracy,
        max_duration_seconds=args.max_duration_seconds,
        min_mechanism_delta_vs_source_overlap=args.min_mechanism_delta_vs_source_overlap,
        max_smoke_val_records=args.max_smoke_val_records,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
