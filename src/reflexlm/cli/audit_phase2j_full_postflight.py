from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reflexlm.cli.audit_phase2j_smoke_postflight import _float, _last_epoch, _load_json, _val_metrics
from reflexlm.llm.candidate_features import CANDIDATE_FEATURE_DIM


DEFAULT_SUMMARY = Path(
    "artifacts/reports/phase2j_source_overlap_hard_slotaware/"
    "phase2j_source_overlap_hard_slotaware_full.training_summary.json"
)
DEFAULT_DATA_HEALTH = Path(
    "artifacts/reports/phase2j_source_overlap_hard_slotaware/"
    "phase2j_source_overlap_hard_slotaware_data_health.json"
)
DEFAULT_SMOKE_POSTFLIGHT = Path(
    "artifacts/reports/phase2j_source_overlap_hard_slotaware/"
    "phase2j_source_overlap_hard_slotaware_smoke_postflight.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/reports/phase2j_source_overlap_hard_slotaware/"
    "phase2j_source_overlap_hard_slotaware_full_postflight.json"
)


def build_phase2j_full_postflight(
    *,
    training_summary_json: str | Path,
    data_health_json: str | Path | None = None,
    smoke_postflight_json: str | Path | None = None,
    min_val_action_accuracy: float = 0.95,
    min_val_command_intent_accuracy: float = 0.95,
    min_val_command_slot_accuracy: float = 0.85,
    min_mechanism_delta_vs_source_overlap: float = 0.01,
    expected_max_train_records: int = 1024,
    expected_max_val_records: int = 512,
) -> dict[str, Any]:
    summary = _load_json(training_summary_json)
    data_health = _load_json(data_health_json)
    smoke_postflight = _load_json(smoke_postflight_json)
    val_metrics = _val_metrics(summary)
    last_epoch = _last_epoch(summary)
    run_manifest = summary.get("run_manifest") or {}
    source_overlap_val = summary.get("source_overlap_command_slot_baseline", {}).get("val", {})
    val_action_accuracy = _float(val_metrics.get("action_accuracy"))
    val_command_intent_accuracy = _float(val_metrics.get("command_intent_accuracy"))
    val_accuracy = _float(val_metrics.get("command_slot_accuracy"))
    source_overlap_accuracy = _float(source_overlap_val.get("accuracy"))
    mechanism_delta = val_accuracy - source_overlap_accuracy
    summary_hashes = summary.get("effective_split_hashes") or {}
    data_hashes = data_health.get("effective_split_hashes") or {}
    train_pairwise = int(last_epoch.get("train_pairwise_encoded_candidates") or 0)
    val_pairwise = int(_float(val_metrics.get("pairwise_encoded_candidates")))
    config = summary.get("config") or {}
    head_config = summary.get("head_config") or {}

    checks = {
        "smoke_postflight_passed": (
            True if not smoke_postflight else smoke_postflight.get("passed") is True
        ),
        "data_health_passed": True if not data_health else data_health.get("passed") is True,
        "summary_exists_and_completed": bool(summary and run_manifest.get("finished_at_utc")),
        "full_limits_recorded": (
            int(config.get("max_train_records") or 0) == expected_max_train_records
            and int(config.get("max_val_records") or 0) == expected_max_val_records
            and int(summary.get("train_examples") or 0) <= expected_max_train_records
            and int(summary.get("val_examples") or 0) <= expected_max_val_records
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
            config.get("latent_fusion") == "additive"
            and head_config.get("latent_fusion") == "additive"
        ),
        "command_identity_latent_dim_recorded": int(head_config.get("nsi_latent_dim") or 0) >= 30,
        "slot_aware_candidate_features_recorded": (
            int(summary.get("command_candidate_feature_dim") or 0) >= CANDIDATE_FEATURE_DIM
            and int(head_config.get("command_candidate_feature_dim") or 0) >= CANDIDATE_FEATURE_DIM
        ),
        "pairwise_disabled_for_phase2j_isolation": (
            summary.get("use_pairwise_command_reranker") is False
            and head_config.get("use_pairwise_command_reranker") is False
            and train_pairwise == 0
            and val_pairwise == 0
        ),
        "effective_hashes_present": bool(summary_hashes.get("phase2c_head_train"))
        and bool(summary_hashes.get("phase2c_head_val")),
        "train_hash_matches_data_health": (
            True
            if not data_hashes
            else summary_hashes.get("phase2c_head_train") == data_hashes.get("phase2j_head_train")
        ),
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
    full_viable = all(checks.values())
    blocked_actions: list[str] = []
    if not full_viable:
        blocked_actions.append("do_not_package_until_phase2j_full_postflight_passes")
    if not checks["val_action_gate_passed"]:
        blocked_actions.append("do_not_package_until_action_gate_passes")
    if not checks["val_command_intent_gate_passed"]:
        blocked_actions.append("do_not_package_until_command_intent_gate_passes")
    if not checks["model_beats_source_overlap_baseline"]:
        blocked_actions.append("do_not_package_without_phase2j_mechanism_increment")
    if not checks["slot_aware_candidate_features_recorded"]:
        blocked_actions.append("do_not_package_without_slot_aware_candidate_feature_evidence")
    return {
        "audit_family": "phase2j_full_postflight",
        "passed": full_viable,
        "full_viable": full_viable,
        "ready_for_package": full_viable,
        "ready_for_sealed_eval": False,
        "allowed_next_action": (
            "run_phase2j_package_only" if full_viable else "revise_phase2j_before_package"
        ),
        "blocked_actions": blocked_actions,
        "checks": checks,
        "metrics": {
            "duration_seconds": _float(run_manifest.get("duration_seconds")),
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
            "command_candidate_feature_dim": int(summary.get("command_candidate_feature_dim") or 0),
        },
        "thresholds": {
            "min_val_action_accuracy": min_val_action_accuracy,
            "min_val_command_intent_accuracy": min_val_command_intent_accuracy,
            "min_val_command_slot_accuracy": min_val_command_slot_accuracy,
            "min_mechanism_delta_vs_source_overlap": min_mechanism_delta_vs_source_overlap,
            "expected_max_train_records": expected_max_train_records,
            "expected_max_val_records": expected_max_val_records,
            "min_command_candidate_feature_dim": CANDIDATE_FEATURE_DIM,
        },
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "data_health_json": str(Path(data_health_json)) if data_health_json else None,
            "smoke_postflight_json": (
                str(Path(smoke_postflight_json)) if smoke_postflight_json else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full postflight gate for Phase2J non-sealed training before package."
    )
    parser.add_argument("--training-summary-json", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--data-health-json", default=str(DEFAULT_DATA_HEALTH))
    parser.add_argument("--smoke-postflight-json", default=str(DEFAULT_SMOKE_POSTFLIGHT))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--min-val-action-accuracy", type=float, default=0.95)
    parser.add_argument("--min-val-command-intent-accuracy", type=float, default=0.95)
    parser.add_argument("--min-val-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--min-mechanism-delta-vs-source-overlap", type=float, default=0.01)
    parser.add_argument("--expected-max-train-records", type=int, default=1024)
    parser.add_argument("--expected-max-val-records", type=int, default=512)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2j_full_postflight(
        training_summary_json=args.training_summary_json,
        data_health_json=args.data_health_json,
        smoke_postflight_json=args.smoke_postflight_json,
        min_val_action_accuracy=args.min_val_action_accuracy,
        min_val_command_intent_accuracy=args.min_val_command_intent_accuracy,
        min_val_command_slot_accuracy=args.min_val_command_slot_accuracy,
        min_mechanism_delta_vs_source_overlap=args.min_mechanism_delta_vs_source_overlap,
        expected_max_train_records=args.expected_max_train_records,
        expected_max_val_records=args.expected_max_val_records,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
