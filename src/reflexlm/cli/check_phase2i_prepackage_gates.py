from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _final_val_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history")
    if not isinstance(history, list) or not history:
        return {}
    latest = history[-1]
    metrics = latest.get("val_metrics") if isinstance(latest, dict) else None
    return metrics if isinstance(metrics, dict) else {}


def _hash_for(hashes: dict[str, Any], split: str) -> str | None:
    if not isinstance(hashes, dict):
        return None
    for name, value in hashes.items():
        if split in str(name).lower() and isinstance(value, str) and value:
            return value
    return None


def _nested_bool(payload: dict[str, Any], *path: str) -> bool:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return False
        value = value.get(key)
    return value is True


def _nested_float(payload: dict[str, Any], *path: str) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_phase2i_prepackage_gate_report(
    *,
    training_summary_json: str | Path,
    data_audit_json: str | Path,
    latent_necessity_audit_json: str | Path | None = None,
    expected_adapter_name: str | None = None,
    min_val_command_slot_accuracy: float = 0.85,
    expected_train_records: int = 1024,
    expected_val_records: int = 512,
    expected_command_candidate_feature_dim: int = 24,
    expected_pairwise_command_fusion: str = "residual",
    expected_pairwise_command_policy: str | None = None,
    expected_pairwise_command_max_length: int | None = None,
    expected_pairwise_command_top_k: int | None = None,
    expected_command_candidate_encoder: str | None = None,
    expected_latent_fusion: str = "additive",
) -> dict[str, Any]:
    summary = _load_json(training_summary_json)
    audit = _load_json(data_audit_json)
    latent_audit = _load_json(latent_necessity_audit_json) if latent_necessity_audit_json else None
    metrics = _final_val_metrics(summary)
    config = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
    head_config = summary.get("head_config", {}) if isinstance(summary.get("head_config"), dict) else {}
    summary_hashes = summary.get("effective_split_hashes", {})
    audit_hashes = audit.get("effective_split_hashes", {})
    summary_train_hash = _hash_for(summary_hashes, "train")
    summary_val_hash = _hash_for(summary_hashes, "val")
    audit_train_hash = _hash_for(audit_hashes, "train")
    audit_val_hash = _hash_for(audit_hashes, "val")
    val_command_slot_accuracy = metrics.get("command_slot_accuracy")
    val_command_slot_count = metrics.get("command_slot_count")
    source_overlap = summary.get("source_overlap_command_slot_baseline")
    slot_intent_distribution = summary.get("slot_intent_distribution")

    use_pairwise = bool(
        summary.get(
            "use_pairwise_command_reranker",
            config.get("use_pairwise_command_reranker", head_config.get("use_pairwise_command_reranker")),
        )
    )
    pairwise_fusion = summary.get(
        "pairwise_command_fusion",
        head_config.get("pairwise_command_fusion", config.get("pairwise_command_fusion")),
    )
    pairwise_policy = summary.get(
        "pairwise_command_policy",
        head_config.get("pairwise_command_policy", config.get("pairwise_command_policy")),
    )
    pairwise_max_length = summary.get(
        "pairwise_command_max_length",
        head_config.get("pairwise_command_max_length", config.get("pairwise_command_max_length")),
    )
    pairwise_top_k = summary.get(
        "pairwise_command_top_k",
        head_config.get("pairwise_command_top_k", config.get("pairwise_command_top_k")),
    )
    command_candidate_encoder = summary.get(
        "command_candidate_encoder",
        head_config.get("command_candidate_encoder", config.get("command_candidate_encoder")),
    )
    latent_fusion = config.get("latent_fusion", head_config.get("latent_fusion"))
    feature_dim = summary.get("command_candidate_feature_dim", head_config.get("command_candidate_feature_dim"))
    low_level_target = summary.get("low_level_qwen_calls_target")
    no_json_motor_target = summary.get("no_json_motor_target")

    checks = {
        "data_audit_passed": audit.get("passed") is True,
        "data_audit_effective_split_hashes_present": _nested_bool(
            audit,
            "checks",
            "phase2i_effective_split_hashes_present",
        ),
        "data_audit_train_val_command_intent_coverage": _nested_bool(
            audit,
            "checks",
            "phase2i_train_val_command_intent_coverage",
        ),
        "data_audit_val_target_command_coverage": _nested_bool(
            audit,
            "checks",
            "phase2i_head_val_target_command_coverage",
        ),
        "data_audit_external_v3_target_overlap_zero": _nested_bool(
            audit,
            "checks",
            "external_v3_has_no_phase2i_command_overlap",
        ),
        "summary_effective_split_hashes_present": bool(summary_train_hash and summary_val_hash),
        "summary_hashes_match_data_audit": bool(
            summary_train_hash
            and summary_val_hash
            and audit_train_hash
            and audit_val_hash
            and summary_train_hash == audit_train_hash
            and summary_val_hash == audit_val_hash
        ),
        "adapter_name_matches": (
            True
            if expected_adapter_name is None
            else summary.get("adapter_name") == expected_adapter_name
        ),
        "train_record_cap_recorded": config.get("max_train_records") == expected_train_records,
        "val_record_cap_recorded": config.get("max_val_records") == expected_val_records,
        "train_examples_match_expected": summary.get("train_examples") == expected_train_records,
        "val_examples_match_expected": summary.get("val_examples") == expected_val_records,
        "pairwise_enabled": use_pairwise is True,
        "pairwise_command_fusion_expected": pairwise_fusion == expected_pairwise_command_fusion,
        "pairwise_command_policy_recorded": isinstance(pairwise_policy, str)
        and pairwise_policy in {"all", "ambiguous_intent"},
        "pairwise_command_policy_expected": (
            True
            if expected_pairwise_command_policy is None
            else pairwise_policy == expected_pairwise_command_policy
        ),
        "pairwise_command_max_length_recorded": isinstance(pairwise_max_length, (int, float))
        and int(pairwise_max_length) > 0,
        "pairwise_command_max_length_expected": (
            True
            if expected_pairwise_command_max_length is None
            else isinstance(pairwise_max_length, (int, float))
            and int(pairwise_max_length) == expected_pairwise_command_max_length
        ),
        "pairwise_command_top_k_expected": (
            True
            if expected_pairwise_command_top_k is None
            else isinstance(pairwise_top_k, (int, float))
            and int(pairwise_top_k) == expected_pairwise_command_top_k
        ),
        "command_candidate_encoder_recorded": command_candidate_encoder in {"backbone", "features_only"},
        "command_candidate_encoder_expected": (
            True
            if expected_command_candidate_encoder is None
            else command_candidate_encoder == expected_command_candidate_encoder
        ),
        "latent_fusion_expected": latent_fusion == expected_latent_fusion,
        "command_candidate_feature_dim_expected": feature_dim == expected_command_candidate_feature_dim,
        "json_text_target_false": summary.get("json_text_target") is False,
        "no_json_motor_target_recorded": no_json_motor_target is True,
        "low_level_qwen_calls_target_zero": low_level_target == 0,
        "slot_intent_distribution_present": isinstance(slot_intent_distribution, dict)
        and isinstance(slot_intent_distribution.get("train"), dict)
        and isinstance(slot_intent_distribution.get("val"), dict),
        "source_overlap_baseline_present": isinstance(source_overlap, dict)
        and isinstance(source_overlap.get("train"), dict)
        and isinstance(source_overlap.get("val"), dict),
        "val_command_slot_count_positive": isinstance(val_command_slot_count, (int, float))
        and float(val_command_slot_count) > 0,
        "val_command_slot_accuracy_min": isinstance(val_command_slot_accuracy, (int, float))
        and float(val_command_slot_accuracy) >= min_val_command_slot_accuracy,
        "latent_necessity_audit_passed": (
            True
            if latent_audit is None
            else latent_audit.get("audit_family") == "phase2i_latent_necessity"
            and latent_audit.get("passed") is True
        ),
        "latent_necessity_architecture_identifiable": (
            True
            if latent_audit is None
            else _nested_bool(
                latent_audit,
                "checks",
                "nsi_latent_command_identity_available",
            )
        ),
    }
    return {
        "gate_family": "phase2i_prepackage_gate",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "val_command_slot_accuracy": (
                float(val_command_slot_accuracy)
                if isinstance(val_command_slot_accuracy, (int, float))
                else None
            ),
            "val_command_slot_count": (
                float(val_command_slot_count)
                if isinstance(val_command_slot_count, (int, float))
                else None
            ),
            "val_command_intent_accuracy": _nested_float(metrics, "command_intent_accuracy"),
            "train_examples": summary.get("train_examples"),
            "val_examples": summary.get("val_examples"),
            "use_pairwise_command_reranker": use_pairwise,
            "pairwise_command_fusion": pairwise_fusion,
            "pairwise_command_policy": pairwise_policy,
            "pairwise_command_max_length": (
                int(pairwise_max_length)
                if isinstance(pairwise_max_length, (int, float))
                else None
            ),
            "pairwise_command_top_k": (
                int(pairwise_top_k)
                if isinstance(pairwise_top_k, (int, float))
                else None
            ),
            "command_candidate_encoder": command_candidate_encoder,
            "latent_fusion": latent_fusion,
            "command_candidate_feature_dim": feature_dim,
            "low_level_qwen_calls_target": low_level_target,
            "json_text_target": summary.get("json_text_target"),
        },
        "effective_split_hashes": {
            "summary_train": summary_train_hash,
            "summary_val": summary_val_hash,
            "data_audit_train": audit_train_hash,
            "data_audit_val": audit_val_hash,
        },
        "inputs": {
            "training_summary_json": str(Path(training_summary_json)),
            "data_audit_json": str(Path(data_audit_json)),
            "expected_adapter_name": expected_adapter_name,
            "min_val_command_slot_accuracy": min_val_command_slot_accuracy,
            "expected_train_records": expected_train_records,
            "expected_val_records": expected_val_records,
            "expected_command_candidate_feature_dim": expected_command_candidate_feature_dim,
            "expected_pairwise_command_fusion": expected_pairwise_command_fusion,
            "expected_pairwise_command_policy": expected_pairwise_command_policy,
            "expected_pairwise_command_max_length": expected_pairwise_command_max_length,
            "expected_pairwise_command_top_k": expected_pairwise_command_top_k,
            "expected_command_candidate_encoder": expected_command_candidate_encoder,
            "expected_latent_fusion": expected_latent_fusion,
            "latent_necessity_audit_json": (
                str(Path(latent_necessity_audit_json)) if latent_necessity_audit_json else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase2I pre-package gates.")
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--data-audit-json", required=True)
    parser.add_argument("--latent-necessity-audit-json")
    parser.add_argument("--expected-adapter-name")
    parser.add_argument("--min-val-command-slot-accuracy", type=float, default=0.85)
    parser.add_argument("--expected-train-records", type=int, default=1024)
    parser.add_argument("--expected-val-records", type=int, default=512)
    parser.add_argument("--expected-command-candidate-feature-dim", type=int, default=24)
    parser.add_argument("--expected-pairwise-command-fusion", default="residual")
    parser.add_argument("--expected-pairwise-command-policy")
    parser.add_argument("--expected-pairwise-command-max-length", type=int)
    parser.add_argument("--expected-pairwise-command-top-k", type=int)
    parser.add_argument("--expected-command-candidate-encoder")
    parser.add_argument("--expected-latent-fusion", default="additive")
    parser.add_argument("--output-json")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()
    report = build_phase2i_prepackage_gate_report(
        training_summary_json=args.training_summary_json,
        data_audit_json=args.data_audit_json,
        latent_necessity_audit_json=args.latent_necessity_audit_json,
        expected_adapter_name=args.expected_adapter_name,
        min_val_command_slot_accuracy=args.min_val_command_slot_accuracy,
        expected_train_records=args.expected_train_records,
        expected_val_records=args.expected_val_records,
        expected_command_candidate_feature_dim=args.expected_command_candidate_feature_dim,
        expected_pairwise_command_fusion=args.expected_pairwise_command_fusion,
        expected_pairwise_command_policy=args.expected_pairwise_command_policy,
        expected_pairwise_command_max_length=args.expected_pairwise_command_max_length,
        expected_pairwise_command_top_k=args.expected_pairwise_command_top_k,
        expected_command_candidate_encoder=args.expected_command_candidate_encoder,
        expected_latent_fusion=args.expected_latent_fusion,
    )
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
