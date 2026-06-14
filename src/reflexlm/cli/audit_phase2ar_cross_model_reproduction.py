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


def _rate(summary: dict[str, Any]) -> float:
    if "success_rate" in summary:
        return float(summary.get("success_rate") or 0.0)
    rows = float(summary.get("rows") or 0)
    successes = float(summary.get("successes") or 0)
    return successes / rows if rows else 0.0


def _config(summary: dict[str, Any]) -> dict[str, Any]:
    config = summary.get("config")
    return config if isinstance(config, dict) else {}


def _split_hashes(summary: dict[str, Any]) -> dict[str, Any]:
    hashes = summary.get("effective_split_hashes")
    return hashes if isinstance(hashes, dict) else {}


def audit_phase2ar_cross_model_reproduction(
    *,
    primary_execution_summary_json: str | Path,
    cross_model_execution_summary_json: str | Path,
    primary_training_summary_json: str | Path,
    cross_model_training_summary_json: str | Path,
    min_success_rate: float = 1.0,
) -> dict[str, Any]:
    primary_execution = _read_json(primary_execution_summary_json)
    cross_execution = _read_json(cross_model_execution_summary_json)
    primary_training = _read_json(primary_training_summary_json)
    cross_training = _read_json(cross_model_training_summary_json)
    primary_config = _config(primary_training)
    cross_config = _config(cross_training)
    primary_model = primary_config.get("base_model_name")
    cross_model = cross_config.get("base_model_name")
    comparable_keys = [
        "quantization",
        "learning_rate",
        "epochs",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "max_length",
        "lora_rank",
        "lora_alpha",
        "command_candidate_encoder",
        "latent_fusion",
        "open_repair_heads_enabled",
        "seed",
    ]
    config_mismatches = {
        key: {"primary": primary_config.get(key), "cross_model": cross_config.get(key)}
        for key in comparable_keys
        if primary_config.get(key) != cross_config.get(key)
    }
    primary_rate = _rate(primary_execution)
    cross_rate = _rate(cross_execution)
    checks = {
        "primary_success_rate_met": primary_rate >= min_success_rate,
        "cross_model_success_rate_met": cross_rate >= min_success_rate,
        "row_counts_match": primary_execution.get("rows") == cross_execution.get("rows"),
        "patch_mode_matches": primary_execution.get("patch_mode")
        == cross_execution.get("patch_mode"),
        "claim_boundary_matches": primary_execution.get("claim_boundary")
        == cross_execution.get("claim_boundary"),
        "base_models_differ": primary_model != cross_model,
        "split_hashes_match": _split_hashes(primary_training) == _split_hashes(cross_training),
        "training_contract_matches_except_model": not config_mismatches,
    }
    passed = all(checks.values())
    return {
        "artifact_family": "phase2ar_cross_model_reproduction_audit",
        "passed": passed,
        "claim_boundary": (
            "Phase2AR cross-model audit supports bounded same-family cross-model "
            "reproduction on nonsealed Phase2AR only. It does not support sealed "
            "cross-model transfer, production autonomy, or open-ended repair."
        ),
        "checks": checks,
        "metrics": {
            "primary_model": primary_model,
            "cross_model": cross_model,
            "primary_success_rate": primary_rate,
            "cross_model_success_rate": cross_rate,
            "row_count": primary_execution.get("rows"),
            "config_mismatches": config_mismatches,
            "primary_seed": primary_config.get("seed"),
            "cross_model_seed": cross_config.get("seed"),
            "split_hashes": _split_hashes(primary_training),
        },
        "supported_claims": ["phase2ar_qwen7b_to_qwen3b_same_family_reproduction_supported"]
        if passed
        else [],
        "unsupported_claims": [
            "sealed_cross_model_transfer",
            "epoch_making_architecture",
            "freeform_patch_generation",
            "open_ended_debugging_generalization",
            "production_autonomy",
        ],
        "blocked_actions": [
            "do_not_claim_sealed_cross_model_transfer_from_nonsealed_phase2ar",
            "do_not_claim_epoch_making_architecture_from_same_family_cross_model_only",
        ],
        "inputs": {
            "primary_execution_summary_json": str(Path(primary_execution_summary_json)),
            "cross_model_execution_summary_json": str(Path(cross_model_execution_summary_json)),
            "primary_training_summary_json": str(Path(primary_training_summary_json)),
            "cross_model_training_summary_json": str(Path(cross_model_training_summary_json)),
            "min_success_rate": min_success_rate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AR cross-model reproduction.")
    parser.add_argument("--primary-execution-summary-json", required=True)
    parser.add_argument("--cross-model-execution-summary-json", required=True)
    parser.add_argument("--primary-training-summary-json", required=True)
    parser.add_argument("--cross-model-training-summary-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    args = parser.parse_args()
    report = audit_phase2ar_cross_model_reproduction(
        primary_execution_summary_json=args.primary_execution_summary_json,
        cross_model_execution_summary_json=args.cross_model_execution_summary_json,
        primary_training_summary_json=args.primary_training_summary_json,
        cross_model_training_summary_json=args.cross_model_training_summary_json,
        min_success_rate=args.min_success_rate,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
