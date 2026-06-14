from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def _rate(summary: dict[str, Any]) -> float:
    if "success_rate" in summary:
        return float(summary.get("success_rate") or 0.0)
    rows = float(summary.get("rows") or 0)
    successes = float(summary.get("successes") or 0)
    return successes / rows if rows else 0.0


def _config(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    config = summary.get("config")
    return config if isinstance(config, dict) else {}


def _split_hashes(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    hashes = summary.get("effective_split_hashes")
    return hashes if isinstance(hashes, dict) else {}


def _same_training_contract(
    primary_training: dict[str, Any] | None,
    reproduction_training: dict[str, Any] | None,
) -> dict[str, Any]:
    primary_config = _config(primary_training)
    reproduction_config = _config(reproduction_training)
    keys = [
        "base_model_name",
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
    ]
    mismatches = {
        key: {
            "primary": primary_config.get(key),
            "reproduction": reproduction_config.get(key),
        }
        for key in keys
        if primary_config.get(key) != reproduction_config.get(key)
    }
    primary_seed = primary_config.get("seed")
    reproduction_seed = reproduction_config.get("seed")
    return {
        "present": bool(primary_config and reproduction_config),
        "same_contract_except_seed_and_names": not mismatches,
        "config_mismatches": mismatches,
        "primary_seed": primary_seed,
        "reproduction_seed": reproduction_seed,
        "seed_changed": primary_seed is not None
        and reproduction_seed is not None
        and primary_seed != reproduction_seed,
        "split_hashes_match": _split_hashes(primary_training)
        == _split_hashes(reproduction_training),
    }


def audit_phase2ar_reproduction(
    *,
    primary_summary_json: str | Path,
    reproduction_summary_json: str | Path,
    reproduction_results_jsonl: str | Path,
    primary_training_summary_json: str | Path | None = None,
    reproduction_training_summary_json: str | Path | None = None,
    min_reproduction_success_rate: float = 1.0,
) -> dict[str, Any]:
    primary = _read_json(primary_summary_json)
    reproduction = _read_json(reproduction_summary_json)
    primary_training = (
        _read_json(primary_training_summary_json)
        if primary_training_summary_json is not None
        else None
    )
    reproduction_training = (
        _read_json(reproduction_training_summary_json)
        if reproduction_training_summary_json is not None
        else None
    )
    training_contract = _same_training_contract(primary_training, reproduction_training)
    rows = _read_jsonl(reproduction_results_jsonl)
    patch_source_counts = Counter(str(row.get("patch_source")) for row in rows)
    success_counts = Counter(str(row.get("success")) for row in rows)
    rollback_head_counts = Counter()
    failure_reasons = Counter()
    for row in rows:
        outputs = row.get("policy_open_repair_outputs")
        if isinstance(outputs, dict):
            rollback_head_counts[str(outputs.get("rollback_safety"))] += 1
            if outputs.get("rollback_safety") != 1:
                failure_reasons["rollback_safety_head_not_authorized"] += 1
        if row.get("patch_source") == "package_runtime_no_patch_authorized":
            failure_reasons["patch_not_authorized"] += 1
        if row.get("success") is not True:
            failure_reasons["row_failed"] += 1

    primary_rate = _rate(primary)
    reproduction_rate = _rate(reproduction)
    checks = {
        "primary_success_rate_perfect": primary_rate >= 1.0,
        "row_counts_match": primary.get("rows") == reproduction.get("rows") == len(rows),
        "reproduction_success_rate_met": reproduction_rate >= min_reproduction_success_rate,
        "reproduction_patch_mode_matches": reproduction.get("patch_mode")
        == primary.get("patch_mode"),
        "reproduction_boundary_matches": reproduction.get("claim_boundary")
        == primary.get("claim_boundary"),
    }
    if training_contract["present"]:
        checks["training_contract_matches_except_seed_and_names"] = bool(
            training_contract["same_contract_except_seed_and_names"]
        )
        checks["effective_split_hashes_match"] = bool(training_contract["split_hashes_match"])
    passed = all(checks.values())
    supported_claims = ["phase2ar_cross_package_reproduction_supported"] if passed else []
    if passed and training_contract["present"] and training_contract["seed_changed"]:
        supported_claims.append("phase2ar_two_seed_reproduction_smoke_supported")
    unsupported_claims = [
        "phase2ar_cross_package_reproduction_supported",
        "multi_seed_reproduction_3plus",
        "cross_model_reproduction",
        "epoch_making_architecture",
    ]
    if passed:
        unsupported_claims = [
            claim
            for claim in unsupported_claims
            if claim != "phase2ar_cross_package_reproduction_supported"
        ]
    if passed and "phase2ar_two_seed_reproduction_smoke_supported" in supported_claims:
        blocked_actions = [
            "do_not_claim_cross_model_reproduction_from_two_seed_single_model_run",
            "do_not_claim_robust_multi_seed_reproduction_until_3plus_seeds",
        ]
    elif passed:
        blocked_actions = [
            "do_not_claim_multi_seed_or_cross_model_reproduction_from_cross_package_only"
        ]
    else:
        blocked_actions = [
            "do_not_claim_cross_package_reproduction_from_failed_run",
            "do_not_claim_multi_seed_or_cross_model_reproduction",
        ]
    return {
        "artifact_family": "phase2ar_reproduction_audit",
        "passed": passed,
        "claim_boundary": (
            "Phase2AR reproduction audit; failed reproductions are negative evidence "
            "and must not be interpreted as cross-model or multi-seed support."
        ),
        "checks": checks,
        "metrics": {
            "primary_success_rate": primary_rate,
            "reproduction_success_rate": reproduction_rate,
            "row_count": len(rows),
            "training_contract": training_contract,
            "patch_source_counts": dict(sorted(patch_source_counts.items())),
            "success_counts": dict(sorted(success_counts.items())),
            "rollback_safety_head_counts": dict(sorted(rollback_head_counts.items())),
            "failure_reasons": dict(sorted(failure_reasons.items())),
        },
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "blocked_actions": blocked_actions,
        "inputs": {
            "primary_summary_json": str(Path(primary_summary_json)),
            "reproduction_summary_json": str(Path(reproduction_summary_json)),
            "reproduction_results_jsonl": str(Path(reproduction_results_jsonl)),
            "primary_training_summary_json": str(Path(primary_training_summary_json))
            if primary_training_summary_json is not None
            else None,
            "reproduction_training_summary_json": str(
                Path(reproduction_training_summary_json)
            )
            if reproduction_training_summary_json is not None
            else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase2AR reproduction evidence.")
    parser.add_argument("--primary-summary-json", required=True)
    parser.add_argument("--reproduction-summary-json", required=True)
    parser.add_argument("--reproduction-results-jsonl", required=True)
    parser.add_argument("--primary-training-summary-json")
    parser.add_argument("--reproduction-training-summary-json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--min-reproduction-success-rate", type=float, default=1.0)
    args = parser.parse_args()
    report = audit_phase2ar_reproduction(
        primary_summary_json=args.primary_summary_json,
        reproduction_summary_json=args.reproduction_summary_json,
        reproduction_results_jsonl=args.reproduction_results_jsonl,
        primary_training_summary_json=args.primary_training_summary_json,
        reproduction_training_summary_json=args.reproduction_training_summary_json,
        min_reproduction_success_rate=args.min_reproduction_success_rate,
    )
    _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
