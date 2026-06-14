from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _latest_val_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history")
    if isinstance(history, list) and history:
        latest = history[-1]
        if isinstance(latest, dict):
            return _dict(latest.get("val_metrics"))
    return _dict(summary.get("val_metrics"))


def _eval_accuracy(payload: dict[str, Any]) -> float | None:
    return _number(_dict(payload.get("eval_metrics")).get("command_slot_accuracy"))


def _eval_count(payload: dict[str, Any]) -> float | None:
    return _number(_dict(payload.get("eval_metrics")).get("command_slot_count"))


def _baseline_accuracy(payload: dict[str, Any], split: str) -> float | None:
    baselines = _dict(payload.get("source_overlap_command_slot_baseline"))
    direct = _number(_dict(baselines.get(split)).get("accuracy"))
    if direct is not None:
        return direct
    if len(baselines) == 1:
        return _number(_dict(next(iter(baselines.values()))).get("accuracy"))
    return None


def _encoded_candidates(payload: dict[str, Any], split: str) -> float | None:
    encodings = _dict(payload.get("pairwise_candidate_encoding"))
    direct = _number(_dict(encodings.get(split)).get("pairwise_scored_candidates"))
    if direct is not None:
        return direct
    if len(encodings) == 1:
        return _number(_dict(next(iter(encodings.values()))).get("pairwise_scored_candidates"))
    return None


def _split_hash(payload: dict[str, Any], split: str) -> str | None:
    hashes = _dict(payload.get("effective_split_hashes"))
    direct = hashes.get(split)
    if isinstance(direct, str) and direct:
        return direct
    prefix = f"{split}_"
    matches = [value for key, value in hashes.items() if key.startswith(prefix)]
    if len(matches) == 1 and isinstance(matches[0], str) and matches[0]:
        return matches[0]
    if len(hashes) == 1:
        only = next(iter(hashes.values()))
        if isinstance(only, str) and only:
            return only
    return None


def build_phase2ah_identity_contrast_postflight(
    *,
    split_manifest_json: str | Path,
    head_manifest_json: str | Path,
    holdout_head_manifest_json: str | Path,
    training_summary_json: str | Path,
    holdout_eval_json: str | Path,
    old_adapter_holdout_eval_json: str | Path,
    adapter_dir: str | Path | None = None,
    min_val_accuracy: float = 0.85,
    min_holdout_accuracy: float = 0.85,
    max_old_adapter_accuracy: float = 0.25,
    min_val_rows: int = 32,
    min_holdout_rows: int = 48,
    expected_source_overlap_accuracy: float = 1.0,
) -> dict[str, Any]:
    split_manifest = _read_json(split_manifest_json)
    head_manifest = _read_json(head_manifest_json)
    holdout_head_manifest = _read_json(holdout_head_manifest_json)
    summary = _read_json(training_summary_json)
    holdout_eval = _read_json(holdout_eval_json)
    old_eval = _read_json(old_adapter_holdout_eval_json)

    val_metrics = _latest_val_metrics(summary)
    val_accuracy = _number(val_metrics.get("command_slot_accuracy"))
    val_count = _number(val_metrics.get("command_slot_count"))
    holdout_accuracy = _eval_accuracy(holdout_eval)
    holdout_count = _eval_count(holdout_eval)
    old_holdout_accuracy = _eval_accuracy(old_eval)
    holdout_source = _baseline_accuracy(holdout_eval, "phase2ah_holdout")
    old_holdout_source = _baseline_accuracy(old_eval, "phase2ah_holdout")

    summary_hashes = _dict(summary.get("effective_split_hashes"))
    head_splits = _dict(head_manifest.get("splits"))
    holdout_head_splits = _dict(holdout_head_manifest.get("splits"))
    head_train_hash = _dict(head_splits.get("train")).get("sha256")
    head_val_hash = _dict(head_splits.get("val")).get("sha256")
    holdout_head_val_hash = _dict(holdout_head_splits.get("val")).get("sha256")
    holdout_eval_hash = _split_hash(holdout_eval, "phase2c_head_phase2ah_holdout")
    old_holdout_eval_hash = _split_hash(old_eval, "phase2c_head_phase2ah_holdout")
    adapter_path = Path(adapter_dir or summary.get("adapter_output_dir") or "")
    adapter_files_present = (
        bool(str(adapter_path))
        and (adapter_path / "native_heads.pt").exists()
        and (adapter_path / "head_config.json").exists()
    )

    split_counts = _dict(split_manifest.get("split_counts"))
    bucket_counts = _dict(split_manifest.get("bucket_counts"))
    val_buckets = _dict(bucket_counts.get("val"))
    holdout_buckets = _dict(bucket_counts.get("holdout"))
    expected_val_rows = _number(split_counts.get("val"))
    expected_holdout_rows = _number(split_counts.get("holdout"))
    source_overlap_ceiling = (
        isinstance(holdout_source, float)
        and abs(holdout_source - expected_source_overlap_accuracy) <= 1e-9
        and isinstance(old_holdout_source, float)
        and abs(old_holdout_source - expected_source_overlap_accuracy) <= 1e-9
    )

    checks = {
        "split_manifest_passed": split_manifest.get("passed") is True,
        "sealed_feedback_absent": _dict(split_manifest.get("checks")).get(
            "sealed_feedback_absent"
        )
        is True,
        "claim_boundary_blocks_natural_trace_claim": split_manifest.get(
            "claim_bearing_natural_trace_evidence"
        )
        is False,
        "unsupported_claims_listed": all(
            claim in set(split_manifest.get("unsupported_claims") or [])
            for claim in [
                "sealed_transfer",
                "production_autonomy",
                "open_ended_debugging_generalization",
                "epoch_making_architecture",
            ]
        ),
        "val_bucket_is_source_correct_identity_wrong": val_buckets
        == {"source_1_identity_0": int(expected_val_rows or -1)},
        "holdout_bucket_is_source_correct_identity_wrong": holdout_buckets
        == {"source_1_identity_0": int(expected_holdout_rows or -1)},
        "training_hashes_present": bool(
            summary_hashes.get("phase2c_head_train")
            and summary_hashes.get("phase2c_head_val")
        ),
        "training_hashes_match_head_manifest": summary_hashes.get(
            "phase2c_head_train"
        )
        == head_train_hash
        and summary_hashes.get("phase2c_head_val") == head_val_hash,
        "holdout_hash_matches_eval_rows": holdout_eval_hash
        == holdout_eval.get("eval_rows_hash")
        == holdout_head_val_hash,
        "old_holdout_hash_matches_eval_rows": old_holdout_eval_hash
        == old_eval.get("eval_rows_hash")
        == holdout_head_val_hash,
        "val_command_slot_count_min": isinstance(val_count, float)
        and val_count >= min_val_rows,
        "holdout_command_slot_count_min": isinstance(holdout_count, float)
        and holdout_count >= min_holdout_rows,
        "holdout_count_matches_manifest": isinstance(holdout_count, float)
        and isinstance(expected_holdout_rows, float)
        and holdout_count == expected_holdout_rows,
        "val_command_slot_accuracy_min": isinstance(val_accuracy, float)
        and val_accuracy >= min_val_accuracy,
        "holdout_command_slot_accuracy_min": isinstance(holdout_accuracy, float)
        and holdout_accuracy >= min_holdout_accuracy,
        "old_adapter_degraded_by_wrong_identity": isinstance(old_holdout_accuracy, float)
        and old_holdout_accuracy <= max_old_adapter_accuracy,
        "new_adapter_beats_old_adapter": isinstance(holdout_accuracy, float)
        and isinstance(old_holdout_accuracy, float)
        and holdout_accuracy > old_holdout_accuracy,
        "source_overlap_ceiling_expected_for_pressure_split": source_overlap_ceiling,
        "pairwise_disabled": summary.get("use_pairwise_command_reranker") is False
        and holdout_eval.get("use_pairwise_command_reranker") is False,
        "pairwise_encoded_candidates_zero": (_encoded_candidates(summary, "train") == 0.0)
        and (_encoded_candidates(summary, "val") == 0.0)
        and (_encoded_candidates(holdout_eval, "phase2ah_holdout") == 0.0),
        "no_json_motor_target": summary.get("no_json_motor_target") is True
        and holdout_eval.get("no_json_motor_target") is True,
        "low_level_qwen_calls_target_zero": summary.get("low_level_qwen_calls_target")
        == 0
        and holdout_eval.get("low_level_qwen_calls_target") == 0,
        "trained_and_evaluated_on_cuda": _dict(summary.get("config")).get("device")
        == "cuda"
        and str(holdout_eval.get("device") or "").startswith("cuda"),
        "adapter_files_present": adapter_files_present,
    }
    passed = all(checks.values())
    full_minus_old = (
        holdout_accuracy - old_holdout_accuracy
        if isinstance(holdout_accuracy, float) and isinstance(old_holdout_accuracy, float)
        else None
    )
    model_minus_source = (
        holdout_accuracy - holdout_source
        if isinstance(holdout_accuracy, float) and isinstance(holdout_source, float)
        else None
    )
    blocked_actions = [
        "do_not_package_phase2ah_from_adversarial_smoke",
        "do_not_run_sealed_phase2ah_from_adversarial_smoke",
        "do_not_claim_natural_public_trace_identity_failure_distribution",
        "do_not_claim_epoch_making_architecture",
        "do_not_claim_open_ended_debugging_generalization",
        "do_not_claim_production_autonomy",
    ]
    if not passed:
        blocked_actions.insert(0, "do_not_scale_phase2ah_training_until_failure_audit")

    return {
        "artifact_family": "phase2ah_identity_contrast_pressure_postflight",
        "passed": passed,
        "ready_for_package": False,
        "ready_for_sealed_eval": False,
        "claim_bearing_natural_trace_evidence": False,
        "claim_bearing_mechanism_evidence": False,
        "adversarial_identity_contrast_pressure_evidence": passed,
        "checks": checks,
        "metrics": {
            "train_examples": summary.get("train_examples"),
            "val_examples": summary.get("val_examples"),
            "holdout_examples": holdout_eval.get("eval_examples"),
            "val_command_slot_accuracy": val_accuracy,
            "holdout_command_slot_accuracy": holdout_accuracy,
            "old_adapter_holdout_command_slot_accuracy": old_holdout_accuracy,
            "holdout_source_overlap_accuracy": holdout_source,
            "holdout_model_minus_source_overlap_accuracy": model_minus_source,
            "holdout_new_minus_old_adapter_accuracy": full_minus_old,
            "val_command_slot_count": val_count,
            "holdout_command_slot_count": holdout_count,
        },
        "thresholds": {
            "min_val_accuracy": min_val_accuracy,
            "min_holdout_accuracy": min_holdout_accuracy,
            "max_old_adapter_accuracy": max_old_adapter_accuracy,
            "min_val_rows": min_val_rows,
            "min_holdout_rows": min_holdout_rows,
            "expected_source_overlap_accuracy": expected_source_overlap_accuracy,
        },
        "blocked_actions": sorted(set(blocked_actions)),
        "allowed_next_action": (
            "design_natural_repo_origin_disjoint_source_1_identity_0_or_measured_control_benchmark"
            if passed
            else "freeze_phase2ah_postflight_failure_and_fix_nonsealed_design"
        ),
        "interpretation": (
            "Phase2AH intentionally sets source-overlap to ceiling while making structured "
            "candidate identity wrong. Passing this postflight means the new adapter learned "
            "to resist an adversarial identity sidecar that misdirects an older adapter. It is "
            "not evidence that the architecture beats source-overlap on natural public traces."
        ),
        "claim_boundary": (
            "Non-sealed adversarial identity-contrast pressure only. No sealed transfer, "
            "production autonomy, open-ended debugging generalization, natural identity-failure "
            "distribution, or epoch-making architecture claim is supported by this artifact."
        ),
        "inputs": {
            "split_manifest_json": str(Path(split_manifest_json)),
            "head_manifest_json": str(Path(head_manifest_json)),
            "holdout_head_manifest_json": str(Path(holdout_head_manifest_json)),
            "training_summary_json": str(Path(training_summary_json)),
            "holdout_eval_json": str(Path(holdout_eval_json)),
            "old_adapter_holdout_eval_json": str(Path(old_adapter_holdout_eval_json)),
            "adapter_dir": str(adapter_path) if str(adapter_path) else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Phase2AH identity-contrast pressure postflight gates."
    )
    parser.add_argument("--split-manifest-json", required=True)
    parser.add_argument("--head-manifest-json", required=True)
    parser.add_argument("--holdout-head-manifest-json", required=True)
    parser.add_argument("--training-summary-json", required=True)
    parser.add_argument("--holdout-eval-json", required=True)
    parser.add_argument("--old-adapter-holdout-eval-json", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--min-val-accuracy", type=float, default=0.85)
    parser.add_argument("--min-holdout-accuracy", type=float, default=0.85)
    parser.add_argument("--max-old-adapter-accuracy", type=float, default=0.25)
    parser.add_argument("--min-val-rows", type=int, default=32)
    parser.add_argument("--min-holdout-rows", type=int, default=48)
    parser.add_argument("--expected-source-overlap-accuracy", type=float, default=1.0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    report = build_phase2ah_identity_contrast_postflight(
        split_manifest_json=args.split_manifest_json,
        head_manifest_json=args.head_manifest_json,
        holdout_head_manifest_json=args.holdout_head_manifest_json,
        training_summary_json=args.training_summary_json,
        holdout_eval_json=args.holdout_eval_json,
        old_adapter_holdout_eval_json=args.old_adapter_holdout_eval_json,
        adapter_dir=args.adapter_dir,
        min_val_accuracy=args.min_val_accuracy,
        min_holdout_accuracy=args.min_holdout_accuracy,
        max_old_adapter_accuracy=args.max_old_adapter_accuracy,
        min_val_rows=args.min_val_rows,
        min_holdout_rows=args.min_holdout_rows,
        expected_source_overlap_accuracy=args.expected_source_overlap_accuracy,
    )
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"] and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
